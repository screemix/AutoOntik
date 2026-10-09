"""
Wikontic-style, ontology-linked, PER-ESSAY extraction over MINE
==================================================================================

Motivation: kg-gen's own MINE evaluation (experiments/MINE/evaluation.py in
the kg-gen repo) builds and evaluates against one SEPARATE graph PER ESSAY --
each essay's queries are retrieved only from that essay's own graph, never a
combined one. This project's own run_judge_eval.py instead evaluates every
essay's queries against ONE graph built from the WHOLE corpus (consistent
with AutoOntic's actual design: a single globally-canonicalized ontology +
KG), which is not apples-to-apples with kg-gen's own reported numbers.

This script closes that gap from the other direction: it takes an ALREADY-
BUILT AutoOntic ontology (type_dedup.pkl / relation_dedup.pkl /
hierarchy_induction.pkl from one pipeline run -- the schema: canonical
types, canonical relations, the induced is-a hierarchy) and, for each essay
INDEPENDENTLY, runs a four-step Wikontic-like extraction against that fixed
schema:

  1. Triplet extraction       -- LLMTripletExtractor.extract_triplets_from_text(),
                                  same as Step 0 everywhere else in this repo.
  2. Entity type linking      -- each extracted subject_type/object_type is
                                  resolved onto the PRE-BUILT type vocabulary:
                                  retrieve candidate types by embedding
                                  similarity, skip the LLM call entirely if
                                  the raw label already IS one of the
                                  candidates (normalized exact match), else
                                  ask the LLM to pick the best candidate (or
                                  none). An unresolved type drops the triplet
                                  (logged, not silently discarded) --
                                  mirrors Wikontic's StructuredInferenceWithDB.
                                  _refine_entity_types / ontology_filtered_triplets.
  3. Relation linking         -- same pattern, against the pre-built relation
                                  vocabulary (Wikontic's _refine_relation,
                                  simplified: this version does not narrow
                                  candidates by domain/range compatibility or
                                  reject on a hierarchy-constraint check the
                                  way Wikontic's _validate_backbone does --
                                  see the module docstring's "Simplifications"
                                  note below).
  4. Entity deduplication     -- WITHIN THIS ESSAY ONLY (never across essays):
                                  a new mention is compared against already-
                                  registered entities that share the same
                                  immediate hierarchy parent (entity_dedup.py's
                                  own _parent_key gate), narrowed further by
                                  embedding similarity, and merged via the
                                  same cluster-verification LLM call
                                  entity_dedup.py itself uses
                                  (verify_cluster_with_llm(..., 'entity')).

Each essay's resolved triplets are written out as one kg-gen-compatible
Graph JSON file (same schema as build_kg_graph.py's single combined output),
so scripts/mine_benchmark/run_judge_eval_per_essay.py can evaluate each
essay's own queries against its own graph only -- never leaking into or
out of any other essay.

Simplifications versus Wikontic's own StructuredInferenceWithDB (kept
deliberately, to bound scope -- noted here rather than silently differing):
  - No domain/range-aware relation-candidate narrowing (Wikontic's
    retrieve_properties_for_entity_type), and no backbone-validation
    rejection step (Wikontic's _validate_backbone). This project's own
    constraints.pkl encodes the same kind of domain/range evidence and
    could narrow candidates the same way; left out of this first version
    to keep the per-triplet LLM-call budget and failure surface smaller.
  - Entity name candidates are also filtered by embedding similarity
    (a floor) before ever reaching the LLM, which Wikontic's own
    retrieve_entity_by_type does not do -- added for the same reason
    sandbox.py's diff_merge_candidates added one: an unfiltered same-type
    candidate list is frequently mostly irrelevant noise.

Usage:
    python -m scripts.mine_benchmark.ontology_linked_extraction \\
        --ontology-dir output-2/mine/checkpoints/run_13 \\
        --essays data/mine/essays.json \\
        --output-dir output/mine/ontology_linked_graphs \\
        --model Openai/Gpt-oss-120b --base-url "$AIRI_BASE_URL" \\
        --api-key-env AIRI_KEY --proxy-key-env PROXY_URL \\
        --limit 5          # pilot on the first 5 essays before the full 101
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.ontodisco.entity_dedup import _parent_key
from src.ontodisco.hierarchy_induction import TypeHierarchy
from src.ontodisco.utils.dedup_base import ContrieverEmbedder, _parse_cluster_response, normalize_label
from src.ontodisco.utils.openai_utils import LLMTripletExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


class OntologyLinker:
    """Holds one fixed ontology (types, relations, hierarchy) plus the
    embedder needed to retrieve candidates against it. Stateless across
    essays -- all per-essay entity-dedup state lives in a fresh
    _EssayState the caller creates per essay, never shared or carried over,
    since entity dedup here is deliberately per-essay only."""

    def __init__(self, ontology_dir: Path, embedder: ContrieverEmbedder, top_k: int = 10):
        type_vocab = _load_pickle(ontology_dir / "type_dedup.pkl")
        relation_vocab = _load_pickle(ontology_dir / "relation_dedup.pkl")
        hierarchy_result = _load_pickle(ontology_dir / "hierarchy_induction.pkl")

        self.hierarchy: TypeHierarchy = hierarchy_result.hierarchy
        self.embedder = embedder
        self.top_k = top_k

        # One label per type_id/relation_id (first one wins on a duplicate
        # canonical label, which shouldn't happen post-dedup but isn't worth
        # raising over here).
        self.type_label_by_id = {tid: t.canonical_label for tid, t in type_vocab.items.items()}
        self.relation_label_by_id = {rid: r.canonical_label for rid, r in relation_vocab.items.items()}

        self.type_ids = list(self.type_label_by_id.keys())
        self.type_labels = [self.type_label_by_id[tid] for tid in self.type_ids]
        self.relation_ids = list(self.relation_label_by_id.keys())
        self.relation_labels = [self.relation_label_by_id[rid] for rid in self.relation_ids]

        logger.info("Embedding %d canonical types and %d canonical relations for candidate retrieval...",
                    len(self.type_labels), len(self.relation_labels))
        self.type_embeddings = embedder.embed(self.type_labels) if self.type_labels else np.zeros((0, 768))
        self.relation_embeddings = embedder.embed(self.relation_labels) if self.relation_labels else np.zeros((0, 768))

        self.type_label_to_id = {}
        for tid, label in self.type_label_by_id.items():
            self.type_label_to_id.setdefault(normalize_label(label), tid)
        self.relation_label_to_id = {}
        for rid, label in self.relation_label_by_id.items():
            self.relation_label_to_id.setdefault(normalize_label(label), rid)

    def _top_k_candidates(self, query_embedding: np.ndarray, pool_embeddings: np.ndarray,
                           pool_labels: list[str]) -> list[str]:
        if pool_embeddings.shape[0] == 0:
            return []
        sims = cosine_similarity(query_embedding.reshape(1, -1), pool_embeddings)[0]
        k = min(self.top_k, len(pool_labels))
        top_idx = np.argsort(-sims)[:k]
        return [pool_labels[i] for i in top_idx]

    def link_entity_type(self, llm, raw_type: str, entity_name: str, sentence: str,
                          embedding: Optional[np.ndarray] = None) -> Optional[str]:
        """Returns a resolved type_id, or None if nothing in the ontology
        genuinely fits (caller drops the triplet)."""
        if embedding is None:
            embedding = self.embedder.embed([raw_type])[0]
        candidates = self._top_k_candidates(embedding, self.type_embeddings, self.type_labels)
        if not candidates:
            return None
        norm_raw = normalize_label(raw_type)
        for c in candidates:
            if normalize_label(c) == norm_raw:
                return self.type_label_to_id[norm_raw]
        result = llm.link_entity_type(entity_name=entity_name, sentence=sentence, candidate_labels=candidates)
        picked = result.get("type")
        return self.type_label_to_id.get(normalize_label(picked)) if picked else None

    def link_relation(self, llm, raw_relation: str, subject_label: str, object_label: str,
                       sentence: str, embedding: Optional[np.ndarray] = None) -> Optional[str]:
        """Returns a resolved relation_id, or None if nothing fits."""
        if embedding is None:
            embedding = self.embedder.embed([raw_relation])[0]
        candidates = self._top_k_candidates(embedding, self.relation_embeddings, self.relation_labels)
        if not candidates:
            return None
        norm_raw = normalize_label(raw_relation)
        for c in candidates:
            if normalize_label(c) == norm_raw:
                return self.relation_label_to_id[norm_raw]
        result = llm.link_relation(
            relation_text=raw_relation, subject_label=subject_label, object_label=object_label,
            sentence=sentence, candidate_labels=candidates,
        )
        picked = result.get("relation")
        return self.relation_label_to_id.get(normalize_label(picked)) if picked else None


class _EssayEntityRegistry:
    """Per-essay (never cross-essay) entity dedup: a new mention is only
    ever compared against already-registered entities sharing the same
    _parent_key (identical type, or siblings under the same immediate
    hierarchy parent) -- the same hard symbolic gate entity_dedup.py uses
    for the whole-corpus case, applied here fresh per essay. Embedding
    similarity further narrows candidates before any LLM call, same
    reasoning as sandbox.py's diff_merge_candidates threshold."""

    SIM_FLOOR = 0.70
    MAX_CANDIDATES = 5

    def __init__(self, hierarchy: TypeHierarchy, embedder: ContrieverEmbedder, llm):
        self.hierarchy = hierarchy
        self.embedder = embedder
        self.llm = llm
        # parent_key -> list of (canonical_name, embedding)
        self.registry: dict[tuple, list[tuple[str, np.ndarray]]] = defaultdict(list)

    def resolve(self, mention_name: str, type_id: str, embedding: Optional[np.ndarray] = None) -> str:
        """Returns the canonical name this mention should use (itself, if
        it's new, or an existing registered entity's name if merged).
        `embedding`: precomputed, to avoid a fresh single-string embed()
        call per mention -- see _link_and_dedup_triplets' batched cache."""
        key = _parent_key(type_id, self.hierarchy)
        bucket = self.registry[key]
        mention_embedding = embedding if embedding is not None else self.embedder.embed([mention_name])[0]
        if not bucket:
            self.registry[key].append((mention_name, mention_embedding))
            return mention_name

        sims = cosine_similarity(mention_embedding.reshape(1, -1),
                                  np.stack([e for _, e in bucket]))[0]
        ranked = sorted(zip(sims, bucket), key=lambda x: -x[0])
        passing = [(name, emb) for sim, (name, emb) in ranked if sim >= self.SIM_FLOOR][: self.MAX_CANDIDATES]

        if not passing:
            self.registry[key].append((mention_name, mention_embedding))
            return mention_name

        candidate_names = [name for name, _ in passing]
        exact_match = next(
            (c for c in candidate_names if c.lower().strip() == mention_name.lower().strip()), None,
        )
        if exact_match is not None:
            # Case-insensitive exact match among candidates -- no need to
            # ask, and reuse the EXISTING entity's casing so later mentions
            # of either spelling keep converging on one consistent name.
            return exact_match

        members_in = [mention_name] + candidate_names
        try:
            raw_groups = self.llm.verify_cluster_with_llm(members_in, surface_form_type="entity")
            exc = None
        except Exception as e:  # noqa: BLE001 -- _parse_cluster_response's own
                                 # failure policy (never merge on error) applies below
            raw_groups = None
            exc = e
        # _parse_cluster_response is the same safety-first fallback every
        # other dedup step in this codebase relies on: a failed call or an
        # unparseable response keeps every member SEPARATE rather than
        # risking an unverified merge (dedup_base.py's own stated policy).
        parsed = _parse_cluster_response(raw_groups, exc, members_in, cluster_id=mention_name)

        # Case-INSENSITIVE comparison, not exact string identity: when
        # mention_name and a candidate are case-variants of the same string
        # (e.g. "urban legend" mention vs. "Urban legend" candidate),
        # _parse_cluster_response's own members_lower dict (built from
        # {m.lower(): m for m in members}) silently collapses them onto one
        # key, so the group it returns may reference the CANDIDATE's casing
        # even though it's the mention's own group. An exact `in` check
        # would miss that and register a redundant case-variant duplicate
        # instead of reusing the existing entity -- measured: "Urban
        # legend" and "urban legend" both survived as distinct entities in
        # a real pilot run before this fix.
        mention_lower = mention_name.lower().strip()
        survivor = mention_name
        for _canonical_label, members in parsed:
            members_lower = [m.lower().strip() for m in members]
            if mention_lower in members_lower and len(members) > 1:
                other = next((m for m in members if m.lower().strip() != mention_lower), None)
                if other is not None:
                    survivor = other
                break

        if survivor == mention_name:
            self.registry[key].append((mention_name, mention_embedding))
        return survivor


def _fold_qualifiers(triplet: dict, relation_label: str, linker: "OntologyLinker") -> str:
    """Fold a triplet's qualifiers into an enriched edge label, e.g.
    "molts (frequency: several times)" -- same convention as
    build_kg_graph.py's _qualifier_enriched_relation, and for the same
    reason documented there: a reified qualifier node is rarely reached by
    kg-gen's retrieval walk (measured on an AutoOntic MINE graph: ~11% of
    entities, ~32% of triples were pure bookkeeping that almost never
    surfaced), while folding into the label keeps the qualifier fact
    reachable in the exact same hop as its base fact, at zero extra nodes.

    Qualifier relations are resolved via a plain exact-match lookup against
    the pre-built relation vocabulary -- no LLM call, no embedding search --
    matching build_kg_graph.py's own weight for qualifiers (lighter than
    main-triplet linking, since a qualifier has no (subject, object) pair
    of its own to give an LLM call real context to adjudicate against).
    The qualifier's object is kept as its raw extracted value; unlike a
    main triplet's subject/object, qualifier values are frequently bare
    dates/quantities/descriptions rather than named entities, so there's no
    natural registry to resolve them against without inventing one.

    Returns relation_label unchanged if there are no parseable qualifiers."""
    qualifiers = triplet.get("qualifiers") or []
    parts = []
    for q in qualifiers:
        if not isinstance(q, dict):
            continue
        q_relation = str(q.get("relation") or "").strip()
        q_object = str(q.get("object") or "").strip()
        if not (q_relation and q_object):
            continue
        rid = linker.relation_label_to_id.get(normalize_label(q_relation))
        if rid is not None:
            q_relation = linker.relation_label_by_id[rid]
        parts.append(f"{q_relation}: {q_object}")
    if not parts:
        return relation_label
    return f"{relation_label} ({'; '.join(parts)})"


def _link_and_dedup_triplets(
    essay_id, triplets: list[dict], context_text: str, linker: OntologyLinker, llm: LLMTripletExtractor,
) -> tuple[dict, dict]:
    """Steps 2-4 only (entity type linking, relation linking, per-essay
    entity dedup) over an already-extracted triplet list -- shared by
    process_essay (fresh extraction, step 1 included) and
    process_existing_triplets (step 1 skipped, triplets already on disk).
    context_text is whatever general context is available to show the
    link_entity_type/link_relation calls (the essay's own text, since raw
    triplets carry no per-triplet evidence span -- see module docstring).

    Returns (graph_dict, stats). graph_dict is kg-gen's {"entities": [...],
    "edges": [...], "relations": [[s, r, o], ...]} schema (see
    build_kg_graph.py's own docstring/schema note).

    Performance: every unique raw type/relation/entity-mention text in this
    essay is embedded ONCE, up front, in a single batched call -- mirrors
    both Wikontic's own structured_inference_with_db.py (which pre-embeds
    "every unique subject/object/relation/entity-type... in a single
    batched call, instead of embedding each one... individually") and this
    codebase's own hierarchy_induction.py embedding-cache pattern. Without
    this, every link_entity_type/link_relation/registry.resolve call
    recomputes its own single-string embedding -- measured cost of skipping
    it: essay-to-essay wall time varying 3-4x with no corresponding jump in
    triplet count. Type/relation linking (read-only against the fixed
    ontology, no shared state) also runs on a thread pool, since each
    triplet's linking is fully independent of every other's -- entity dedup
    is NOT parallelized, since _EssayEntityRegistry.resolve() mutates
    shared per-essay state and correctness there depends on seeing earlier
    merges before deciding the next one."""
    valid = [
        t for t in triplets
        if t.get("subject") and t.get("object") and t.get("subject_type")
        and t.get("object_type") and t.get("relation")
    ]
    num_extracted = len(triplets)

    unique_texts = set()
    for t in valid:
        unique_texts.update((t["subject_type"], t["object_type"], t["relation"], t["subject"], t["object"]))
    unique_texts = list(unique_texts)
    embedding_cache: dict[str, np.ndarray] = {}
    if unique_texts:
        vectors = linker.embedder.embed(unique_texts)
        embedding_cache = dict(zip(unique_texts, vectors))

    def _link_one(t: dict):
        sentence = t.get("evidence") or context_text[:500]
        subj_type_id = linker.link_entity_type(
            llm, t["subject_type"], t["subject"], sentence, embedding=embedding_cache.get(t["subject_type"]))
        obj_type_id = linker.link_entity_type(
            llm, t["object_type"], t["object"], sentence, embedding=embedding_cache.get(t["object_type"]))
        if subj_type_id is None or obj_type_id is None:
            return (t, None, None, None, "type_dropped")
        subj_type_label = linker.type_label_by_id[subj_type_id]
        obj_type_label = linker.type_label_by_id[obj_type_id]
        relation_id = linker.link_relation(
            llm, t["relation"], subj_type_label, obj_type_label, sentence,
            embedding=embedding_cache.get(t["relation"]),
        )
        if relation_id is None:
            return (t, subj_type_id, obj_type_id, None, "relation_dropped")
        return (t, subj_type_id, obj_type_id, relation_id, "ok")

    with ThreadPoolExecutor(max_workers=8) as executor:
        resolved = list(executor.map(_link_one, valid))

    registry = _EssayEntityRegistry(linker.hierarchy, linker.embedder, llm)
    entities: set[str] = set()
    edges: set[str] = set()
    relations: list[list[str]] = []
    num_type_dropped = 0
    num_relation_dropped = 0

    for t, subj_type_id, obj_type_id, relation_id, status in resolved:
        if status == "type_dropped":
            num_type_dropped += 1
            continue
        if status == "relation_dropped":
            num_relation_dropped += 1
            continue
        relation_label = linker.relation_label_by_id[relation_id]
        relation_label = _fold_qualifiers(t, relation_label, linker)
        canonical_subj = registry.resolve(t["subject"], subj_type_id, embedding=embedding_cache.get(t["subject"]))
        canonical_obj = registry.resolve(t["object"], obj_type_id, embedding=embedding_cache.get(t["object"]))

        entities.add(canonical_subj)
        entities.add(canonical_obj)
        edges.add(relation_label)
        relations.append([canonical_subj, relation_label, canonical_obj])

    prompt_tokens, completion_tokens = llm.calculate_used_tokens()
    stats = {
        "essay_id": essay_id,
        "num_extracted": num_extracted,
        "num_kept": len(relations),
        "num_type_dropped": num_type_dropped,
        "num_relation_dropped": num_relation_dropped,
        "num_entities": len(entities),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    graph = {"entities": sorted(entities), "edges": sorted(edges), "relations": relations}
    return graph, stats


def process_essay(
    essay_id, essay_text: str, linker: OntologyLinker, llm: LLMTripletExtractor,
) -> tuple[dict, dict]:
    """Full four-step pipeline, INCLUDING fresh extraction (step 1) over
    essay_text. Prefer process_existing_triplets instead when the triplets
    this essay's ontology was built from are already on disk -- re-running
    extraction is a second, independent LLM pass that will not reproduce
    the original extraction exactly, which is a confound you don't want
    when isolating "what does per-essay scoping alone cost" (see
    CLAUDE.md-adjacent discussion in this module's git history / session
    notes)."""
    llm.reset_tokens()
    raw = llm.extract_triplets_from_text(essay_text)
    triplets = raw.get("triplets", []) if isinstance(raw, dict) else []
    return _link_and_dedup_triplets(essay_id, triplets, essay_text, linker, llm)


def process_existing_triplets(
    essay_id, triplets: list[dict], context_text: str, linker: OntologyLinker, llm: LLMTripletExtractor,
) -> tuple[dict, dict]:
    """Steps 2-4 only, over triplets ALREADY extracted (e.g. the exact ones
    the base ontology was built from, data/mine/triplets_gptoss_v2.jsonl) --
    no fresh LLM extraction call. This is the right mode for isolating the
    cost of per-essay scoping + ontology re-linking alone, without also
    confounding it with a second, independent extraction pass's own
    variance relative to the first."""
    llm.reset_tokens()
    return _link_and_dedup_triplets(essay_id, triplets, context_text, linker, llm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology-dir", required=True,
                         help="A pipeline run_<n> checkpoint dir holding type_dedup.pkl / "
                              "relation_dedup.pkl / hierarchy_induction.pkl, e.g. "
                              "output-2/mine/checkpoints/run_13 (unseeded gpt-oss MINE ontology).")
    parser.add_argument("--essays", default="data/mine/essays.json")
    parser.add_argument("--existing-triplets", default=None,
                         help="Path to an already-extracted triplets JSONL (e.g. "
                              "data/mine/triplets_gptoss_v2.jsonl, the EXACT triplets --ontology-dir "
                              "was built from) with a 'source_essay_id' field per row. When given, "
                              "step 1 (fresh LLM extraction) is SKIPPED entirely -- each essay's "
                              "triplets are sliced from this file instead, and only steps 2-4 "
                              "(entity type linking, relation linking, per-essay entity dedup) run. "
                              "This isolates per-essay-scoping + re-linking cost from the added "
                              "variance of a second, independent extraction pass.")
    parser.add_argument("--output-dir", default="output/mine/ontology_linked_graphs")
    parser.add_argument("--model", default="Openai/Gpt-oss-120b")
    parser.add_argument("--base-url", default=os.environ.get("AIRI_BASE_URL",
                         "https://inference.airi.net:46783/v1"),
                         help="Defaults to $AIRI_BASE_URL if set -- the configs/*.yaml files' "
                              "hardcoded ':46783' port is stale; .env's AIRI_BASE_URL "
                              "(https://inference.airi.net/v1/, no custom port) is the one "
                              "confirmed working.")
    parser.add_argument("--api-key-env", default="AIRI_KEY")
    parser.add_argument("--proxy-key-env", default="PROXY_URL")
    parser.add_argument("--contriever-model", default="facebook/contriever")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=10, help="Candidate types/relations shown per link call.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N essays (pilot run).")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env!r} is not set")
    proxy = os.environ.get(args.proxy_key_env) if args.proxy_key_env else None

    llm = LLMTripletExtractor(api_key=api_key, model=args.model, base_url=args.base_url, proxy=proxy)
    embedder = ContrieverEmbedder(model_name=args.contriever_model, device=args.device)
    linker = OntologyLinker(Path(args.ontology_dir), embedder, top_k=args.top_k)

    with open(args.essays, "r", encoding="utf-8") as f:
        essays = json.load(f)
    if args.limit:
        essays = essays[: args.limit]

    existing_by_essay: dict = {}
    if args.existing_triplets:
        with open(args.existing_triplets, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                existing_by_essay.setdefault(row["source_essay_id"], []).append(row)
        logger.info("Loaded %d pre-extracted triplets across %d essays from %s (fresh extraction SKIPPED)",
                    sum(len(v) for v in existing_by_essay.values()), len(existing_by_essay),
                    args.existing_triplets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    start = time.time()
    for i, essay in enumerate(essays, start=1):
        essay_id, text = essay["id"], essay["content"]
        logger.info("[%d/%d] Processing essay %s (%r)...", i, len(essays), essay_id, essay.get("topic", ""))
        if args.existing_triplets:
            triplets = existing_by_essay.get(essay_id, [])
            graph, stats = process_existing_triplets(essay_id, triplets, text, linker, llm)
        else:
            graph, stats = process_essay(essay_id, text, linker, llm)
        out_path = output_dir / f"essay_{essay_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        all_stats.append(stats)
        logger.info(
            "  essay %s: %d extracted -> %d kept (%d type-dropped, %d relation-dropped), %d entities",
            essay_id, stats["num_extracted"], stats["num_kept"],
            stats["num_type_dropped"], stats["num_relation_dropped"], stats["num_entities"],
        )

    elapsed = time.time() - start
    usage = llm.get_usage_snapshot()
    summary = {
        "ontology_dir": str(args.ontology_dir),
        "num_essays": len(essays),
        "elapsed_seconds": elapsed,
        "usage": usage,
        "per_essay": all_stats,
    }
    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Done: %d essays in %.1fs. Cost so far: $%.4f. Summary written to %s",
                len(essays), elapsed, usage.get("cost_usd", 0.0), summary_path)


if __name__ == "__main__":
    main()
