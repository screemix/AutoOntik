"""
MuSiQue QA evaluation over an already-built AutoOnto KG
=========================================================

Re-implements the question-answering evaluation from Wikontic
(https://github.com/screemix/Wikontic, see
inference_and_eval/qa_eval_musique.py and
src/wikontic/utils/base_inference_with_db.py) against AutoOnto's own KG
representation instead of Wikontic's MongoDB/Qdrant-backed one.

Wikontic builds one small per-sample KG per MuSiQue instance (all triplets
extracted from that instance's ~20 paragraphs) and answers each question by:
  1. extracting entity mentions from the question with an LLM,
  2. retrieving similar entity names from THAT SAMPLE's own KG via embedding
     search, refined by a second LLM call ("which of these retrieved
     candidates are actually relevant"),
  3. walking up to `hop_depth` hops outward from the linked entities to
     collect supporting (subject, relation, object, qualifiers) triplets,
  4. answering the question from those triplets with a final LLM call.
(See base_inference_with_db.py's identify_relevant_entities_from_question_
with_llm / answer_question_with_llm -- this script mirrors that path, NOT
the multi-step "QA collapsing" mode, which Wikontic itself only uses
opt-in via --multi-step-qa; both accept the same fidelity trade-off.)

AutoOnto canonicalizes entities/types/relations GLOBALLY across the whole
corpus, not per-sample -- there is no per-sample vector index the way
Wikontic's MongoDB/Qdrant triplets_db has one. So instead this script:
  - resolves every raw triplet (data/musique_initial_triplets.jsonl, tagged
    with "sample_id") to its canonical entity/relation ids using a
    pipeline run's relation_dedup.pkl / type_dedup.pkl / entity_dedup.pkl
    checkpoints (same surface_to_id lookup pattern as
    scripts/text2kgbench_eval/run_judge_eval.py's _resolve_type_label /
    _resolve_relation_label),
  - then restricts retrieval and hop-walking to just that one sample_id's
    own resolved triplets, so the "per-sample subgraph" Wikontic gets for
    free from its DB's sample_id filter is reconstructed here explicitly.
  - Only relation_dedup.pkl / type_dedup.pkl / entity_dedup.pkl are needed
    -- CanonicalEntity.primary_type_id was already computed against the
    induced TypeHierarchy at pipeline-build time, and QA retrieval itself
    is hierarchy-agnostic in Wikontic too (the induced ontology gates KG
    CONSTRUCTION, not QA retrieval) -- so hierarchy_induction.pkl and
    constraints.pkl are not loaded.

Scoring: Wikontic's own qa_eval_musique.py logs raw vs. normalize()'d
strings for eyeballing but never computes a metric -- there is no scoring
script anywhere in that repo. This script keeps Wikontic's exact
normalize() (unidecode -> lowercase -> digit-group punctuation collapse ->
punctuation-to-space -> whitespace collapse) and adds the missing piece:
standard SQuAD/MuSiQue-style exact-match and token-F1, scored against
{answer} + answer_aliases and taking the max over all references.

Usage:
    python -m scripts.musique_qa_eval.run_qa_eval \\
        --output-dir output \\
        --run-number 14 \\
        --qa-model openai/gpt-oss-120b \\
        --output output/musique_qa/run_14_qa_results.json

    # Or point --checkpoints-dir straight at a run_<n> folder instead of
    # --output-dir/--run-number.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import string
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from unidecode import unidecode

from src.ontodisco.utils.dedup_base import ContrieverEmbedder, normalize_label
from src.ontodisco.utils.openai_utils import LLMTripletExtractor
from src.ontodisco.entity_dedup import _make_compound, _normalize_compound, _parse_compound

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


# ═══════════════════════════════════════════════════════════════════════════════
#  Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _latest_run_dir(output_dir: Path) -> Path:
    checkpoints_root = output_dir / "checkpoints"
    numbers = []
    if checkpoints_root.exists():
        for child in checkpoints_root.iterdir():
            m = _RUN_DIR_RE.match(child.name)
            if child.is_dir() and m:
                numbers.append(int(m.group(1)))
    if not numbers:
        raise FileNotFoundError(f"No run_<n> checkpoint directory found under {checkpoints_root}")
    return checkpoints_root / f"run_{max(numbers)}"


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  Resolving raw triplets to canonical (per-sample) subgraphs
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SampleEdge:
    subject_id: str
    subject_name: str
    subject_type: str
    relation_label: str
    object_id: str
    object_name: str
    object_type: str
    qualifiers: list = field(default_factory=list)


def _resolve_type(raw_type: str, type_vocab) -> Optional[tuple[str, str]]:
    type_id = type_vocab.surface_to_id.get(normalize_label(raw_type)) or type_vocab.surface_to_id.get(raw_type)
    if not type_id:
        return None
    return type_id, type_vocab.items[type_id].canonical_label


def _resolve_relation(raw_relation: str, relation_vocab) -> Optional[str]:
    relation_id = relation_vocab.surface_to_id.get(raw_relation) or \
        relation_vocab.surface_to_id.get(normalize_label(raw_relation))
    if not relation_id:
        return None
    return relation_vocab.items[relation_id].canonical_label


def _resolve_entity(raw_name: str, raw_type: str, type_vocab, entity_vocab) -> Optional[tuple[str, str, str]]:
    """Reconstruct the SAME compound "name [canonical type label]" string
    entity_dedup.collect_entity_surface_forms() built for this exact
    mention, and look it up in entity_vocab.surface_to_id. Returns
    (entity_id, display_name, type_label), or None if either the type or
    the resulting compound never resolved (extraction noise that didn't
    survive canonicalization -- same discipline as run_judge_eval.py)."""
    raw_name = raw_name.strip()
    if not raw_name or not raw_type:
        return None
    type_result = _resolve_type(raw_type, type_vocab)
    if type_result is None:
        return None
    _, type_label = type_result

    compound = _make_compound(raw_name, type_label)
    entity_id = entity_vocab.surface_to_id.get(compound)
    if entity_id is None:
        entity_id = entity_vocab.surface_to_id.get(_normalize_compound(compound))
    if entity_id is None:
        return None

    entity = entity_vocab.items[entity_id]
    display_name, _ = _parse_compound(entity.canonical_label)
    return entity_id, display_name, type_label


def build_sample_graphs(
    triplets_by_sample: dict[str, list[dict]], type_vocab, relation_vocab, entity_vocab,
) -> dict[str, list[SampleEdge]]:
    """One canonical (subject_id, relation_label, object_id) subgraph per
    MuSiQue sample_id -- the analogue of Wikontic's sample_id-scoped
    MongoDB/Qdrant `triplets` collection."""
    graphs: dict[str, list[SampleEdge]] = {}
    n_dropped = 0
    n_total = 0
    for sample_id, triplets in triplets_by_sample.items():
        edges: list[SampleEdge] = []
        seen: set[tuple[str, str, str]] = set()
        for t in triplets:
            n_total += 1
            subj = _resolve_entity(t.get("subject", ""), t.get("subject_type", ""), type_vocab, entity_vocab)
            obj = _resolve_entity(t.get("object", ""), t.get("object_type", ""), type_vocab, entity_vocab)
            relation_label = _resolve_relation(t.get("relation", "").strip(), relation_vocab)
            if not subj or not obj or not relation_label:
                n_dropped += 1
                continue

            key = (subj[0], relation_label, obj[0])
            if key in seen:
                continue
            seen.add(key)
            edges.append(SampleEdge(
                subject_id=subj[0], subject_name=subj[1], subject_type=subj[2],
                relation_label=relation_label,
                object_id=obj[0], object_name=obj[1], object_type=obj[2],
                qualifiers=t.get("qualifiers", []),
            ))
        graphs[sample_id] = edges

    logger.info(
        "Built %d per-sample subgraphs from %d raw triplets (%d dropped: "
        "subject/object/relation didn't resolve to a canonical id)",
        len(graphs), n_total, n_dropped,
    )
    return graphs


def sample_entity_index(edges: list[SampleEdge]) -> dict[str, tuple[str, str]]:
    """entity_id -> (display_name, type_label), for every entity appearing
    in this sample's subgraph -- the retrieval candidate pool for one
    question (Wikontic's per-sample vector-search scope)."""
    index: dict[str, tuple[str, str]] = {}
    for e in edges:
        index[e.subject_id] = (e.subject_name, e.subject_type)
        index[e.object_id] = (e.object_name, e.object_type)
    return index


# ═══════════════════════════════════════════════════════════════════════════════
#  Retrieval + multi-hop supporting-triplet collection
#  (mirrors base_inference_with_db.py's identify_relevant_entities_from_
#  question_with_llm / get_1_hop_supporting_triplets / answer_question_with_llm)
# ═══════════════════════════════════════════════════════════════════════════════

def identify_relevant_entities(
    question: str,
    entity_index: dict[str, tuple[str, str]],
    embedder: ContrieverEmbedder,
    extractor: LLMTripletExtractor,
    embed_lock: threading.Lock,
    top_k: int = 10,
) -> list[str]:
    """Returns a list of entity_ids linked to the question, within this
    sample's own entity_index."""
    if not entity_index:
        return []

    entity_ids = list(entity_index.keys())
    names = [entity_index[eid][0] for eid in entity_ids]
    with embed_lock:
        name_embs = embedder.embed(names)

    try:
        mentions = extractor.extract_entities_from_question(question)
    except Exception:
        logger.exception("extract_entities_from_question failed for question=%r", question)
        mentions = []
    if isinstance(mentions, dict):
        mentions = [mentions]
    if not isinstance(mentions, list):
        mentions = []

    linked: list[tuple[str, str, str]] = []      # exact surface matches -- skip LLM re-ranking
    identified: list[tuple[str, str, str]] = []   # embedding-retrieved candidates needing LLM re-ranking
    seen_identified: set[str] = set()

    for mention in mentions:
        mention = str(mention).strip()
        if not mention:
            continue
        with embed_lock:
            mention_emb = embedder.embed([mention])[0]
        sims = name_embs @ mention_emb
        k = min(top_k, len(entity_ids))
        top_idx = np.argsort(-sims)[:k]
        candidates = [(entity_ids[i], entity_index[entity_ids[i]][0], entity_index[entity_ids[i]][1]) for i in top_idx]

        exact = [c for c in candidates if c[1].lower() == mention.lower()]
        if exact:
            linked.extend(exact)
        else:
            for c in candidates:
                if c[0] not in seen_identified:
                    seen_identified.add(c[0])
                    identified.append(c)

    if identified:
        entity_list = [{"entity": name, "entity_type": etype} for _, name, etype in identified]
        name_to_candidate = {name: (eid, name, etype) for eid, name, etype in identified}
        try:
            ranked = extractor.identify_relevant_entities(question, entity_list)
        except Exception:
            logger.exception("identify_relevant_entities failed for question=%r", question)
            ranked = []
        if not isinstance(ranked, list):
            ranked = []
        for r in ranked:
            name = r.get("entity") if isinstance(r, dict) else None
            if name in name_to_candidate:
                linked.append(name_to_candidate[name])

    seen_ids: set[str] = set()
    result: list[str] = []
    for eid, _name, _etype in linked:
        if eid not in seen_ids:
            seen_ids.add(eid)
            result.append(eid)
    if not result:
        logger.error("No entities identified/linked for question=%r", question)
    return result


def bfs_supporting_edges(
    seed_entity_ids: list[str], edges: list[SampleEdge], hop_depth: int = 5,
) -> list[SampleEdge]:
    """Same accumulation as get_1_hop_supporting_triplets called in a loop
    by answer_question_with_llm: each hop only (re-)searches entities
    newly discovered by the previous hop, capped at hop_depth hops."""
    supporting: list[SampleEdge] = []
    supporting_keys: set[tuple[str, str, str]] = set()
    searched: set[str] = set()
    frontier: set[str] = set(seed_entity_ids)

    for _ in range(hop_depth):
        if not frontier:
            break
        for e in edges:
            if e.subject_id in frontier or e.object_id in frontier:
                key = (e.subject_id, e.relation_label, e.object_id)
                if key not in supporting_keys:
                    supporting_keys.add(key)
                    supporting.append(e)
        searched |= frontier
        new_frontier: set[str] = set()
        for e in supporting:
            if e.subject_id not in searched:
                new_frontier.add(e.subject_id)
            if e.object_id not in searched:
                new_frontier.add(e.object_id)
        frontier = new_frontier

    return supporting


def format_triplets_for_prompt(edges: list[SampleEdge], use_qualifiers: bool) -> list[dict]:
    out = []
    for e in edges:
        d = {"subject": e.subject_name, "relation": e.relation_label, "object": e.object_name}
        if use_qualifiers:
            d["qualifiers"] = e.qualifiers
        out.append(d)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Scoring -- Wikontic's normalize() + standard SQuAD/MuSiQue-style EM/F1
#  (Wikontic itself never computes a metric, only logs normalize()'d
#  strings for eyeballing -- see module docstring)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(input_string: str) -> str:
    input_string = unidecode(str(input_string))
    input_string = input_string.lower()
    input_string = re.sub(r"(?<=\d)[,\.](?=\d)", "", input_string)
    input_string = re.sub(f"[{re.escape(string.punctuation)}]", " ", input_string)
    input_string = re.sub(r"\s+", " ", input_string)
    return input_string.strip()


def _f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(pred: str, gold_answer: str, gold_aliases: list[str]) -> tuple[bool, float]:
    references = [gold_answer] + list(gold_aliases or [])
    em = any(normalize(pred) == normalize(ref) for ref in references)
    f1 = max((_f1(pred, ref) for ref in references), default=0.0)
    return em, f1


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-sample pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def answer_one_sample(
    sample_id: str, gold: dict, edges: list[SampleEdge],
    embedder: ContrieverEmbedder, extractor: LLMTripletExtractor, embed_lock: threading.Lock,
    hop_depth: int, top_k: int, use_qualifiers: bool,
) -> dict:
    question = gold["question"]

    if not edges:
        return {
            "sample_id": sample_id, "question": question,
            "gold_answer": gold.get("answer", ""), "predicted_answer": "",
            "num_edges": 0, "num_linked_entities": 0, "num_supporting_triplets": 0,
            "em": False, "f1": 0.0, "error": "no_resolved_triplets_for_sample",
        }

    entity_index = sample_entity_index(edges)
    linked_ids = identify_relevant_entities(question, entity_index, embedder, extractor, embed_lock, top_k=top_k)
    supporting = bfs_supporting_edges(linked_ids, edges, hop_depth=hop_depth)

    try:
        answer = extractor.answer_question(question, format_triplets_for_prompt(supporting, use_qualifiers))
    except Exception:
        logger.exception("answer_question failed for sample_id=%s question=%r", sample_id, question)
        answer = ""
    if not isinstance(answer, str):
        answer = json.dumps(answer)

    em, f1 = score_answer(answer, gold.get("answer", ""), gold.get("answer_aliases", []))

    return {
        "sample_id": sample_id, "question": question,
        "gold_answer": gold.get("answer", ""), "gold_answer_aliases": gold.get("answer_aliases", []),
        "predicted_answer": answer,
        "num_edges": len(edges), "num_linked_entities": len(linked_ids),
        "num_supporting_triplets": len(supporting),
        "em": em, "f1": f1,
    }


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    em_rate = sum(r["em"] for r in results) / n if n else 0.0
    avg_f1 = sum(r["f1"] for r in results) / n if n else 0.0
    n_no_graph = sum(1 for r in results if r.get("error") == "no_resolved_triplets_for_sample")
    n_no_entities = sum(1 for r in results if r.get("num_linked_entities") == 0 and r.get("num_edges", 0) > 0)
    return {
        "num_samples": n,
        "exact_match": em_rate,
        "f1": avg_f1,
        "num_samples_with_no_resolved_subgraph": n_no_graph,
        "num_samples_with_no_linked_entities": n_no_entities,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", help="Pipeline output_dir -- the latest run_<n> checkpoint is used "
                                              "unless --run-number/--checkpoints-dir is given.")
    parser.add_argument("--run-number", type=int, help="Explicit run_<n> under --output-dir/checkpoints.")
    parser.add_argument("--checkpoints-dir", help="Explicit output_dir/checkpoints/run_<n> dir "
                                                     "(overrides --output-dir/--run-number).")
    parser.add_argument("--triplets", default="data/musique_initial_triplets.jsonl",
                         help="Raw triplets JSONL, one dict per line, tagged with sample_id.")
    parser.add_argument("--qa-dataset", default="data/musique_qa_test.json",
                         help="MuSiQue QA test set: JSON list of {id, question, answer, answer_aliases, ...}.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=None,
                         help="Evaluate only the first N sample_ids (after intersecting KG and QA dataset).")
    parser.add_argument("--sample-ids", nargs="+", default=None,
                         help="Evaluate only these specific sample_ids.")
    parser.add_argument("--hop-depth", type=int, default=5,
                         help="Max BFS hops outward from linked entities (Wikontic's answer_question_with_llm default).")
    parser.add_argument("--top-k", type=int, default=10,
                         help="Embedding-retrieval shortlist size per extracted question entity mention.")
    parser.add_argument("--use-qualifiers", action="store_true", default=True)
    parser.add_argument("--no-use-qualifiers", action="store_false", dest="use_qualifiers")
    parser.add_argument("--qa-model", default="Openai/Gpt-oss-120b",
                         help="LLM used to answer questions -- independent of whatever model built the KG.")
    parser.add_argument("--qa-base-url", default="https://inference.airi.net:46783/v1")
    parser.add_argument("--qa-api-key-env", default="AIRI_KEY", help="Env var holding the QA LLM's API key.")
    parser.add_argument("--embedding-model", default="facebook/contriever")
    parser.add_argument("--device", default=None, help="'cuda' | 'cpu' | None (auto)")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    if args.checkpoints_dir:
        run_dir = Path(args.checkpoints_dir)
    elif args.output_dir and args.run_number is not None:
        run_dir = Path(args.output_dir) / "checkpoints" / f"run_{args.run_number}"
    elif args.output_dir:
        run_dir = _latest_run_dir(Path(args.output_dir))
    else:
        raise SystemExit("Pass either --checkpoints-dir, or --output-dir (optionally with --run-number)")

    logger.info("Loading canonical vocabularies from %s", run_dir)
    relation_vocab = _load_pickle(run_dir / "relation_dedup.pkl")
    type_vocab = _load_pickle(run_dir / "type_dedup.pkl")
    entity_vocab = _load_pickle(run_dir / "entity_dedup.pkl")

    raw_triplets = _load_jsonl(args.triplets)
    triplets_by_sample: dict[str, list[dict]] = {}
    for t in raw_triplets:
        triplets_by_sample.setdefault(t["sample_id"], []).append(t)
    sample_graphs = build_sample_graphs(triplets_by_sample, type_vocab, relation_vocab, entity_vocab)

    qa_dataset = _load_json(args.qa_dataset)
    id2sample = {x["id"]: x for x in qa_dataset}

    sample_ids = sorted(set(sample_graphs) & set(id2sample))
    if args.sample_ids:
        sample_ids = [sid for sid in args.sample_ids if sid in sample_ids]
    elif args.num_samples:
        sample_ids = sample_ids[: args.num_samples]
    logger.info("Evaluating %d samples (KG ∩ QA dataset)", len(sample_ids))

    api_key = os.environ.get(args.qa_api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.qa_api_key_env!r} is not set")
    extractor = LLMTripletExtractor(api_key=api_key, model=args.qa_model, base_url=args.qa_base_url)
    embedder = ContrieverEmbedder(model_name=args.embedding_model, device=args.device)
    embed_lock = threading.Lock()

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                answer_one_sample, sid, id2sample[sid], sample_graphs[sid],
                embedder, extractor, embed_lock, args.hop_depth, args.top_k, args.use_qualifiers,
            ): sid
            for sid in sample_ids
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            results.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                logger.info("Answered %d/%d questions. Cost so far: $%.4f",
                            completed, len(futures), extractor.calculate_cost())

    results.sort(key=lambda r: r["sample_id"])
    summary = aggregate(results)
    summary["run_dir"] = str(run_dir)
    summary["qa_model"] = args.qa_model
    summary["hop_depth"] = args.hop_depth
    summary["usage"] = extractor.get_usage_snapshot()

    output = {"summary": summary, "per_sample": results}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(
        "EM=%.2f%% F1=%.2f%% over %d samples (%d with no resolved subgraph, %d with no linked entities). "
        "Cost=$%.4f. Results written to %s",
        summary["exact_match"] * 100, summary["f1"] * 100, summary["num_samples"],
        summary["num_samples_with_no_resolved_subgraph"], summary["num_samples_with_no_linked_entities"],
        summary["usage"]["cost_usd"], output_path,
    )


if __name__ == "__main__":
    main()
