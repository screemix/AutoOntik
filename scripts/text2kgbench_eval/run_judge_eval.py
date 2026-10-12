"""
LLM-as-a-judge evaluation against Text2KGBench gold triples + ontology
============================================================================

Standalone (only needs pickle + the openai client via LLMTripletExtractor --
no torch/sklearn import required, unlike the ontology pipeline itself), so it
can be re-run against an already-built pipeline output any number of times,
e.g. to try a different --judge-model, without re-running the (expensive,
LLM-heavy) ontology pipeline.

The judge model is DELIBERATELY independent of whatever model built the KG
(--judge-model/--judge-base-url/--judge-api-key-env, not the pipeline's own
--config) -- comparing qwen-built vs gpt-oss-built KGs against a fixed
third-party judge (e.g. gpt-4o) avoids a model grading its own homework, and
keeps the judge constant across every run being compared. Mirrors
scripts/mine_benchmark/run_judge_eval.py's --judge-model/--judge-base-url/
--judge-api-key-env convention.

Reads:
  - triplets.jsonl written by extract_triplets.py (each row tagged with
    source_sentence_id AND source_ontology_id).
  - relation_dedup.pkl / type_dedup.pkl checkpoints written by
    run_pipeline_text2kgbench.py, to resolve every raw extracted triplet to
    its canonical relation_id / type labels (surface_to_id lookups -- same
    pattern constraints.py's own Stage 0 uses). Hierarchy/entity checkpoints
    aren't needed: this eval scores relation and type canonicalization, not
    hierarchy structure or entity resolution.
  - ground_truth.jsonl + ontology.ttl for ONE OR MORE domains (--ontology-ids)
    under --data-dir/<ontology-id>/. Text2KGBench's ground truth carries no
    per-triple types (see ontology.py's docstring), so domain/range for the
    type judge always come from the RIGHT domain's ontology file -- tracked
    per sentence via source_ontology_id, not assumed to be a single ontology
    for the whole run. This is what makes it possible to score a run of
    run_pipeline_text2kgbench.py over a MULTI-DOMAIN COMBINED corpus (one
    shared KG spanning several Text2KGBench domains, see
    run_multi_domain.sh's "combined" mode) as well as a single-domain one --
    passing one --ontology-ids value reduces to the single-domain case.

Reports two levels of aggregation: "overall" (every domain pooled together)
and "by_domain" (one summary per --ontology-ids value, computed by filtering
results to that domain's sentences). For a single-domain run these are
mostly redundant (by_domain has one entry matching overall); for a
multi-domain COMBINED run, by_domain is what lets you compare directly
against N separate single-domain runs' own summaries (run_multi_domain.sh's
compare_domain_reports.py does exactly that comparison).

Matching algorithm, per sentence (see CLAUDE.md's eval design discussion for
the full reasoning trail):
  1. Iterate from the GOLD side, not the generated side: for each gold
     triple, ask the LLM to pick its match (if any) from the REMAINING
     (not yet consumed) bucket of generated triples for that sentence, via
     LLMTripletExtractor.match_triple_with_llm(). Once a candidate is
     matched it is removed from the bucket (greedy consumption -- accepted
     as an approximation given how small each sentence's bucket is, not
     worth a real bipartite assignment solver).
  2. A gold triple with no match is a genuine recall miss. A generated
     triple left over in the bucket after all gold triples are processed is
     reported as an "extra", NOT scored as a false positive -- Text2KGBench's
     gold ontology covers a small curated relation vocabulary (~15-20
     relations per domain), so a schema-free pipeline can correctly extract
     true facts the gold ontology simply doesn't cover.
  3. For every matched pair, compare our resolved subject_type/object_type
     against the gold relation's domain/range (from the ontology file) via
     LLMTripletExtractor.compare_types_with_llm(), swapping which side
     compares to which when the match was "inverse".

Error-handling discipline: a judge call that raises (auth/rate-limit/network
failure) is NEVER folded into a "no match"/"not_related" judgment -- that's
exactly the failure mode scripts/mine_benchmark/run_judge_eval.py's docstring
documents from a real incident (a 402 Insufficient Credits error on every
call silently became a bogus 0.0% accuracy). Call failures are tracked
separately and excluded from recall, with a loud warning if they're a large
fraction of the total.

Usage:
    # Single domain, judged by gpt-4o via OpenRouter regardless of what built the KG
    python -m scripts.text2kgbench_eval.run_judge_eval \\
        --triplets data/text2kgbench/wikidata_tekgen/ont_2_music/triplets.jsonl \\
        --data-dir data/text2kgbench/wikidata_tekgen \\
        --ontology-ids ont_2_music \\
        --output-dir output/text2kgbench/wikidata_tekgen/ont_2_music \\
        --output output/text2kgbench/wikidata_tekgen/ont_2_music/judge_results.json \\
        --judge-model openai/gpt-4o --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_KEY

    # Multiple domains scored against one combined pipeline run
    python -m scripts.text2kgbench_eval.run_judge_eval \\
        --triplets data/text2kgbench/wikidata_tekgen/_combined/triplets.jsonl \\
        --data-dir data/text2kgbench/wikidata_tekgen \\
        --ontology-ids ont_2_music ont_1_movie \\
        --output-dir output/text2kgbench/wikidata_tekgen/_combined \\
        --output output/text2kgbench/wikidata_tekgen/_combined/judge_results.json \\
        --judge-model openai/gpt-4o --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.ontodisco.utils.dedup_base import normalize_label
from src.ontodisco.utils.openai_utils import LLMTripletExtractor
from src.ontodisco.relation_dedup import update_relation_type_map
from src.ontodisco.constraints import ConstraintConfig, induce_constraints

from scripts.text2kgbench_eval.ontology import Text2KGOntology, parse_ontology

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")
TYPE_JUDGE_LABELS = ("exact", "more_general", "more_narrow", "not_related")


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


def _resolve_type(raw_type: str, type_vocab) -> tuple[str | None, str | None]:
    """Returns (type_id, canonical_label), mirroring _resolve_relation_label.
    The id is needed for Constraint Plausibility's ancestor lookups in
    TypeHierarchy; the label is what the existing Domain/Range Type
    Matching judge call already consumes."""
    type_id = type_vocab.surface_to_id.get(normalize_label(raw_type)) or type_vocab.surface_to_id.get(raw_type)
    if not type_id:
        return None, None
    return type_id, type_vocab.items[type_id].canonical_label


def _is_ancestor_or_equal(candidate_ancestor_id: str, node_id: str, hierarchy) -> bool:
    """True iff candidate_ancestor_id is node_id itself or one of its
    ancestors in hierarchy's single-parent is-a chain. Used both to SELECT
    which constraint signature a matched pair's types fall under, and as
    the free containment sanity check (see Constraint Plausibility, module
    docstring) -- which should always return True by construction, since a
    constraint's (domain,range) is the LCA over exactly the triples that
    fed it, including -- assuming no pipeline bug -- this one."""
    current = node_id
    seen = set()
    while current is not None and current not in seen:
        if current == candidate_ancestor_id:
            return True
        seen.add(current)
        parents = hierarchy.parents.get(current)
        current = parents[0] if parents else None
    return False


def _resolve_relation_label(raw_relation: str, relation_vocab) -> tuple[str | None, str | None]:
    relation_id = relation_vocab.surface_to_id.get(raw_relation) or \
        relation_vocab.surface_to_id.get(normalize_label(raw_relation))
    if not relation_id:
        return None, None
    return relation_id, relation_vocab.items[relation_id].canonical_label


def build_sentence_buckets(triplets: list[dict], relation_vocab, type_vocab) -> dict[str, list[dict]]:
    """Resolve every raw extracted triplet to its canonical relation/type
    labels and group by source_sentence_id. Triplets missing that tag, or
    that never resolved to a canonical id (extraction noise that didn't
    survive canonicalization), are dropped and counted."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    n_dropped = 0
    for t in triplets:
        sentence_id = t.get("source_sentence_id")
        raw_subject = t.get("subject", "").strip()
        raw_relation = t.get("relation", "").strip()
        raw_object = t.get("object", "").strip()
        raw_subject_type = t.get("subject_type", "").strip()
        raw_object_type = t.get("object_type", "").strip()
        if not (sentence_id and raw_subject and raw_relation and raw_object):
            n_dropped += 1
            continue

        relation_id, relation_label = _resolve_relation_label(raw_relation, relation_vocab)
        subject_type_id, subject_type_label = _resolve_type(raw_subject_type, type_vocab) if raw_subject_type else (None, None)
        object_type_id, object_type_label = _resolve_type(raw_object_type, type_vocab) if raw_object_type else (None, None)

        buckets[sentence_id].append({
            "subject": raw_subject, "relation": raw_relation, "object": raw_object,
            "relation_id": relation_id, "relation_label": relation_label,
            "subject_type_id": subject_type_id, "subject_type_label": subject_type_label,
            "object_type_id": object_type_id, "object_type_label": object_type_label,
        })

    if n_dropped:
        logger.warning("%d/%d generated triplets dropped (missing sentence id, or empty subject/relation/object)",
                        n_dropped, len(triplets))
    return dict(buckets)


def build_constraint_index(constraints: list) -> dict[str, list]:
    """relation_id -> every RelationConstraint for it (any strength tier,
    including BELOW_HINT when induce_constraints was called with
    include_below_threshold=True -- see Constraint Plausibility, module
    docstring: we score every candidate signature a matched pair might
    select, not just the ones that cleared hint_threshold)."""
    index: dict[str, list] = defaultdict(list)
    for c in constraints:
        index[c.relation_id].append(c)
    return dict(index)


def _select_constraint(relation_id, subject_type_id, object_type_id, constraint_index, hierarchy):
    """The constraint for relation_id whose (domain_type_id, range_type_id)
    both contain (ancestor-or-equal) this pair's own types -- see
    _is_ancestor_or_equal. A relation spanning disconnected hierarchy
    partitions can carry several constraints; this picks the one the
    pair's own types actually witness, not an arbitrary one. Returns
    (constraint, containment_ok) -- containment_ok is the free sanity
    check (expected True always; see module docstring)."""
    candidates = constraint_index.get(relation_id, [])
    if not candidates or not subject_type_id or not object_type_id:
        return None, None
    for c in candidates:
        domain_ok = _is_ancestor_or_equal(c.domain_type_id, subject_type_id, hierarchy)
        range_ok = _is_ancestor_or_equal(c.range_type_id, object_type_id, hierarchy)
        if domain_ok and range_ok:
            return c, True
    # No stored signature contains this pair -- either a true pipeline bug
    # (constraint computed over a different triplet set than this matched
    # instance), or the relation legitimately has zero candidates (shouldn't
    # happen if constraint_index was built with include_below_threshold=True
    # and this relation has at least one resolvable triplet). Return the
    # first candidate anyway so the containment violation is visible rather
    # than silently treated as "no constraint".
    return candidates[0], False


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-sentence matching + type judging
# ═══════════════════════════════════════════════════════════════════════════════

def _judge_sentence(
    record: dict, bucket: list[dict], ontology: Text2KGOntology, extractor: LLMTripletExtractor,
    constraint_index: dict | None = None, hierarchy=None, type_label_by_id: dict | None = None,
    direction_by_relation: dict | None = None,
) -> dict:
    """Match every gold triple for one sentence against the (mutable) bucket
    of generated triples for that sentence, consuming a candidate once
    matched, then type-judge each matched pair's domain/range against the
    gold ontology. See module docstring for the full algorithm and why
    iteration starts from the gold side."""
    remaining = list(bucket)
    gold_results = []
    call_failures = 0

    for gold in record.get("triples", []):
        if not remaining:
            gold_results.append({"gold": gold, "matched": False, "reason": "no_candidates_left"})
            continue

        candidates = [(e["subject"], e["relation"], e["object"]) for e in remaining]
        try:
            match_resp = extractor.match_triple_with_llm(gold["sub"], gold["rel"], gold["obj"], candidates)
        except Exception:
            logger.exception("match_triple_with_llm failed for sentence %s, gold triple %r", record["id"], gold)
            call_failures += 1
            gold_results.append({"gold": gold, "matched": False, "reason": "judge_call_failed"})
            continue

        idx = match_resp.get("match")
        if idx is None:
            gold_results.append({"gold": gold, "matched": False, "reason": "no_match"})
            continue

        direction = match_resp.get("direction", "same")
        matched_entry = remaining.pop(idx)

        # ── Type judge for this matched pair ──────────────────────────────
        gold_rel_info = ontology.relations.get(gold["rel"], {})
        gold_domain = gold_rel_info.get("domain")
        gold_range = gold_rel_info.get("range")

        if direction == "inverse":
            our_domain_side = matched_entry["object_type_label"]
            our_range_side = matched_entry["subject_type_label"]
        else:
            our_domain_side = matched_entry["subject_type_label"]
            our_range_side = matched_entry["object_type_label"]

        domain_pair = (our_domain_side, gold_domain) if (our_domain_side and gold_domain) else None
        range_pair = (our_range_side, gold_range) if (our_range_side and gold_range) else None

        type_judgment = {"domain": None, "range": None}
        if domain_pair or range_pair:
            try:
                type_judgment = extractor.compare_types_with_llm(domain_pair, range_pair)
            except Exception:
                logger.exception("compare_types_with_llm failed for sentence %s, gold triple %r", record["id"], gold)
                call_failures += 1
                type_judgment = {"domain": None, "range": None, "judge_call_failed": True}

        # ── Constraint Plausibility (Track B): same rubric, different
        # comparison target -- the relation's MINED CONSTRAINT against gold,
        # not this instance's own assigned type against gold (that's Track
        # A above). See module docstring. Selection uses the pair's own
        # (un-swapped) types; the swap-on-inverse applies to GOLD's side,
        # exactly mirroring Track A's swap, so both tracks share one
        # direction convention. ──
        constraint_judgment = {"domain": None, "range": None}
        constraint_info = None
        if constraint_index is not None and hierarchy is not None:
            # The constraint's own (domain,range) is stored in the canonical
            # relation's direction-CORRECTED forward sense (induce_constraints'
            # internal _apply_direction). This triple's raw surface form may
            # itself be the inverted one (e.g. "served by" vs. canonical
            # "serves") -- apply the SAME per-surface-form correction before
            # selecting/containment-checking, exactly mirroring
            # constraints.py's _apply_direction, or every triple extracted via
            # an inverted surface form spuriously fails containment.
            surface_direction = (direction_by_relation or {}).get(
                matched_entry["relation_id"], {}
            ).get(normalize_label(matched_entry["relation"]), "forward")
            if surface_direction == "inverted":
                subj_for_constraint = matched_entry["object_type_id"]
                obj_for_constraint = matched_entry["subject_type_id"]
            else:
                subj_for_constraint = matched_entry["subject_type_id"]
                obj_for_constraint = matched_entry["object_type_id"]
            constraint, containment_ok = _select_constraint(
                matched_entry["relation_id"], subj_for_constraint,
                obj_for_constraint, constraint_index, hierarchy,
            )
            if constraint is not None:
                if direction == "inverse":
                    gold_domain_for_c, gold_range_for_c = gold_range, gold_domain
                else:
                    gold_domain_for_c, gold_range_for_c = gold_domain, gold_range
                c_domain_label = (type_label_by_id or {}).get(constraint.domain_type_id)
                c_range_label = (type_label_by_id or {}).get(constraint.range_type_id)
                c_domain_pair = (c_domain_label, gold_domain_for_c) if (c_domain_label and gold_domain_for_c) else None
                c_range_pair = (c_range_label, gold_range_for_c) if (c_range_label and gold_range_for_c) else None
                if c_domain_pair or c_range_pair:
                    try:
                        constraint_judgment = extractor.compare_types_with_llm(c_domain_pair, c_range_pair)
                    except Exception:
                        logger.exception("compare_types_with_llm (constraint) failed for sentence %s, gold triple %r",
                                          record["id"], gold)
                        call_failures += 1
                        constraint_judgment = {"domain": None, "range": None, "judge_call_failed": True}
                constraint_info = {
                    "relation_id": constraint.relation_id,
                    "domain_type_id": constraint.domain_type_id, "range_type_id": constraint.range_type_id,
                    "domain_type_label": c_domain_label, "range_type_label": c_range_label,
                    "support": constraint.support, "total": constraint.total,
                    "pca_confidence": constraint.pca_confidence, "strength": constraint.strength.value,
                    "containment_ok": containment_ok,
                }

        gold_results.append({
            "gold": gold, "matched": True, "direction": direction,
            "generated": {
                "subject": matched_entry["subject"], "relation": matched_entry["relation"],
                "object": matched_entry["object"],
                "subject_type": matched_entry["subject_type_label"],
                "object_type": matched_entry["object_type_label"],
            },
            "gold_domain": gold_domain, "gold_range": gold_range,
            "type_judgment": type_judgment,
            "constraint": constraint_info,
            "constraint_judgment": constraint_judgment,
        })

    return {
        "sentence_id": record["id"], "sent": record.get("sent", ""),
        "source_ontology_id": record.get("_source_ontology_id"),
        "gold_results": gold_results, "extras": remaining, "call_failures": call_failures,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Aggregation (reused for the overall summary AND each domain's own summary)
# ═══════════════════════════════════════════════════════════════════════════════

def summarize(results: list[dict], total_generated: int, *, ontology_id, judge_model: str) -> dict:
    """Compute the same recall / direction / extras / type-judgment-distribution
    summary from any subset of _judge_sentence results -- called once over
    every result (the "overall" summary) and once per domain (the
    "by_domain" breakdown), so a multi-domain combined run is directly
    comparable to N separate single-domain runs' own summaries."""
    total_gold = sum(len(r["gold_results"]) for r in results)
    total_call_failures = sum(r["call_failures"] for r in results)
    matched = [g for r in results for g in r["gold_results"] if g["matched"]]
    unmatched = [g for r in results for g in r["gold_results"] if not g["matched"]]
    failed_calls = [g for g in unmatched if g["reason"] == "judge_call_failed"]
    genuine_misses = [g for g in unmatched if g["reason"] != "judge_call_failed"]

    total_scored = total_gold - len(failed_calls)
    recall = len(matched) / total_scored if total_scored else None

    same_count = sum(1 for g in matched if g["direction"] == "same")
    inverse_count = sum(1 for g in matched if g["direction"] == "inverse")
    total_extras = sum(len(r["extras"]) for r in results)

    def _label_distribution(side: str, judgment_key: str = "type_judgment") -> dict:
        dist = {label: 0 for label in TYPE_JUDGE_LABELS}
        dist["no_gold_type"] = 0
        dist["judge_call_failed"] = 0
        for g in matched:
            tj = g[judgment_key]
            if tj.get("judge_call_failed"):
                dist["judge_call_failed"] += 1
                continue
            label = tj.get(side)
            if label in TYPE_JUDGE_LABELS:
                dist[label] += 1
            else:
                dist["no_gold_type"] += 1
        return dist

    # ── Constraint Plausibility (Track B) aggregates ──────────────────────
    with_constraint = [g for g in matched if g.get("constraint") is not None]
    containment_violations = [g for g in with_constraint if g["constraint"]["containment_ok"] is False]
    strength_counts: dict[str, int] = defaultdict(int)
    for g in with_constraint:
        strength_counts[g["constraint"]["strength"]] += 1

    return {
        "ontology_id": ontology_id,
        "judge_model": judge_model,
        "total_gold_triples": total_gold,
        "total_generated_triples": total_generated,
        "total_call_failures": total_call_failures,
        "recall_excl_call_failures": recall,
        "matched_same_direction": same_count,
        "matched_inverse_direction": inverse_count,
        "unmatched_genuine": len(genuine_misses),
        "unmatched_judge_call_failed": len(failed_calls),
        "extra_generated_triples": total_extras,
        "extra_rate": total_extras / total_generated if total_generated else None,
        "domain_type_judgment_distribution": _label_distribution("domain"),
        "range_type_judgment_distribution": _label_distribution("range"),
        "constraint_domain_judgment_distribution": _label_distribution("domain", "constraint_judgment"),
        "constraint_range_judgment_distribution": _label_distribution("range", "constraint_judgment"),
        "constraint_matched_pairs_with_constraint": len(with_constraint),
        "constraint_matched_pairs_without_constraint": len(matched) - len(with_constraint),
        "constraint_containment_violations": len(containment_violations),
        "constraint_strength_tier_counts": dict(strength_counts),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def _load_existing_results(output_path: Path) -> list[dict]:
    """Resume support: if --output already exists (a previous run was
    interrupted, e.g. by an API account running out of credits mid-domain),
    return its per_sentence results so already-judged sentences aren't
    re-paid-for. Returns [] if the file doesn't exist or can't be parsed
    (a truncated/corrupt partial write from a hard kill -- better to redo
    that one file's sentences than crash on it)."""
    if not output_path.exists():
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("per_sentence", [])
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not parse existing %s, starting fresh", output_path)
        return []


def _write_output(
    results: list[dict], ground_truth: list[dict], buckets: dict, args, output_path: Path,
) -> dict:
    """Compute overall + by_domain summaries from `results` and write the
    full output file. Called periodically during judging (not just once at
    the end) so a mid-run interruption -- e.g. an API account running out
    of credits -- leaves a valid, already-resumable file on disk instead of
    losing every dollar spent so far."""
    overall_ontology_id = args.ontology_ids[0] if len(args.ontology_ids) == 1 else "+".join(args.ontology_ids)
    overall_summary = summarize(
        results, total_generated=sum(len(b) for b in buckets.values()),
        ontology_id=overall_ontology_id, judge_model=args.judge_model,
    )

    by_domain = {}
    for ontology_id in args.ontology_ids:
        domain_sentence_ids = {r["id"] for r in ground_truth if r["_source_ontology_id"] == ontology_id}
        domain_results = [r for r in results if r["sentence_id"] in domain_sentence_ids]
        domain_total_generated = sum(len(buckets.get(sid, [])) for sid in domain_sentence_ids)
        by_domain[ontology_id] = summarize(
            domain_results, total_generated=domain_total_generated,
            ontology_id=ontology_id, judge_model=args.judge_model,
        )

    output = {"overall": overall_summary, "by_domain": by_domain, "per_sentence": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp_path.replace(output_path)  # atomic on POSIX -- never leaves a half-written judge_results.json
    return {"overall": overall_summary, "by_domain": by_domain}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", required=True,
                         help="Combined triplets.jsonl -- one file regardless of how many "
                              "--ontology-ids are given (see extract_triplets.py's --ontology-id "
                              "tagging for how a multi-domain corpus stays one file).")
    parser.add_argument("--data-dir", required=True,
                         help="Parent dir containing <ontology-id>/{ground_truth.jsonl,ontology.ttl} "
                              "for each requested domain, e.g. data/text2kgbench/wikidata_tekgen.")
    parser.add_argument("--ontology-ids", nargs="+", required=True,
                         help="One or more ontology ids to evaluate, e.g. ont_2_music ont_1_movie. "
                              "Each sentence is scored against ITS OWN domain's ontology (tracked via "
                              "source_ontology_id), so this also works against a single combined "
                              "pipeline run spanning several domains, not just one domain at a time.")
    parser.add_argument("--output-dir", help="Pipeline output_dir -- the latest run_<n> checkpoint "
                                              "is used unless --checkpoints-dir is given explicitly.")
    parser.add_argument("--checkpoints-dir", help="Explicit output_dir/checkpoints/run_<n> dir "
                                                     "(overrides --output-dir's auto-latest lookup).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-4o",
                         help="Chat-completions model name for the judge LLM -- deliberately "
                              "independent of whatever model built the KG (see module docstring). "
                              "Must match the exact casing/slug the endpoint expects.")
    parser.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-api-key-env", default="OPENROUTER_KEY",
                         help="Env var holding the judge LLM's API key.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--constraint-plausibility", action="store_true",
                         help="Also run Constraint Plausibility (Track B, see module docstring): "
                              "for every matched pair, select the relation's mined domain/range "
                              "constraint (recomputed here with include_below_threshold=True so "
                              "no candidate is excluded) and judge it against gold with the same "
                              "rubric Domain/Range Type Matching already uses. Needs "
                              "hierarchy_induction.pkl in the checkpoint dir (relation_dedup.pkl / "
                              "type_dedup.pkl are already required either way) and re-runs relation "
                              "direction classification (one LLM call per canonical relation, using "
                              "--judge-model/--judge-base-url/--judge-api-key-env -- the judge "
                              "model, not the pipeline's own backbone, since this is pure "
                              "evaluation-time recomputation of a symbolic LCA, not a rebuild of "
                              "the KG itself).")
    args = parser.parse_args()

    if args.checkpoints_dir:
        run_dir = Path(args.checkpoints_dir)
    elif args.output_dir:
        run_dir = _latest_run_dir(Path(args.output_dir))
    else:
        raise SystemExit("Pass either --checkpoints-dir or --output-dir")

    logger.info("Loading canonical vocabularies from %s", run_dir)
    relation_vocab = _load_pickle(run_dir / "relation_dedup.pkl")
    type_vocab = _load_pickle(run_dir / "type_dedup.pkl")

    data_dir = Path(args.data_dir)
    ontologies: dict[str, Text2KGOntology] = {}
    ground_truth: list[dict] = []
    for ontology_id in args.ontology_ids:
        domain_dir = data_dir / ontology_id
        ontology = parse_ontology(domain_dir / "ontology.ttl")
        ontologies[ontology_id] = ontology
        records = _load_jsonl(str(domain_dir / "ground_truth.jsonl"))
        for r in records:
            r["_source_ontology_id"] = ontology_id
        ground_truth.extend(records)
        logger.info("Domain %s: %d classes, %d relations, %d ground-truth sentences",
                    ontology_id, len(ontology.class_label_by_iri), len(ontology.relations), len(records))

    triplets = _load_jsonl(args.triplets)
    buckets = build_sentence_buckets(triplets, relation_vocab, type_vocab)
    logger.info("Loaded %d generated triplets across %d sentences (%d domains)",
                len(triplets), len(buckets), len(args.ontology_ids))

    api_key = os.environ.get(args.judge_api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.judge_api_key_env!r} is not set")
    extractor = LLMTripletExtractor(api_key=api_key, model=args.judge_model, base_url=args.judge_base_url)

    constraint_index = None
    hierarchy = None
    type_label_by_id = None
    direction_by_relation = None
    if args.constraint_plausibility:
        logger.info("=== Constraint Plausibility enabled: loading hierarchy, recomputing constraints "
                     "(include_below_threshold=True) ===")
        hierarchy_result = _load_pickle(run_dir / "hierarchy_induction.pkl")
        hierarchy = hierarchy_result.hierarchy

        type_label_by_id = {tid: item.canonical_label for tid, item in type_vocab.items.items()}
        for st in hierarchy_result.synthesized_types:
            type_label_by_id.setdefault(st.type_id, st.canonical_label)

        # update_relation_type_map is NOT idempotent (see pipeline.py) --
        # relation_dedup.pkl is checkpointed BEFORE this bridge runs in the
        # original pipeline, so it must be (re-)applied here exactly once.
        relation_vocab_for_constraints = update_relation_type_map(relation_vocab, type_vocab)
        constraints, direction_by_relation = induce_constraints(
            triplets=triplets, type_vocab=type_vocab, relation_vocab=relation_vocab_for_constraints,
            hierarchy=hierarchy, llm_extractor=extractor, config=ConstraintConfig(),
            include_below_threshold=True,
        )
        constraint_index = build_constraint_index(constraints)
        logger.info("Constraint index: %d candidate signatures across %d relations",
                    len(constraints), len(constraint_index))

    output_path = Path(args.output)
    loaded = _load_existing_results(output_path)
    # A sentence that hit call failures (e.g. every remaining call failing
    # once an account runs out of credits) still gets a result dict back
    # from _judge_sentence -- it doesn't raise. Treating that as "done"
    # would permanently lock in a call-failure result instead of retrying
    # it once the underlying issue (credits, rate limit, ...) is fixed, so
    # only clean (zero call-failure) sentences count as already judged.
    results = [r for r in loaded if r.get("call_failures", 0) == 0]
    done_ids = {r["sentence_id"] for r in results}
    if loaded:
        logger.info("Resuming: %d/%d sentences already judged cleanly, %d had call failures and will "
                    "be retried, %d remaining", len(done_ids), len(ground_truth),
                    len(loaded) - len(done_ids), len(ground_truth) - len(done_ids))

    tasks = [
        (record, buckets.get(record["id"], []), ontologies[record["_source_ontology_id"]])
        for record in ground_truth if record["id"] not in done_ids
    ]

    # Checkpointed periodically (not just once at the end): an API account
    # running out of credits mid-run (a real, observed failure mode) must
    # not lose every dollar of judging already paid for -- see
    # _write_output's docstring and _load_existing_results above, which is
    # what makes re-running this exact command afterward a resume rather
    # than a full re-judge from scratch.
    if tasks:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_judge_sentence, record, bucket, ontology, extractor,
                                 constraint_index, hierarchy, type_label_by_id, direction_by_relation): record["id"]
                for record, bucket, ontology in tasks
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                results.append(future.result())
                if completed % 25 == 0 or completed == len(futures):
                    logger.info("Judged %d/%d sentences", completed, len(futures))
                    _write_output(results, ground_truth, buckets, args, output_path)

    summaries = _write_output(results, ground_truth, buckets, args, output_path)
    overall_summary, by_domain = summaries["overall"], summaries["by_domain"]

    total_call_failures = overall_summary["total_call_failures"]
    total_gold = overall_summary["total_gold_triples"]
    # Same discipline as scripts/mine_benchmark/run_judge_eval.py: a judge
    # call failure must never look like a "0" result in the summary above
    # (it's excluded from recall via total_scored inside summarize()) -- but
    # it also must not be silently invisible, so it's surfaced loudly here.
    if total_call_failures:
        logger.error(
            "%d/%d judge calls FAILED (excluded from recall, NOT counted as misses). "
            "If this is a large fraction (e.g. an account ran out of credits mid-run), fix the "
            "underlying cause and re-run this exact command -- already-judged sentences resume "
            "instead of being re-paid-for, but the sentences that hit call failures will be retried.",
            total_call_failures, total_gold,
        )

    recall = overall_summary["recall_excl_call_failures"]
    logger.info(
        "Overall recall (excl. call failures): %s. Same=%d Inverse=%d Extras=%d/%d. Results written to %s",
        f"{recall:.2%}" if recall is not None else "N/A",
        overall_summary["matched_same_direction"], overall_summary["matched_inverse_direction"],
        overall_summary["extra_generated_triples"], overall_summary["total_generated_triples"], output_path,
    )
    for ontology_id, s in by_domain.items():
        r = s["recall_excl_call_failures"]
        logger.info("  %s: recall=%s (%d/%d gold matched)",
                    ontology_id, f"{r:.2%}" if r is not None else "N/A",
                    s["matched_same_direction"] + s["matched_inverse_direction"], s["total_gold_triples"])

    if args.constraint_plausibility:
        logger.info(
            "Constraint Plausibility: %d/%d matched pairs had an applicable constraint "
            "(%d did not), %d containment violation(s) [should be 0 -- see module docstring]. "
            "Domain distribution: %s. Range distribution: %s.",
            overall_summary["constraint_matched_pairs_with_constraint"],
            overall_summary["constraint_matched_pairs_with_constraint"]
                + overall_summary["constraint_matched_pairs_without_constraint"],
            overall_summary["constraint_matched_pairs_without_constraint"],
            overall_summary["constraint_containment_violations"],
            overall_summary["constraint_domain_judgment_distribution"],
            overall_summary["constraint_range_judgment_distribution"],
        )


if __name__ == "__main__":
    main()
