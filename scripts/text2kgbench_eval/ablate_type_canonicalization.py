"""
Ablation: score domain/range type-judgment using RAW (as-extracted) types
instead of canonicalized ones
================================================================================

run_judge_eval.py's triple-matching/recall step already operates on raw,
uncanonicalized subject/relation/object strings -- canonicalization
(relation_dedup.pkl / type_dedup.pkl) never touches that step. The *only*
place canonicalization actually feeds into scoring is the domain/range
type-judgment step: each matched triple's subject_type/object_type is
resolved through type_dedup.pkl before being compared against the gold
ontology's declared domain/range class.

This script isolates that one mechanism: it reuses the ALREADY-COMPUTED
match decisions (matched/not, direction) from an existing judge_results.json
-- so recall/extra-rate are identical by construction -- and re-runs ONLY
the type-judgment LLM call, using each matched triple's RAW subject_type/
object_type (looked up from the original triplets.jsonl by
(sentence_id, subject, relation, object)) instead of the canonical label
run_judge_eval.py originally used.

Usage:
    python -m scripts.text2kgbench_eval.ablate_type_canonicalization \\
        --judge-results output/text2kgbench/wikidata_tekgen/gpt_oss/ont_2_music/judge_results.json \\
        --triplets data/text2kgbench/wikidata_tekgen/ont_2_music/gpt_oss/triplets.jsonl \\
        --output output/text2kgbench/wikidata_tekgen/gpt_oss/ont_2_music/judge_results_raw_types.json \\
        --judge-model openai/gpt-4o --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env MY_OPENROUTER_KEY
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.ontodisco.utils.openai_utils import LLMTripletExtractor
from scripts.text2kgbench_eval.run_judge_eval import summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_raw_type_lookup(triplets: list[dict]) -> dict[tuple, tuple]:
    """(sentence_id, subject, relation, object) -> (raw_subject_type, raw_object_type),
    keyed on the exact raw strings run_judge_eval.py already matched against."""
    lookup = {}
    for t in triplets:
        sid = t.get("source_sentence_id")
        subj = t.get("subject", "").strip()
        rel = t.get("relation", "").strip()
        obj = t.get("object", "").strip()
        if not (sid and subj and rel and obj):
            continue
        key = (sid, subj, rel, obj)
        if key not in lookup:  # first occurrence wins -- duplicates within a sentence are rare
            lookup[key] = (t.get("subject_type", "").strip(), t.get("object_type", "").strip())
    return lookup


def _rejudge_sentence(sentence_result: dict, lookup: dict, extractor: LLMTripletExtractor) -> dict:
    """Return a copy of sentence_result with every matched gold_results
    entry's type_judgment recomputed from raw types. Unmatched entries pass
    through unchanged."""
    new_gold_results = []
    call_failures = 0
    for g in sentence_result["gold_results"]:
        if not g.get("matched"):
            new_gold_results.append(g)
            continue

        gen = g["generated"]
        key = (sentence_result["sentence_id"], gen["subject"], gen["relation"], gen["object"])
        raw_types = lookup.get(key)
        if raw_types is None:
            logger.warning("No raw-type lookup hit for %s -- keeping original (canonical) type_judgment", key)
            new_gold_results.append(g)
            continue
        raw_subject_type, raw_object_type = raw_types

        gold_domain = g.get("gold_domain")
        gold_range = g.get("gold_range")
        if g.get("direction") == "inverse":
            our_domain_side, our_range_side = raw_object_type, raw_subject_type
        else:
            our_domain_side, our_range_side = raw_subject_type, raw_object_type

        domain_pair = (our_domain_side, gold_domain) if (our_domain_side and gold_domain) else None
        range_pair = (our_range_side, gold_range) if (our_range_side and gold_range) else None

        type_judgment = {"domain": None, "range": None}
        if domain_pair or range_pair:
            try:
                type_judgment = extractor.compare_types_with_llm(domain_pair, range_pair)
            except Exception:
                logger.exception("compare_types_with_llm failed for sentence %s", sentence_result["sentence_id"])
                call_failures += 1
                type_judgment = {"domain": None, "range": None, "judge_call_failed": True}

        new_g = dict(g)
        new_g["type_judgment"] = type_judgment
        new_g["generated"] = dict(gen, subject_type=raw_subject_type, object_type=raw_object_type)
        new_gold_results.append(new_g)

    new_result = dict(sentence_result)
    new_result["gold_results"] = new_gold_results
    new_result["call_failures"] = call_failures
    return new_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-results", required=True, help="Existing judge_results.json to reuse matches from.")
    parser.add_argument("--triplets", required=True, help="Original raw triplets.jsonl for this run.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-4o")
    parser.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-api-key-env", default="OPENROUTER_KEY")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    existing = _load_json(args.judge_results)
    per_sentence = existing["per_sentence"]
    triplets = _load_jsonl(args.triplets)
    lookup = build_raw_type_lookup(triplets)
    logger.info("Loaded %d sentence results, %d raw-type lookup entries", len(per_sentence), len(lookup))

    api_key = os.environ.get(args.judge_api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.judge_api_key_env!r} is not set")
    extractor = LLMTripletExtractor(api_key=api_key, model=args.judge_model, base_url=args.judge_base_url)

    output_path = Path(args.output)
    if output_path.exists():
        done = {r["sentence_id"]: r for r in _load_json(str(output_path)).get("per_sentence", [])
                if r.get("call_failures", 0) == 0}
    else:
        done = {}
    logger.info("Resuming: %d/%d sentences already re-judged cleanly", len(done), len(per_sentence))

    todo = [s for s in per_sentence if s["sentence_id"] not in done]
    results = list(done.values())

    if todo:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_rejudge_sentence, s, lookup, extractor): s["sentence_id"] for s in todo}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                results.append(future.result())
                if completed % 25 == 0 or completed == len(futures):
                    logger.info("Re-judged %d/%d sentences", completed, len(futures))

    # Recompute summary the same way run_judge_eval.py does, from scratch
    # over the re-judged per-sentence results.
    total_generated = sum(len(s["gold_results"]) + len(s.get("extras", [])) for s in results)
    # total_generated for extra_rate isn't quite right via that formula (extras
    # already excludes matched-and-consumed candidates) -- reuse original file's
    # total_generated_triples instead, since the generated-triple pool itself
    # never changes in this ablation.
    total_generated = existing["overall"]["total_generated_triples"]
    overall = summarize(results, total_generated=total_generated,
                         ontology_id=existing["overall"]["ontology_id"], judge_model=args.judge_model)

    output = {"overall": overall, "per_sentence": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Done. Recall (should match original): %.4f (was %.4f)",
                overall["recall_excl_call_failures"] or -1,
                existing["overall"]["recall_excl_call_failures"] or -1)
    logger.info("Domain type distribution (raw): %s", overall["domain_type_judgment_distribution"])
    logger.info("Range type distribution (raw): %s", overall["range_type_judgment_distribution"])
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
