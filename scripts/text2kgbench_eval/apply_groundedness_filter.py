"""
Post-filter: recompute Text2KGBench recall/metrics excluding gold triples
that are not grounded in their sentence
================================================================================

Reads an already-computed judge_results.json (from run_judge_eval.py) plus
each referenced domain's groundedness.json sidecar (from
check_groundedness.py), and recomputes the SAME summary metrics
(run_judge_eval.summarize()) after dropping any gold triple flagged
"grounded": false from every sentence's gold_results list before
aggregation. Total generated-triple counts are carried over unchanged from
the input file's own summary, since filtering only removes items from the
GOLD side (the denominator), not the extracted side.

Does not re-run any LLM matching -- this is a pure recomputation over
already-collected judge results, which is what makes it a post-filter
rather than a full re-evaluation. A gold triple whose sentence has no
groundedness data yet (not checked, or a checked call failed) is KEPT
as-is rather than dropped, matching the same fail-open policy
check_triple_groundedness_with_llm already applies to unparseable
responses -- silently dropping on missing data would bias recall upward
without anyone noticing.

Usage:
    python -m scripts.text2kgbench_eval.apply_groundedness_filter \\
        --judge-results output/text2kgbench/wikidata_tekgen/qwen/ont_2_music/judge_results.json \\
        --data-dir data/text2kgbench/wikidata_tekgen \\
        --output output/text2kgbench/wikidata_tekgen/qwen/ont_2_music/judge_results_grounded_only.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.text2kgbench_eval.run_judge_eval import summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_groundedness(data_dir: Path, ontology_ids: list[str]) -> dict[str, list[dict]]:
    merged: dict[str, list[dict]] = {}
    for ontology_id in ontology_ids:
        path = data_dir / ontology_id / "groundedness.json"
        if not path.exists():
            logger.warning("No groundedness.json for %s at %s -- its gold triples are kept unfiltered", ontology_id, path)
            continue
        merged.update(_load_json(str(path)))
    return merged


def filter_per_sentence(per_sentence: list[dict], groundedness: dict[str, list[dict]]) -> tuple[list[dict], int, int]:
    """Returns (filtered per_sentence, n_dropped, n_kept_unfiltered).
    n_kept_unfiltered counts gold triples whose sentence had no usable
    groundedness data (missing entirely, or a length mismatch against
    gold_results -- unsafe to zip positionally) and were therefore kept."""
    filtered = []
    n_dropped = 0
    n_kept_unfiltered = 0
    for sentence_result in per_sentence:
        sid = sentence_result["sentence_id"]
        gold_results = sentence_result["gold_results"]
        grounds = groundedness.get(sid)

        if grounds is None or len(grounds) != len(gold_results):
            if grounds is not None:
                logger.warning(
                    "Sentence %s: groundedness has %d entries but gold_results has %d -- "
                    "keeping this sentence's gold triples unfiltered", sid, len(grounds), len(gold_results),
                )
            n_kept_unfiltered += len(gold_results)
            filtered.append(sentence_result)
            continue

        kept_gold = [g for g, ground in zip(gold_results, grounds) if ground.get("grounded", True)]
        n_dropped += len(gold_results) - len(kept_gold)
        new_sentence_result = dict(sentence_result)
        new_sentence_result["gold_results"] = kept_gold
        filtered.append(new_sentence_result)
    return filtered, n_dropped, n_kept_unfiltered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-results", required=True)
    parser.add_argument("--data-dir", required=True,
                         help="Parent dir containing <ontology-id>/groundedness.json for each domain "
                              "referenced in --judge-results.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = _load_json(args.judge_results)
    per_sentence = data["per_sentence"]
    ontology_ids = list(data["by_domain"].keys())

    groundedness = _load_groundedness(Path(args.data_dir), ontology_ids)
    filtered_per_sentence, n_dropped, n_kept_unfiltered = filter_per_sentence(per_sentence, groundedness)
    logger.info("Dropped %d ungrounded gold triples; %d gold triples had no usable groundedness data (kept unfiltered)",
                n_dropped, n_kept_unfiltered)

    overall = data["overall"]
    filtered_overall = summarize(
        filtered_per_sentence, total_generated=overall["total_generated_triples"],
        ontology_id=overall["ontology_id"], judge_model=overall["judge_model"],
    )

    filtered_by_domain = {}
    for ontology_id, domain_summary in data["by_domain"].items():
        domain_results = [r for r in filtered_per_sentence if r.get("source_ontology_id") == ontology_id]
        filtered_by_domain[ontology_id] = summarize(
            domain_results, total_generated=domain_summary["total_generated_triples"],
            ontology_id=ontology_id, judge_model=domain_summary["judge_model"],
        )

    output = {
        "raw_overall": overall,
        "filtered_overall": filtered_overall,
        "raw_by_domain": data["by_domain"],
        "filtered_by_domain": filtered_by_domain,
        "n_dropped_ungrounded": n_dropped,
        "n_kept_unfiltered_missing_groundedness": n_kept_unfiltered,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    raw_recall = overall["recall_excl_call_failures"]
    filt_recall = filtered_overall["recall_excl_call_failures"]
    logger.info("Recall: raw=%s -> filtered (grounded-only)=%s. Results written to %s",
                f"{raw_recall:.2%}" if raw_recall is not None else "N/A",
                f"{filt_recall:.2%}" if filt_recall is not None else "N/A",
                output_path)
    for ontology_id in ontology_ids:
        r = filtered_by_domain[ontology_id]["recall_excl_call_failures"]
        logger.info("  %s: filtered recall=%s", ontology_id, f"{r:.2%}" if r is not None else "N/A")


if __name__ == "__main__":
    main()
