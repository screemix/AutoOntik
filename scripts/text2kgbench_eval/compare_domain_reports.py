"""
Compare per-domain metrics: N separate single-domain KGs vs one combined
multi-domain KG
================================================================================

Reads:
  - --separate: judge_results.json from N single-domain runs (run_all.sh),
    each carrying that domain's own "overall" summary.
  - --combined: judge_results.json from ONE combined multi-domain run
    (run_multi_domain.sh's "combined" mode), carrying a "by_domain" breakdown
    (see run_judge_eval.py's multi-domain --ontology-ids support).

Matches domains by ontology_id and prints/writes a side-by-side comparison:
does building one shared ontology across several domains help or hurt each
individual domain's recall/extras, compared to giving that domain its own
dedicated pipeline run? This is the actual question a "does schema-free
discovery separate unrelated domains correctly" experiment is asking.

Usage:
    python -m scripts.text2kgbench_eval.compare_domain_reports \\
        --separate output/text2kgbench/wikidata_tekgen/ont_2_music/judge_results.json \\
                   output/text2kgbench/wikidata_tekgen/ont_1_movie/judge_results.json \\
        --combined output/text2kgbench/wikidata_tekgen/_combined_ont_2_music_ont_1_movie/judge_results.json \\
        --output output/text2kgbench/wikidata_tekgen/multi_domain_comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_METRICS = ("recall_excl_call_failures", "extra_rate", "total_gold_triples", "total_generated_triples")


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x) -> str:
    return f"{x:.2%}" if isinstance(x, float) else "N/A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separate", nargs="+", required=True,
                         help="judge_results.json paths from single-domain runs.")
    parser.add_argument("--combined", required=True,
                         help="judge_results.json path from one combined multi-domain run.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    separate_by_domain = {}
    for path in args.separate:
        summary = _load(path)["overall"]
        separate_by_domain[summary["ontology_id"]] = summary

    combined_by_domain = _load(args.combined)["by_domain"]

    all_domains = sorted(set(separate_by_domain) | set(combined_by_domain))
    comparison = {}
    for domain in all_domains:
        sep = separate_by_domain.get(domain)
        comb = combined_by_domain.get(domain)
        comparison[domain] = {
            "separate": {m: sep[m] for m in _METRICS} if sep else None,
            "combined": {m: comb[m] for m in _METRICS} if comb else None,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    logger.info("%-20s %14s %14s %12s %12s", "domain", "recall(separate)", "recall(combined)", "extra(sep)", "extra(comb)")
    for domain in all_domains:
        sep, comb = comparison[domain]["separate"], comparison[domain]["combined"]
        logger.info(
            "%-20s %14s %14s %12s %12s", domain,
            _fmt(sep["recall_excl_call_failures"]) if sep else "N/A",
            _fmt(comb["recall_excl_call_failures"]) if comb else "N/A",
            _fmt(sep["extra_rate"]) if sep else "N/A",
            _fmt(comb["extra_rate"]) if comb else "N/A",
        )

    logger.info("Comparison written to %s", output_path)


if __name__ == "__main__":
    main()
