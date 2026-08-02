"""
Compare QA metrics across several run_qa_eval.py outputs
============================================================

For comparing KGs built with different models / max_depth / seed_roots
(each its own pipeline run_<n>, each already scored by run_qa_eval.py into
its own qa_results.json) side by side on exact-match / F1.

Usage:
    python -m scripts.musique_qa_eval.compare_qa_reports \\
        --reports output/musique_qa/run_9_qa_results.json:gpt_oss_depth50 \\
                  output/musique_qa/run_10_qa_results.json:qwen_depth50 \\
                  output/musique_qa/run_14_qa_results.json:gpt_oss_dolce_depth50 \\
        --output output/musique_qa/comparison.json

Each --reports entry is PATH[:LABEL]; LABEL defaults to the report's
run_dir (from its own summary) if omitted.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_METRICS = ("exact_match", "f1", "num_samples",
            "num_samples_with_no_resolved_subgraph", "num_samples_with_no_linked_entities")


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x) -> str:
    return f"{x:.2%}" if isinstance(x, float) else str(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", nargs="+", required=True, help="PATH[:LABEL] entries.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    comparison = {}
    for entry in args.reports:
        path, _, label = entry.partition(":")
        summary = _load(path)["summary"]
        label = label or summary.get("run_dir", path)
        comparison[label] = {m: summary.get(m) for m in _METRICS}
        comparison[label]["qa_model"] = summary.get("qa_model")
        comparison[label]["source"] = path

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    logger.info("%-30s %10s %10s %10s", "run", "EM", "F1", "n")
    for label, s in sorted(comparison.items(), key=lambda kv: kv[1]["exact_match"] or 0, reverse=True):
        logger.info("%-30s %10s %10s %10s", label, _fmt(s["exact_match"]), _fmt(s["f1"]), s["num_samples"])

    logger.info("Comparison written to %s", output_path)


if __name__ == "__main__":
    main()
