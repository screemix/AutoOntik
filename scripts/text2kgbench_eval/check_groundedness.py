"""
Check whether Text2KGBench gold triples are actually grounded in their sentence
====================================================================================

Motivation: a quick literal-substring check found that 33.2% of gold
subjects (and 42.5% of gold subjects-or-objects) do not appear verbatim in
their paired sentence -- overwhelmingly because Text2KGBench's test
sentences are pulled from mid-article (Wikipedia-derived) text, and the
subject is only established anaphorically by the source article's own
topic, not restated in every sentence. Recall measured on the subset where
the subject IS present in the sentence is 51.8%, vs. 33.4% otherwise -- an
18-point gap. No extractor given only the isolated test sentence (as
Text2KGBench's own test/ground_truth format provides) can recover a fact
whose subject it was never shown.

This script asks an LLM judge (see LLMTripletExtractor.
check_triple_groundedness_with_llm(), prompts/gold_triple_groundedness.txt)
a more precise question than a literal substring check can answer: does
THIS sentence, taken in isolation, actually support this gold triple at
all -- allowing for name-form variation (e.g. "Charles Spurgeon Johnson"
vs. "Charles S. Johnson" still counts as grounded), but not for anaphoric
reference to context outside the sentence.

Groundedness is a property of the (sentence, gold triple) pair ALONE -- it
does not depend on what any model extracted, so this is run ONCE per
domain and the result is reused across every model's judge_results.json
(see apply_groundedness_filter.py). Output is a per-domain sidecar file,
<data-dir>/<ontology-id>/groundedness.json:
    {"ont_2_music_test_1": [{"grounded": true}, {"grounded": false, "missing": "subject"}, ...], ...}
one list entry per gold triple, in the SAME order as that sentence's
"triples" list in ground_truth.jsonl (matches run_judge_eval.py's
per-sentence gold_results order exactly, which is what makes the post-hoc
filter in apply_groundedness_filter.py a plain positional zip).

Resumable per sentence id, same pattern as extract_triplets.py.

Usage:
    python -m scripts.text2kgbench_eval.check_groundedness \\
        --data-dir data/text2kgbench/wikidata_tekgen \\
        --ontology-ids ont_1_movie ont_2_music ont_3_sport ont_4_book ont_5_military \\
                       ont_6_computer ont_7_space ont_8_politics ont_9_nature ont_10_culture \\
        --judge-model openai/gpt-4o --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv

from src.ontodisco.utils.openai_utils import LLMTripletExtractor

# Unlike extract_triplets.py/run_judge_eval.py, this script never imports
# src.ontodisco.pipeline (it doesn't need pipeline config), so it doesn't
# get .env loaded as a side effect of that import chain (pipeline.py ->
# relation_dedup.py -> dedup_base.py, which calls dotenv.load_dotenv() at
# module level) -- load it explicitly instead.
dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_existing(output_path: Path) -> dict:
    if not output_path.exists():
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_sentence(record: dict, get_extractor) -> tuple[str, list[dict] | None, str | None]:
    """Returns (sentence_id, per-triple groundedness list, error_reason).
    error_reason set (list None) only on a hard call failure -- never on a
    parse failure, which check_triple_groundedness_with_llm already fails
    open on internally."""
    extractor = get_extractor()
    sentence_id = record["id"]
    sent = record["sent"]
    results = []
    try:
        for gold in record.get("triples", []):
            r = extractor.check_triple_groundedness_with_llm(sent, gold["sub"], gold["rel"], gold["obj"])
            results.append(r)
    except Exception:
        logger.exception("Groundedness check failed for sentence %s", sentence_id)
        return sentence_id, None, "call_failed"
    return sentence_id, results, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                         help="Parent dir containing <ontology-id>/ground_truth.jsonl for each domain.")
    parser.add_argument("--ontology-ids", nargs="+", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-4o")
    parser.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-api-key-env", default="OPENROUTER_KEY")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    api_key = os.environ.get(args.judge_api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.judge_api_key_env!r} is not set")

    thread_local = threading.local()

    def get_extractor() -> LLMTripletExtractor:
        ext = getattr(thread_local, "extractor", None)
        if ext is None:
            ext = LLMTripletExtractor(api_key=api_key, model=args.judge_model, base_url=args.judge_base_url)
            thread_local.extractor = ext
        return ext

    for ontology_id in args.ontology_ids:
        domain_dir = Path(args.data_dir) / ontology_id
        ground_truth = _load_jsonl(str(domain_dir / "ground_truth.jsonl"))
        output_path = domain_dir / "groundedness.json"
        existing = _load_existing(output_path)

        todo = [r for r in ground_truth if r["id"] not in existing]
        if not todo:
            logger.info("%s: all %d sentences already checked, skipping", ontology_id, len(ground_truth))
            continue
        logger.info("%s: checking %d/%d sentences (%d already cached)",
                    ontology_id, len(todo), len(ground_truth), len(existing))

        results = dict(existing)
        num_failed = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_check_sentence, record, get_extractor): record["id"] for record in todo}
            completed = 0
            for future in as_completed(futures):
                sentence_id, groundedness, error = future.result()
                completed += 1
                if error is not None:
                    num_failed += 1
                    continue
                results[sentence_id] = groundedness
                if completed % 50 == 0 or completed == len(todo):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    logger.info("%s: checked %d/%d (%d call failures so far)",
                                ontology_id, completed, len(todo), num_failed)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        total_triples = sum(len(v) for v in results.values())
        ungrounded = sum(1 for v in results.values() for t in v if not t.get("grounded", True))
        logger.info("%s: done. %d/%d sentences checked (%d call failures), %d/%d gold triples flagged ungrounded (%.1f%%)",
                    ontology_id, len(results), len(ground_truth), num_failed,
                    ungrounded, total_triples, ungrounded / total_triples * 100 if total_triples else 0.0)


if __name__ == "__main__":
    main()
