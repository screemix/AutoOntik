"""
Step 0 -- Typed Triplet Extraction over Text2KGBench sentences
==================================================================

Runs LLMTripletExtractor.extract_triplets_from_text() once per ground-truth
sentence (Text2KGBench sentences are single sentences, well within one
completion -- no chunking needed, same reasoning as
scripts/mine_benchmark/extract_triplets.py). Every extracted raw triplet is
tagged with "source_sentence_id" (the ground-truth record's "id" field) AND
"source_ontology_id" (the --ontology-id this run was called with) so
run_judge_eval.py can (a) bucket generated triples per sentence for matching
against gold, and (b) in a multi-domain run, know WHICH domain's ontology to
score a given sentence's matches against. Downstream pipeline code
(src/ontodisco/pipeline.py and friends) only ever reads
subject/subject_type/relation/object/object_type, so the extra keys are
harmless (same pattern as scripts/mine_benchmark/extract_triplets.py's
source_essay_id tagging).

Resumable per sentence id: re-running skips ids already present in --output.
This is what makes a multi-domain COMBINED corpus possible without a
separate "multi-domain extract" script -- call this once per domain, each
time pointed at the SAME --output file (see run_multi_domain.sh's "combined"
mode): sentence ids are already globally unique across domains (Text2KGBench
prefixes them with the ontology id, e.g. "ont_2_music_test_1"), so appending
each domain's sentences in turn builds one combined triplets.jsonl that
run_pipeline_text2kgbench.py can then run as a single shared ontology/KG
across every domain.

Extraction runs CONCURRENTLY (--max-workers, default 8) -- unlike
scripts/mine_benchmark/extract_triplets.py's sequential loop, sequential
extraction over hundreds of sentences per domain (some Text2KGBench domains
have 600+) becomes the dominant cost at multi-domain scale. Each worker
thread gets its OWN LLMTripletExtractor instance rather than sharing one:
extract_triplets_from_text()'s retry bookkeeping (_refine_attempt /
_prev_error) is per-instance, read-modify-write, and NOT guarded by a lock
the way get_completion()'s token counters are -- sharing one instance across
threads would let one sentence's retry error message get attached to a
different sentence's retry attempt. Per-instance token/cost counters are
summed across all worker extractors at the end instead.

Usage:
    python -m scripts.text2kgbench_eval.extract_triplets \\
        --ontology-id ont_2_music \\
        --ground-truth data/text2kgbench/wikidata_tekgen/ont_2_music/ground_truth.jsonl \\
        --output data/text2kgbench/wikidata_tekgen/ont_2_music/triplets.jsonl \\
        --config configs/gpt_oss.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.ontodisco.pipeline import load_config
from src.ontodisco.utils.openai_utils import LLMTripletExtractor

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


def _already_done(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "source_sentence_id" in row:
                done.add(row["source_sentence_id"])
    return done


def _extract_one(record: dict, get_extractor) -> tuple[str, list[dict] | None, str | None]:
    """Returns (sentence_id, triplets, error_reason). triplets is None iff
    error_reason is set. get_extractor is called HERE, inside the function
    the ThreadPoolExecutor actually runs on the worker thread -- calling it
    at the submit() call site instead would evaluate it in the submitting
    (main) thread every time, handing every task the same extractor
    instance and defeating the whole point of per-thread isolation."""
    extractor = get_extractor()
    sentence_id = record["id"]
    sent = record["sent"]
    # See scripts/mine_benchmark/extract_triplets.py for why this reset is
    # needed every call: retry state is tracked at the LLMTripletExtractor
    # INSTANCE level, not per-call -- safe here because each thread has its
    # OWN extractor (see module docstring).
    extractor.reset_error_state()
    try:
        result = extractor.extract_triplets_from_text(sent)
    except Exception:
        logger.exception("Extraction failed for sentence %s, skipping", sentence_id)
        return sentence_id, None, "extraction_failed"

    # Not every model wraps its output in {"triplets": [...]} the way the
    # prompt asks -- observed empirically: gpt-oss does, but Qwen3
    # (configs/qwen.yaml) consistently returns a bare JSON list instead.
    # Accept either shape rather than silently skipping (and therefore
    # losing) every sentence for models that return a list.
    if isinstance(result, dict) and "triplets" in result:
        triplets = result["triplets"]
    elif isinstance(result, list):
        triplets = result
    else:
        logger.warning("Sentence %s: unparseable extraction response, skipping", sentence_id)
        return sentence_id, None, "unparseable"

    return sentence_id, [t for t in triplets if isinstance(t, dict)], None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology-id", required=True,
                         help="Text2KGBench ontology id these sentences belong to (e.g. ont_2_music) -- "
                              "tagged onto every triplet as source_ontology_id.")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/gpt_oss.yaml",
                         help="Pipeline YAML config to read the LLM client settings from "
                              "(model / api_key_env / base_url).")
    parser.add_argument("--max-workers", type=int, default=8,
                         help="Concurrent sentence extractions (each gets its own extractor "
                              "instance -- see module docstring).")
    args = parser.parse_args()

    config = load_config(args.config)
    api_key = os.environ.get(config.llm.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {config.llm.api_key_env!r} is not set")
    proxy = os.environ.get(config.llm.proxy_key_env) if config.llm.proxy_key_env else None

    thread_local = threading.local()
    all_extractors: list[LLMTripletExtractor] = []
    extractors_lock = threading.Lock()

    def get_extractor() -> LLMTripletExtractor:
        ext = getattr(thread_local, "extractor", None)
        if ext is None:
            ext = LLMTripletExtractor(api_key=api_key, model=config.llm.model, base_url=config.llm.base_url, proxy=proxy)
            thread_local.extractor = ext
            with extractors_lock:
                all_extractors.append(ext)
        return ext

    records = _load_jsonl(args.ground_truth)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _already_done(output_path)
    todo = [r for r in records if r["id"] not in done_ids]
    if done_ids:
        logger.info("Resuming: %d/%d sentences already extracted, %d remaining", len(done_ids), len(records), len(todo))

    num_triplets_written = 0
    num_failed = 0
    write_lock = threading.Lock()

    with open(output_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_extract_one, record, get_extractor): record["id"] for record in todo}
            completed = 0
            for future in as_completed(futures):
                sentence_id, triplets, error = future.result()
                completed += 1
                if error is not None:
                    num_failed += 1
                else:
                    with write_lock:
                        for triplet in triplets:
                            triplet["source_sentence_id"] = sentence_id
                            triplet["source_ontology_id"] = args.ontology_id
                            out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                        out_f.flush()
                    num_triplets_written += len(triplets)
                if completed % 25 == 0 or completed == len(todo):
                    logger.info("Extracted %d/%d sentences (%d triplets so far, %d failed)",
                                completed, len(todo), num_triplets_written, num_failed)

    total_prompt = sum(e.calculate_used_tokens()[0] for e in all_extractors)
    total_completion = sum(e.calculate_used_tokens()[1] for e in all_extractors)
    total_cost = sum(e.calculate_cost() for e in all_extractors)
    logger.info(
        "Done. %d triplets written to %s. %d sentences failed. Tokens: prompt=%d completion=%d cost=$%.4f",
        num_triplets_written, output_path, num_failed, total_prompt, total_completion, total_cost,
    )


if __name__ == "__main__":
    main()
