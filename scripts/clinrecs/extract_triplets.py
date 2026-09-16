"""
Step 0 -- Typed Triplet Extraction over clinrecs text chunks
==================================================================

Runs LLMTripletExtractor.extract_triplets_from_text() once per chunk in
the (already field-converted -- see convert_ids.py) clinrecs corpus,
reading each record's "text" field. Every extracted raw triplet keeps the
standard schema (subject, subject_type, relation, object, object_type,
qualifiers) exactly as produced by extract_triplets_from_text() -- nothing
renamed or restructured -- tagged additionally with sample_id and
source_text_id, copied straight from the source record, for provenance.
This is the same sample_id convention already used for MuSiQue
(CLAUDE.md sec. 17), so downstream per-sample scoping works the same way
without new code. Downstream pipeline code (src/ontodisco/pipeline.py and
friends) only ever reads subject/subject_type/relation/object/object_type,
so the extra keys are harmless.

Extraction runs CONCURRENTLY (--max-workers, default 8) -- at ~28k chunks,
sequential extraction is not viable. Each worker thread gets its OWN
LLMTripletExtractor instance rather than sharing one: extract_triplets_
from_text()'s retry bookkeeping (_refine_attempt / _prev_error) is
per-instance and not safe to share across threads (same reasoning as
scripts/text2kgbench_eval/extract_triplets.py).

Resumable per chunk (sample_id/source_text_id/chunk_idx -- one section can
have several chunks, so chunk_idx is part of the resumability key, not
just section-level).

Usage:
    # Pilot run first -- see module note on scale before running unbounded.
    python -m scripts.clinrecs.extract_triplets \\
        --input data/clinrecs_chunked_converted.jsonl \\
        --output data/clinrecs_triplets.jsonl \\
        --config configs/clinrecs_config.yaml \\
        --limit 20

    # Full run
    python -m scripts.clinrecs.extract_triplets \\
        --input data/clinrecs_chunked_converted.jsonl \\
        --output data/clinrecs_triplets.jsonl \\
        --config configs/clinrecs_config.yaml
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


def _load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _chunk_id(rec: dict) -> str:
    return f"{rec.get('sample_id', '')}::{rec.get('source_text_id', '')}::{rec.get('chunk_idx', '')}"


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
            if "source_chunk_id" in row:
                done.add(row["source_chunk_id"])
    return done


def _extract_one(record: dict, get_extractor) -> tuple[str, list[dict] | None, str | None]:
    """Returns (chunk_id, triplets, error_reason). triplets is None iff
    error_reason is set. get_extractor is called HERE, on the worker
    thread that actually runs this function -- calling it at the submit()
    call site instead would hand every task the same extractor instance
    (see scripts/text2kgbench_eval/extract_triplets.py for why that's
    wrong)."""
    extractor = get_extractor()
    chunk_id = _chunk_id(record)
    text = record.get("text", "")
    extractor.reset_error_state()
    try:
        result = extractor.extract_triplets_from_text(text)
    except Exception:
        logger.exception("Extraction failed for chunk %s, skipping", chunk_id)
        return chunk_id, None, "extraction_failed"

    # Not every model wraps its output in {"triplets": [...]} the way the
    # prompt asks -- some models return a bare JSON list instead (observed
    # with Qwen3, see scripts/text2kgbench_eval/extract_triplets.py).
    # Accept either shape rather than silently losing every triplet.
    if isinstance(result, dict) and "triplets" in result:
        triplets = result["triplets"]
    elif isinstance(result, list):
        triplets = result
    else:
        logger.warning("Chunk %s: unparseable extraction response, skipping", chunk_id)
        return chunk_id, None, "unparseable"

    return chunk_id, [t for t in triplets if isinstance(t, dict)], None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/clinrecs_chunked_converted.jsonl")
    parser.add_argument("--output", default="data/clinrecs_triplets.jsonl")
    parser.add_argument("--config", default="configs/clinrecs_config.yaml",
                         help="Pipeline YAML config to read the LLM client settings from "
                              "(model / api_key_env / base_url / proxy_key_env).")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N chunks (pilot run) -- the full corpus "
                              "is ~28k chunks, worth sanity-checking on a small slice first.")
    parser.add_argument("--ru", action="store_true",
                         help="Use the Russian extraction prompt "
                              "(prompts/prompt_1_with_types_and_qualifiers_ru.txt) instead of the "
                              "English default -- clinrecs text is Russian-language clinical "
                              "guidance, so this is normally what you want.")
    args = parser.parse_args()

    config = load_config(args.config)
    api_key = os.environ.get(config.llm.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {config.llm.api_key_env!r} is not set")
    proxy = os.environ.get(config.llm.proxy_key_env) if config.llm.proxy_key_env else None
    if config.llm.proxy_key_env and not proxy:
        logger.warning(
            "config.llm.proxy_key_env=%r is set but that environment variable is empty -- "
            "proceeding without a proxy", config.llm.proxy_key_env,
        )

    # LLMTripletExtractor's default system_prompt_paths always maps
    # "triplet_extraction" to the English prompt -- this is the only prompt
    # this script ever uses (it never calls cluster verification or any
    # other LLMTripletExtractor method), so a single-key override is enough
    # rather than needing to restate the full default mapping.
    system_prompt_paths = None
    if args.ru:
        system_prompt_paths = {"triplet_extraction": "prompt_1_with_types_and_qualifiers_ru.txt"}
        logger.info("Using the Russian extraction prompt (--ru)")

    thread_local = threading.local()
    all_extractors: list[LLMTripletExtractor] = []
    extractors_lock = threading.Lock()

    def get_extractor() -> LLMTripletExtractor:
        ext = getattr(thread_local, "extractor", None)
        if ext is None:
            ext = LLMTripletExtractor(api_key=api_key, model=config.llm.model,
                                       base_url=config.llm.base_url, proxy=proxy,
                                       system_prompt_paths=system_prompt_paths)
            thread_local.extractor = ext
            with extractors_lock:
                all_extractors.append(ext)
        return ext

    records = _load_jsonl(args.input, limit=args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _already_done(output_path)
    todo = [r for r in records if _chunk_id(r) not in done_ids]
    if done_ids:
        logger.info("Resuming: %d/%d chunks already extracted, %d remaining",
                    len(done_ids), len(records), len(todo))

    num_triplets_written = 0
    num_failed = 0
    write_lock = threading.Lock()

    with open(output_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_extract_one, record, get_extractor): record for record in todo}
            completed = 0
            for future in as_completed(futures):
                record = futures[future]
                chunk_id, triplets, error = future.result()
                completed += 1
                if error is not None:
                    num_failed += 1
                else:
                    with write_lock:
                        for triplet in triplets:
                            triplet["sample_id"] = record.get("sample_id")
                            triplet["source_text_id"] = record.get("source_text_id")
                            triplet["source_chunk_id"] = chunk_id
                            out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                        out_f.flush()
                    num_triplets_written += len(triplets)
                if completed % 25 == 0 or completed == len(todo):
                    logger.info("Extracted %d/%d chunks (%d triplets so far, %d failed)",
                                completed, len(todo), num_triplets_written, num_failed)

    total_prompt = sum(e.calculate_used_tokens()[0] for e in all_extractors)
    total_completion = sum(e.calculate_used_tokens()[1] for e in all_extractors)
    total_cost = sum(e.calculate_cost() for e in all_extractors)
    logger.info(
        "Done. %d triplets written to %s. %d chunks failed. Tokens: prompt=%d completion=%d cost=$%.4f",
        num_triplets_written, output_path, num_failed, total_prompt, total_completion, total_cost,
    )


if __name__ == "__main__":
    main()
