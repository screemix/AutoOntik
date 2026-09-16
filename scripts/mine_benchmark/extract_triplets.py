"""
Step 0 -- Typed Triplet Extraction over the MINE essays
==========================================================

Runs LLMTripletExtractor.extract_triplets_from_text() once per essay (Step 0
is implemented as a callable but NOT orchestrated as a corpus-level pipeline
step anywhere in src/ontodisco -- see CLAUDE.md sec. 3). This script is the
missing corpus-level driver for the MINE benchmark specifically: one call per
essay (each essay is a few hundred to ~1000 words, well within a single
completion, so no chunking is needed -- consistent with there being no
chunking module in this repo).

Every triplet is tagged with its source essay ("source_essay_id" /
"source_essay_topic") purely for traceability. Downstream pipeline code
(src/ontodisco/pipeline.py and friends) only ever reads
subject/subject_type/relation/object/object_type from these dicts, so the
extra keys are harmless.

Crucially, ALL essays' triplets are written into ONE combined JSONL file --
there is no per-essay boundary preserved anywhere in run_pipeline() itself,
so feeding this single file to it produces exactly ONE shared ontology and
ONE shared KG spanning every essay, rather than 101 separate per-essay
graphs (which is what kg-gen's own MINE eval does).

Resumable: re-running skips essay ids already present in the output file.

Usage:
    python -m scripts.mine_benchmark.extract_triplets \\
        --essays data/mine/essays.json \\
        --output data/mine/triplets.jsonl \\
        --config configs/gpt_oss.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from src.ontodisco.pipeline import load_config
from src.ontodisco.utils.openai_utils import LLMTripletExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _already_done(output_path: Path) -> set[int]:
    done: set[int] = set()
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
            if "source_essay_id" in row:
                done.add(row["source_essay_id"])
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-passes", type=int, default=2,
                        help="Extra full passes over essays that failed, before giving up on them. "
                             "A permanently-skipped essay contributes ZERO triplets, so every query "
                             "about it scores near 0: on the gpt-4o-mini run, 6 skipped essays cost "
                             "90 of 1500 queries (6%% of MINE) and ~3.5 accuracy points.")
    parser.add_argument("--essays", default="data/mine/essays.json")
    parser.add_argument("--output", default="data/mine/triplets.jsonl")
    parser.add_argument("--config", default="configs/gpt_oss.yaml",
                         help="Pipeline YAML config to read the LLM client settings from "
                              "(model / api_key_env / base_url) -- reused so extraction "
                              "uses the same model as the rest of the run.")
    args = parser.parse_args()

    config = load_config(args.config)
    api_key = os.environ.get(config.llm.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {config.llm.api_key_env!r} is not set")
    proxy = os.environ.get(config.llm.proxy_key_env) if config.llm.proxy_key_env else None

    extractor = LLMTripletExtractor(api_key=api_key, model=config.llm.model, base_url=config.llm.base_url, proxy=proxy)

    with open(args.essays, "r", encoding="utf-8") as f:
        essays = json.load(f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _already_done(output_path)
    if done_ids:
        logger.info("Resuming: %d/%d essays already extracted", len(done_ids), len(essays))

    num_triplets_written = 0
    num_failed = 0
    pending = [e for e in essays if e["id"] not in done_ids]
    failed_essays: list = []
    with open(output_path, "a", encoding="utf-8") as out_f:
      for _pass in range(args.retry_passes + 1):
        if _pass:
            if not failed_essays:
                break
            logger.info("Retry pass %d/%d over %d essay(s) that failed: %s",
                        _pass, args.retry_passes, len(failed_essays),
                        [e["id"] for e in failed_essays])
            pending, failed_essays = failed_essays, []
        for essay in pending:
            essay_id = essay["id"]
            if essay_id in done_ids:
                continue

            logger.info("Extracting essay %s: %r (%d chars)", essay_id, essay["topic"], len(essay["content"]))
            # LLMTripletExtractor tracks retry state (_refine_attempt /
            # _prev_error) at the INSTANCE level, not per-call -- without
            # resetting it here, essay N's system prompt gets a fabricated
            # "(Previous attempt #N-1 failed with error: None...)" tacked
            # on, carried over from essay N-1's successful (non-error) call
            # simply incrementing the same counter. Reset before every essay
            # so each one starts from a clean retry state.
            extractor.reset_error_state()
            try:
                result = extractor.extract_triplets_from_text(essay["content"])
            except Exception:
                logger.exception("Extraction failed for essay %s, queued for retry", essay_id)
                failed_essays.append(essay)
                continue

            if not isinstance(result, dict) or "triplets" not in result:
                logger.warning("Essay %s: unparseable extraction response, queued for retry", essay_id)
                failed_essays.append(essay)
                continue

            triplets = result["triplets"]
            for triplet in triplets:
                if not isinstance(triplet, dict):
                    continue
                triplet["source_essay_id"] = essay_id
                triplet["source_essay_topic"] = essay["topic"]
                out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
            out_f.flush()
            num_triplets_written += len(triplets)
            done_ids.add(essay_id)
            logger.info("Essay %s: extracted %d triplets (running total: %d)",
                        essay_id, len(triplets), num_triplets_written)

    num_failed = len(failed_essays)
    if failed_essays:
        logger.error(
            "%d essay(s) STILL failed after %d retry pass(es) and contribute NO triplets: %s -- "
            "every benchmark query about them will score ~0",
            num_failed, args.retry_passes, [e["id"] for e in failed_essays],
        )

    prompt_tokens, completion_tokens = extractor.calculate_used_tokens()
    logger.info(
        "Done. %d triplets written to %s. %d essays failed. Tokens: prompt=%d completion=%d cost=$%.4f",
        num_triplets_written, output_path, num_failed, prompt_tokens, completion_tokens,
        extractor.calculate_cost(),
    )


if __name__ == "__main__":
    main()
