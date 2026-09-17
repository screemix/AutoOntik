"""
Step 0 -- Typed Triplet Extraction over the BIE biomedical NER benchmark
==========================================================================

Same role as scripts/mine_benchmark/extract_triplets.py (one LLMTripletExtractor
call per item, Step 0 isn't corpus-orchestrated anywhere in src/ontodisco -- see
CLAUDE.md sec. 3), adapted for data/BIE_test_extracted.json's schema instead of
MINE's essays.json schema.

Source file shape: a JSON file whose top-level value is a ONE-ELEMENT list
wrapping the real list of 33998 items, each:
    {"task_name": "aimed", "text": "# This is the text to analyze\\ntext = \"...\"\\n\\n# ...",
     "labels": "[\\n    Gene_or_Genome(span=\"Cdk2\"), ...]"}
"text" is itself a code-completion-style prompt wrapping the real passage --
_extract_passage() pulls out just the quoted string between `text = "` and the
following `"\\n\\n#` so our own extraction prompt sees the raw biomedical text,
not the wrapping fixture. "labels" is pre-existing NER span annotations (entity
+ coarse type only, no relations) -- unused here; Step 0 extracts full
subject/relation/object triplets from scratch, same as the MINE driver.

Supports a task-stratified pilot sample (--sample-per-task) instead of the full
33998 items, since a first full run's cost/time is unknown up front -- see the
chat this was built from: a single gpt-oss call took 37.8s for one 1.2KB
passage, and even short passages can run long due to gpt-oss's reasoning-token
verbosity, making a blind full run a multi-week commitment.

Usage:
    # Pilot: 3 items from each of the 29 tasks (~87 items)
    python -m scripts.bie_benchmark.extract_triplets \\
        --source data/BIE_test_extracted.json \\
        --output data/bie/triplets_pilot.jsonl \\
        --config configs/mine_gptoss.yaml \\
        --sample-per-task 3

    # Full run (all 33998 items) -- only after the pilot's density/cost/time
    # numbers are in hand and scope is confirmed:
    python -m scripts.bie_benchmark.extract_triplets \\
        --source data/BIE_test_extracted.json \\
        --output data/bie/triplets.jsonl \\
        --config configs/mine_gptoss.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path

from src.ontodisco.pipeline import load_config
from src.ontodisco.utils.openai_utils import LLMTripletExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TEXT_RE = re.compile(r'text = "(.*?)"\n\n#', re.DOTALL)


def _extract_passage(wrapped: str) -> str:
    m = _TEXT_RE.search(wrapped)
    return m.group(1) if m else wrapped


def _load_items(source_path: str) -> list[dict]:
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Top level is a 1-element list wrapping the real list of items.
    return data[0] if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list) else data


def _select_items(items: list[dict], sample_per_task: int | None, seed: int) -> list[dict]:
    if not sample_per_task:
        return items
    rng = random.Random(seed)
    by_task: dict[str, list[dict]] = {}
    for idx, item in enumerate(items):
        item["_item_id"] = idx  # stable id, assigned before any filtering/sampling
        by_task.setdefault(item.get("task_name", "unknown"), []).append(item)
    selected: list[dict] = []
    for task, task_items in by_task.items():
        selected.extend(rng.sample(task_items, min(sample_per_task, len(task_items))))
    selected.sort(key=lambda it: it["_item_id"])
    return selected


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
            if "source_item_id" in row:
                done.add(row["source_item_id"])
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/BIE_test_extracted.json")
    parser.add_argument("--output", default="data/bie/triplets.jsonl")
    parser.add_argument("--config", default="configs/mine_gptoss.yaml",
                         help="Pipeline YAML config to read the LLM client settings from.")
    parser.add_argument("--sample-per-task", type=int, default=None,
                         help="If set, sample this many items per task_name (task-stratified pilot) "
                              "instead of processing the full item list.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--retry-passes", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    api_key = os.environ.get(config.llm.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {config.llm.api_key_env!r} is not set")
    proxy = os.environ.get(config.llm.proxy_key_env) if config.llm.proxy_key_env else None
    extractor = LLMTripletExtractor(api_key=api_key, model=config.llm.model, base_url=config.llm.base_url, proxy=proxy)

    all_items = _load_items(args.source)
    for idx, item in enumerate(all_items):
        item.setdefault("_item_id", idx)
    items = _select_items(all_items, args.sample_per_task, args.sample_seed)
    logger.info("Loaded %d items from %s (%d selected%s)",
                len(all_items), args.source, len(items),
                f", stratified {args.sample_per_task}/task" if args.sample_per_task else "")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _already_done(output_path)
    if done_ids:
        logger.info("Resuming: %d/%d selected items already extracted", len(done_ids), len(items))

    num_triplets_written = 0
    num_failed = 0
    pending = [it for it in items if it["_item_id"] not in done_ids]
    failed_items: list = []
    with open(output_path, "a", encoding="utf-8") as out_f:
      for _pass in range(args.retry_passes + 1):
        if _pass:
            if not failed_items:
                break
            logger.info("Retry pass %d/%d over %d item(s) that failed", _pass, args.retry_passes, len(failed_items))
            pending, failed_items = failed_items, []
        for item in pending:
            item_id = item["_item_id"]
            if item_id in done_ids:
                continue
            passage = _extract_passage(item["text"])
            logger.info("Extracting item %d [%s] (%d chars)", item_id, item["task_name"], len(passage))

            extractor.reset_error_state()
            try:
                result = extractor.extract_triplets_from_text(passage)
            except Exception:
                logger.exception("Extraction failed for item %d, queued for retry", item_id)
                failed_items.append(item)
                continue

            if not isinstance(result, dict) or "triplets" not in result:
                logger.warning("Item %d: unparseable extraction response, queued for retry", item_id)
                failed_items.append(item)
                continue

            triplets = result["triplets"]
            for triplet in triplets:
                if not isinstance(triplet, dict):
                    continue
                triplet["source_item_id"] = item_id
                triplet["source_task_name"] = item["task_name"]
                out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
            out_f.flush()
            num_triplets_written += len(triplets)
            done_ids.add(item_id)
            logger.info("Item %d: extracted %d triplets (running total: %d)", item_id, len(triplets), num_triplets_written)

    num_failed = len(failed_items)
    if failed_items:
        logger.error("%d item(s) STILL failed after %d retry pass(es): %s",
                      num_failed, args.retry_passes, [it["_item_id"] for it in failed_items])

    prompt_tokens, completion_tokens = extractor.calculate_used_tokens()
    logger.info(
        "Done. %d triplets written to %s. %d items failed. Tokens: prompt=%d completion=%d cost=$%.4f",
        num_triplets_written, output_path, num_failed, prompt_tokens, completion_tokens,
        extractor.calculate_cost(),
    )


if __name__ == "__main__":
    main()
