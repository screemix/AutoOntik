"""
Download the MINE benchmark (essays + generated factual queries)
====================================================================

Source: the "josancamon/kg-gen-MINE-evaluation-dataset" Hugging Face dataset
built by kg-gen's experiments/MINE/upload_dataset.py
(https://github.com/stair-lab/kg-gen/blob/main/experiments/MINE). It bundles,
per essay: essay_topic, essay_content, and generated_queries (a list of
factual statements extracted from the essay -- these double as both the
retrieval query AND the "correct answer" the judge checks for, exactly as
kg-gen's own experiments/MINE/_1_evaluation.py uses them).

We only need essay_content (to build our own KG from) and generated_queries
(to evaluate against it) -- not kg-gen's own pre-generated kggen/graphrag_kg/
openie_kg fields, since we're building our own single ontology + KG for the
whole corpus.

Pulled via the public datasets-server REST API (plain `requests`, paginated
100 rows at a time) rather than the `datasets` library, so this script has
no dependency beyond what's already in this repo's venv.

Usage:
    python -m scripts.mine_benchmark.download_mine --limit 12
    python -m scripts.mine_benchmark.download_mine   # full 101 essays
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET = "josancamon/kg-gen-MINE-evaluation-dataset"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100


def fetch_all_rows(dataset: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        page_length = PAGE_SIZE if limit is None else min(PAGE_SIZE, limit - len(rows))
        if page_length <= 0:
            break
        resp = None
        for attempt in range(5):
            resp = requests.get(
                ROWS_URL,
                params={"dataset": dataset, "config": "default", "split": "train",
                         "offset": offset, "length": page_length},
                timeout=60,
            )
            if resp.status_code < 500:
                break
            logger.warning("datasets-server returned %d (attempt %d/5), retrying...",
                            resp.status_code, attempt + 1)
            time.sleep(2 * (attempt + 1))
        resp.raise_for_status()
        payload = resp.json()
        page_rows = [r["row"] for r in payload.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
        logger.info("Fetched %d rows so far", len(rows))
        if len(page_rows) < page_length:
            break  # last page
        if limit is not None and len(rows) >= limit:
            break
    return rows if limit is None else rows[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only download the first N essays (pilot run). Default: all.")
    parser.add_argument("--output-dir", default="data/mine",
                         help="Directory to write essays.json / queries.json into.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MINE dataset (%s) from HF datasets-server%s",
                DATASET, f", limit={args.limit}" if args.limit else "")
    rows = fetch_all_rows(DATASET, limit=args.limit)
    logger.info("Downloaded %d essays", len(rows))

    essays = []
    queries_by_essay = []
    for row in rows:
        essay_id = row["id"]
        essays.append({
            "id": essay_id,
            "topic": row.get("essay_topic", ""),
            "content": row.get("essay_content", ""),
        })
        queries_by_essay.append({
            "id": essay_id,
            "topic": row.get("essay_topic", ""),
            # Each query IS the "correct answer" statement to check for --
            # same convention kg-gen's own MINE eval uses (query text ==
            # correct_answer text, see _1_evaluation.py).
            "queries": row.get("generated_queries", []),
        })

    essays_path = output_dir / "essays.json"
    queries_path = output_dir / "queries.json"
    with open(essays_path, "w", encoding="utf-8") as f:
        json.dump(essays, f, indent=2, ensure_ascii=False)
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries_by_essay, f, indent=2, ensure_ascii=False)

    total_queries = sum(len(q["queries"]) for q in queries_by_essay)
    logger.info("Wrote %s (%d essays) and %s (%d total queries)",
                essays_path, len(essays), queries_path, total_queries)


if __name__ == "__main__":
    main()
