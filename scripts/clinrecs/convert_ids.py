"""
Convert data/clinrecs_chunked.jsonl's field names to this codebase's
standard sample-scoped provenance convention
================================================================================

data/clinrecs_chunked.jsonl uses dataset-specific field names (cr_id,
section_id) for what the rest of this codebase calls sample_id /
source_text_id elsewhere -- e.g. MuSiQue's musique_initial_triplets.jsonl
tags every raw triplet with sample_id, the exact MuSiQue instance id
(CLAUDE.md sec. 17), which scripts/musique_qa_eval/run_qa_eval.py's
per-sample scoping (build_sample_graphs / sample_entity_index) depends on.
This script renames just cr_id -> sample_id and section_id ->
source_text_id, leaving every other field (text, cr_name, mkb,
section_title, chunk_idx, total_chunks, text_no_header, char_count)
untouched, so scripts/clinrecs/extract_triplets.py doesn't need any
clinrecs-specific field-name knowledge baked into it.

Usage:
    python -m scripts.clinrecs.convert_ids \\
        --input data/clinrecs_chunked.jsonl \\
        --output data/clinrecs_chunked_converted.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def convert(input_path: str, output_path: str) -> int:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(input_path, "r", encoding="utf-8") as in_f, open(out, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "cr_id" in rec:
                rec["sample_id"] = rec.pop("cr_id")
            if "section_id" in rec:
                rec["source_text_id"] = rec.pop("section_id")
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/clinrecs_chunked.jsonl")
    parser.add_argument("--output", default="data/clinrecs_chunked_converted.jsonl")
    args = parser.parse_args()

    n = convert(args.input, args.output)
    logger.info("Converted %d records: cr_id -> sample_id, section_id -> source_text_id. Wrote %s", n, args.output)


if __name__ == "__main__":
    main()
