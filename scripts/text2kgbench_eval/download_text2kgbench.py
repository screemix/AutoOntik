"""
Download Text2KGBench ground-truth + ontology files for one domain
========================================================================

Source: cenguix/Text2KGBench (https://github.com/cenguix/Text2KGBench) --
an ontology-guided text-to-KG benchmark. We use it in REVERSE of its own
intended task: instead of feeding the gold ontology into extraction, we
extract with our own schema-free pipeline and score the induced ontology +
triples against Text2KGBench's gold ontology/triples (see CLAUDE.md's eval
design discussion).

Both upstream subsets are supported -- verified to share an identical file
layout and ontology TTL statement shape (owl:Class/owl:ObjectProperty with
rdfs:label/subClassOf/domain/range), just under different id namespaces:
  - "wikidata_tekgen": 10 ontologies, e.g. ont_2_music, ont_1_movie.
    Some domains' ontologies have a real rdfs:subClassOf class hierarchy
    (e.g. ont_2_music: single/album/composed_musical_work -> musical_work).
  - "dbpedia_webnlg": 19 ontologies, e.g. ont_2_musicalwork, ont_19_film.
    Checked ont_2_musicalwork.ttl specifically: 0 rdfs:subClassOf statements
    -- a flat class list, no hierarchy. Worth re-checking per-domain before
    relying on ontology.py's parent_label() for a DBpedia-WebNLG ontology,
    since this may not hold for every domain in this subset.

Only "ground_truth/" is fetched here (test-split sentences bundled with
their gold triples) -- NOT "train/" (single-triple-per-line records meant
for fine-tuning/few-shot-prompting the benchmark's own ontology-guided
generation task, out of scope for a zero-shot schema-free eval).

Usage:
    python -m scripts.text2kgbench_eval.download_text2kgbench --ontology-id ont_2_music
    python -m scripts.text2kgbench_eval.download_text2kgbench --source dbpedia_webnlg --ontology-id ont_2_musicalwork
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_ROOT = "https://raw.githubusercontent.com/cenguix/Text2KGBench/main/data"
SOURCES = ("wikidata_tekgen", "dbpedia_webnlg")


def download_ontology(source: str, ontology_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_base = f"{RAW_ROOT}/{source}"
    gt_url = f"{raw_base}/ground_truth/{ontology_id}_ground_truth.jsonl"
    ttl_url = f"{raw_base}/ontologies/owl/{ontology_id}.ttl"

    logger.info("Fetching ground truth: %s", gt_url)
    gt_resp = requests.get(gt_url, timeout=60)
    gt_resp.raise_for_status()
    (output_dir / "ground_truth.jsonl").write_text(gt_resp.text, encoding="utf-8")

    logger.info("Fetching ontology: %s", ttl_url)
    ttl_resp = requests.get(ttl_url, timeout=60)
    ttl_resp.raise_for_status()
    (output_dir / "ontology.ttl").write_text(ttl_resp.text, encoding="utf-8")

    n_sentences = sum(1 for line in gt_resp.text.splitlines() if line.strip())
    logger.info("Downloaded %s: %d ground-truth sentences -> %s", ontology_id, n_sentences, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="wikidata_tekgen", choices=SOURCES,
                         help="Which Text2KGBench subset to pull from.")
    parser.add_argument("--ontology-id", default="ont_2_music",
                         help="Ontology id within --source, e.g. ont_2_music (wikidata_tekgen) "
                              "or ont_2_musicalwork (dbpedia_webnlg).")
    parser.add_argument("--output-dir", default="data/text2kgbench",
                         help="Parent dir; files land in <output-dir>/<source>/<ontology-id>/.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.source / args.ontology_id
    download_ontology(args.source, args.ontology_id, output_dir)


if __name__ == "__main__":
    main()
