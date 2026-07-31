"""
Run the ontology discovery pipeline over the combined MINE triplets
======================================================================

Thin wrapper around src.ontodisco.pipeline.run_pipeline(): loads a base
config (configs/gpt_oss.yaml by default), overrides input_path/output_dir/
corpus_id to point at the MINE corpus, and runs all five orchestrated steps
(Relation Canonicalization -> Type Canonicalization -> Hierarchy Induction ->
Entity Deduplication -> Constraint Induction).

Because extract_triplets.py already wrote every essay's triplets into ONE
combined JSONL file with no per-essay boundary, this produces exactly ONE
shared ontology (type hierarchy + relation vocabulary + domain/range
constraints) and ONE shared KG (canonical entity vocabulary) spanning the
whole MINE corpus -- not 101 separate per-essay graphs.

Usage:
    python -m scripts.mine_benchmark.run_pipeline_mine \\
        --config configs/gpt_oss.yaml \\
        --input data/mine/triplets.jsonl \\
        --output-dir output/mine \\
        --corpus-id mine_benchmark_pilot
    # add --resume to continue the most recent run_<n> after a crash
"""

from __future__ import annotations

import argparse
import logging

from src.ontodisco.pipeline import load_config, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gpt_oss.yaml")
    parser.add_argument("--input", default="data/mine/triplets.jsonl")
    parser.add_argument("--output-dir", default="output/mine")
    parser.add_argument("--corpus-id", default="mine_benchmark")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.input_path = args.input
    config.output_dir = args.output_dir
    config.corpus_id = args.corpus_id

    logger.info(
        "Running ontodisco pipeline: input=%s output_dir=%s model=%s resume=%s",
        config.input_path, config.output_dir, config.llm.model, args.resume,
    )
    result = run_pipeline(config, resume=args.resume)

    logger.info(
        "Pipeline complete: %d canonical types, %d canonical relations, %d canonical entities, "
        "%d hierarchy edges (%d roots), %d constraints",
        result.type_vocab.num_canonical_types,
        result.relation_vocab.num_canonical_relations,
        result.entity_vocab.num_canonical_entities,
        len(result.hierarchy_result.hierarchy.edges),
        len(result.hierarchy_result.hierarchy.roots),
        len(result.constraints),
    )


if __name__ == "__main__":
    main()
