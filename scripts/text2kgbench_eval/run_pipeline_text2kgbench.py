"""
Run the ontology discovery pipeline over the extracted Text2KGBench triplets
=================================================================================

Thin wrapper around src.ontodisco.pipeline.run_pipeline(): loads a base
config, overrides input_path/output_dir/corpus_id, and runs all five
orchestrated steps (Relation Canonicalization -> Type Canonicalization ->
Hierarchy Induction -> Entity Deduplication -> Constraint Induction).

Mirrors scripts/mine_benchmark/run_pipeline_mine.py. All sentences for one
ontology-domain go into one combined triplets.jsonl (see extract_triplets.py),
so this produces one shared ontology + KG for that domain, which
run_judge_eval.py then scores against Text2KGBench's gold ontology/triples
for the same domain.

Usage:
    python -m scripts.text2kgbench_eval.run_pipeline_text2kgbench \\
        --config configs/gpt_oss.yaml \\
        --input data/text2kgbench/ont_2_music/triplets.jsonl \\
        --output-dir output/text2kgbench/ont_2_music \\
        --corpus-id text2kgbench_ont_2_music
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corpus-id", default="text2kgbench")
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
