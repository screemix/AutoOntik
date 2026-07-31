#!/usr/bin/env bash
# End-to-end MINE benchmark run: download -> extract -> ontology/KG pipeline
# -> graph construction -> LLM-as-a-judge evaluation.
#
# Two Python environments are involved on purpose (see run_judge_eval.py's
# docstring for why): this repo's own venv (Python 3.8) runs everything
# through KG construction; a separate conda env ("mine_eval", Python
# >=3.10) runs the judge evaluation, because kg-gen (used there for
# retrieval, for fidelity with kg-gen's own MINE methodology) requires
# Python >=3.10.
#
# One-time setup for the judge env, if you haven't already:
#   conda create -y -n mine_eval python=3.11
#   conda activate mine_eval && pip install kg-gen
#
# Usage:
#   ./scripts/mine_benchmark/run_all.sh [--limit N] [--resume] [--use-qualifiers]
#
#   --limit N          Only download/run on the first N essays (pilot run).
#                       Omit for the full 101-essay benchmark.
#   --resume           Pass --resume through to the pipeline step (skip steps
#                       with an existing checkpoint in output_dir/checkpoints/run_<n>).
#   --use-qualifiers   Fold extracted qualifiers (e.g. "point in time: 1903")
#                       into the graph instead of discarding them -- see
#                       build_kg_graph.py's docstring. Writes/reads
#                       kg_graph_qualifiers.json instead of kg_graph.json, so
#                       a plain rerun without this flag never clobbers it --
#                       you can build and evaluate both variants side by side.

set -euo pipefail
cd "$(dirname "$0")/../.."

LIMIT=""
RESUME=""
CONFIG="configs/gpt_oss.yaml"
CORPUS_ID="mine_benchmark"
DATA_DIR="data/mine"
OUTPUT_DIR="output/mine"
JUDGE_MODEL="Openai/gpt-4o"
JUDGE_API_KEY_ENV="OPENROUTER_KEY"
USE_QUALIFIERS=""
GRAPH_FILENAME="kg_graph.json"
JUDGE_URL="https://openrouter.ai/api/v1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="--limit $2"; CORPUS_ID="mine_benchmark_pilot"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-api-key-env) JUDGE_API_KEY_ENV="$2"; shift 2 ;;
    --use-qualifiers) USE_QUALIFIERS="--use-qualifiers"; GRAPH_FILENAME="kg_graph_qualifiers.json"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo "=== [venv] 1/4: Downloading MINE essays + queries ==="
source venv/bin/activate
python -m scripts.mine_benchmark.download_mine --output-dir "$DATA_DIR" $LIMIT

echo "=== [venv] 2/4: Extracting triplets (Step 0) into a single combined corpus ==="
python -m scripts.mine_benchmark.extract_triplets \
  --essays "$DATA_DIR/essays.json" \
  --output "$DATA_DIR/triplets.jsonl" \
  --config "$CONFIG"

echo "=== [venv] 3/4: Running the ontology discovery pipeline (single ontology + KG) ==="
python -m scripts.mine_benchmark.run_pipeline_mine \
  --config "$CONFIG" \
  --input "$DATA_DIR/triplets.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --corpus-id "$CORPUS_ID" \
  $RESUME

echo "=== [venv] 4/4: Building the kg-gen-format KG graph ==="
python -m scripts.mine_benchmark.build_kg_graph \
  --triplets "$DATA_DIR/triplets.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --graph-output "$OUTPUT_DIR/$GRAPH_FILENAME" \
  $USE_QUALIFIERS
deactivate

echo "=== [conda mine_eval] Running LLM-as-a-judge evaluation ==="
# NOTE: mine_eval lives under ~/miniconda3, but this shell's default `conda`
# (from ~/.bashrc) resolves to ~/anaconda3, which doesn't see it -- so we
# source miniconda3's conda.sh explicitly rather than trusting whatever
# `conda info --base` returns for the ambient shell.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate mine_eval
RESULTS_FILENAME="judge_results.json"
[[ -n "$USE_QUALIFIERS" ]] && RESULTS_FILENAME="judge_results_qualifiers.json"
python -m scripts.mine_benchmark.run_judge_eval \
  --graph "$OUTPUT_DIR/$GRAPH_FILENAME" \
  --queries "$DATA_DIR/queries.json" \
  --output "$OUTPUT_DIR/$RESULTS_FILENAME" \
  --judge-model "$JUDGE_MODEL" \
  --judge-api-key-env "$JUDGE_API_KEY_ENV" \
  --judge-base-url "$JUDGE_URL"
conda deactivate

echo "=== Done. Results: $OUTPUT_DIR/$RESULTS_FILENAME ==="
