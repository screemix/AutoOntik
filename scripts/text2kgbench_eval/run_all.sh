#!/usr/bin/env bash
# End-to-end Text2KGBench eval: download -> extract -> ontology/KG pipeline -> judge eval.
#
# Unlike scripts/mine_benchmark/run_all.sh, this doesn't need a second Python
# environment: the judge step only needs pickle + the openai client (already
# in this repo's venv), no kg-gen/sentence-transformers retrieval dependency
# -- matching works directly off the pipeline's own pickled relation/type
# vocabularies rather than a constructed retrieval graph.
#
# The model that BUILDS the KG (--config) and the model that JUDGES it
# (--judge-model/--judge-base-url/--judge-api-key-env) are independent --
# comparing e.g. a qwen-built KG against a gpt-oss-built KG under a fixed
# third-party judge (gpt-4o by default) requires the judge to stay constant
# across runs. Extracted triplets and pipeline output are scoped per model
# (MODEL_TAG, derived from --config's filename) so running this twice with
# different --config values for the SAME --ontology-id produces two
# separate KGs instead of the second run silently reusing the first
# model's already-extracted triplets (extraction is resumable by sentence
# id, so a shared path would look "already done" to the second model).
#
# Usage:
#   ./scripts/text2kgbench_eval/run_all.sh [--source wikidata_tekgen] [--ontology-id ont_2_music] \
#       [--resume] [--config configs/gpt_oss.yaml] \
#       [--judge-model openai/gpt-4o] [--judge-base-url https://openrouter.ai/api/v1] [--judge-api-key-env OPENROUTER_KEY]

set -euo pipefail
cd "$(dirname "$0")/../.."

SOURCE="wikidata_tekgen"
ONTOLOGY_ID="ont_2_music"
RESUME=""
CONFIG="configs/gpt_oss.yaml"
JUDGE_MODEL="openai/gpt-4o"
JUDGE_BASE_URL="https://openrouter.ai/api/v1"
JUDGE_API_KEY_ENV="OPENROUTER_KEY"
DATA_DIR="data/text2kgbench"
OUTPUT_DIR="output/text2kgbench"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --ontology-id) ONTOLOGY_ID="$2"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
    --judge-api-key-env) JUDGE_API_KEY_ENV="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

MODEL_TAG="$(basename "$CONFIG" .yaml)"
ONT_DATA_DIR="$DATA_DIR/$SOURCE/$ONTOLOGY_ID"
MODEL_DATA_DIR="$ONT_DATA_DIR/$MODEL_TAG"
ONT_OUTPUT_DIR="$OUTPUT_DIR/$SOURCE/$MODEL_TAG/$ONTOLOGY_ID"

source venv/bin/activate

echo "=== 1/4: Downloading Text2KGBench ontology + ground truth ($SOURCE/$ONTOLOGY_ID) ==="
python -m scripts.text2kgbench_eval.download_text2kgbench --source "$SOURCE" --ontology-id "$ONTOLOGY_ID" --output-dir "$DATA_DIR"

echo "=== 2/4: Extracting triplets with $MODEL_TAG (Step 0), tagged per source sentence ==="
python -m scripts.text2kgbench_eval.extract_triplets \
  --ontology-id "$ONTOLOGY_ID" \
  --ground-truth "$ONT_DATA_DIR/ground_truth.jsonl" \
  --output "$MODEL_DATA_DIR/triplets.jsonl" \
  --config "$CONFIG"

echo "=== 3/4: Running the ontology discovery pipeline ($MODEL_TAG) ==="
python -m scripts.text2kgbench_eval.run_pipeline_text2kgbench \
  --config "$CONFIG" \
  --input "$MODEL_DATA_DIR/triplets.jsonl" \
  --output-dir "$ONT_OUTPUT_DIR" \
  --corpus-id "text2kgbench_${SOURCE}_${ONTOLOGY_ID}_${MODEL_TAG}" \
  $RESUME

echo "=== 4/4: LLM-as-a-judge evaluation ($JUDGE_MODEL) against gold triples + ontology ==="
python -m scripts.text2kgbench_eval.run_judge_eval \
  --triplets "$MODEL_DATA_DIR/triplets.jsonl" \
  --data-dir "$DATA_DIR/$SOURCE" \
  --ontology-ids "$ONTOLOGY_ID" \
  --output-dir "$ONT_OUTPUT_DIR" \
  --output "$ONT_OUTPUT_DIR/judge_results.json" \
  --judge-model "$JUDGE_MODEL" \
  --judge-base-url "$JUDGE_BASE_URL" \
  --judge-api-key-env "$JUDGE_API_KEY_ENV"

deactivate
echo "=== Done. Results: $ONT_OUTPUT_DIR/judge_results.json ==="
