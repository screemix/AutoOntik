#!/usr/bin/env bash
# Full Text2KGBench sweep: both models (qwen, gpt-oss) x both benchmark
# sources (wikidata_tekgen: 10 domains, dbpedia_webnlg: 19 domains), each as
# a run_multi_domain.sh --mode both (separate-KG-per-domain AND one
# combined-KG-per-source), judged throughout by a fixed --judge-model so
# every comparison shares the same judge. This is a genuinely long job
# (thousands of sentences x 2 models, ~58 separate pipeline runs + 4
# combined runs, each with 5 LLM-heavy dedup/hierarchy/constraint steps) --
# meant to be launched with nohup and left running, not watched live.
#
# Each of the 4 (source, model) combinations logs to its own file under
# $LOG_DIR and continues independently if one fails outright -- see
# run_multi_domain.sh's own per-domain failure tolerance for the finer-
# grained version of the same policy.
#
# Usage:
#   nohup ./scripts/text2kgbench_eval/run_full_sweep.sh > logs/text2kgbench_sweep/_top.log 2>&1 &
#   disown
#   # then monitor: tail -f logs/text2kgbench_sweep/*.log

set -uo pipefail
cd "$(dirname "$0")/../.."

JUDGE_MODEL="openai/gpt-4o"
JUDGE_BASE_URL="https://openrouter.ai/api/v1"
JUDGE_API_KEY_ENV="OPENROUTER_KEY"
LOG_DIR="logs/text2kgbench_sweep"
mkdir -p "$LOG_DIR"

WIKIDATA_IDS="ont_1_movie ont_2_music ont_3_sport ont_4_book ont_5_military ont_6_computer ont_7_space ont_8_politics ont_9_nature ont_10_culture"
DBPEDIA_IDS="ont_1_university ont_2_musicalwork ont_3_airport ont_4_building ont_5_athlete ont_6_politician ont_7_company ont_8_celestialbody ont_9_astronaut ont_10_comicscharacter ont_11_meanoftransportation ont_12_monument ont_13_food ont_14_writtenwork ont_15_sportsteam ont_16_city ont_17_artist ont_18_scientist ont_19_film"

CONFIGS="configs/qwen.yaml configs/gpt_oss.yaml"
SOURCES_AND_IDS=(
  "wikidata_tekgen|$WIKIDATA_IDS"
  "dbpedia_webnlg|$DBPEDIA_IDS"
)

START_TS=$(date +%s)
echo "=== Full sweep started at $(date) ==="

for CONFIG in $CONFIGS; do
  for ENTRY in "${SOURCES_AND_IDS[@]}"; do
    SOURCE="${ENTRY%%|*}"
    IDS="${ENTRY#*|}"
    MODEL_TAG="$(basename "$CONFIG" .yaml)"
    LOG_FILE="$LOG_DIR/${SOURCE}_${MODEL_TAG}.log"

    echo "=== [$( date +%H:%M:%S )] Starting $SOURCE / $MODEL_TAG -- log: $LOG_FILE ==="
    ./scripts/text2kgbench_eval/run_multi_domain.sh \
      --source "$SOURCE" \
      --ontology-ids "$IDS" \
      --config "$CONFIG" \
      --judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL" --judge-api-key-env "$JUDGE_API_KEY_ENV" \
      --mode both \
      > "$LOG_FILE" 2>&1
    STATUS=$?
    if [[ $STATUS -ne 0 ]]; then
      echo "!!! [$( date +%H:%M:%S )] $SOURCE / $MODEL_TAG exited with status $STATUS -- see $LOG_FILE" >&2
    else
      echo "=== [$( date +%H:%M:%S )] Finished $SOURCE / $MODEL_TAG ==="
    fi
  done
done

END_TS=$(date +%s)
echo "=== Full sweep finished at $(date) (elapsed: $(( (END_TS - START_TS) / 60 )) minutes) ==="
echo "Per-combination logs: $LOG_DIR/*.log"
echo "Per-combination comparison reports: output/text2kgbench/<source>/<model_tag>/multi_domain_comparison.json"
