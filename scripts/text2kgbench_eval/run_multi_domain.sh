#!/usr/bin/env bash
# Multi-domain Text2KGBench evaluation: runs BOTH experiments for a set of
# domains and reports them side by side:
#   - "separate": N independent single-domain pipeline runs (N separate KGs/
#     ontologies), each scored against its own domain -- this is just
#     run_all.sh looped once per domain.
#   - "combined": ONE pipeline run over all domains' sentences pooled into a
#     single corpus (one shared KG/ontology spanning every domain), scored
#     per-domain via run_judge_eval.py's --ontology-ids (multi-domain
#     support) -- tests whether schema-free discovery can tell unrelated
#     domains apart on its own, not just whether it can build one clean
#     ontology when only ever shown one domain at a time.
#
# As in run_all.sh: --config (the model that BUILDS the KG) and
# --judge-model/--judge-base-url/--judge-api-key-env (the model that JUDGES
# it) are independent, and every path is scoped by MODEL_TAG (derived from
# --config's filename) so re-running with a different --config for the same
# --ontology-ids never silently reuses another model's extracted triplets.
#
# Usage:
#   ./scripts/text2kgbench_eval/run_multi_domain.sh \
#       --ontology-ids "ont_2_music ont_1_movie ont_3_sport" \
#       [--source wikidata_tekgen] [--resume] [--config configs/gpt_oss.yaml] \
#       [--judge-model openai/gpt-4o] [--judge-base-url https://openrouter.ai/api/v1] [--judge-api-key-env OPENROUTER_KEY] \
#       [--mode separate|combined|both]
#
#   --mode separate   Only run the N-independent-KGs experiment.
#   --mode combined   Only run the one-shared-KG experiment.
#   --mode both        (default) Run both, then write a side-by-side comparison.

# NOTE: does NOT use `set -e` -- a full sweep across many domains is a long,
# unattended job, and one domain's transient failure (network hiccup, a
# malformed LLM response that exhausts retries) must not abort every domain
# after it. Each risky step below is individually checked and logged instead;
# see FAILED_DOMAINS at the end.
set -uo pipefail
cd "$(dirname "$0")/../.."

SOURCE="wikidata_tekgen"
ONTOLOGY_IDS=""
RESUME=""
CONFIG="configs/gpt_oss.yaml"
JUDGE_MODEL="openai/gpt-4o"
JUDGE_BASE_URL="https://openrouter.ai/api/v1"
JUDGE_API_KEY_ENV="OPENROUTER_KEY"
MODE="both"
DATA_DIR="data/text2kgbench"
OUTPUT_DIR="output/text2kgbench"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --ontology-ids) ONTOLOGY_IDS="$2"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
    --judge-api-key-env) JUDGE_API_KEY_ENV="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$ONTOLOGY_IDS" ]]; then
  echo "Pass --ontology-ids \"ont_2_music ont_1_movie ...\" (space-separated, quoted)" >&2
  exit 1
fi
read -ra IDS <<< "$ONTOLOGY_IDS"
if [[ ${#IDS[@]} -lt 2 ]]; then
  echo "Pass at least 2 --ontology-ids -- for a single domain, use run_all.sh directly." >&2
  exit 1
fi

MODEL_TAG="$(basename "$CONFIG" .yaml)"
COMBINED_ID="_combined_$(IFS=_; echo "${IDS[*]}")"

source venv/bin/activate

FAILED_DOMAINS=()

echo "=== Downloading ${#IDS[@]} domains from $SOURCE ==="
for ID in "${IDS[@]}"; do
  python -m scripts.text2kgbench_eval.download_text2kgbench --source "$SOURCE" --ontology-id "$ID" --output-dir "$DATA_DIR" \
    || { echo "!!! Download failed for $ID, skipping it entirely for this run" >&2; FAILED_DOMAINS+=("download:$ID"); }
done

if [[ "$MODE" == "separate" || "$MODE" == "both" ]]; then
  echo "=== SEPARATE mode ($MODEL_TAG): one pipeline run per domain (${IDS[*]}) ==="
  for ID in "${IDS[@]}"; do
    ./scripts/text2kgbench_eval/run_all.sh \
      --source "$SOURCE" --ontology-id "$ID" --config "$CONFIG" $RESUME \
      --judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL" --judge-api-key-env "$JUDGE_API_KEY_ENV" \
      || { echo "!!! SEPARATE run failed for $ID, continuing with remaining domains" >&2; FAILED_DOMAINS+=("separate:$ID"); }
  done
fi

if [[ "$MODE" == "combined" || "$MODE" == "both" ]]; then
  echo "=== COMBINED mode ($MODEL_TAG): one pipeline run across all domains (${IDS[*]}) ==="
  COMBINED_DATA_DIR="$DATA_DIR/$SOURCE/$COMBINED_ID/$MODEL_TAG"
  COMBINED_OUTPUT_DIR="$OUTPUT_DIR/$SOURCE/$MODEL_TAG/$COMBINED_ID"
  mkdir -p "$COMBINED_DATA_DIR"

  for ID in "${IDS[@]}"; do
    # If SEPARATE mode already extracted this domain with this exact model
    # (the common case when --mode both), reuse those triplets instead of
    # re-running extraction from scratch -- avoids paying for (and waiting
    # on) every sentence's extraction twice. Dedup-aware append (checks
    # source_sentence_id already in the combined file) so this is also safe
    # to re-run under --resume without duplicating rows.
    PER_DOMAIN_TRIPLETS="$DATA_DIR/$SOURCE/$ID/$MODEL_TAG/triplets.jsonl"
    if [[ -f "$PER_DOMAIN_TRIPLETS" ]]; then
      echo "--- Seeding combined corpus with already-extracted $ID triplets ---"
      python3 - "$PER_DOMAIN_TRIPLETS" "$COMBINED_DATA_DIR/triplets.jsonl" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
existing = set()
try:
    with open(dst) as f:
        for line in f:
            line = line.strip()
            if line:
                existing.add(json.loads(line).get("source_sentence_id"))
except FileNotFoundError:
    pass
with open(src) as f, open(dst, "a") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if json.loads(line).get("source_sentence_id") not in existing:
            out.write(line + "\n")
PYEOF
    fi
    echo "--- Extracting $ID with $MODEL_TAG into the combined corpus (skips sentences already seeded above) ---"
    python -m scripts.text2kgbench_eval.extract_triplets \
      --ontology-id "$ID" \
      --ground-truth "$DATA_DIR/$SOURCE/$ID/ground_truth.jsonl" \
      --output "$COMBINED_DATA_DIR/triplets.jsonl" \
      --config "$CONFIG" \
      || { echo "!!! Extraction failed for $ID in combined corpus, continuing with remaining domains" >&2; FAILED_DOMAINS+=("combined-extract:$ID"); }
  done

  echo "--- Running the ontology discovery pipeline over the combined corpus ---"
  python -m scripts.text2kgbench_eval.run_pipeline_text2kgbench \
    --config "$CONFIG" \
    --input "$COMBINED_DATA_DIR/triplets.jsonl" \
    --output-dir "$COMBINED_OUTPUT_DIR" \
    --corpus-id "text2kgbench_${SOURCE}_${COMBINED_ID}_${MODEL_TAG}" \
    $RESUME \
    || { echo "!!! Combined pipeline run FAILED -- combined-mode judging will be skipped" >&2; FAILED_DOMAINS+=("combined-pipeline:$COMBINED_ID"); }

  if [[ ! " ${FAILED_DOMAINS[*]} " == *" combined-pipeline:$COMBINED_ID "* ]]; then
    echo "--- Judging the combined KG, scored per-domain ---"
    python -m scripts.text2kgbench_eval.run_judge_eval \
      --triplets "$COMBINED_DATA_DIR/triplets.jsonl" \
      --data-dir "$DATA_DIR/$SOURCE" \
      --ontology-ids "${IDS[@]}" \
      --output-dir "$COMBINED_OUTPUT_DIR" \
      --output "$COMBINED_OUTPUT_DIR/judge_results.json" \
      --judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL" --judge-api-key-env "$JUDGE_API_KEY_ENV" \
      || { echo "!!! Combined judging FAILED" >&2; FAILED_DOMAINS+=("combined-judge:$COMBINED_ID"); }
  fi
fi

if [[ "$MODE" == "both" ]]; then
  echo "=== Comparison: separate-KG vs combined-KG recall per domain ($MODEL_TAG) ==="
  COMBINED_RESULTS="$OUTPUT_DIR/$SOURCE/$MODEL_TAG/$COMBINED_ID/judge_results.json"
  SEPARATE_PATHS=()
  for ID in "${IDS[@]}"; do
    P="$OUTPUT_DIR/$SOURCE/$MODEL_TAG/$ID/judge_results.json"
    [[ -f "$P" ]] && SEPARATE_PATHS+=("$P") || echo "  (skipping $ID in comparison -- no judge_results.json, that domain failed)"
  done
  if [[ -f "$COMBINED_RESULTS" && ${#SEPARATE_PATHS[@]} -gt 0 ]]; then
    python -m scripts.text2kgbench_eval.compare_domain_reports \
      --separate "${SEPARATE_PATHS[@]}" \
      --combined "$COMBINED_RESULTS" \
      --output "$OUTPUT_DIR/$SOURCE/$MODEL_TAG/multi_domain_comparison.json"
  else
    echo "!!! Skipping comparison -- combined results and/or every separate result is missing" >&2
  fi
fi

if [[ ${#FAILED_DOMAINS[@]} -gt 0 ]]; then
  echo "=== Completed with ${#FAILED_DOMAINS[@]} failure(s): ${FAILED_DOMAINS[*]} ===" >&2
  echo "Re-run with --resume to retry -- already-completed steps/domains are checkpointed and skipped." >&2
else
  echo "=== All domains completed successfully ==="
fi

deactivate
echo "=== Done. ==="
