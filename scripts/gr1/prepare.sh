#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"
export HF_ENDPOINT=${VJEPA_HF_ENDPOINT:-https://huggingface.co}

PYTHON=${PYTHON:-python}
: "${DATA_ROOT:?Set DATA_ROOT to the GR-1 LeRobot dataset root}"
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$DATA_ROOT/artifacts}
TEXT_GPU=${TEXT_GPU:-0}
TEXT_MODEL=${TEXT_MODEL:-google/t5-v1_1-xxl}
DATASET_REVISION=${DATASET_REVISION:-ea7ac0b68f87da62f1e726771bba0fe74300802f}
ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-16}
TEXT_CONTEXT_LENGTH=${TEXT_CONTEXT_LENGTH:-48}

if (( ACTION_CHUNK_SIZE <= 0 || ACTION_CHUNK_SIZE % 4 != 0 )); then
  echo "ACTION_CHUNK_SIZE must be a positive multiple of 4" >&2
  exit 64
fi

mapfile -t DATASETS < <(find "$DATA_ROOT" -mindepth 3 -maxdepth 3 -path '*/meta/info.json' -printf '%h\n' | xargs -r -n1 dirname | sort)
if (( ${#DATASETS[@]} != 24 )); then
  echo "Expected 24 GR-1 task roots under $DATA_ROOT, found ${#DATASETS[@]}" >&2
  exit 1
fi

mkdir -p "$ARTIFACT_ROOT/text_embeddings"
"$PYTHON" scripts/gr1/compute_dataset_stats.py \
  --data-root "$DATA_ROOT" \
  --output "$ARTIFACT_ROOT/dataset_stats_absolute_minmax_ck${ACTION_CHUNK_SIZE}.json" \
  --action-chunk-size "$ACTION_CHUNK_SIZE" --expected-tasks 24 \
  --expected-episodes 1000 --revision "$DATASET_REVISION"

CUDA_VISIBLE_DEVICES="$TEXT_GPU" "$PYTHON" -m vjepa_policy.text_embeddings \
  --dataset-dirs "${DATASETS[@]}" \
  --instruction-field remarks \
  --cache-dir "$ARTIFACT_ROOT/text_embeddings" \
  --model-name "$TEXT_MODEL" \
  --context-length "$TEXT_CONTEXT_LENGTH" --device cuda

echo "[done] GR-1 stats and text embeddings are in $ARTIFACT_ROOT"
