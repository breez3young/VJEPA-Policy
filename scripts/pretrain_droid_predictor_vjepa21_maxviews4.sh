#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

: "${ENCODER_CHECKPOINT:?Set ENCODER_CHECKPOINT to a V-JEPA 2.1 ViT-L checkpoint}"
export OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runs/droid_predictor_pretraining_vjepa2_1_vitl_maxviews4_prts_5fps_bs192_100k}

exec bash "$ROOT/scripts/pretrain_droid_predictor.sh" \
  --encoder vjepa2_1_vitl \
  --encoder-checkpoint-key ema_encoder \
  --max-views 4 \
  "$@"
