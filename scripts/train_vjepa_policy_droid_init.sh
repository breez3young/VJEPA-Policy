#!/usr/bin/env bash
set -euo pipefail

# LIBERO policy fine-tuning initialized from the completed DROID predictor
# checkpoint.  Only Predictor weights are imported; the Action Expert and all
# optimizer/scheduler state start new.  The 48+48 packed state contract must
# match the DROID predictor's proprio projection.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/configs/recipes/libero.env"

: "${LIBERO_DATA_ROOT:?Set LIBERO_DATA_ROOT to the directory containing the four LIBERO datasets}"
: "${VJEPA2_ENCODER_CHECKPOINT:?Set VJEPA2_ENCODER_CHECKPOINT to the frozen encoder checkpoint}"
: "${TEXT_EMBEDDING_CACHE:?Set TEXT_EMBEDDING_CACHE to the LIBERO T5 cache}"
DATASET_STATS=${DATASET_STATS:-}
: "${PREDICTOR_INIT:?Set PREDICTOR_INIT to the DROID predictor checkpoint to transfer}"
export PREDICTOR_INIT
export OUTDIR=${OUTDIR:-$ROOT/runs/libero_vjepa_policy_droid_init_packed48_t5len128_gbs128_step21360_seed7}

exec bash "$ROOT/scripts/train_vjepa_policy.sh" "$@"
