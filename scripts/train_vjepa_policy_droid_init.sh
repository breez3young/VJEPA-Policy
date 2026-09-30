#!/usr/bin/env bash
set -euo pipefail

# LIBERO policy fine-tuning initialized from the completed DROID predictor
# checkpoint.  Only Predictor weights are imported; the Action Expert and all
# optimizer/scheduler state start new.  The 48+48 packed state contract must
# match the DROID predictor's proprio projection.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

: "${LIBERO_DATA_ROOT:?Set LIBERO_DATA_ROOT to the directory containing the four LIBERO datasets}"
: "${VJEPA2_ENCODER_CHECKPOINT:?Set VJEPA2_ENCODER_CHECKPOINT to the frozen encoder checkpoint}"
: "${TEXT_EMBEDDING_CACHE:?Set TEXT_EMBEDDING_CACHE to the LIBERO T5 cache}"
DATASET_STATS=${DATASET_STATS:-}
export ENCODER=${ENCODER:-vjepa2_vitl}
export ENCODER_FAMILY=${ENCODER_FAMILY:-vjepa2}
export ENCODER_MODEL_NAME=${ENCODER_MODEL_NAME:-vit_large}
export ENCODER_CHECKPOINT_KEY=${ENCODER_CHECKPOINT_KEY:-target_encoder}
export CONTEXT_LEN=${CONTEXT_LEN:-128}
export MAX_STATE_DIM=${MAX_STATE_DIM:-48}
export PREDICTOR_INIT=${PREDICTOR_INIT:-$ROOT/runs/droid_predictor_pretraining_prts_5fps_bs192_100k/checkpoint_step100000.pt}
export ACTIVATION_CHECKPOINTING_BLOCKS=${ACTIVATION_CHECKPOINTING_BLOCKS:-8}
export SAVE_EVERY=${SAVE_EVERY:-4000}
export OUTDIR=${OUTDIR:-$ROOT/runs/libero_vjepa_policy_droid_init_packed48_t5len128_gbs128_step21360_seed7}

exec bash "$ROOT/scripts/train_vjepa_policy.sh" "$@"
