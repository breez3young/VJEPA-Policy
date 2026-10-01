#!/usr/bin/env bash
set -euo pipefail

# Full DROID predictor-only pre-training.  The loader samples the 15-Hz source
# at 5 Hz (source-frame stride 3), keeps ten frames, and drops only incomplete
# eight-frame future windows at each episode tail.  The default run uses the
# requested global batch size of 192 (24 samples per rank on 8 GPUs) and
# 100,000 optimizer steps.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/configs/recipes/droid.env"
PYTHON=${PYTHON:-python}
ACCELERATE=${ACCELERATE:-accelerate}
: "${ENCODER_CHECKPOINT:?Set ENCODER_CHECKPOINT to a pretrained visual encoder}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the DROID LeRobot dataset}"
: "${TEXT_CACHE_DIR:?Set TEXT_CACHE_DIR to the DROID T5 cache}"
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runs/droid_predictor_pretraining_prts_5fps_bs192_100k}
NUM_EPOCHS=${NUM_EPOCHS:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
RECYCLE_WORKERS_EVERY=${RECYCLE_WORKERS_EVERY:-500}

if (( NUM_GPUS != 8 )); then
  echo "DROID pretraining requires the configured 8-GPU launch; got NUM_GPUS=$NUM_GPUS" >&2
  exit 2
fi
if (( NUM_GPUS * BATCH_SIZE != GLOBAL_BATCH_SIZE )); then
  echo "Expected global batch size $GLOBAL_BATCH_SIZE, got $((NUM_GPUS * BATCH_SIZE))" >&2
  exit 2
fi

cd "$ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ACCELERATE" launch --num_processes="$NUM_GPUS" --mixed_precision=bf16 \
  scripts/pretrain_droid_predictor.py \
  --dataset-root "$DATASET_ROOT" \
  --encoder-checkpoint "$ENCODER_CHECKPOINT" \
  --text-cache-dir "$TEXT_CACHE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$NUM_EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --augmentation prts_crop_rotate \
  --num-workers "$NUM_WORKERS" \
  --recycle-workers-every "$RECYCLE_WORKERS_EVERY" \
  --save-every 10000 \
  --mixed-precision bf16 \
  "$@"
