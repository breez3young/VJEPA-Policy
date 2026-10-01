#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/configs/recipes/libero.env"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

PYTHON=${PYTHON:-python}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-scripts/train_vjepa_policy.py}
# Keep the checked global batch consistent with the actual training arguments.
for argument in "$@"; do
  case "$argument" in
    --batch-size|--batch-size=*|--gradient-accumulation-steps|--gradient-accumulation-steps=*)
      echo "Set BATCH_SIZE / GRAD_ACCUM in the environment so GLOBAL_BATCH_SIZE can be checked" >&2
      exit 64 ;;
  esac
done
: "${LIBERO_DATA_ROOT:?Set LIBERO_DATA_ROOT to the directory containing the four LeRobot LIBERO datasets}"
: "${VJEPA2_ENCODER_CHECKPOINT:?Set VJEPA2_ENCODER_CHECKPOINT to the pretrained V-JEPA2 encoder checkpoint}"
: "${TEXT_EMBEDDING_CACHE:?Set TEXT_EMBEDDING_CACHE to the precomputed instruction embeddings}"
DATASET_STATS=${DATASET_STATS:-}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
SEED=${SEED:-7}
PRED_DEPTH=${PRED_DEPTH:-24}
PRED_EMBED_DIM=${PRED_EMBED_DIM:-1024}
PRED_NUM_HEADS=${PRED_NUM_HEADS:-16}
ACTION_HIDDEN_SIZE=${ACTION_HIDDEN_SIZE:-512}
CONDITION_NUM_HEADS=${CONDITION_NUM_HEADS:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
RECYCLE_WORKERS_EVERY=${RECYCLE_WORKERS_EVERY:-0}
CROP_SIZE=${CROP_SIZE:-224}
VIEW_LAYOUT=${VIEW_LAYOUT:-independent}
PREDICTOR_ROPE_FREQUENCY_PAIRING=${PREDICTOR_ROPE_FREQUENCY_PAIRING:-corrected}
ENCODER_INTERPOLATE_ROPE=${ENCODER_INTERPOLATE_ROPE:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
PREDICTOR_INIT=${PREDICTOR_INIT:-}
OUTDIR=${OUTDIR:-./runs/vjepa_policy_vitl_gbs128_seed${SEED}}

ROPE_ARGS=()
ROPE_ARGS+=(--predictor-rope-frequency-pairing "$PREDICTOR_ROPE_FREQUENCY_PAIRING")
case "$ENCODER_INTERPOLATE_ROPE" in
  1|true|TRUE) ROPE_ARGS+=(--encoder-interpolate-rope) ;;
  0|false|FALSE) ROPE_ARGS+=(--no-encoder-interpolate-rope) ;;
  *) echo "ENCODER_INTERPOLATE_ROPE must be 0 or 1" >&2; exit 64 ;;
esac

if (( ACTIVATION_CHECKPOINTING_BLOCKS < 0 || ACTIVATION_CHECKPOINTING_BLOCKS > PRED_DEPTH )); then
  echo "ACTIVATION_CHECKPOINTING_BLOCKS must be between 0 and PRED_DEPTH" >&2
  exit 64
fi
ENCODER_ARGS=()
if [[ -n "$ENCODER" ]]; then
  ENCODER_ARGS+=(--encoder "$ENCODER")
fi
if (( NUM_GPUS * BATCH_SIZE * GRAD_ACCUM != GLOBAL_BATCH_SIZE )); then
  echo "Expected global batch $GLOBAL_BATCH_SIZE; got $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))" >&2
  exit 64
fi
echo "global_batch_size=$GLOBAL_BATCH_SIZE"

DATASETS=(
  "$LIBERO_DATA_ROOT/libero_object_no_noops_1.0.0_lerobot"
  "$LIBERO_DATA_ROOT/libero_goal_no_noops_1.0.0_lerobot"
  "$LIBERO_DATA_ROOT/libero_spatial_no_noops_1.0.0_lerobot"
  "$LIBERO_DATA_ROOT/libero_10_no_noops_1.0.0_lerobot"
)
STATS_ARGS=()
if [[ -n "$DATASET_STATS" ]]; then
  STATS_ARGS+=(--dataset-stats "$DATASET_STATS")
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTDIR"

PREDICTOR_INIT_ARGS=()
if [[ -n "$PREDICTOR_INIT" ]]; then
  if [[ ! -s "$PREDICTOR_INIT" ]]; then
    echo "PREDICTOR_INIT does not point to a readable checkpoint: $PREDICTOR_INIT" >&2
    exit 64
  fi
  if (( MAX_STATE_DIM <= 0 )); then
    echo "Set MAX_STATE_DIM=48 when initializing from a fixed-width DROID predictor" >&2
    exit 64
  fi
  PREDICTOR_INIT_ARGS+=(--predictor-init "$PREDICTOR_INIT")
fi

"$PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_GPUS" --num_machines 1 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision "$MIXED_PRECISION" --dynamo_backend no \
  "$TRAIN_SCRIPT" \
    --dataset-dirs "${DATASETS[@]}" \
    "${STATS_ARGS[@]}" --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
    --lang-dim 4096 --encoder-checkpoint "$VJEPA2_ENCODER_CHECKPOINT" \
    "${ENCODER_ARGS[@]}" \
    --encoder-family "$ENCODER_FAMILY" --model-name "$ENCODER_MODEL_NAME" \
    --encoder-checkpoint-key "$ENCODER_CHECKPOINT_KEY" \
    --crop-size "$CROP_SIZE" --patch-size 16 --tubelet-size 2 \
    --num-frames 33 --past-frames 4 --video-frame-stride 4 --context-tubelets 1 \
    --context-len "$CONTEXT_LEN" \
    --view-layout "$VIEW_LAYOUT" \
    --action-normalization QUANTILE --state-normalization QUANTILE \
    --action-chunk-size 32 --action-dim 7 --proprio-dim 8 \
    --max-state-dim "$MAX_STATE_DIM" \
    --action-hidden-size "$ACTION_HIDDEN_SIZE" --action-num-layers "$PRED_DEPTH" \
    --condition-num-heads "$CONDITION_NUM_HEADS" \
    --action-loss-weight 1.0 --action-num-inference-steps 10 \
    --pred-depth "$PRED_DEPTH" --pred-embed-dim "$PRED_EMBED_DIM" \
    --pred-num-heads "$PRED_NUM_HEADS" --num-mask-tokens 10 \
    --activation-checkpointing-blocks "$ACTIVATION_CHECKPOINTING_BLOCKS" \
    "${ROPE_ARGS[@]}" \
    --seed "$SEED" --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
    --recycle-workers-every "$RECYCLE_WORKERS_EVERY" \
    --lr 1e-4 --weight-decay 0.01 --max-steps "$MAX_STEPS" --num-epochs 10 \
    --mixed-precision "$MIXED_PRECISION" --log-every 10 --save-every "$SAVE_EVERY" \
    --output-dir "$OUTDIR" --wandb-name "$(basename "$OUTDIR")" \
    "${PREDICTOR_INIT_ARGS[@]}" "$@" \
  2>&1 | tee -a "$OUTDIR/train.log"

echo "[done] checkpoint: $OUTDIR/checkpoint_step$(printf '%06d' "$MAX_STEPS").pt"
