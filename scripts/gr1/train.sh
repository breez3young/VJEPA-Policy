#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

ENCODER=${1:-vjepa2_1_vitl}
: "${WEIGHTS:?Set WEIGHTS to the directory containing pretrained encoder checkpoints}"
case "$ENCODER" in
  vjepa2_vitg)
    ENCODER_FAMILY=vjepa2
    ENCODER_MODEL_NAME=vit_giant_xformers
    ENCODER_CHECKPOINT_KEY=encoder
    ENCODER_CHECKPOINT=$WEIGHTS/vitg.pt
    ;;
  vjepa2_vitl)
    ENCODER_FAMILY=vjepa2
    ENCODER_MODEL_NAME=vit_large
    ENCODER_CHECKPOINT_KEY=target_encoder
    ENCODER_CHECKPOINT=$WEIGHTS/vitl.pt
    ;;
  vjepa2_1_vitl)
    ENCODER_FAMILY=vjepa2_1
    ENCODER_MODEL_NAME=vit_large
    ENCODER_CHECKPOINT_KEY=ema_encoder
    ENCODER_CHECKPOINT=$WEIGHTS/vjepa2_1_vitl_dist_vitG_384.pt
    ;;
  vjepa2_1_vitg)
    ENCODER_FAMILY=vjepa2_1
    ENCODER_MODEL_NAME=vit_giant_xformers
    ENCODER_CHECKPOINT_KEY=target_encoder
    ENCODER_CHECKPOINT=$WEIGHTS/vjepa2_1_vitg_384.pt
    ;;
  *) echo "Unknown encoder: $ENCODER" >&2; exit 64 ;;
esac

PYTHON=${PYTHON:-python}
: "${DATA_ROOT:?Set DATA_ROOT to the GR-1 LeRobot dataset root}"
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$DATA_ROOT/artifacts}
DATASET_REVISION=${DATASET_REVISION:-unknown}
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-64}
GRAD_ACCUM=${GRAD_ACCUM:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
MAX_STEPS=${MAX_STEPS:-50000}
SEED=${SEED:-7}
ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-16}
TEXT_CONTEXT_LENGTH=${TEXT_CONTEXT_LENGTH:-48}
ACTIVATION_CHECKPOINTING_BLOCKS=${ACTIVATION_CHECKPOINTING_BLOCKS:-0}
NUM_WORKERS=${NUM_WORKERS:-16}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
OUTDIR=${OUTDIR:-$ROOT/runs/gr1_${ENCODER}_absolute_minmax_ck${ACTION_CHUNK_SIZE}_t5len${TEXT_CONTEXT_LENGTH}_gbs${GLOBAL_BATCH_SIZE}_step${MAX_STEPS}_seed${SEED}}

if (( NUM_GPUS * BATCH_SIZE * GRAD_ACCUM != GLOBAL_BATCH_SIZE )); then
  echo "Expected global batch size $GLOBAL_BATCH_SIZE, got $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))" >&2
  exit 64
fi
if (( ACTION_CHUNK_SIZE <= 0 || ACTION_CHUNK_SIZE % 4 != 0 )); then
  echo "ACTION_CHUNK_SIZE must be a positive multiple of 4" >&2
  exit 64
fi
NUM_FRAMES=$((ACTION_CHUNK_SIZE + 1))
mapfile -t DATASETS < <(find "$DATA_ROOT" -mindepth 3 -maxdepth 3 -path '*/meta/info.json' -printf '%h\n' | xargs -r -n1 dirname | sort)
if (( ${#DATASETS[@]} != 24 )); then
  echo "Expected 24 GR-1 task roots under $DATA_ROOT, found ${#DATASETS[@]}" >&2
  exit 1
fi

ACTION_STATE_INDICES=(0 1 2 3 4 5 6 22 23 24 25 26 27 28 7 8 9 10 11 12 29 30 31 32 33 34 41 42 43)
STATS=$ARTIFACT_ROOT/dataset_stats_absolute_minmax_ck${ACTION_CHUNK_SIZE}.json
TEXT_CACHE=$ARTIFACT_ROOT/text_embeddings
[[ -s "$STATS" ]] || { echo "Missing $STATS; run scripts/gr1/prepare.sh first" >&2; exit 1; }
"$PYTHON" scripts/gr1/validate_dataset.py \
  --data-root "$DATA_ROOT" --expected-tasks 24 --expected-episodes 1000 \
  --revision "$DATASET_REVISION" --stats "$STATS"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTDIR"

ENCODER_ARGS=()
case "$ENCODER" in
  vjepa2_vitl|vjepa2_1_vitl) ENCODER_ARGS+=(--encoder "$ENCODER") ;;
esac

"$PYTHON" -m accelerate.commands.launch \
  --num_processes "$NUM_GPUS" --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  scripts/train_vjepa_policy.py \
    --dataset-dirs "${DATASETS[@]}" \
    --dataset-stats "$STATS" --text-cache-dir "$TEXT_CACHE" \
    --context-len "$TEXT_CONTEXT_LENGTH" \
    --instruction-field remarks \
    --action-indices "${ACTION_STATE_INDICES[@]}" \
    --state-indices "${ACTION_STATE_INDICES[@]}" \
    --clip-normalized \
    --checkpoint "$ENCODER_CHECKPOINT" "${ENCODER_ARGS[@]}" --encoder-family "$ENCODER_FAMILY" \
    --model-name "$ENCODER_MODEL_NAME" --encoder-checkpoint-key "$ENCODER_CHECKPOINT_KEY" \
    --encoder-interpolate-rope --crop-size 224 --patch-size 16 --tubelet-size 2 \
    --num-frames "$NUM_FRAMES" --past-frames 4 --video-frame-stride 4 --context-tubelets 1 \
    --require-full-future-window \
    --view-layout independent --video-backend torchcodec \
    --action-normalization MIN_MAX --state-normalization IDENTITY \
    --action-chunk-size "$ACTION_CHUNK_SIZE" --action-dim 29 --proprio-dim 29 --max-state-dim 0 \
    --proprio-encoding sincos \
    --action-hidden-size 512 --action-num-layers 24 --condition-num-heads 8 \
    --action-loss-weight 1.0 --action-num-inference-steps 10 \
    --pred-depth 24 --pred-embed-dim 1024 --pred-num-heads 16 --num-mask-tokens 10 \
    --predictor-rope-frequency-pairing corrected \
    --activation-checkpointing-blocks "$ACTIVATION_CHECKPOINTING_BLOCKS" \
    --seed "$SEED" --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
    --recycle-workers-every 500 \
    --lr 1e-4 --weight-decay 0.01 --max-steps "$MAX_STEPS" --num-epochs 10 \
    --mixed-precision bf16 --log-every 10 --save-every 5000 \
    --output-dir "$OUTDIR" --wandb-name "$(basename "$OUTDIR")" \
  2>&1 | tee -a "$OUTDIR/train.log"
