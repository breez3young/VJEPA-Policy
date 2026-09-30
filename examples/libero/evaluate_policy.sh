#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

RUN_DIR=${1:?usage: evaluate_policy.sh RUN_DIR [CHECKPOINT]}
CHECKPOINT=${2:-$RUN_DIR/checkpoint_step021360.pt}
: "${VJEPA2_ENCODER_CHECKPOINT:?Set VJEPA2_ENCODER_CHECKPOINT}"
: "${TEXT_EMBEDDING_CACHE:?Set TEXT_EMBEDDING_CACHE}"

POLICY_PYTHON=${POLICY_PYTHON:-python}
LIBERO_PYTHON=${LIBERO_PYTHON:-python}
LIBERO_TORCH_LIB=${LIBERO_TORCH_LIB:-}
LIBERO_MUJOCO_GL=${LIBERO_MUJOCO_GL:-egl}
POLICY_SERVER_SCRIPT=${POLICY_SERVER_SCRIPT:-examples/libero/serve_policy.py}
EVAL_DIR=${EVAL_DIR:-$RUN_DIR/libero_eval_r16_50trials}
DATASET_STATS=${DATASET_STATS:-$RUN_DIR/dataset_stats.json}
BASE_PORT=${BASE_PORT:-14001}
REPLAN_STEPS=${REPLAN_STEPS:-16}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
EVAL_SEED=${EVAL_SEED:-7}
# Empty means recover the serving contract from the checkpoint.  The serving
# config defaults still cover metadata-free canonical checkpoints.
POLICY_PRECISION=${POLICY_PRECISION:-}
POLICY_T5_LEN=${POLICY_T5_LEN:-}
POLICY_CROP_SIZE=${POLICY_CROP_SIZE:-}
POLICY_VIEW_LAYOUT=${POLICY_VIEW_LAYOUT:-}
POLICY_VIEWS=${POLICY_VIEWS:-}
# Empty means recover the value saved in the policy checkpoint. Set either
# variable explicitly when evaluating a legacy or intentionally overridden run.
POLICY_ENCODER_INTERPOLATE_ROPE=${POLICY_ENCODER_INTERPOLATE_ROPE:-}
POLICY_CORRECTED_PREDICTOR_ROPE=${POLICY_CORRECTED_PREDICTOR_ROPE:-}
ENCODER=${ENCODER:-}
ENCODER_FAMILY=${ENCODER_FAMILY:-}
ENCODER_MODEL_NAME=${ENCODER_MODEL_NAME:-}
ENCODER_CHECKPOINT_KEY=${ENCODER_CHECKPOINT_KEY:-}
PRED_DEPTH=${PRED_DEPTH:-24}
PRED_EMBED_DIM=${PRED_EMBED_DIM:-1024}
PRED_NUM_HEADS=${PRED_NUM_HEADS:-16}
ACTION_HIDDEN_SIZE=${ACTION_HIDDEN_SIZE:-512}
ACTION_NUM_LAYERS=${ACTION_NUM_LAYERS:-24}
IFS=',' read -r -a GPU_IDS <<<"${EVAL_GPUS:-0,1,2,3}"
ALL_SUITES=(libero_spatial libero_object libero_goal libero_10)
IFS=',' read -r -a SUITES <<<"${EVAL_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Missing policy checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -s "$DATASET_STATS" ]]; then
  echo "Missing normalization statistics: $DATASET_STATS" >&2
  exit 1
fi
if [[ ! -f "$POLICY_SERVER_SCRIPT" ]]; then
  echo "Missing policy server script: $POLICY_SERVER_SCRIPT" >&2
  exit 1
fi
if (( ${#SUITES[@]} == 0 )); then
  echo "EVAL_SUITES must contain at least one suite" >&2
  exit 64
fi
declare -A requested_suites=()
for suite in "${SUITES[@]}"; do
  if [[ ! " ${ALL_SUITES[*]} " =~ " $suite " ]]; then
    echo "Unknown LIBERO suite: $suite" >&2
    exit 64
  fi
  if [[ -v "requested_suites[$suite]" ]]; then
    echo "EVAL_SUITES must not contain duplicates: $suite" >&2
    exit 64
  fi
  requested_suites[$suite]=1
done
if (( ${#GPU_IDS[@]} < ${#SUITES[@]} )); then
  echo "EVAL_GPUS must contain at least ${#SUITES[@]} comma-separated GPU IDs" >&2
  exit 1
fi
mkdir -p "$EVAL_DIR"
SUMMARY_LOG="$EVAL_DIR/summary.txt"

{
  echo "LIBERO evaluation"
  echo "checkpoint: $CHECKPOINT"
  echo "policy_server_script: $POLICY_SERVER_SCRIPT"
  echo "started: $(date -Is)"
  echo "replan_steps: $REPLAN_STEPS"
  echo "frame_stride: 4"
  echo "trials_per_task: $NUM_TRIALS_PER_TASK"
  echo "seed: $EVAL_SEED"
  echo "mujoco_gl: $LIBERO_MUJOCO_GL"
  echo "t5_len: $POLICY_T5_LEN"
  echo "requested_suites: ${SUITES[*]}"
  echo "view_layout: $POLICY_VIEW_LAYOUT"
  echo "crop_size: $POLICY_CROP_SIZE"
  echo "encoder_endpoint_rope: ${POLICY_ENCODER_INTERPOLATE_ROPE:-checkpoint}"
  echo "corrected_predictor_rope: $POLICY_CORRECTED_PREDICTOR_ROPE"
  echo
} >"$SUMMARY_LOG"

run_suite() (
  set -uo pipefail
  suite=$1
  gpu=$2
  port=$3
  server_log="$EVAL_DIR/${suite}_server.log"
  client_log="$EVAL_DIR/${suite}.log"
  status_file="$EVAL_DIR/${suite}.status"
  results_file="$EVAL_DIR/$suite/${suite}_eval_results.txt"
  server_pid=""

  cleanup() {
    if [[ -n "$server_pid" ]]; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  if [[ -s "$results_file" ]]; then
    echo 0 >"$status_file"
    echo "[$(date -Is)] [$suite] reusing $results_file"
    exit 0
  fi

  rope_args=()
  if [[ -n "$POLICY_CORRECTED_PREDICTOR_ROPE" ]]; then
    case "$POLICY_CORRECTED_PREDICTOR_ROPE" in
      1|true|TRUE) rope_args+=(--corrected-predictor-rope True) ;;
      0|false|FALSE) rope_args+=(--corrected-predictor-rope False) ;;
      *) echo "POLICY_CORRECTED_PREDICTOR_ROPE must be 0/1 when set" >&2; exit 64 ;;
    esac
  fi
  if [[ -n "$POLICY_ENCODER_INTERPOLATE_ROPE" ]]; then
    case "$POLICY_ENCODER_INTERPOLATE_ROPE" in
      1|true|TRUE) rope_args+=(--encoder-interpolate-rope True) ;;
      0|false|FALSE) rope_args+=(--encoder-interpolate-rope False) ;;
      *) echo "POLICY_ENCODER_INTERPOLATE_ROPE must be 0/1 when set" >&2; exit 64 ;;
    esac
  fi
  serving_args=()
  if [[ -n "$POLICY_PRECISION" ]]; then
    serving_args+=(--precision "$POLICY_PRECISION")
  fi
  if [[ -n "$POLICY_T5_LEN" ]]; then
    serving_args+=(--t5-len "$POLICY_T5_LEN")
  fi
  if [[ -n "$POLICY_CROP_SIZE" ]]; then
    serving_args+=(--crop-size "$POLICY_CROP_SIZE")
  fi
  if [[ -n "$POLICY_VIEW_LAYOUT" ]]; then
    serving_args+=(--view-layout "$POLICY_VIEW_LAYOUT")
  fi
  view_args=()
  if [[ -n "$POLICY_VIEWS" ]]; then
    IFS=',' read -r -a policy_views <<<"$POLICY_VIEWS"
    view_args+=(--views "${policy_views[@]}")
  fi
  encoder_args=()
  if [[ -n "$ENCODER" ]]; then
    encoder_args+=(--encoder "$ENCODER")
  fi
  if [[ -n "$ENCODER_FAMILY" ]]; then
    encoder_args+=(--encoder-family "$ENCODER_FAMILY")
  fi
  if [[ -n "$ENCODER_MODEL_NAME" ]]; then
    encoder_args+=(--model-name "$ENCODER_MODEL_NAME")
  fi
  if [[ -n "$ENCODER_CHECKPOINT_KEY" ]]; then
    encoder_args+=(--encoder-checkpoint-key "$ENCODER_CHECKPOINT_KEY")
  fi

  echo running >"$status_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$POLICY_PYTHON" "$POLICY_SERVER_SCRIPT" \
    --ckpt "$CHECKPOINT" \
    --pretrained-encoder "$VJEPA2_ENCODER_CHECKPOINT" \
    "${encoder_args[@]}" \
    --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
    --dataset-stats "$DATASET_STATS" \
    "${serving_args[@]}" \
    "${view_args[@]}" \
    "${rope_args[@]}" \
    --pred-depth "$PRED_DEPTH" \
    --pred-embed-dim "$PRED_EMBED_DIM" \
    --pred-num-heads "$PRED_NUM_HEADS" \
    --action-hidden-size "$ACTION_HIDDEN_SIZE" \
    --action-num-layers "$ACTION_NUM_LAYERS" \
    --seed "$EVAL_SEED" \
    --port "$port" \
    >"$server_log" 2>&1 &
  server_pid=$!

  ready=false
  for _ in $(seq 1 300); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[$suite] policy server exited during startup" >&2
      tail -100 "$server_log" >&2
      echo 1 >"$status_file"
      exit 1
    fi
    if curl --noproxy '*' -fs --max-time 1 "http://127.0.0.1:$port/healthz" \
      >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 2
  done
  if [[ "$ready" != true ]]; then
    echo "[$suite] policy server did not become ready" >&2
    tail -100 "$server_log" >&2
    echo 1 >"$status_file"
    exit 1
  fi

  set +e
  client_env=(
    "CUDA_VISIBLE_DEVICES=$gpu"
    "MUJOCO_GL=$LIBERO_MUJOCO_GL"
    "LD_LIBRARY_PATH=${LIBERO_TORCH_LIB:+$LIBERO_TORCH_LIB:}${LD_LIBRARY_PATH:-}"
    "OMP_NUM_THREADS=1"
    "MKL_NUM_THREADS=1"
    "OPENBLAS_NUM_THREADS=1"
    "NUMEXPR_NUM_THREADS=1"
  )
  if [[ "$LIBERO_MUJOCO_GL" == "egl" ]]; then
    # These variables must be present before run_libero_client imports
    # MuJoCo/robosuite.  CUDA_VISIBLE_DEVICES is intentionally kept aligned
    # with the physical EGL id used by this suite worker.
    client_env+=(
      "PYOPENGL_PLATFORM=egl"
      "EGL_PLATFORM=device"
      "MUJOCO_EGL_DEVICE_ID=$gpu"
    )
    env -u LIBGL_ALWAYS_SOFTWARE \
      -u http_proxy -u HTTP_PROXY -u https_proxy -u HTTPS_PROXY \
      "${client_env[@]}" \
      "$LIBERO_PYTHON" \
      examples/libero/run_libero_client.py \
        --args.host 127.0.0.1 \
        --args.port "$port" \
        --args.task-suite-name "$suite" \
        --args.replan-steps "$REPLAN_STEPS" \
        --args.frame-stride 4 \
        --args.resize-size 256 \
        --args.num-trials-per-task "$NUM_TRIALS_PER_TASK" \
        --args.seed "$EVAL_SEED" \
        --args.video-out-path "$EVAL_DIR/$suite" \
        >"$client_log" 2>&1
  else
    env -u http_proxy -u HTTP_PROXY -u https_proxy -u HTTPS_PROXY \
      "${client_env[@]}" \
      "$LIBERO_PYTHON" \
      examples/libero/run_libero_client.py \
        --args.host 127.0.0.1 \
        --args.port "$port" \
        --args.task-suite-name "$suite" \
        --args.replan-steps "$REPLAN_STEPS" \
        --args.frame-stride 4 \
        --args.resize-size 256 \
        --args.num-trials-per-task "$NUM_TRIALS_PER_TASK" \
        --args.seed "$EVAL_SEED" \
        --args.video-out-path "$EVAL_DIR/$suite" \
        >"$client_log" 2>&1
  fi
  client_status=$?
  set -e

  if (( client_status == 0 )) && [[ -s "$results_file" ]]; then
    echo 0 >"$status_file"
    exit 0
  fi
  if (( client_status == 0 )); then
    client_status=2
  fi
  echo "$client_status" >"$status_file"
  exit "$client_status"
)

worker_pids=()
for index in "${!SUITES[@]}"; do
  run_suite \
    "${SUITES[$index]}" \
    "${GPU_IDS[$index]}" \
    "$((BASE_PORT + index))" \
    >"$EVAL_DIR/${SUITES[$index]}_runner.log" 2>&1 &
  worker_pids+=("$!")
  sleep 2
done

set +e
for worker_pid in "${worker_pids[@]}"; do
  wait "$worker_pid"
done
set -e

failed=()
requested_failed=()
total_success=0
total_episodes=0
for suite in "${SUITES[@]}"; do
  results_file="$EVAL_DIR/$suite/${suite}_eval_results.txt"
  status=missing
  if [[ -s "$results_file" ]]; then
    status=0
  elif [[ -f "$EVAL_DIR/${suite}.status" ]]; then
    status=$(<"$EVAL_DIR/${suite}.status")
  fi
  {
    echo "[$suite] exit_status: $status"
    if [[ -s "$results_file" ]]; then
      sed 's/^/  /' "$results_file"
    else
      echo "  missing results: $results_file"
    fi
    echo
  } >>"$SUMMARY_LOG"
  if [[ "$status" != 0 ]] || [[ ! -s "$results_file" ]]; then
    failed+=("$suite")
    if [[ -v "requested_suites[$suite]" ]]; then
      requested_failed+=("$suite")
    fi
    continue
  fi
  successes=$(awk -F': ' '$1 == "Total success" {print $2}' "$results_file")
  episodes=$(awk -F': ' '$1 == "Total episodes" {print $2}' "$results_file")
  total_success=$((total_success + successes))
  total_episodes=$((total_episodes + episodes))
done

{
  echo "finished: $(date -Is)"
  echo "successful_suites: $((${#SUITES[@]} - ${#failed[@]}))/${#SUITES[@]}"
  echo "failed_suites: ${failed[*]:-none}"
  if (( total_episodes > 0 )); then
    awk -v successes="$total_success" -v episodes="$total_episodes" \
      'BEGIN {printf "overall: %d/%d = %.2f%%\n", successes, episodes, 100 * successes / episodes}'
  fi
} >>"$SUMMARY_LOG"

if (( ${#requested_failed[@]} > 0 )); then
  echo "Requested LIBERO evaluation incomplete: ${requested_failed[*]}" >&2
  exit 1
fi

echo "[done] LIBERO evaluation summary: $SUMMARY_LOG"
