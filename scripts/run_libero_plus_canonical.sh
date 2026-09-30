#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_libero_plus_canonical.sh [options]

Run the canonical 10,030-episode LIBERO-Plus benchmark with one policy
server per GPU and a shared global shard set. The default topology is four
GPUs, eight simulator clients per server, 32 shards, and batch size 8.

Required options:
  --server-config PATH    run-local V-JEPA model-server config
  --checkpoint PATH       policy checkpoint
  --pretrained-encoder PATH V-JEPA encoder checkpoint
  --dataset-stats PATH    action normalization statistics
  --text-cache-dir DIR    canonical fixed-length 128-token cache
  --output-dir DIR        persistent result directory

Common options:
  --harness-root DIR      vla-evaluation-harness checkout
  --benchmark-config PATH benchmark YAML (default: configs/libero_plus_all.yaml)
  --allow-incomplete       validate SQLite integrity without full 10,030 coverage
  --gpus LIST             policy server GPU ids (default: 0,1,2,3)
  --sim-gpus LIST         simulator GPU ids (default: same as --gpus)
  --clients-per-gpu N     simulator clients per policy server (default: 8)
  --batch-size N          server batch size (default: clients per server)
  --base-port N           first localhost server port (default: 31100)
  --eval-id ID            stable evaluation id (default: generated)
  --server-python PATH    Python used by the policy server
  --bench-env PATH        environment containing vla-eval
EOF
  exit "${1:-0}"
}

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${VLA_WORKSPACE:-$(cd -- "$ROOT/.." && pwd)}"
HARNESS="${VLA_HARNESS_ROOT:-$WORKSPACE/vla-evaluation-harness}"
CONDA_ROOT="${VLA_CONDA_ROOT:-$WORKSPACE/../miniconda3}"
BENCH_ENV="${VLA_BENCH_ENV:-$CONDA_ROOT/envs/libero-plus-eval}"
SERVER_PYTHON="${VJEPA_SERVER_PYTHON:-$CONDA_ROOT/envs/VLA_JEPA/bin/python}"
SERVER_CONFIG="${VJEPA_SERVER_CONFIG:-}"
CHECKPOINT="${VJEPA_POLICY_CHECKPOINT:-}"
PRETRAINED_ENCODER="${VJEPA_PRETRAINED_ENCODER:-}"
DATASET_STATS="${VJEPA_DATASET_STATS:-}"
TEXT_CACHE_DIR="${VJEPA_TEXT_CACHE_DIR:-}"
BENCHMARK_CONFIG="$ROOT/configs/libero_plus_all.yaml"
GPU_LIST="0,1,2,3"
SIM_GPU_LIST=""
CLIENTS_PER_GPU=8
BATCH_SIZE=""
BASE_PORT=31100
OUTPUT_DIR=""
EVAL_ID=""
ALLOW_INCOMPLETE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --harness-root) HARNESS="$2"; shift 2 ;;
    --server-config) SERVER_CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --pretrained-encoder) PRETRAINED_ENCODER="$2"; shift 2 ;;
    --dataset-stats) DATASET_STATS="$2"; shift 2 ;;
    --text-cache-dir) TEXT_CACHE_DIR="$2"; shift 2 ;;
    --benchmark-config) BENCHMARK_CONFIG="$2"; shift 2 ;;
    --allow-incomplete) ALLOW_INCOMPLETE=true; shift ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --sim-gpus) SIM_GPU_LIST="$2"; shift 2 ;;
    --clients-per-gpu) CLIENTS_PER_GPU="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --eval-id) EVAL_ID="$2"; shift 2 ;;
    --server-python) SERVER_PYTHON="$2"; shift 2 ;;
    --bench-env) BENCH_ENV="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$SERVER_CONFIG" ]] || { echo "--server-config is required" >&2; exit 2; }
[[ -n "$CHECKPOINT" ]] || { echo "--checkpoint is required" >&2; exit 2; }
[[ -n "$PRETRAINED_ENCODER" ]] || { echo "--pretrained-encoder is required" >&2; exit 2; }
[[ -n "$DATASET_STATS" ]] || { echo "--dataset-stats is required" >&2; exit 2; }
[[ -n "$TEXT_CACHE_DIR" ]] || { echo "--text-cache-dir is required" >&2; exit 2; }
[[ -n "$OUTPUT_DIR" ]] || { echo "--output-dir is required" >&2; exit 2; }
[[ -x "$BENCH_ENV/bin/vla-eval" ]] || { echo "Missing vla-eval: $BENCH_ENV/bin/vla-eval" >&2; exit 1; }
[[ -x "$SERVER_PYTHON" ]] || { echo "Missing server Python: $SERVER_PYTHON" >&2; exit 1; }
[[ -f "$SERVER_CONFIG" && -f "$BENCHMARK_CONFIG" ]] || { echo "Missing evaluation config" >&2; exit 1; }
[[ -s "$CHECKPOINT" ]] || { echo "Missing policy checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -s "$PRETRAINED_ENCODER" ]] || { echo "Missing pretrained encoder: $PRETRAINED_ENCODER" >&2; exit 1; }
[[ -s "$DATASET_STATS" ]] || { echo "Missing dataset statistics: $DATASET_STATS" >&2; exit 1; }
[[ -d "$TEXT_CACHE_DIR" ]] || { echo "Missing text cache: $TEXT_CACHE_DIR" >&2; exit 1; }
[[ -d "$HARNESS/src" && -f "$HARNESS/scripts/libero_plus_runtime_env.sh" ]] || { echo "Missing harness checkout: $HARNESS" >&2; exit 1; }
source "$HARNESS/scripts/libero_plus_runtime_env.sh"
libero_plus_setup_imagemagick
if ! "$BENCH_ENV/bin/python" -c 'from wand.api import library' >/dev/null 2>&1; then
  echo "ImageMagick/MagickWand is unavailable to $BENCH_ENV/bin/python; install MagickWand or set VLA_IMAGEMAGICK_HOME to a compatible runtime" >&2
  exit 1
fi

for value in "$CLIENTS_PER_GPU" "$BASE_PORT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Numeric options must be positive integers" >&2; exit 2; }
done
if [[ -z "$BATCH_SIZE" ]]; then BATCH_SIZE="$CLIENTS_PER_GPU"; fi
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "--batch-size must be positive" >&2; exit 2; }
(( BATCH_SIZE <= CLIENTS_PER_GPU )) || { echo "--batch-size cannot exceed --clients-per-gpu" >&2; exit 2; }
[[ -n "$SIM_GPU_LIST" ]] || SIM_GPU_LIST="$GPU_LIST"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
IFS=',' read -r -a SIM_GPUS <<< "$SIM_GPU_LIST"
(( ${#GPUS[@]} > 0 && ${#GPUS[@]} == ${#SIM_GPUS[@]} )) || { echo "--gpus and --sim-gpus must have equal non-zero lengths" >&2; exit 2; }
if [[ -z "$EVAL_ID" ]]; then EVAL_ID="libero-plus-$(date -u +%Y%m%dT%H%M%SZ)"; fi
[[ "$EVAL_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid --eval-id" >&2; exit 2; }

NUM_SERVERS="${#GPUS[@]}"
NUM_SHARDS=$((NUM_SERVERS * CLIENTS_PER_GPU))
mkdir -p "$(dirname -- "$OUTPUT_DIR")"
OUTPUT_DIR="$(cd -- "$(dirname -- "$OUTPUT_DIR")" && pwd)/$(basename -- "$OUTPUT_DIR")"
if [[ -e "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output directory is not empty: $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
SERVER_CONFIG="$(readlink -f "$SERVER_CONFIG")"
CHECKPOINT="$(readlink -f "$CHECKPOINT")"
PRETRAINED_ENCODER="$(readlink -f "$PRETRAINED_ENCODER")"
DATASET_STATS="$(readlink -f "$DATASET_STATS")"
TEXT_CACHE_DIR="$(readlink -f "$TEXT_CACHE_DIR")"
BENCHMARK_CONFIG="$(readlink -f "$BENCHMARK_CONFIG")"

# Check the canonical cache contract without adding a repository-specific
# digest or checksum requirement. The harness still owns cache file loading.
"$SERVER_PYTHON" - "$TEXT_CACHE_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"cache manifest is missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
checks = {
    "task_count": manifest.get("task_count") == 10030,
    "unique_prompt_count": manifest.get("unique_prompt_count") == 1549,
    "context_length": manifest.get("context_length") == 128,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("canonical cache contract failed: " + ", ".join(failed))
print("validated canonical cache: 10030 tasks, 1549 prompts, fixed length 128")
PY

export PATH="$BENCH_ENV/bin:$PATH"
export PYTHONPATH="$ROOT/src:$HARNESS/src${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

LIBERO_ROOT="${VLA_LIBERO_PLUS_ROOT:-$WORKSPACE/LIBERO-plus/libero/libero}"
if [[ -z "${VLA_LIBERO_PLUS_ASSETS:-}" ]]; then
  echo "VLA_LIBERO_PLUS_ASSETS must point to the unpacked LIBERO-plus-0/assets directory" >&2
  exit 1
fi
LIBERO_ASSETS="$(readlink -f -- "$VLA_LIBERO_PLUS_ASSETS" 2>/dev/null || true)"
if [[ ! -d "$LIBERO_ASSETS" || ! -f "$LIBERO_ASSETS/wall_frames.stl" || ! -d "$LIBERO_ASSETS/textures" || ! -d "$LIBERO_ROOT/bddl_files" || ! -d "$LIBERO_ROOT/init_files" ]]; then
  echo "Missing LIBERO-Plus assets or source tree; check VLA_LIBERO_PLUS_ASSETS=$VLA_LIBERO_PLUS_ASSETS" >&2
  exit 1
fi
[[ -d "$LIBERO_ROOT" ]] || { echo "Missing LIBERO-Plus source tree: $LIBERO_ROOT" >&2; exit 1; }

LOCAL_DIR="/dev/shm/vjepa-libero-plus-${EVAL_ID}"
[[ ! -e "$LOCAL_DIR" ]] || { echo "Local run directory already exists: $LOCAL_DIR" >&2; exit 1; }
mkdir -p "$LOCAL_DIR"
RUNTIME_LIBERO_CONFIG="$LOCAL_DIR/libero-config"
mkdir -p "$RUNTIME_LIBERO_CONFIG"
python3 - "$RUNTIME_LIBERO_CONFIG/config.yaml" "$LIBERO_ROOT" "$LIBERO_ASSETS" <<'PY'
from pathlib import Path
import sys

config_path, root, assets = map(Path, sys.argv[1:])
config_path.write_text(
    "\n".join(
        [
            f"assets: {assets}",
            f"bddl_files: {root / 'bddl_files'}",
            f"benchmark_root: {root}",
            f"datasets: {root.parent / 'datasets'}",
            f"init_states: {root / 'init_files'}",
            "",
        ]
    )
)
PY
export LIBERO_CONFIG_PATH="$RUNTIME_LIBERO_CONFIG"
SERVER_PIDS=()
SHARD_PIDS=()
cleanup() {
  for pid in "${SHARD_PIDS[@]:-}"; do [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -- "-$pid" 2>/dev/null || true; done
  for pid in "${SHARD_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

cd "$HARNESS"
for index in "${!GPUS[@]}"; do
  port=$((BASE_PORT + index))
  gpu="${GPUS[$index]}"
  log="$LOCAL_DIR/model-server-gpu${gpu}.log"
  CUDA_VISIBLE_DEVICES="$gpu" setsid "$BENCH_ENV/bin/vla-eval" serve \
    -c "$SERVER_CONFIG" --python "$SERVER_PYTHON" \
    --arg "port=$port" --arg "max_batch_size=$BATCH_SIZE" \
    --arg "checkpoint=$CHECKPOINT" \
    --arg "pretrained_encoder=$PRETRAINED_ENCODER" \
    --arg "dataset_stats=$DATASET_STATS" \
    --arg "text_cache_dir=$TEXT_CACHE_DIR" >"$log" 2>&1 &
  SERVER_PIDS+=("$!")
done

for index in "${!GPUS[@]}"; do
  port=$((BASE_PORT + index))
  deadline=$((SECONDS + 900))
  until curl --noproxy '*' -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "${SERVER_PIDS[$index]}" 2>/dev/null || { tail -100 "$LOCAL_DIR/model-server-gpu${GPUS[$index]}.log" >&2; exit 1; }
    (( SECONDS < deadline )) || { echo "Server on port $port did not become ready" >&2; exit 1; }
    sleep 1
  done
done

echo "Running $NUM_SHARDS shards on servers GPU=$GPU_LIST, simulator GPU=$SIM_GPU_LIST, clients/server=$CLIENTS_PER_GPU, batch=$BATCH_SIZE"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  server_index=$((shard % NUM_SERVERS))
  sim_gpu="${SIM_GPUS[$server_index]}"
  port=$((BASE_PORT + server_index))
  log="$LOCAL_DIR/shard-$(printf '%03d' "$shard").log"
  env -u LIBGL_ALWAYS_SOFTWARE \
    CUDA_VISIBLE_DEVICES="$sim_gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    EGL_PLATFORM=device MUJOCO_EGL_DEVICE_ID="$sim_gpu" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$BENCH_ENV/bin/vla-eval" run -c "$BENCHMARK_CONFIG" --eval-id "$EVAL_ID" \
    --output-dir "$LOCAL_DIR" --render gpu --server-url "ws://127.0.0.1:$port" \
    --param send_wrist_image=true --param send_state=true --param quat_no_antipodal=true \
    --no-docker --shard-id "$shard" --num-shards "$NUM_SHARDS" >"$log" 2>&1 &
  SHARD_PIDS+=("$!")
done

failed=0
for index in "${!SHARD_PIDS[@]}"; do
  wait "${SHARD_PIDS[$index]}" || failed=$((failed + 1))
  SHARD_PIDS[$index]=""
done
if (( failed != 0 )); then
  echo "$failed/$NUM_SHARDS simulator shards failed; raw run retained at $LOCAL_DIR" >&2
  exit 1
fi

DB="$LOCAL_DIR/recording-$EVAL_ID.sqlite"
[[ -f "$DB" ]] || { echo "Missing result database: $DB" >&2; exit 1; }
ALLOW_INCOMPLETE="$ALLOW_INCOMPLETE" "$SERVER_PYTHON" - "$DB" <<'PY'
import json
import os
import sqlite3
import sys

allow_incomplete = os.environ.get("ALLOW_INCOMPLETE") == "true"
expected = {
    "libero_plus_spatial": 2402,
    "libero_plus_object": 2518,
    "libero_plus_goal": 2591,
    "libero_plus_10": 2519,
}
con = sqlite3.connect(sys.argv[1])
checkpoint = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
if checkpoint[0] != 0 or checkpoint[1] != checkpoint[2]:
    raise SystemExit(f"SQLite WAL checkpoint failed: {checkpoint}")
if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("SQLite integrity check failed")
rows = con.execute("SELECT eval_id, safe_name FROM eval_metadata ORDER BY safe_name").fetchall()
if allow_incomplete:
    total = con.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0]
    statuses = dict(con.execute(
        "SELECT status, COUNT(*) FROM episode_results GROUP BY status ORDER BY status"
    ).fetchall())
    print(
        "validated incomplete LIBERO-Plus SQLite: "
        f"metadata_rows={len(rows)} episode_rows={total} statuses={statuses}"
    )
else:
    if len(rows) != 4:
        raise SystemExit(f"expected four suite metadata rows, got {len(rows)}")
    seen = set()
    for eval_id, safe_name in rows:
        suite_name = safe_name.removeprefix("LIBEROPlusBenchmark_")
        expected_count = expected.get(suite_name)
        records = con.execute(
            "SELECT episode_id, status, context FROM episode_results WHERE eval_id = ?",
            (eval_id,),
        ).fetchall()
        unique = set()
        task_ids = set()
        episode_ids = set()
        statuses = set()
        suites = set()
        for episode_id, status, context in records:
            payload = json.loads(context or "{}")
            suite = payload.get("suite")
            task_id = payload.get("task_id")
            unique.add((suite, task_id, episode_id))
            suites.add(suite)
            task_ids.add(task_id)
            episode_ids.add(episode_id)
            statuses.add(status)
        expected_suite = suite_name.removeprefix("libero_plus_")
        if (
            expected_count is None
            or len(records) != expected_count
            or len(unique) != len(records)
            or suites != {f"libero_{expected_suite}"}
            or task_ids != set(range(expected_count))
            or episode_ids != {0}
            or not statuses <= {"success", "fail", "error"}
        ):
            raise SystemExit(
                f"coverage failure {suite_name}: rows={len(records)} unique={len(unique)} "
                f"tasks={len(task_ids)} statuses={statuses}"
            )
        seen.add(suite_name)
    total = con.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0]
    if seen != set(expected) or total != 10030:
        raise SystemExit(f"expected 10030 episodes, got {total}")
    print("validated LIBERO-Plus coverage: 10030 episodes")
PY

mkdir -p "$OUTPUT_DIR/logs"
"$BENCH_ENV/bin/vla-eval" merge --db "$DB" --output-dir "$OUTPUT_DIR" >"$LOCAL_DIR/merge.log" 2>&1
cp "$DB" "$OUTPUT_DIR/recording-$EVAL_ID.sqlite"
cp "$LOCAL_DIR"/model-server-*.log "$LOCAL_DIR"/shard-*.log "$LOCAL_DIR/merge.log" "$OUTPUT_DIR/logs/"
printf 'eval_id=%s\nserver_gpus=%s\nsimulator_gpus=%s\nservers=%s\nshards=%s\nclients_per_server=%s\nbatch_size=%s\n' \
  "$EVAL_ID" "$GPU_LIST" "$SIM_GPU_LIST" "$NUM_SERVERS" "$NUM_SHARDS" "$CLIENTS_PER_GPU" "$BATCH_SIZE" \
  >"$OUTPUT_DIR/run_topology.txt"
echo "Results written to $OUTPUT_DIR"
