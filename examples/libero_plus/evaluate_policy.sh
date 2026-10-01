#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${LIBERO_PLUS_PYTHON:?Set LIBERO_PLUS_PYTHON to the dedicated Plus simulator interpreter}"
: "${LIBERO_PLUS_MANIFEST:?Run examples/libero_plus/prepare.py first}"
: "${LIBERO_CONFIG_PATH:?Set the isolated Plus config directory}"
export EVAL_BENCHMARK=libero-plus
export LIBERO_PYTHON="$LIBERO_PLUS_PYTHON"
export NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-1}
export RECORD_VIDEO=${RECORD_VIDEO:-0}
export EVAL_DIR=${EVAL_DIR:-${1:?usage: evaluate_policy.sh RUN_DIR [CHECKPOINT]}/libero_plus_$(date -u +%Y%m%dT%H%M%SZ)}
exec bash "$ROOT/examples/libero/evaluate_policy.sh" "$@"
