#!/usr/bin/env bash
set -euo pipefail

# Fresh LIBERO policy training: Predictor and Action Expert are both created
# from their seeded constructors.  State conditioning follows the DROID
# contract: 48 padded native dimensions concatenated with a 48-d validity mask.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/configs/recipes/libero.env"

: "${LIBERO_DATA_ROOT:?Set LIBERO_DATA_ROOT to the directory containing the four LIBERO datasets}"
: "${VJEPA2_ENCODER_CHECKPOINT:?Set VJEPA2_ENCODER_CHECKPOINT to the frozen encoder checkpoint}"
: "${TEXT_EMBEDDING_CACHE:?Set TEXT_EMBEDDING_CACHE to the LIBERO T5 cache}"
DATASET_STATS=${DATASET_STATS:-}
export OUTDIR=${OUTDIR:-$ROOT/runs/libero_vjepa_policy_fresh_packed48_t5len128_gbs128_step21360_seed7}

# This entry point is intentionally fresh; an inherited predictor-init path
# must not silently turn it into the DROID-initialized experiment.
export PREDICTOR_INIT=

exec bash "$ROOT/scripts/train_vjepa_policy.sh" "$@"
