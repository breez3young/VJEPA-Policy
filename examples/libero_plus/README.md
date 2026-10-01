# LIBERO-Plus evaluation

Evaluate a policy trained on LIBERO on **10,030 perturbed tasks**, without further fine-tuning. Both From Scratch and Pretrained Predictor use this protocol. The public workflow reuses the [LIBERO policy server and client](../libero/README.md), with a separate Plus simulator environment and a task manifest for prompt/score validation.

The interface was reviewed on **2026-10-01** against official [LIBERO-Plus commit `4976dc3`](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182). Plus registers the same Python package and suite names as LIBERO: use separate environments and separate `LIBERO_CONFIG_PATH` directories. The rollout pattern follows [OpenPI's LIBERO example](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero), with the additional two-frame context required by V-JEPA Policy.

## Simulator setup

Run from the V-JEPA Policy repository root on Linux. The official Plus environment needs MagickWand for image perturbations. For example, on Ubuntu:

```bash
sudo apt-get install libexpat1 libfontconfig1-dev libmagickwand-dev libgl1 libegl1 libosmesa6
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus
git clone https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_ROOT"
git -C "$LIBERO_PLUS_ROOT" checkout --detach 4976dc30028e805ff8094b55501d532c48fec182

conda create -n vjepa-libero-plus python=3.10 pip -y
conda activate vjepa-libero-plus
export LIBERO_PLUS_PYTHON="$CONDA_PREFIX/bin/python"
"$LIBERO_PLUS_PYTHON" -m pip install --upgrade pip
# Torch loads simulator initial states; policy inference runs in another environment.
"$LIBERO_PLUS_PYTHON" -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
"$LIBERO_PLUS_PYTHON" -m pip install -r examples/libero_plus/requirements.txt
"$LIBERO_PLUS_PYTHON" -m pip install -e "$LIBERO_PLUS_ROOT"
```

Download `assets.zip` from the official [Sylvest/LIBERO-plus dataset](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main). Extract it under `$LIBERO_PLUS_ROOT/libero/libero/`, producing `assets/textures`, `assets/new_objects`, and the other official asset directories. BDDL tasks and initial states come from the pinned Git checkout. Save the archive's SHA-256 with your experiment. Torch 2.5.1 preserves the upstream state loader's behavior; check its `weights_only` handling before upgrading Torch for these state files. The dependency recipe is not a complete lockfile or a GPU validation claim.

## Prepare tasks and text embeddings

Copy [the environment template](../../configs/libero_plus_eval.env), edit paths, and source it in a clean shell. Use the same policy checkpoint, visual encoder, and statistics as LIBERO, plus a new cache directory for Plus instructions. Set `LIBERO_PLUS_PYTHON` in the local template to the interpreter path captured above, and `POLICY_PYTHON` to the interpreter in the policy conda environment.

```bash
cp configs/libero_plus_eval.env configs/libero_plus_eval.local.env
# Edit the copy, then load it.
source configs/libero_plus_eval.local.env

"$LIBERO_PLUS_PYTHON" examples/libero_plus/prepare.py \
  --libero-plus-root "$LIBERO_PLUS_ROOT" \
  --config-dir "$LIBERO_CONFIG_PATH" --output "$LIBERO_PLUS_MANIFEST"

# Use the policy environment with the training package and T5 dependencies installed.
"$POLICY_PYTHON" examples/libero_plus/cache_text.py \
  --manifest "$LIBERO_PLUS_MANIFEST" --cache-dir "$TEXT_EMBEDDING_CACHE" \
  --model-name google/t5-v1_1-xxl --context-length 128 --device cuda
```

`prepare.py` writes the isolated simulator config before importing LIBERO, checks the Git revision and classification checksum, and enumerates all four suites in task order 0. Non-language perturbations use canonical task instructions; language perturbations retain the rewritten BDDL instructions. The manifest records task IDs, names, categories, difficulty levels, and actual instructions. Unique prompts are counted from those instructions, rather than assumed from a historical cache.

`cache_text.py` uses the training prompt format, tokenization, and tensor payloads. It rejects instruction truncation and records the manifest hash, T5 identity, context length, and filenames. Interrupted caches remain marked incomplete; rerun with the same configuration to finish. Use the run's exact T5 snapshot if it differs from the model ID above.

## Launch

```bash
"$LIBERO_PLUS_PYTHON" scripts/check_environment.py --profile libero-plus
"$POLICY_PYTHON" scripts/check_environment.py --profile policy --require-cuda
export EVAL_DIR="$RUN_DIR/libero_plus_$(date -u +%Y%m%dT%H%M%SZ)"
bash examples/libero_plus/evaluate_policy.sh "$RUN_DIR" "$CHECKPOINT"
```

The launcher validates the cache, starts one server/client pair per selected suite, waits for `/healthz`, and requires a fresh output directory. Four suites need four GPU entries in parallel. Set `EVAL_SERIAL=1 EVAL_GPUS=0` to evaluate all four suites sequentially on one GPU in a single result directory. This example has no sharding or automatic resume. The client sends current and past frames, rotates both camera images by 180 degrees, and executes 16 actions per plan.

Rendering defaults to EGL. Select `LIBERO_MUJOCO_GL=osmesa` for CPU rendering with OSMesa installed; inference still uses a GPU. Record the renderer with the score. Video recording defaults off; set `RECORD_VIDEO=1` when needed.

## Protocol and results

| Suite | Episodes |
| --- | ---: |
| `libero_spatial` | 2,402 |
| `libero_object` | 2,518 |
| `libero_goal` | 2,591 |
| `libero_10` | 2,519 |

Each task uses one episode and the first initial state returned by the official suite, seed 7, and ten settling steps. Step limits are 220, 280, 300, and 520 for the four suites. Preserve frame stride 4, 256px rendering, and the checkpoint's view/state/action settings for the reference policy.

Outputs include per-episode `episodes.jsonl`, per-suite summaries, logs, copied task/cache manifests, and aggregate `results.json`. A full score requires 10,030 unique task identities, matching instructions and manifest hashes, and one boolean outcome per task. Simulator/server errors abort the run and are not counted as completed failed episodes. Overall success is episode-weighted; category scores include their denominators.

Revalidate results without a GPU or simulator:

```bash
python scripts/validate_eval_results.py libero-plus "$EVAL_DIR" \
  --manifest "$EVAL_DIR/task_manifest.json"
```

For a one-task diagnostic, keep the full manifest/cache and launch with:

```bash
EVAL_SUITES=libero_spatial EVAL_GPUS=0 MAX_TASKS=1 ALLOW_INCOMPLETE=1 \
  EVAL_DIR="$RUN_DIR/plus_smoke_$(date -u +%Y%m%dT%H%M%SZ)" \
  bash examples/libero_plus/evaluate_policy.sh "$RUN_DIR" "$CHECKPOINT"
```

Partial runs have `complete: false`. Keep the repository revision, environment report, weights/statistics identities, asset checksum, and manifests with results. GPU rollouts must validate the native runner before claiming paper-score reproduction.

## External harness

These public commands do not require `vla-evaluation-harness`. At the reviewed [harness commit `41bd8a4`](https://github.com/allenai/vla-evaluation-harness/tree/41bd8a44dbe5ba0f7902adb5b7c121c1c174dc84), the public tree has the Plus benchmark but no V-JEPA Policy model-server adapter or the previously assumed `libero_plus_runtime_env.sh`. Internal harness runs can continue separately with their own adapter and recorded revision.
