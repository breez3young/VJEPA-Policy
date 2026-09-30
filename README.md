# V-JEPA Policy

<p align="center">
  <strong>World-action models on a frozen predictive visual latent space</strong>
</p>

<p align="center">
  <a href="https://github.com/breez3young/VJEPA-Policy"><img src="https://img.shields.io/badge/code-GitHub-181717?style=flat&logo=github" alt="Code"></a>
  <a href="https://github.com/breez3young/VJEPA-Policy/blob/release/LICENSE"><img src="https://img.shields.io/badge/license-MIT-2f6f68?style=flat" alt="MIT license"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-%3E%3D3.10-3776ab?style=flat&logo=python&logoColor=white" alt="Python 3.10 or newer"></a>
  <a href="https://arxiv.org/abs/2609.37250"><img src="https://img.shields.io/badge/arXiv-2609.37250-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv:2609.37250"></a>
</p>

V-JEPA Policy learns robot actions from a frozen predictive visual latent space.
The trainable part has two modules:

1. a language and state conditioned predictor for future visual latents;
2. a flow matching action expert that attends to the predictor context.

The released recipe uses V-JEPA 2.1 ViT-L and T5-XXL as frozen encoders. The
visual encoder is a registry entry, so a different visual latent substrate can
be selected without changing the predictor or action expert. Checkpoints,
datasets, simulator assets, text caches, and evaluation outputs stay outside Git.

**Paper:** [V-JEPA Policy: Building Effective World-Action Models on Predictive
Visual Latents](https://arxiv.org/pdf/2609.37250).

## Results at a glance

The default policy has 0.9B total parameters, 0.6B trainable parameters, a
frozen V-JEPA 2.1 ViT-L encoder, and a frozen T5-XXL text encoder.

| Initialization | LIBERO | LIBERO-Plus | RoboCasa-GR1 |
| --- | ---: | ---: | ---: |
| From scratch | 97.25 | 79.25 | 50.92 |
| DROID predictor | **98.70** | **91.50** | **55.58** |

Scores are success rates in percent. The DROID row transfers only the future
predictor; the action expert is initialized from scratch downstream.

## Support matrix

| Capability | Entry point | Status |
| --- | --- | --- |
| Predictor pretraining on DROID | `scripts/pretrain_droid_predictor_vjepa21_maxviews4.sh` | Reproduction recipe |
| Predictor pretraining on any dataset | `scripts/pretrain_predictor.py` | Dataset factory |
| Policy post-training on LeRobot | `scripts/train_vjepa_policy.sh` | Built-in adapter |
| Policy post-training on any dataset | `scripts/train_vjepa_policy.py` | Dataset factory |
| LIBERO train/eval | `examples/libero/` | Included; simulator is external |
| RoboCasa-GR1 train/eval | `scripts/gr1/`, `examples/gr1/` | Rollout environment is external |
| LIBERO-Plus | External `vla-evaluation-harness` | Integration contract; not vendored |
| Alternate visual encoders | `src/vjepa_policy/encoders/` | Registry and geometry contract |

## Layout

```text
src/vjepa_policy/
  encoders/                 frozen encoder registry and geometry contracts
  datasets/                 LeRobot, DROID, transforms, normalization
  models/                   predictor, action expert, policy, V-JEPA backbones
  policy_serving/           LIBERO and GR-1 checkpoint adapters
  dataset_api.py            small custom dataset factory contract
  trainer.py                Accelerate training loop and checkpoints
scripts/
  pretrain_predictor.py     generic predictor-only pretraining
  pretrain_droid_predictor.py
  train_vjepa_policy.py     generic post-training on LeRobot or an adapter
  train_vjepa_policy.sh     LIBERO reference launcher
  gr1/                      GR-1 preparation and training
examples/
  libero/                   websocket server, client, and evaluation launcher
  gr1/                      GR-1 server and rollout evaluator
```

The repository contains source only. Checkpoints, datasets, simulator assets,
logs, and evaluation outputs are intentionally kept outside Git.

## Install

```bash
git clone --branch release https://github.com/breez3young/VJEPA-Policy.git
cd VJEPA-Policy
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Python 3.10 or newer is required. The paper recipes use BF16 on multi-GPU NVIDIA
systems; memory depends on image size, views, and activation checkpointing. The
optional `libero` extra installs the local LIBERO simulator dependencies. The
LIBERO websocket server also needs the `openpi-client` package. GR-1 uses the
official Isaac-GR00T and RoboCasa environments and must be installed separately.

```bash
python -m pip install -e '.[libero]'
python -m pip install openpi-client
```

Check the install and list the stable encoder IDs before downloading weights:

```bash
python - <<'PY'
import torch
from vjepa_policy.encoders import encoder_names
print("torch", torch.__version__)
print("encoders", ", ".join(encoder_names()))
PY
```

## Choose an encoder

List the built-in registry IDs:

```bash
python -c 'from vjepa_policy.encoders import encoder_names; print("\n".join(encoder_names()))'
```

The stable registry exposes `vjepa2_vitl`, `vjepa2_1_vitl`, DINOv2, DINOv3,
WAN2.2, and geometry variants. A custom encoder implements the `EncoderSpec`
contract and calls `register_encoder`; training and serving then use the same
`--encoder` argument. The encoder must expose a stable latent grid and feature
width so the predictor can build its mask and projection layers.

To inspect the exact registry for a checkout:

```bash
python -c 'from vjepa_policy.encoders import encoder_names; print("\\n".join(encoder_names()))'
```

## Dataset contract

The built-in post-training path reads one or more LeRobot roots and discovers
camera features from `meta/info.json`. For another storage format, implement a
small factory in an importable module:

```python
def build_dataset(args, include_action=True):
    dataset = MyDataset(args)
    return dataset
```

Each sample must contain:

```text
video         [views, channels, frames, height, width] or [channels, frames, height, width]
context       [text_length, text_width] T5 features
context_mask  [text_length] boolean mask
proprio       [state_width]
```

Post-training samples also contain `action` and `action_is_pad`. Pass
`--dataset-factory package.module:build_dataset`, `--views ...`,
`--action-dim`, and `--proprio-dim`. The stock collator supplies the causal
latent masks. A factory may return `(dataset, collator)` when it needs custom
decoding or batching.

This keeps dataset code at the adapter boundary. The optimizer, checkpoint
format, visual encoder, predictor, and action expert stay unchanged.

The same boundary is used by policy post-training:

```bash
accelerate launch --num_processes 4 scripts/train_vjepa_policy.py \
  --dataset-factory my_data:build_dataset \
  --views camera.front camera.wrist \
  --encoder vjepa2_1_vitl \
  --encoder-checkpoint /path/to/vjepa2_1_vitl.pt \
  --text-cache-dir /path/to/t5_cache \
  --action-dim 7 --proprio-dim 8 --max-state-dim 48 \
  --context-len 128 --action-chunk-size 32 \
  --output-dir runs/my_policy
```

## Paper recipes

### DROID predictor

Set paths in [`configs/droid_vjepa21.env`](configs/droid_vjepa21.env), build the
129-token cache, then launch the 8-GPU, global-batch-192 recipe:

```bash
source configs/droid_vjepa21.env
python scripts/cache_droid_t5.py --dataset-root "$DATASET_ROOT" \
  --cache-dir "$TEXT_CACHE_DIR" --model-name google/t5-v1_1-xxl \
  --context-length 129 --device cuda --batch-size 1
bash scripts/pretrain_droid_predictor_vjepa21_maxviews4.sh
```

Use `MAX_STEPS=20` and a separate `OUTPUT_DIR` for a smoke run. For arbitrary
datasets, use `scripts/pretrain_predictor.py` with the factory contract above.

### LIBERO policy

Build a 128-token instruction cache, set paths in
[`configs/libero_vjepa21.env`](configs/libero_vjepa21.env), then choose scratch
or DROID initialization:

```bash
source configs/libero_vjepa21.env
bash scripts/train_vjepa_policy_fresh_packed48.sh

export PREDICTOR_INIT=/path/to/droid/checkpoint_step100000.pt
bash scripts/train_vjepa_policy_droid_init.sh
```

The paper setting is two independent 224px views, 32-step actions, global batch
128, and 21,360 updates. DROID uses a 129-token cache; LIBERO uses 128. The
When DROID has four view-embedding rows and LIBERO has two inputs, the loader
automatically maps source rows `[0, 1]` into the two-row policy embedding; all
other predictor parameters still load strictly. Use `--predictor-view-map` to
override that mapping for a custom camera order.

For arbitrary post-training datasets, call `scripts/train_vjepa_policy.py`
with `--dataset-factory`, `--views`, `--action-dim`, and `--proprio-dim`.

### RoboCasa-GR1

Follow [`examples/gr1/README.md`](examples/gr1/README.md), then run
`scripts/gr1/prepare.sh` and `scripts/gr1/train.sh vjepa2_1_vitl`. The rollout
environment is external.

## Evaluation

For LIBERO, set the matching encoder, 128-token cache, Python environments, and
output directory. Use `EVAL_SUITES=libero_spatial NUM_TRIALS_PER_TASK=1` for a
smoke run; the paper protocol uses four suites and 50 trials per task.

```bash
export VJEPA2_ENCODER_CHECKPOINT=/path/to/encoder.pt
export TEXT_EMBEDDING_CACHE=/path/to/libero_t5_len128
export POLICY_PYTHON=/path/to/policy/python
export LIBERO_PYTHON=/path/to/libero/python
export EVAL_DIR=/path/to/results/libero
bash examples/libero/evaluate_policy.sh /path/to/run_dir \
  /path/to/run_dir/checkpoint_step021360.pt
```

LIBERO-Plus uses [`allenai/vla-evaluation-harness`](https://github.com/allenai/vla-evaluation-harness)
with the V-JEPA server/cache integration. A clean upstream clone does not
include that integration; use a V-JEPA-enabled checkout containing
`scripts/libero_plus_runtime_env.sh`,
`src/vla_eval/model_servers/vjepa_policy.py`, and
`configs/model_servers/vjepa_policy/vjepa2_1_libero.yaml`.

Set `VLA_HARNESS_ROOT`, `VLA_BENCH_ENV` (contains `vla-eval`),
`VJEPA_SERVER_PYTHON`, `VLA_LIBERO_PLUS_ROOT`, and
`VLA_LIBERO_PLUS_ASSETS`. Set `VLA_IMAGEMAGICK_HOME` when MagickWand is not
available. Create a server config from the harness V-JEPA template using the
same policy checkpoint, frozen encoder, 128-token cache, and `dataset_stats`.

Run the paper topology with four servers, eight simulator clients per server,
32 shards, and batch size 8:

```bash
export VJEPA_SERVER_CONFIG=/path/to/vjepa2_1_droid_init_libero_plus.yaml
scripts/run_libero_plus_canonical.sh \
  --harness-root "$VLA_HARNESS_ROOT" \
  --server-config "$VJEPA_SERVER_CONFIG" \
  --checkpoint /path/to/checkpoint_step021360.pt \
  --pretrained-encoder /path/to/vjepa2_1_vitl.pt \
  --dataset-stats /path/to/dataset_stats.json \
  --text-cache-dir /path/to/libero_plus_t5_len128 \
  --gpus 0,1,2,3 --sim-gpus 0,1,2,3 \
  --clients-per-gpu 8 --batch-size 8 \
  --output-dir /path/to/results/libero-plus
```

The complete paper run contains 10,030 episodes. For a smoke run, use a
benchmark YAML with `max_tasks: 1` and add `--allow-incomplete`.

## Checkpoints and artifacts

Keep encoder weights, predictor or policy checkpoints, normalization statistics,
and T5 caches in an artifact store or local data directory. A policy checkpoint
and its `dataset_stats.json` must come from the same training run.

```text
checkpoint_step*.pt
dataset_stats.json
encoder checkpoint or model directory
T5 cache (DROID: embeddings.bin, index.json, manifest.json;
          LIBERO/GR-1: prompt-indexed *.pt files)
resolved command and environment metadata
```

`dataset_stats.json` must come from the same dataset and normalization settings
as the checkpoint. `--predictor-init` imports only predictor weights; `--resume`
restores training progress and optimizer scheduling.

## Troubleshooting

- **Shape or mask mismatch:** keep `num_frames`, `tubelet_size`, `crop_size`,
  views, and encoder ID identical between training and serving. Read the
  serialized `encoder_spec` in checkpoint metadata.
- **Text cache failure:** use the same context length at cache construction and
  training. Cache construction disables truncation and fails on an overlong
  prompt instead of silently changing the contract.
- **Legacy LIBERO checkpoint:** older scratch checkpoints without packed-state
  metadata use the variable-width state contract. Serve them with
  `--max-state-dim 0`, the matching 32-token cache, and the camera views saved
  by that run. DROID-init checkpoints use `max_state_dim=48` and the 128-token
  cache shown above.
- **Out of memory:** lower per-device `BATCH_SIZE`, enable more activation
  checkpointing blocks, or use gradient accumulation while preserving the
  intended global batch.
- **LIBERO/LIBERO-Plus EGL abort:** use the harness sharded runner with one
  simulator process per shard, `MUJOCO_GL=egl`, and a fresh output directory.
  Keep the selected simulator GPUs exclusive and idle; concurrent MuJoCo or
  RoboTwin jobs can leave native workers in D-state before the first episode.
  If the host has no usable EGL device, run the simulator in its supported
  software backend and record that setting with the result.

## Citation

Please cite the arXiv preprint:

```bibtex
@misc{zhang2026vjepapolicybuildingeffective,
  title={V-JEPA Policy: Building Effective World-Action Models on Predictive Visual Latents},
  author={Yang Zhang and Jiangyuan Zhao and Chenyou Fan and Jiayu Hu and Xiu Yuan and Chenjia Bai and Xiu Li},
  year={2026},
  eprint={2609.37250},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.37250},
}
```

## License

This repository is released under the [MIT License](LICENSE). Upstream visual
encoders, T5 checkpoints, datasets, simulator assets, and evaluation harnesses
retain their own licenses and terms.
