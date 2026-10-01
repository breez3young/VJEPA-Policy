# Setup

## Environments

The package requires Python 3.10+. Use Linux/NVIDIA for the reference BF16 training and benchmark recipes. Launchers use Bash 4+ (`mapfile`, associative arrays) and GNU tools. A macOS checkout is suitable for editing and lightweight checks, not these simulator recipes.

Create a dedicated conda environment for policy training and serving:

```bash
conda create -n vjepa-policy python=3.10 pip -y
conda activate vjepa-policy
export POLICY_PYTHON="$CONDA_PREFIX/bin/python"
```

Install a [PyTorch build](https://pytorch.org/get-started/locally/) compatible with your GPU driver in this environment, then install from the repository root:

```bash
python -m pip install -e .
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'from vjepa_policy.encoders import encoder_names; print("\n".join(encoder_names()))'
```

Memory depends on the encoder, views, batch size, and activation checkpointing; this repository does not specify a measured minimum-VRAM guarantee.

| Environment | Additional setup | Used by |
| --- | --- | --- |
| Policy training/serving | `python -m pip install -e '.[serving]'` for serving | `POLICY_PYTHON`, `VJEPA_SERVER_PYTHON` |
| LIBERO simulation | Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and assets; install `openpi-client`, `tyro`, `imageio[ffmpeg]`, `termcolor` plus compatible simulator dependencies | `LIBERO_PYTHON` |
| LIBERO-Plus | Pinned Plus checkout and assets, separate simulator environment, MagickWand and EGL/OSMesa; see [the native guide](../examples/libero_plus/README.md) | `LIBERO_PLUS_PYTHON`, `LIBERO_CONFIG_PATH` |
| RoboCasa-GR1 | Official [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) and its RoboCasa-GR1 environment | Dedicated policy/simulator environments |

Use separate conda environments for the policy and each simulator when their dependencies differ. Record the active environment's `$CONDA_PREFIX/bin/python` in the corresponding local configuration. Keep LIBERO and LIBERO-Plus in different environments because they provide the same Python package. For RoboCasa-GR1, follow the upstream environment recipe in the table above.

For a combined LIBERO environment, `python -m pip install -e '.[serving,libero]'` adds the Python dependencies listed by this project. It does **not** install LIBERO itself or download BDDL files, initial states, or meshes. When using a separate simulator environment, follow LIBERO's installation instructions there and keep its tested simulator versions; the local extra pins robosuite 1.4.1 and MuJoCo 3.2.3. The simulator client does not need the training package installed.

The websocket transport lives in this repository and uses `openpi-client`'s message format. The upstream `openpi-client` package provides a client, not a `websocket_policy_server` module.

## Artifacts

Supply the following before starting a recipe:

| Artifact | Contract |
| --- | --- |
| Frozen visual encoder | Match the checkpoint's registry ID and checkpoint key; obtain V-JEPA 2.1 weights from [Meta's release](https://github.com/facebookresearch/vjepa2) |
| Training dataset | LeRobot roots with `meta/info.json`, task metadata, parquet data, and videos; exact LIBERO names are in [Training](training.md#libero) |
| T5 model | `google/t5-v1_1-xxl` or the exact snapshot used by the run; needed to build caches |
| Text cache | DROID: 129-token memmap; LIBERO: 128-token prompt files; GR-1: 48-token prompt files by default |
| Policy checkpoint | `checkpoint_step*.pt` from the desired run |
| Normalization statistics | The corresponding run's `dataset_stats.json`; GR-1 uses its chunk-specific absolute-action statistics |
| Simulator assets | Install separately for the selected benchmark |

Train policies using the supplied recipes. LIBERO and RoboCasa-GR1 policies and the DROID-pretrained predictor are planned checkpoint releases; links will be added to the [release plan](../README.md#checkpoint-release-plan) when available. Keep each trained checkpoint with its statistics, configuration, encoder identity, and text-cache provenance for evaluation.

## Local configuration

Copy a recipe to an ignored local file and edit that copy:

```bash
cp configs/libero_vjepa21.env configs/libero_vjepa21.local.env
# Set LIBERO_DATA_ROOT, VJEPA2_ENCODER_CHECKPOINT, TEXT_EMBEDDING_CACHE.
source configs/libero_vjepa21.local.env
```

Templates and launchers share defaults in `configs/recipes/`. Launchers **do not load your `.local.env` files automatically**; source the edited template explicitly. Use absolute paths to artifacts outside the checkout, quoted when they contain spaces. The `/path/to/...` entries are placeholders to replace, not defaults that will work after cloning. Keep only task-relevant variables in an environment snapshot; do not record authentication tokens.

## Common failures

- **Missing T5 tokenizer:** reinstall package dependencies, including SentencePiece.
- **Missing prompt:** use the correct instruction source and cache length; cache construction disables truncation. LIBERO-Plus needs perturbation prompts too.
- **Encoder/view/state mismatch:** compare the saved serving metadata with the selected encoder and statistics before overriding a shape.
- **CUDA or simulator import errors:** check the selected interpreter and its dependencies independently; a successful policy import does not test MuJoCo.
- **EGL native abort:** preserve logs and retry in a new evaluation directory. For Plus, select `LIBERO_MUJOCO_GL=osmesa` for CPU rendering and record that choice. Never count a renderer crash as a completed failed episode.
