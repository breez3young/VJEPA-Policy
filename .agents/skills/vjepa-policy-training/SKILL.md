---
name: vjepa-policy-training
description: Prepare data and text caches, run VJEPA-Policy predictor-only pretraining or downstream joint training, and resume checkpoints. Use for From Scratch and Pretrained Predictor workflows, with LIBERO as the reference example and support for custom datasets.
---

# V-JEPA Policy training

Run from the repository root. Read [Setup](../../../docs/setup.md) and [Training](../../../docs/training.md); use the [GR-1 skill](../vjepa-policy-gr1/SKILL.md) for its separate data/state preparation.

## Select the current entry point

| Task | Source |
| --- | --- |
| Predictor-only pretraining on DROID | `scripts/pretrain_droid_predictor_vjepa21_maxviews4.sh` |
| From Scratch (LIBERO example) | `scripts/train_vjepa_policy_fresh_packed48.sh` |
| Pretrained Predictor (LIBERO example) | `scripts/train_vjepa_policy_droid_init.sh` |
| Custom predictor dataset | `scripts/pretrain_predictor.py` |
| Custom policy dataset / LeRobot | `scripts/train_vjepa_policy.py` |

Copy the relevant `configs/*.env` template to an ignored `.local.env`, resolve artifact paths, and source it explicitly. Check the launcher and Python parser when changing flags; not every shell script forwards arguments. Prefer the user's requested recipe and hardware; label adaptations to the reference protocol.

## Non-obvious contracts

- Use **From Scratch** and **Pretrained Predictor** in user-facing descriptions. Predictor-only pretraining uses DROID videos, instructions, and observed state without action labels or an action expert. Downstream training jointly optimizes both modules, with the action expert initialized from scratch in both variants. The visual and text encoders remain frozen.

- DROID uses a 129-token memmap cache built by `scripts/cache_droid_t5.py`; LIBERO uses 128-token prompt files built by `vjepa_policy.text_embeddings`. Its default is 128; pass other recipes' lengths explicitly.
- The DROID reference shell launcher enforces eight GPUs. Use its Python entry point with Accelerate for another topology, not an ineffective environment override.
- The fresh wrapper clears inherited `PREDICTOR_INIT`; predictor transfer needs the explicit source path and packed state 48. Only predictor weights transfer. `--resume` is a different operation and restores training progress.
- Global batch is GPUs × per-device batch × accumulation. Keep it fixed for a reproduction, or record the changed value for an experiment.
- Templates and launchers share `configs/recipes/*.env`. Existing environment variables override defaults; source the local template for artifact paths. LIBERO defaults to V-JEPA 2.1 ViT-L, state width 48, and text length 128.
- New checkpoints serialize serving metadata. Preserve the generated dataset statistics and cache/encoder provenance for later inference.

Use a fresh output directory for a smoke run. Report whether cache construction, data loading, forward/backward, optimizer updates, and checkpoint saving were actually tested. Do not infer training success from imports or CLI help alone. For custom data/encoders follow [Extension contracts](../../../docs/extensions.md).
