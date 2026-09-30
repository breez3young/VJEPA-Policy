# RoboCasa GR-1

The GR-1 LeRobot release stores a 44-dimensional state and absolute action. This
experiment selects the official `robocasa_gr1_tabletop` 29D order:

```text
29D order: left_arm, right_arm, left_hand, right_hand, waist
state condition: concat(sin(state), cos(state)), 29D -> 58D
action target: absolute 29D action
```

State is not normalized. Exact global action minima and maxima are obtained by
reducing the exact per-task minima and maxima, then map absolute actions to
`[-1, 1]` with clipping. Preparation validates all
24 task roots, including the 20 Hz 44D schema, ego-view resolution, codec ranges,
exactly 1,000 parquet/video episodes, and non-empty remarks. It then binds the
statistics to a deterministic local-data manifest. Download and prepare the exact
ABot-M0 task allowlist with:

```bash
python scripts/gr1/download_dataset.py
bash scripts/gr1/prepare.sh  # ACTION_CHUNK_SIZE=16 by default
```

The downloader uses the Hugging Face Hub API to enumerate and download only the
24 allowlisted task directories from
`nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`. It does not invoke Git or
Git LFS, scan the full repository tree, or use the separate
`PhysicalAI-Robotics-GR00T-Teleop-Sim` release. Hub metadata and incomplete files
under the local `.cache/huggingface` directory make repeated invocations resumable.

After choosing the best frozen encoder from the LIBERO ablation and running a
micro-batch/activation-checkpointing preflight:

```bash
bash scripts/gr1/train.sh  # V-JEPA2 ViT-L, target_encoder from vitl.pt
```

The default run uses 50,000 optimizer steps and global batch 1024 on four GPUs:
micro-batch 32 with eight accumulation steps. Run a real forward/backward preflight
before increasing the micro-batch. The 16-step action chunk uses video offsets
`[-4, 0, 4, 8, 12, 16]`: two context frames and four future frames. Training excludes
episode-tail samples without the complete future target while retaining padded past
context at episode starts.

The chunk-32 follow-up is selected consistently for preparation and training:

```bash
ACTION_CHUNK_SIZE=32 bash scripts/gr1/prepare.sh
ACTION_CHUNK_SIZE=32 bash scripts/gr1/train.sh
```

`ACTION_CHUNK_SIZE` must be divisible by the fixed video stride of four. The launcher
sets `num_frames=ACTION_CHUNK_SIZE+1`, so chunk 32 uses eight future frames.
