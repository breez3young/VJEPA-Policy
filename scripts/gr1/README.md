# RoboCasa-GR1 preparation and training

Start with [Setup](../../docs/setup.md). Copy `configs/gr1_vjepa21.env` to `configs/gr1_vjepa21.local.env`, set `DATA_ROOT` and `WEIGHTS`, then source it. `WEIGHTS` must contain `vjepa2_1_vitl_dist_vitG_384.pt` for the default encoder.

```bash
source configs/gr1_vjepa21.local.env
python -m scripts.gr1.download_dataset --local-dir "$DATA_ROOT"
bash scripts/gr1/prepare.sh
bash scripts/gr1/train.sh vjepa2_1_vitl
```

The downloader uses the Hugging Face Hub API for the 24 task directories listed in `scripts/gr1/dataset_manifest.py` from `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`. It is resumable through the Hub cache. This is not the separate GR00T-Teleop-Sim dataset.

Preparation checks the 20 Hz, 44D schema, camera/video data, task remarks, and 1,000 episodes per task (24,000 total). It computes exact action minima/maxima and binds the statistics to a deterministic local-data manifest. Artifacts are written to `ARTIFACT_ROOT`:

```text
dataset_stats_absolute_minmax_ck16.json
text_embeddings/                         # remarks, T5 length 48 by default
```

The policy selects the 29D order `left_arm, right_arm, left_hand, right_hand, waist`. State conditioning is `concat(sin(state), cos(state))`, giving 58D, without normalization. Targets are absolute 29D actions, normalized to `[-1, 1]` using exact min/max values and clipping. LIBERO statistics and packed state do not apply to this recipe.

The local template and `train.sh` share `configs/recipes/gr1.env`: 50,000 updates and global batch 256 (4 GPUs × batch 64 × accumulation 1). Existing environment values take precedence. Preserve the intended batch when changing GPU count. The shell script accepts only the encoder name; it does not forward arbitrary training flags. Use the Python entry point for custom training options.

The default chunk 16 samples video offsets `[-4, 0, 4, 8, 12, 16]`: two observed context frames and four future frames. The launcher's `num_frames=17` describes the source window; the sampled model clip has six frames. Tail samples without complete future targets are excluded, while episode-start past context is padded.

For chunk 32, use matching settings during both preparation and training:

```bash
ACTION_CHUNK_SIZE=32 bash scripts/gr1/prepare.sh
ACTION_CHUNK_SIZE=32 bash scripts/gr1/train.sh vjepa2_1_vitl
```

The action chunk must be divisible by four. Chunk 32 has a ten-frame sampled clip and its own `dataset_stats_absolute_minmax_ck32.json`. See the [GR-1 evaluation guide](../../examples/gr1/README.md) for serving those horizons.
