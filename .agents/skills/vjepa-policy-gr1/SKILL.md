---
name: vjepa-policy-gr1
description: Prepare, train, serve, or evaluate VJEPA-Policy on RoboCasa-GR1 with the 29D absolute-action and sin/cos state contract. Use for GR-1 work with the external Isaac-GR00T simulator environment.
---

# V-JEPA Policy RoboCasa-GR1

Run from the repository root. Read [preparation/training](../../../scripts/gr1/README.md) and [serving/evaluation](../../../examples/gr1/README.md). GR-1 uses Isaac-GR00T's policy protocol; the LIBERO websocket client and state representation do not apply.

## Data and training

Use `configs/gr1_vjepa21.env` via a local copy. Download with `python -m scripts.gr1.download_dataset --local-dir "$DATA_ROOT"`, then run `scripts/gr1/prepare.sh`. The allowlist is in `scripts/gr1/dataset_manifest.py`: 24 tasks, 1,000 episodes each, 20 Hz source data. Validate the dataset and its manifest-bound statistics before training.

The raw schema is 44D. The model selects the 29D order left arm, right arm, left hand, right hand, waist; state conditioning is 58D sin/cos, without state normalization. Actions are absolute 29D, with exact min/max normalization. Do not apply LIBERO's 48+48 packed state or quantile statistics.

The default action chunk is 16, text length 48, language field `remarks`, and sampled model clip length 6. Keep preparation, training, and serving horizons consistent. Chunk 32 needs its own statistics and ten-frame serving clip. The template and standalone launcher share global batch 256 / accumulation 1 from `configs/recipes/gr1.env`. The training shell accepts an encoder name, not arbitrary forwarded training arguments.

## Serve and evaluate

Use `examples/gr1/serve_policy.py` in the policy environment with the matching Isaac-GR00T import path. It uses explicit encoder/family/key and horizon flags; do not assume the metadata-driven LIBERO CLI applies unchanged. Run `examples/gr1/evaluate_policy.py` in the supported simulator environment.

The paper protocol evaluates 24 tasks with 50 episodes each (1,200 episodes). The CLI defaults to 50; the guide also passes it explicitly. Use seed 7 and execution horizon 16 for the documented recipe. Apply the same evaluation to the From Scratch and Pretrained Predictor variants. Restrict `--task`, `--n-episodes`, and `--n-envs` only for a requested diagnostic or recorded protocol change. Compare action chunk and execution horizon separately. Video recording is opt-in.

For the paper protocol, require `completed_tasks=24`, exactly 50 boolean outcomes in each task's `successes` list, and a consistent `macro_success_rate` in the JSON report. It is written after each task, so an existing JSON file may still be partial. Report coverage and external environment revision with the score; preserve logs and stop only this run's processes.
