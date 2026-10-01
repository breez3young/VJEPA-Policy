# RoboCasa-GR1 evaluation

GR-1 uses the official Isaac-GR00T ZeroMQ/msgpack policy interface. Install Isaac-GR00T in the policy environment and use its supported RoboCasa-GR1 simulator environment for rollouts. See [Setup](../../docs/setup.md) and [dataset preparation](../../scripts/gr1/README.md).

## Serve the policy

From this repository root, start the server with your checkpoint and its prepared statistics/cache. Set `GR00T_ROOT` to your Isaac-GR00T checkout if it is not already installed as a package:

```bash
export GR00T_ROOT=/path/to/Isaac-GR00T
export PYTHONPATH="$GR00T_ROOT:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python examples/gr1/serve_policy.py \
  --checkpoint /path/to/policy/checkpoint.pt \
  --pretrained-encoder /path/to/vjepa2_1_vitl_dist_vitG_384.pt \
  --text-cache-dir /path/to/gr1/artifacts/text_embeddings \
  --dataset-stats /path/to/gr1/artifacts/dataset_stats_absolute_minmax_ck16.json \
  --encoder-family vjepa2_1 --encoder-model-name vit_large \
  --encoder-checkpoint-key ema_encoder \
  --t5-len 48 --action-chunk-size 16 --num-frames 6 \
  --host 127.0.0.1 --port 5555
```

This CLI uses explicit GR-1 defaults; it does not expose the same optional metadata-driven flags as the LIBERO server. Match all settings to the training run, especially encoder identity, text length, chunk size, and normalization.

## Evaluation

The paper evaluates all 24 tasks with **50 episodes per task** (1,200 episodes) and reports the mean task success rate. In a second terminal with the dedicated simulator environment active, run:

```bash
python examples/gr1/evaluate_policy.py \
  --host 127.0.0.1 --port 5555 \
  --n-episodes 50 --n-envs 5 --seed 7 --execution-horizon 16 \
  --output runs/gr1_eval/results.json
```

The CLI defaults to 50 episodes per task, matching the paper. The command keeps `--n-episodes 50` explicit for either the From Scratch or Pretrained Predictor variant.

## Results

The JSON report is updated after each task. Require `completed_tasks=24`, 50 boolean episode outcomes in each task's `successes` list (1,200 total), and a consistent `macro_success_rate` for the paper protocol. Keep the simulator revision, command, checkpoint, and artifact provenance with the report. Choose a fresh output path for a new experiment.

## Diagnostic runs

For one diagnostic rollout, add `--task PnPBottleToCabinetClose --n-episodes 1 --n-envs 1` and use a separate output path. Video recording is opt-in via `--video-dir`; it records the egocentric wire view. A diagnostic result is not the full benchmark.

## Action horizon

The default server predicts 16 absolute actions from a six-frame model clip. For a chunk-32 checkpoint, use `--action-chunk-size 32 --num-frames 10` and the chunk-32 statistics. The client's `--execution-horizon` is independent: retain 16 to execute half of a predicted 32-step chunk before observing again.
