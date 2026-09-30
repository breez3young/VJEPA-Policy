# RoboCasa GR-1 Evaluation

The server uses Isaac-GR00T's ZeroMQ/msgpack protocol. Keep the simulator in its
dedicated environment and make both repositories importable without modifying the
LIBERO environment.

Start the policy server in the V-JEPA training environment:

```bash
export PYTHONPATH=/path/to/Isaac-GR00T:/path/to/VJEPA-Policy-revised/src
python examples/gr1/serve_policy.py \
  --checkpoint /path/to/checkpoint.pt \
  --pretrained-encoder /path/to/vjepa2_1_vitl_dist_vitG_384.pt \
  --text-cache-dir /path/to/text_embeddings \
  --dataset-stats /path/to/dataset_stats_absolute_minmax_ck16.json \
  --encoder-family vjepa2_1 \
  --encoder-model-name vit_large \
  --encoder-checkpoint-key ema_encoder
```

Run the official rollout client from the dedicated RoboCasa GR-1 environment:

```bash
export PYTHONPATH=/path/to/Isaac-GR00T:/path/to/VJEPA-Policy-revised
python examples/gr1/evaluate_policy.py \
  --host 127.0.0.1 --port 5555 \
  --n-episodes 20 --n-envs 5 --seed 7 \
  --output runs/gr1_eval/results.json
```

Video recording is disabled by default for the full evaluation. For a smoke test,
pass `--video-dir runs/gr1_eval/smoke_videos`; only the policy's egocentric wire
view is recorded.

The default policy predicts 16 absolute actions from a six-frame model clip:
two real `[-4, 0]` context frames and four future mask slots. The official
rollout executes the first 16 and then requests a new chunk with updated context.
For a chunk-32 checkpoint, start the server with
`--action-chunk-size 32 --num-frames 10`; the rollout execution horizon can be
changed independently with `--execution-horizon`.
