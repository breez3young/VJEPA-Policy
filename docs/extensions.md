# Dataset and encoder contracts

## Custom datasets

The built-in policy trainer reads LeRobot roots and camera features from `meta/info.json`. For another format, implement an importable factory:

```python
def build_dataset(args, include_action=True):
    return MyDataset(args, include_action=include_action)
```

Pass `--dataset-factory package.module:build_dataset`, ordered `--views`, `--action-dim`, and `--proprio-dim` to `scripts/train_vjepa_policy.py`. For predictor-only work use `scripts/pretrain_predictor.py`.

| Sample key | Shape / meaning |
| --- | --- |
| `video` | `[views, channels, frames, height, width]` or `[channels, frames, height, width]` |
| `context` | `[text_length, text_width]` cached text features |
| `context_mask` | `[text_length]` boolean mask |
| `proprio` | `[state_width]`, consistent with the chosen state encoding |
| `action` | `[chunk_size, action_dim]`, policy training only |
| `action_is_pad` | `[chunk_size]` padding mask, policy training only |

A factory can return a dataset, `(dataset, collator)`, or a mapping with `dataset` and `collator`. The stock collator supplies latent masks. See [src/vjepa_policy/dataset_api.py](../src/vjepa_policy/dataset_api.py) and the selected training parser for the exact contract. Dataset adapters own decoding and normalization; keep those details out of the predictor and action expert.

## Visual encoders

```bash
python -c 'from vjepa_policy.encoders import encoder_names; print("\n".join(encoder_names()))'
```

The registry includes V-JEPA 2/2.1 ViT-L, DINO, and WAN variants. Treat the actual registry as the list of IDs; historical ViT-G family/model combinations are not necessarily registry IDs. Implement `EncoderSpec` and a factory, then register it with `register_encoder(name, spec, factory)` in an imported module.

`EncoderSpec` describes input geometry, latent grid, temporal stride, and feature width. These drive predictor projections and masks. Keep physical encoder input geometry distinct from the predictor's logical grid. LIBERO serving uses the same registry through `examples/libero/serve_policy.py`; there is no separate alternate-latent server entry point in this checkout.

New checkpoints serialize `model_topology` and `policy_serving` metadata. Validate that training and serving agree on camera order, resize mode, frame layout, normalization, state packing, and RoPE settings before comparing encoders.
