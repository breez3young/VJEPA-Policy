"""Pretrain a future-latent predictor on any dataset adapter.

The adapter is deliberately small.  Each sample must provide ``video``
(``[V,C,T,H,W]`` or ``[C,T,H,W]``), ``context`` (`[L,D]`),
``context_mask`` (`[L]`), and ``proprio`` (`[S]`).  A factory is imported with
``--dataset-factory module:function`` and receives this argparse namespace.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from vjepa_policy.dataset_api import PredictorBatchCollator, build_dataset
from vjepa_policy.encoders import build_encoder, encoder_names, encoder_spec
from vjepa_policy.models.predictor_pretraining import PredictorPretraining
from vjepa_policy.models.wm_predictor import build_world_model_predictor
from vjepa_policy.state import packed_state_dim
from vjepa_policy.trainer import Trainer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-factory", required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--encoder", choices=encoder_names(), default="vjepa2_1_vitl")
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--encoder-checkpoint-key", default=None)
    parser.add_argument("--output-dir", default="./runs/predictor_pretraining")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--context-tubelets", type=int, default=1)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--proprio-dim", type=int, required=True)
    parser.add_argument("--max-state-dim", type=int, default=48)
    parser.add_argument("--pred-depth", type=int, default=24)
    parser.add_argument("--pred-embed-dim", type=int, default=1024)
    parser.add_argument("--pred-num-heads", type=int, default=16)
    parser.add_argument("--num-mask-tokens", type=int, default=10)
    parser.add_argument("--lang-dim", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--activation-checkpointing-blocks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--predictor-init", default=None)
    parser.add_argument("--no-encoder-interpolate-rope", action="store_true")
    parser.add_argument("--legacy-predictor-rope", action="store_true")
    parser.add_argument("--wandb-project", default="vjepa_policy")
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args(argv)


def _seed(seed: int) -> None:
    rank = int(os.environ.get("RANK", "0"))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def _load_predictor(model, path: str | None) -> None:
    if not path:
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("predictor") if isinstance(payload, dict) else None
    if state is None and isinstance(payload, dict) and all(
        torch.is_tensor(value) for value in payload.values()
    ):
        state = payload
    if state is None:
        raise ValueError(f"{path} is not a predictor state dict or policy checkpoint")
    model.predictor.load_state_dict(state, strict=True)


def main(argv=None):
    args = parse_args(argv)
    if args.image_size % 16:
        raise ValueError("image-size must be divisible by 16")
    if args.max_views is None:
        args.max_views = len(args.views)
    if args.max_views < len(args.views):
        raise ValueError("max-views must cover every declared view")
    _seed(args.seed)
    device = _device()
    spec = encoder_spec(args.encoder)
    checkpoint_key = args.encoder_checkpoint_key or spec.checkpoint_key
    encoder = build_encoder(
        args.encoder,
        checkpoint=args.encoder_checkpoint,
        checkpoint_key=checkpoint_key,
        image_size=(args.image_size, args.image_size),
        video_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        interpolate_rope=not args.no_encoder_interpolate_rope,
    )
    layout = spec.layout_for_clip(
        video_frames=args.num_frames,
        image_size=(args.image_size, args.image_size),
        num_views=len(args.views),
        context_steps=args.context_tubelets,
    )
    predictor = build_world_model_predictor(
        img_size=(args.image_size, args.image_size),
        patch_size=spec.input_patch_size,
        num_frames=args.num_frames,
        tubelet_size=spec.temporal_stride,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=args.pred_embed_dim,
        depth=args.pred_depth,
        num_heads=args.pred_num_heads,
        num_mask_tokens=args.num_mask_tokens,
        lang_dim=args.lang_dim,
        proprio_dim=packed_state_dim(args.max_state_dim),
        use_activation_checkpointing=args.activation_checkpointing_blocks > 0,
        activation_checkpointing_blocks=args.activation_checkpointing_blocks,
        interpolate_rope=True,
        corrected_rope_frequency_pairing=not args.legacy_predictor_rope,
        num_views=len(args.views),
        max_views=args.max_views,
        latent_grid=(layout.grid_depth, layout.grid_height, layout.grid_width),
        canonical_spatial_grid=spec.canonical_spatial_grid,
    )
    model = PredictorPretraining(
        encoder,
        predictor,
        proprio_dim=args.proprio_dim,
        max_state_dim=args.max_state_dim,
        latent_layout=layout,
    ).to(device)
    _load_predictor(model, args.predictor_init)
    model.encoder_name = spec.name
    model.encoder_spec = spec.to_dict()
    model.encoder_checkpoint_key = checkpoint_key
    model.camera_keys = tuple(args.views)
    model.policy_serving = {"config": {"stage": "predictor_pretraining", "encoder": spec.name}}
    dataset, custom_collator = build_dataset(
        args.dataset_factory, args, include_action=False
    )
    collator = custom_collator or PredictorBatchCollator(layout)
    Trainer(
        model,
        dataset,
        collator,
        cfg={
            "output_dir": args.output_dir,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "mixed_precision": args.mixed_precision,
            "save_every": args.save_every,
            "prefetch_factor": args.prefetch_factor,
            "train_seed": args.seed,
            "data_seed": args.seed,
            "wandb_project": args.wandb_project,
            "wandb_name": args.wandb_name,
            "training_stage": "predictor_pretraining",
        },
    ).train()


if __name__ == "__main__":
    main()
