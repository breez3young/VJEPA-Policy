"""Embodied pre-training of the language-conditioned V-JEPA predictor on DROID.

The ViT-L target encoder is frozen.  This entry point deliberately constructs
no Action Expert and optimizes only ``WorldModelPredictor`` parameters.  It
uses the DROID-specific 15 Hz -> 5 Hz sample offsets (ten frames: one context
tubelet and four future tubelets) rather than the contiguous policy sampler.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from vjepa_policy.data import CausalPatchMask
from vjepa_policy.datasets.droid import (
    DROID_BENCH_CACHE_ROOT,
    DROID_CONTEXT_TUBELETS,
    DROID_FPS,
    DROID_MAX_FUTURE_OFFSET,
    DROID_NUM_FRAMES,
    DROID_PROMPT,
    DROID_ROOT,
    DROID_SOURCE_FRAME_STRIDE,
    DROID_STATE_KEY,
    DROID_TARGET_FPS,
    DROID_T5_CONTEXT_LENGTH,
    DROID_TUBELET_SIZE,
    DROID_VIDEO_OFFSETS,
    DROID_VIDEO_KEYS,
    DroidPretrainingDataset,
    PackedT5EmbeddingCache,
    build_droid_video_transform,
)
from vjepa_policy.encoders import build_encoder, encoder_spec
from vjepa_policy.models.predictor_pretraining import PredictorPretraining
from vjepa_policy.models.wm_predictor import build_world_model_predictor
from vjepa_policy.state import DEFAULT_MAX_STATE_DIM, packed_state_dim
from vjepa_policy.trainer import Trainer
from vjepa_policy.utils.logging import get_logger


logger = get_logger(__name__, force=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DROID_ROOT)
    parser.add_argument(
        "--encoder",
        choices=("vjepa2_vitl", "vjepa2_1_vitl"),
        default="vjepa2_vitl",
    )
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--encoder-checkpoint-key", default="target_encoder")
    parser.add_argument("--text-cache-dir", default=DROID_BENCH_CACHE_ROOT)
    parser.add_argument(
        "--output-dir",
        default="./runs/droid_predictor_pretraining_prts_5fps_bs192_100k",
    )
    parser.add_argument(
        "--augmentation",
        choices=("current", "letterbox", "vjepa2_ac", "prts_crop_rotate"),
        default="prts_crop_rotate",
        help="PRTS-style crop+rotate, current letterbox, or V-JEPA2-AC reference crop.",
    )
    parser.add_argument("--context-length", type=int, default=DROID_T5_CONTEXT_LENGTH)
    parser.add_argument("--prompt-template", default=DROID_PROMPT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=DROID_TUBELET_SIZE)
    parser.add_argument("--num-frames", type=int, default=DROID_NUM_FRAMES)
    parser.add_argument("--context-tubelets", type=int, default=DROID_CONTEXT_TUBELETS)
    parser.add_argument("--pred-depth", type=int, default=24)
    parser.add_argument("--pred-embed-dim", type=int, default=1024)
    parser.add_argument("--pred-num-heads", type=int, default=16)
    parser.add_argument("--num-mask-tokens", type=int, default=10)
    parser.add_argument(
        "--max-views",
        type=int,
        default=len(DROID_VIDEO_KEYS),
        help="View-embedding table capacity; DROID still activates exactly two rows.",
    )
    parser.add_argument("--lang-dim", type=int, default=4096)
    parser.add_argument("--proprio-dim", type=int, default=8)
    parser.add_argument(
        "--max-state-dim",
        type=int,
        default=DEFAULT_MAX_STATE_DIM,
        help="Native state width before concatenating the validity mask (default: 48).",
    )
    parser.add_argument("--activation-checkpointing-blocks", type=int, default=8)
    parser.add_argument(
        "--predictor-rope-frequency-pairing",
        choices=("corrected", "legacy"),
        default="corrected",
    )
    parser.add_argument(
        "--encoder-interpolate-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--predictor-init",
        default=None,
        help="Optional checkpoint containing a predictor state_dict.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Per-device batch size (24 fits the measured 80-GiB H100 budget).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Data-loader workers per rank (4 sustained the measured throughput without excess RSS).",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--recycle-workers-every",
        type=int,
        default=500,
        help="Respawn video workers periodically to bound the torchcodec RSS leak.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--loss-exp", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--wandb-project", default="vjepa_policy")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--video-backend", choices=("torchcodec", "pyav"), default="torchcodec"
    )
    return parser.parse_args(argv)


def seed_process(seed: int) -> None:
    rank = int(os.environ.get("RANK", "0"))
    rank_seed = int(seed) + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def make_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


class DroidPredictorCollator:
    """Stack DROID samples and add one causal context/target mask pair."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        tubelet_size: int,
        num_frames: int,
        num_views: int,
        context_tubelets: int,
    ):
        mask = CausalPatchMask(
            image_size=(image_size, image_size),
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            video_frames=num_frames,
            context_tubelets=context_tubelets,
            num_views=num_views,
        )
        self._ctx_idx = mask.ctx_idx
        self._tgt_idx = mask.tgt_idx

    def __call__(self, batch):
        clips = torch.stack([sample["video"] for sample in batch])
        language = torch.stack([sample["context"] for sample in batch])
        language_mask = torch.stack([sample["context_mask"] for sample in batch])
        state = torch.stack([sample["proprio"] for sample in batch])
        batch_size = len(batch)
        masks_enc = [self._ctx_idx.unsqueeze(0).expand(batch_size, -1).clone()]
        masks_pred = [self._tgt_idx.unsqueeze(0).expand(batch_size, -1).clone()]
        return clips, language, language_mask, masks_enc, masks_pred, state


def _load_predictor_state(
    model: PredictorPretraining, checkpoint_path: str | None
) -> None:
    if not checkpoint_path:
        return
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("predictor") if isinstance(payload, dict) else None
    if state is None:
        if not isinstance(payload, dict) or not all(
            torch.is_tensor(v) for v in payload.values()
        ):
            raise ValueError(f"{checkpoint_path} is not a predictor checkpoint")
        state = payload
    result = model.predictor.load_state_dict(state, strict=True)
    logger.info("loaded predictor initialization from %s: %s", checkpoint_path, result)


def build_model(args, device: torch.device):
    if args.image_size % args.patch_size:
        raise ValueError("image-size must be divisible by patch-size")
    if args.num_frames != DROID_NUM_FRAMES or args.tubelet_size != DROID_TUBELET_SIZE:
        raise ValueError(
            "DROID predictor pretraining requires 10 frames and tubelet size 2"
        )
    if args.context_length != DROID_T5_CONTEXT_LENGTH:
        raise ValueError(
            f"DROID T5 cache is intentionally untruncated at {DROID_T5_CONTEXT_LENGTH} tokens; "
            f"got context-length={args.context_length}"
        )
    if args.max_state_dim <= 0:
        raise ValueError("DROID pretraining requires a positive --max-state-dim")
    if args.max_views < len(DROID_VIDEO_KEYS):
        raise ValueError(
            f"max-views={args.max_views} is smaller than the "
            f"{len(DROID_VIDEO_KEYS)} active DROID views"
        )
    if args.proprio_dim > args.max_state_dim:
        raise ValueError(
            f"proprio-dim={args.proprio_dim} exceeds max-state-dim={args.max_state_dim}"
        )
    model_proprio_dim = packed_state_dim(args.max_state_dim)
    spec = encoder_spec(args.encoder)
    encoder = build_encoder(
        args.encoder,
        checkpoint=args.encoder_checkpoint,
        checkpoint_key=args.encoder_checkpoint_key,
        image_size=(args.image_size, args.image_size),
        video_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        interpolate_rope=args.encoder_interpolate_rope,
    )
    layout = spec.layout_for_clip(
        video_frames=args.num_frames,
        image_size=(args.image_size, args.image_size),
        num_views=len(DROID_VIDEO_KEYS),
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
        proprio_dim=model_proprio_dim,
        use_activation_checkpointing=args.activation_checkpointing_blocks > 0,
        activation_checkpointing_blocks=args.activation_checkpointing_blocks,
        interpolate_rope=True,
        corrected_rope_frequency_pairing=args.predictor_rope_frequency_pairing
        == "corrected",
        num_views=len(DROID_VIDEO_KEYS),
        max_views=args.max_views,
        latent_grid=(layout.grid_depth, layout.grid_height, layout.grid_width),
        canonical_spatial_grid=spec.canonical_spatial_grid,
    )
    model = PredictorPretraining(
        encoder,
        predictor,
        proprio_dim=args.proprio_dim,
        max_state_dim=args.max_state_dim,
        loss_exp=args.loss_exp,
        latent_layout=layout,
    )
    _load_predictor_state(model, args.predictor_init)
    model.camera_keys = tuple(DROID_VIDEO_KEYS)
    model.view_layout = "independent"
    model.encoder_family = spec.family
    model.encoder_model_name = spec.model_name
    model.encoder_checkpoint_key = args.encoder_checkpoint_key
    model.encoder_name = spec.name
    model.encoder_spec = spec.to_dict()
    model.latent_layout = layout
    model.state_indices = tuple(range(args.proprio_dim))
    model.instruction_field = (
        "language_instruction|language_instruction_2|language_instruction_3"
    )
    model.policy_serving = {
        "config": {
            "stage": "droid_predictor_pretraining",
            "encoder": spec.name,
            "encoder_checkpoint_key": args.encoder_checkpoint_key,
            "image_size": args.image_size,
            "num_frames": args.num_frames,
            "tubelet_size": args.tubelet_size,
            "context_tubelets": args.context_tubelets,
            "future_frames": 8,
            "camera_keys": list(DROID_VIDEO_KEYS),
            "max_views": args.max_views,
            "t5_len": args.context_length,
            "proprio_dim": args.proprio_dim,
            "max_state_dim": args.max_state_dim,
            "packed_proprio_dim": model_proprio_dim,
            "augmentation": args.augmentation,
            "fps": DROID_TARGET_FPS,
            "source_fps": DROID_FPS,
            "source_frame_stride": DROID_SOURCE_FRAME_STRIDE,
            "sampling_interval_seconds": 1.0 / DROID_TARGET_FPS,
            "video_offsets": list(DROID_VIDEO_OFFSETS),
            "max_steps": args.max_steps,
        }
    }
    return model.to(device), layout


def build_dataset(args):
    cache = PackedT5EmbeddingCache(
        args.text_cache_dir, context_length=args.context_length
    )
    transform_name = (
        "letterbox" if args.augmentation == "current" else args.augmentation
    )
    dataset = DroidPretrainingDataset(
        args.dataset_root,
        video_keys=DROID_VIDEO_KEYS,
        state_key=DROID_STATE_KEY,
        video_transform=build_droid_video_transform(
            transform_name, size=args.image_size
        ),
        text_cache=cache,
        prompt_template=args.prompt_template,
        video_backend=args.video_backend,
    )
    collator = DroidPredictorCollator(
        image_size=args.image_size,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        num_frames=args.num_frames,
        num_views=len(DROID_VIDEO_KEYS),
        context_tubelets=args.context_tubelets,
    )
    return dataset, collator


def main(argv=None):
    args = parse_args(argv)
    seed_process(args.seed)
    device = make_device()
    model, _ = build_model(args, device)
    dataset, collator = build_dataset(args)
    if dataset.state_dim != args.proprio_dim:
        raise ValueError(
            f"Configured proprio-dim={args.proprio_dim}, DROID dataset has "
            f"state_dim={dataset.state_dim}"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size = args.batch_size * world_size
    anchor_count = len(dataset)
    optimizer_steps_per_epoch = anchor_count // global_batch_size
    dropped_samples = anchor_count % global_batch_size
    source_total_frames = int(dataset.info.get("total_frames", 0) or 0)
    logger.info(
        "DROID epoch contract: source_frames=%d complete_window_anchors=%d "
        "global_batch_size=%d optimizer_steps_per_epoch=%d dropped_samples=%d "
        "future_offset=%d source_fps=%d target_fps=%d",
        source_total_frames,
        anchor_count,
        global_batch_size,
        optimizer_steps_per_epoch,
        dropped_samples,
        DROID_MAX_FUTURE_OFFSET,
        DROID_FPS,
        DROID_TARGET_FPS,
    )
    model.policy_serving["config"].update(
        {
            "source_total_frames": source_total_frames,
            "source_total_episodes": int(dataset.info.get("total_episodes", dataset.num_episodes)),
            "complete_window_anchors": anchor_count,
            "max_future_offset": DROID_MAX_FUTURE_OFFSET,
            "global_batch_size": global_batch_size,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "dropped_samples_per_epoch": dropped_samples,
        }
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.encoder.parameters())
    logger.info(
        "predictor trainable parameters=%d; frozen encoder parameters=%d",
        trainable,
        frozen,
    )
    trainer = Trainer(
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
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "mixed_precision": args.mixed_precision,
            "log_every": args.log_every,
            "save_every": args.save_every,
            "wandb_project": args.wandb_project,
            "wandb_name": args.wandb_name,
            "prefetch_factor": args.prefetch_factor,
            "recycle_workers_every": args.recycle_workers_every,
            "resume_from": None,
            "train_seed": args.seed,
            "data_seed": args.seed,
            "training_stage": "droid_predictor_pretraining",
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
