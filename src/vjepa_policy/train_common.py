# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Shared data, optimization, and runtime utilities for
# scripts/train_vjepa_policy.py.

import argparse
import os
import random

import numpy as np
import torch

from vjepa_policy.data import CausalPatchMask
from vjepa_policy.dataset_api import build_dataset
from vjepa_policy.datasets import (
    QuadrantViewCombiner,
    VideoClipTransform,
    build_patch_valid_mask,
)
from vjepa_policy.trainer import Trainer
from vjepa_policy.utils.logging import get_logger
from vjepa_policy.state import DEFAULT_MAX_STATE_DIM

logger = get_logger(__name__, force=True)

def _strip_prefix(state_dict, prefix="module.backbone."):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def add_common_args(p):
    """Args shared by both the world-model and the policy training scripts."""
    # data
    p.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=None,
        help="LeRobotDataset roots; omit when using --dataset-factory",
    )
    p.add_argument(
        "--dataset-factory",
        default=None,
        help=(
            "Optional module:function adapter. It returns a Dataset or "
            "(Dataset, collator) using the public sample contract."
        ),
    )
    p.add_argument("--text-cache-dir", required=True, help="Dir with precomputed T5 embeddings")
    p.add_argument("--camera-key", default=None,
                   help="Explicit single-view LeRobot feature; omitted discovers all views")
    p.add_argument("--views", nargs="+", default=None,
                   help="Explicit LeRobot video features arranged by --view-layout "
                        "(e.g. observation.images.image observation.images.wrist_image).")
    p.add_argument(
        "--view-layout",
        default="independent",
        choices=["quadrant", "horizontal", "independent"],
        help="How explicit --views are represented; independent preserves a view dimension.",
    )
    p.add_argument("--dataset-stats", default=None,
                   help="Shared dataset_stats.json; omitted means aggregate LeRobot metadata stats.")
    p.add_argument(
        "--action-indices", nargs="+", type=int, default=None,
        help="Optional ordered subset of raw LeRobot action dimensions.",
    )
    p.add_argument(
        "--state-indices", nargs="+", type=int, default=None,
        help="Optional ordered subset of raw LeRobot state dimensions.",
    )
    p.add_argument(
        "--relative-action-indices", nargs="+", type=int, default=None,
        help="Projected action dimensions represented relative to the current state.",
    )
    p.add_argument(
        "--instruction-field", choices=("task", "remarks"), default="task",
        help="Read language from task metadata or per-episode remarks.",
    )
    p.add_argument("--output-dir", default="./runs/vjepa_policy")
    # clip / model
    p.add_argument(
        "--checkpoint",
        "--encoder-checkpoint",
        dest="checkpoint",
        required=True,
        help="Pretrained checkpoint used to initialize the frozen encoder",
    )
    p.add_argument(
        "--encoder",
        default=None,
        help="Stable encoder registry id (canonical default: vjepa2_vitl)",
    )
    # Retained for Route 2 and historical launcher compatibility.  Canonical
    # launchers should use --encoder so invalid family/model/key combinations
    # cannot be assembled accidentally.
    p.add_argument("--model-name", default="vit_large")
    p.add_argument(
        "--encoder-family",
        choices=("vjepa2", "vjepa2_1"),
        default="vjepa2",
    )
    p.add_argument(
        "--encoder-checkpoint-key",
        default="target_encoder",
        help="State-dict entry in --checkpoint (for example encoder, target_encoder, ema_encoder)",
    )
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--num-frames", type=int, default=33,
                   help="Contiguous current/future steps; actions use [0, num_frames - 1).")
    p.add_argument("--past-frames", type=int, default=4,
                   help="Number of native dataset steps sampled before the current frame.")
    p.add_argument("--video-frame-stride", type=int, default=4,
                   help="Stride used to downsample range(-past_frames, num_frames).")
    p.add_argument("--video-backend", default=None, choices=["torchcodec", "pyav"])
    p.add_argument(
        "--require-full-future-window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude episode-tail samples that cannot provide every positive frame offset.",
    )
    p.add_argument("--action-normalization", default="MIN_MAX",
                   choices=["MIN_MAX", "MEAN_STD", "QUANTILE", "IDENTITY"])
    p.add_argument("--state-normalization", default="MIN_MAX",
                   choices=["MIN_MAX", "MEAN_STD", "QUANTILE", "IDENTITY"])
    p.add_argument(
        "--clip-normalized", action=argparse.BooleanOptionalAction, default=False,
        help="Clip normalized action and state values to [-1, 1].",
    )
    p.add_argument("--context-tubelets", type=int, default=1,
                   help="Number of leading tubelets used as context; the rest are the future target")
    p.add_argument("--pred-depth", type=int, default=12)
    p.add_argument("--pred-embed-dim", type=int, default=384)
    p.add_argument("--pred-num-heads", type=int, default=12)
    p.add_argument("--num-mask-tokens", type=int, default=10,
                   help="Keep all pretrained mask tokens; forward uses mask_index=1 (zero-shot V-JEPA2).")
    p.add_argument("--use-activation-checkpointing", action="store_true")
    p.add_argument("--lang-dim", type=int, default=4096)
    p.add_argument("--context-len", type=int, default=32, help="Cached text sequence length")
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--proprio-dim", type=int, default=8)
    p.add_argument(
        "--max-state-dim",
        type=int,
        default=DEFAULT_MAX_STATE_DIM,
        help=(
            "Fixed native-state width for the zero-padded state + validity-mask "
            "conditioning contract. Pass 0 to use the legacy variable-width path."
        ),
    )
    p.add_argument(
        "--proprio-encoding",
        choices=("identity", "sincos"),
        default="identity",
        help="Encode selected raw state before conditioning both policy experts.",
    )
    # optimization
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="DataLoader prefetch per worker (persistent workers, so a larger lead is fine).")
    p.add_argument("--recycle-workers-every", type=int, default=0,
                   help="Recreate the iterator every N optimizer steps to reclaim torchcodec worker "
                        "memory (0 only recycles at epoch boundaries).")
    p.add_argument("--resume", default=None,
                   help="Resume from a {predictor, action_expert, step} checkpoint: reloads weights, "
                        "jumps global_step forward, and fast-forwards the LR schedule.")
    p.add_argument(
        "--predictor-init",
        default=None,
        help=(
            "Initialize only the predictor from a DROID checkpoint. Unlike "
            "--resume, this does not restore step, scheduler, or Action Expert."
        ),
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=7,
                   help="Base training/data seed; process stochasticity uses seed + global rank.")
    p.add_argument("--loss-exp", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--wandb-project", default="vjepa_policy")
    p.add_argument("--wandb-name", default=None)
    return p


def seed_process(seed):
    """Seed model/runtime randomness per rank; data shuffling keeps the base seed."""
    rank = int(os.environ.get("RANK", "0"))
    rank_seed = int(seed) + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    logger.info(
        "[seed] train_seed=%d rank=%d rank_seed=%d data_seed=%d",
        seed,
        rank,
        rank_seed,
        seed,
    )
    return rank_seed


def make_device():
    """Under `accelerate launch` every process sees all GPUs; bind to its own by LOCAL_RANK."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def get_model_video_frames(args):
    from vjepa_policy.datasets.lerobot_video import make_video_offsets

    return len(
        make_video_offsets(
            num_frames=args.num_frames,
            past_frames=args.past_frames,
            video_stride=args.video_frame_stride,
        )
    )


def get_camera_keys(args):
    if not args.dataset_dirs and not args.dataset_factory and not args.views and not args.camera_key:
        raise ValueError("Provide --dataset-dirs, --dataset-factory, --views, or --camera-key")
    if args.dataset_factory and not args.views and not args.camera_key:
        raise ValueError("Custom dataset factories must declare --views or --camera-key")
    if args.views:
        keys = args.views
    elif args.camera_key:
        keys = [args.camera_key]
    else:
        from vjepa_policy.datasets.lerobot_video import discover_video_keys

        keys = discover_video_keys(args.dataset_dirs)
    view_layout = getattr(args, "view_layout", "independent")
    if view_layout == "quadrant" and len(keys) > 4:
        raise ValueError(f"The quadrant layout supports at most 4 views, got {len(keys)}")
    if view_layout == "horizontal" and len(keys) < 2:
        raise ValueError(f"The horizontal layout requires at least 2 views, got {len(keys)}")
    return list(keys)


def get_model_image_size(args):
    view_layout = getattr(args, "view_layout", "independent")
    if view_layout == "independent":
        if args.crop_size % args.patch_size:
            raise ValueError("crop_size must be divisible by patch_size for independent views")
        return args.crop_size, args.crop_size
    if args.views or not args.camera_key:
        camera_keys = get_camera_keys(args)
        if view_layout == "horizontal":
            if args.crop_size % args.patch_size:
                raise ValueError("crop_size must be divisible by patch_size for horizontal views")
            return args.crop_size, args.crop_size * len(camera_keys)
        if args.crop_size % (2 * args.patch_size):
            raise ValueError("crop_size must be divisible by 2 * patch_size for quadrant views")
    return args.crop_size, args.crop_size


class VJEPABatchCollator:
    """Stack FastWAM dictionary samples and add V-JEPA causal patch masks."""

    def __init__(self, image_size=None, patch_size=None, tubelet_size=None, video_frames=None,
                 context_tubelets=1, include_action=False, num_views=1, layout=None):
        mask = (
            CausalPatchMask.from_layout(layout)
            if layout is not None
            else CausalPatchMask(
                image_size,
                patch_size,
                tubelet_size,
                video_frames,
                context_tubelets,
                num_views=num_views,
            )
        )
        self._ctx_idx = mask.ctx_idx
        self._tgt_idx = mask.tgt_idx
        self.n_ctx = mask.n_ctx
        self.layout = layout
        self.num_views = mask.num_views
        self.include_action = include_action

    def __call__(self, batch):
        clips = torch.stack([sample["video"] for sample in batch])
        lang = torch.stack([sample["context"] for sample in batch])
        lang_mask = torch.stack([sample["context_mask"] for sample in batch])
        batch_size = len(batch)
        masks_enc = [self._ctx_idx.unsqueeze(0).expand(batch_size, -1).clone()]
        masks_pred = [self._tgt_idx.unsqueeze(0).expand(batch_size, -1).clone()]
        if not self.include_action:
            return clips, lang, lang_mask, masks_enc, masks_pred
        action = torch.stack([sample["action"] for sample in batch])
        action_pad = torch.stack([sample["action_is_pad"] for sample in batch])
        proprio = torch.stack([sample["proprio"] for sample in batch])
        return clips, lang, lang_mask, masks_enc, masks_pred, action, action_pad, proprio


def build_dataset_collator(args, action_chunk_size, layout=None):
    """Build the official-LeRobot-backed clip/action dataset and mask collator."""
    if args.dataset_factory:
        dataset, custom_collator = build_dataset(
            args.dataset_factory, args, include_action=action_chunk_size is not None
        )
        if custom_collator is not None:
            return dataset, custom_collator
        if layout is None:
            raise ValueError("A custom dataset factory requires an encoder layout")
        from vjepa_policy.dataset_api import PredictorBatchCollator

        return dataset, PredictorBatchCollator(
            layout, include_action=action_chunk_size is not None
        )
    if not args.dataset_dirs:
        raise ValueError("--dataset-dirs is required for the built-in LeRobot adapter")
    from vjepa_policy.datasets.lerobot_video import LeRobotVideoDataset
    camera_keys = get_camera_keys(args)
    image_size = get_model_image_size(args)
    expected_action_chunk_size = args.num_frames - 1
    if action_chunk_size is not None and action_chunk_size != expected_action_chunk_size:
        raise ValueError(
            f"action_chunk_size must equal num_frames - 1 ({expected_action_chunk_size}), "
            f"got {action_chunk_size}"
        )
    view_layout = getattr(args, "view_layout", "independent")
    explicit_quadrant = view_layout == "quadrant" and (
        bool(args.views) or (not args.camera_key and len(camera_keys) > 1)
    )
    independent_views = view_layout == "independent"
    video_transform = VideoClipTransform(
        (args.crop_size // 2, args.crop_size // 2)
        if explicit_quadrant
        else (args.crop_size, args.crop_size)
    )
    if explicit_quadrant:
        view_combiner = QuadrantViewCombiner(
            image_size, fill_value=video_transform.normalized_black
        )
    elif independent_views:
        view_combiner = "independent"
    else:
        view_combiner = "horizontal"
    dataset = LeRobotVideoDataset(
        dataset_dirs=args.dataset_dirs,
        video_keys=camera_keys,
        num_frames=args.num_frames,
        past_frames=args.past_frames,
        video_stride=args.video_frame_stride,
        action_indices=args.action_indices,
        state_indices=args.state_indices,
        relative_action_indices=args.relative_action_indices,
        video_transform=video_transform,
        view_combiner=view_combiner,
        text_cache_dir=args.text_cache_dir,
        context_len=args.context_len,
        instruction_field=args.instruction_field,
        action_normalization=args.action_normalization,
        state_normalization=args.state_normalization,
        clip_normalized=args.clip_normalized,
        normalization_stats_path=args.dataset_stats,
        stats_output_path=os.path.join(args.output_dir, "dataset_stats.json"),
        video_backend=args.video_backend,
        require_full_future_window=args.require_full_future_window,
    )
    if dataset.action_dim != args.action_dim:
        raise ValueError(f"Configured action_dim={args.action_dim}, dataset has {dataset.action_dim}")
    if dataset.state_dim != args.proprio_dim:
        raise ValueError(f"Configured proprio_dim={args.proprio_dim}, dataset has {dataset.state_dim}")
    if args.max_state_dim:
        if args.proprio_encoding != "identity":
            raise ValueError(
                "--max-state-dim requires --proprio-encoding identity; "
                "pass --max-state-dim 0 for legacy variable-width conditioning"
            )
        if args.proprio_dim > args.max_state_dim:
            raise ValueError(
                f"proprio_dim={args.proprio_dim} exceeds max_state_dim={args.max_state_dim}"
            )
    normalizer_metadata = getattr(dataset.normalizer, "metadata", {})
    proprio_metadata = (
        normalizer_metadata.get("proprio_encoding")
        if isinstance(normalizer_metadata, dict)
        else None
    )
    if proprio_metadata is not None:
        encoded_dim = args.proprio_dim * (2 if args.proprio_encoding == "sincos" else 1)
        expected_metadata = {
            "type": args.proprio_encoding,
            "raw_dim": args.proprio_dim,
            "encoded_dim": encoded_dim,
        }
        if proprio_metadata != expected_metadata:
            raise ValueError(
                "Proprio encoding metadata does not match training configuration: "
                f"expected {expected_metadata}, got {proprio_metadata}"
            )
    collator = VJEPABatchCollator(
        image_size=image_size if layout is None else None,
        patch_size=args.patch_size if layout is None else None,
        tubelet_size=args.tubelet_size if layout is None else None,
        video_frames=get_model_video_frames(args) if layout is None else None,
        context_tubelets=args.context_tubelets,
        include_action=action_chunk_size is not None,
        num_views=len(camera_keys) if independent_views else 1,
        layout=layout,
    )
    return dataset, collator


def build_patch_valid(args, layout=None):
    """Return the static valid-token mask for an explicit quadrant layout."""
    if getattr(args, "view_layout", "independent") != "quadrant":
        return None
    camera_keys = get_camera_keys(args)
    if len(camera_keys) == 1 and not args.views:
        return None
    if layout is not None:
        if layout.num_views != 1:
            raise ValueError("quadrant layout must expose one composite latent view")
        if layout.grid_height % 2 or layout.grid_width % 2:
            raise ValueError("quadrant layout requires an even latent spatial grid")
        rows = torch.arange(layout.grid_height).unsqueeze(1)
        columns = torch.arange(layout.grid_width).unsqueeze(0)
        quadrants = (
            (rows // (layout.grid_height // 2)) * 2
            + columns // (layout.grid_width // 2)
        )
        return (quadrants < len(camera_keys)).flatten().repeat(layout.grid_depth)
    return build_patch_valid_mask(
        crop_size=args.crop_size,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        video_frames=get_model_video_frames(args),
        num_views=len(camera_keys),
    )


def run_training(args, model, dataset, collator):
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
            "resume_from": args.resume,
            "train_seed": args.seed,
            "data_seed": args.seed,
        },
    )
    trainer.train()
