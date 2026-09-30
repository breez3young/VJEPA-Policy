"""Isolated builders for latent-space ablations against the V-JEPA2-L baseline."""

from __future__ import annotations

import torch

from vjepa_policy.experimental.latent_space.dino_encoder import (
    load_dino_latent_encoder,
)
from vjepa_policy.experimental.latent_space.wan22_encoder import (
    WAN22_LATENT_CHANNELS,
    build_wan22_policy_mask,
    load_wan22_latent_encoder,
)
from vjepa_policy.models import vision_transformer as video_vit
from vjepa_policy.models.wm_predictor import build_world_model_predictor


_BASELINE_ENCODER_DIM = 1024
_WAN_SHAPE_SPECIFIC_PREDICTOR_KEYS = {
    "predictor_embed.weight",
    "predictor_proj.weight",
    "predictor_proj.bias",
}


def _model_image_size(args):
    from vjepa_policy.train_common import get_model_image_size

    return get_model_image_size(args)


def _model_video_frames(args):
    from vjepa_policy.train_common import get_model_video_frames

    return get_model_video_frames(args)


def _camera_keys(args):
    from vjepa_policy.train_common import get_camera_keys

    return get_camera_keys(args)


def validate_latent_ablation_config(args) -> None:
    """Reject configuration drift beyond the requested latent encoder."""
    expected = {
        "encoder_family": "vjepa2",
        "model_name": "vit_large",
        "encoder_checkpoint_key": "target_encoder",
        "crop_size": 224,
        "patch_size": 16,
        "tubelet_size": 2,
        "context_tubelets": 1,
        "view_layout": "independent",
        "pred_depth": 24,
        "pred_embed_dim": 1024,
        "pred_num_heads": 16,
        "action_hidden_size": 512,
        "action_num_layers": 24,
        "condition_num_heads": 8,
        "predictor_rope_frequency_pairing": "corrected",
        "encoder_interpolate_rope": True,
    }
    mismatches = {
        name: (getattr(args, name, None), value)
        for name, value in expected.items()
        if getattr(args, name, None) != value
    }
    if _model_image_size(args) != (224, 224):
        mismatches["model_image_size"] = (_model_image_size(args), (224, 224))
    if _model_video_frames(args) != 10:
        mismatches["model_video_frames"] = (_model_video_frames(args), 10)
    camera_keys = _camera_keys(args)
    if len(camera_keys) != 2:
        mismatches["camera_count"] = (len(camera_keys), 2)
    if mismatches:
        raise ValueError(
            f"latent ablation must match the V-JEPA2-L 97.3% baseline: {mismatches}"
        )


def _consume_baseline_encoder_initialization(args) -> None:
    """Advance RNG exactly as the canonical V-JEPA2-L encoder construction does."""
    encoder = video_vit.vit_large(
        img_size=_model_image_size(args),
        patch_size=args.patch_size,
        num_frames=_model_video_frames(args),
        tubelet_size=args.tubelet_size,
        uniform_power=False,
        use_rope=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        interpolate_rope=args.encoder_interpolate_rope,
    )
    del encoder


def build_latent_encoder(args):
    """Build a frozen alternate encoder without perturbing downstream baseline RNG."""
    validate_latent_ablation_config(args)
    _consume_baseline_encoder_initialization(args)

    family = args.latent_encoder
    with torch.random.fork_rng(devices=[]):
        if family in ("dinov2", "dinov3"):
            return load_dino_latent_encoder(
                family,
                args.latent_checkpoint,
                image_microbatch_size=args.image_microbatch_size,
            )
        if family == "wan2_2":
            return load_wan22_latent_encoder(
                args.latent_checkpoint,
                fastwam_src_dir=args.fastwam_src_dir,
                video_microbatch_size=args.video_microbatch_size,
            )
    raise ValueError(f"unsupported latent encoder {family!r}")


def build_latent_predictor(*, latent_encoder: str, **kwargs):
    """Keep the baseline Predictor initialization except Wan's 48-D endpoints."""
    if latent_encoder in ("dinov2", "dinov3"):
        if kwargs.get("embed_dim") != _BASELINE_ENCODER_DIM:
            raise ValueError("DINO ViT-L must expose 1024-D latent tokens")
        return build_world_model_predictor(**kwargs)
    if latent_encoder != "wan2_2":
        raise ValueError(f"unsupported latent encoder {latent_encoder!r}")
    if kwargs.get("embed_dim") != WAN22_LATENT_CHANNELS:
        raise ValueError("Wan2.2 VAE38 must expose 48-D latent tokens")

    baseline_kwargs = dict(kwargs)
    baseline_kwargs["embed_dim"] = _BASELINE_ENCODER_DIM
    baseline = build_world_model_predictor(**baseline_kwargs)
    with torch.random.fork_rng(devices=[]):
        predictor = build_world_model_predictor(**kwargs)

    baseline_state = baseline.state_dict()
    predictor_state = predictor.state_dict()
    shape_specific = {
        name
        for name, value in baseline_state.items()
        if name not in predictor_state or value.shape != predictor_state[name].shape
    }
    if shape_specific != _WAN_SHAPE_SPECIFIC_PREDICTOR_KEYS:
        raise RuntimeError(
            f"unexpected Wan2.2 Predictor shape differences: {sorted(shape_specific)}"
        )
    compatible = {
        name: value
        for name, value in baseline_state.items()
        if name not in shape_specific
    }
    result = predictor.load_state_dict(compatible, strict=False)
    if set(result.missing_keys) != _WAN_SHAPE_SPECIFIC_PREDICTOR_KEYS:
        raise RuntimeError(
            f"unexpected missing Wan2.2 Predictor keys: {result.missing_keys}"
        )
    if result.unexpected_keys:
        raise RuntimeError(
            f"unexpected Wan2.2 Predictor keys: {result.unexpected_keys}"
        )
    return predictor


def build_latent_dataset_collator(args, action_chunk_size):
    from vjepa_policy.train_common import build_dataset_collator

    dataset, collator = build_dataset_collator(args, action_chunk_size)
    if args.latent_encoder == "wan2_2":
        mask = build_wan22_policy_mask(
            image_size=_model_image_size(args),
            num_views=len(_camera_keys(args)),
        )
        collator._ctx_idx = mask.ctx_idx
        collator._tgt_idx = mask.tgt_idx
        collator.n_ctx = mask.n_ctx
    return dataset, collator
