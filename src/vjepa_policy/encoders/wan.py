"""Wan2.2 VAE factories for compact latent-token experiments."""

from __future__ import annotations

from dataclasses import replace

from vjepa_policy.experimental.latent_space.wan22_encoder import (
    load_wan22_compact_latent_encoder,
)
from vjepa_policy.experimental.latent_space.wan22_stride2_encoder import (
    load_wan22_stride2_compact_latent_encoder,
)

from .base import EncoderSpec


WAN22_VAE38_SPEC = EncoderSpec(
    name="wan22_vae38",
    family="wan2_2",
    model_name="vae38_compact",
    checkpoint_key="encoder_quant_conv",
    input_patch_size=16,
    source_patch_size=16,
    temporal_mode="causal_stride",
    temporal_stride=4,
    temporal_offset=0,
    latent_spatial_grid=(14, 14),
    interpolate_rope=True,
    canonical_spatial_grid=(14, 14),
)

WAN22_STRIDE2_SPEC = EncoderSpec(
    name="wan22_stride2",
    family="wan2_2_stride2",
    model_name="vae38_compact",
    checkpoint_key="encoder_quant_conv",
    input_patch_size=16,
    source_patch_size=16,
    temporal_mode="causal_stride",
    temporal_stride=4,
    temporal_offset=0,
    latent_spatial_grid=(14, 14),
    interpolate_rope=True,
    canonical_spatial_grid=(14, 14),
)


def _validate_checkpoint_key(checkpoint_key: str | None) -> None:
    # Wan weights are a standalone safetensors file, not a torch payload with
    # named entries.  Accept the common launcher defaults as no-op aliases.
    if checkpoint_key not in (
        None,
        "encoder_quant_conv",
        "target_encoder",
        "local_pretrained",
    ):
        raise ValueError(
            "Wan2.2 checkpoints are standalone safetensors files; unsupported "
            f"checkpoint key {checkpoint_key!r}"
        )


def _attach_spec(encoder, spec: EncoderSpec):
    encoder.spec = spec
    encoder._vjepa_policy_family = spec.family
    encoder._vjepa_policy_model_name = spec.model_name
    encoder._vjepa_policy_checkpoint_key = spec.checkpoint_key
    return encoder


def build_wan22_vae38(
    *,
    checkpoint: str,
    image_size: tuple[int, int],
    video_frames: int,
    checkpoint_key: str | None = None,
    fastwam_src_dir: str | None = None,
    video_microbatch_size: int | None = None,
    num_views: int | None = None,
    interpolate_rope: bool = True,
    **kwargs,
):
    del kwargs
    _validate_checkpoint_key(checkpoint_key)
    if tuple(image_size) != (224, 224):
        raise ValueError(f"Wan2.2 VAE38 expects 224x224 views, got {tuple(image_size)}")
    load_kwargs = {"video_microbatch_size": video_microbatch_size, "num_views": num_views}
    if fastwam_src_dir is not None:
        load_kwargs["fastwam_src_dir"] = fastwam_src_dir
    encoder = load_wan22_compact_latent_encoder(checkpoint, **load_kwargs)
    return _attach_spec(
        encoder,
        replace(WAN22_VAE38_SPEC, interpolate_rope=bool(interpolate_rope)),
    )


def build_wan22_stride2(
    *,
    checkpoint: str,
    image_size: tuple[int, int],
    video_frames: int,
    checkpoint_key: str | None = None,
    fastwam_src_dir: str | None = None,
    video_microbatch_size: int | None = None,
    num_views: int | None = None,
    interpolate_rope: bool = True,
    **kwargs,
):
    del kwargs
    _validate_checkpoint_key(checkpoint_key)
    if tuple(image_size) != (224, 224):
        raise ValueError(
            f"Wan2.2 stride-2 VAE38 expects 224x224 views, got {tuple(image_size)}"
        )
    load_kwargs = {"video_microbatch_size": video_microbatch_size, "num_views": num_views}
    if fastwam_src_dir is not None:
        load_kwargs["fastwam_src_dir"] = fastwam_src_dir
    encoder = load_wan22_stride2_compact_latent_encoder(checkpoint, **load_kwargs)
    return _attach_spec(
        encoder,
        replace(WAN22_STRIDE2_SPEC, interpolate_rope=bool(interpolate_rope)),
    )


WAN2_2_SPEC = replace(WAN22_VAE38_SPEC, name="wan2_2")
WAN22_STRIDE2_T5_SPEC = replace(WAN22_STRIDE2_SPEC, name="wan22_stride2_t5")


def build_wan2_2(**kwargs):
    encoder = build_wan22_vae38(**kwargs)
    return _attach_spec(
        encoder,
        replace(
            WAN2_2_SPEC,
            interpolate_rope=encoder.spec.interpolate_rope,
        ),
    )


def build_wan22_stride2_t5(**kwargs):
    # The old launcher name now resolves to the compact implementation in the
    # registry.  The standalone geometry module still exposes its historical
    # fixed-T=5 class for reproducibility of prior runs.
    encoder = build_wan22_stride2(**kwargs)
    return _attach_spec(
        encoder,
        replace(
            WAN22_STRIDE2_T5_SPEC,
            interpolate_rope=encoder.spec.interpolate_rope,
        ),
    )


__all__ = [
    "WAN22_VAE38_SPEC",
    "WAN22_STRIDE2_SPEC",
    "WAN2_2_SPEC",
    "WAN22_STRIDE2_T5_SPEC",
    "build_wan22_vae38",
    "build_wan2_2",
    "build_wan22_stride2",
    "build_wan22_stride2_t5",
]
