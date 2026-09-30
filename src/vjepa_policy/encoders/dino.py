"""DINOv2/v3 factories for the frozen encoder registry."""

from __future__ import annotations

from dataclasses import replace

from vjepa_policy.experimental.latent_space.dino_encoder import (
    load_dino_latent_encoder,
)
from vjepa_policy.experimental.latent_space.dino_native16 import (
    load_dinov2_native16_latent_encoder,
)

from .base import EncoderSpec


DINOV2_VITL_ALIGNED14_SPEC = EncoderSpec(
    name="dinov2_vitl_aligned14",
    family="dinov2",
    model_name="dinov2_vitl_aligned14",
    checkpoint_key="local_pretrained",
    # Logical policy grid: 224 / 16 = 14.  The backbone itself is ViT-L/14.
    input_patch_size=16,
    source_patch_size=14,
    temporal_mode="sampled_frame",
    temporal_stride=2,
    temporal_offset=1,
    latent_spatial_grid=None,
    interpolate_rope=True,
    canonical_spatial_grid=(14, 14),
)

DINOV2_VITL_NATIVE16_SPEC = EncoderSpec(
    name="dinov2_vitl_native16",
    family="dinov2",
    model_name="dinov2_vitl_native16",
    checkpoint_key="local_pretrained",
    # Logical grid is 16x16, matching the native ViT-L/14 output at 224px.
    input_patch_size=14,
    source_patch_size=14,
    temporal_mode="sampled_frame",
    temporal_stride=2,
    temporal_offset=1,
    latent_spatial_grid=None,
    interpolate_rope=True,
    canonical_spatial_grid=(16, 16),
)

DINOV3_VITL_NATIVE14_SPEC = EncoderSpec(
    name="dinov3_vitl_native14",
    family="dinov3",
    model_name="dinov3_vitl_native14",
    checkpoint_key="local_pretrained",
    # DINOv3 ViT-L/16 at 224px natively emits a 14x14 patch grid.
    input_patch_size=16,
    source_patch_size=16,
    temporal_mode="sampled_frame",
    temporal_stride=2,
    temporal_offset=1,
    latent_spatial_grid=None,
    interpolate_rope=True,
    canonical_spatial_grid=(14, 14),
)


def _attach_spec(encoder, spec: EncoderSpec):
    encoder.spec = spec
    encoder._vjepa_policy_model_name = spec.model_name
    encoder._vjepa_policy_checkpoint_key = spec.checkpoint_key
    return encoder


def build_dinov2_vitl_aligned14(
    *,
    checkpoint: str,
    image_size: tuple[int, int],
    video_frames: int,
    checkpoint_key: str | None = None,
    image_microbatch_size: int | None = None,
    frame_stride: int = 2,
    frame_offset: int = 1,
    num_views: int | None = None,
    interpolate_rope: bool = True,
    **kwargs,
):
    del kwargs
    if checkpoint_key not in (None, "local_pretrained", "target_encoder"):
        raise ValueError(
            "DINO encoders load a Hugging Face directory directly; "
            f"checkpoint key {checkpoint_key!r} is unsupported"
        )
    image_size = tuple(image_size)
    if any(value % 16 for value in image_size):
        raise ValueError(f"DINOv2 aligned14 image size must be divisible by 16, got {image_size}")
    output_grid = tuple(value // 16 for value in image_size)
    encoder = load_dino_latent_encoder(
        "dinov2",
        checkpoint,
        image_microbatch_size=image_microbatch_size,
        output_grid=output_grid,
        interpolate_spatial=True,
        logical_patch_size=16,
        frame_stride=frame_stride,
        frame_offset=frame_offset,
        num_views=num_views,
        input_size=tuple(image_size),
        video_frames=video_frames,
    )
    runtime_spec = replace(
        DINOV2_VITL_ALIGNED14_SPEC,
        latent_spatial_grid=None,
        temporal_stride=frame_stride,
        temporal_offset=frame_offset,
        interpolate_rope=bool(interpolate_rope),
    )
    return _attach_spec(encoder, runtime_spec)


def build_dinov2_vitl_native16(
    *,
    checkpoint: str,
    image_size: tuple[int, int],
    video_frames: int,
    checkpoint_key: str | None = None,
    image_microbatch_size: int | None = None,
    frame_stride: int = 2,
    frame_offset: int = 1,
    num_views: int | None = None,
    interpolate_rope: bool = True,
    **kwargs,
):
    del kwargs
    if checkpoint_key not in (None, "local_pretrained", "target_encoder"):
        raise ValueError(
            "DINO encoders load a Hugging Face directory directly; "
            f"checkpoint key {checkpoint_key!r} is unsupported"
        )
    image_size = tuple(image_size)
    if any(value % 14 for value in image_size):
        raise ValueError(f"DINOv2 native16 image size must be divisible by 14, got {image_size}")
    encoder = load_dinov2_native16_latent_encoder(
        checkpoint,
        image_microbatch_size=image_microbatch_size,
        num_views=num_views,
        input_size=image_size,
        frame_stride=frame_stride,
        frame_offset=frame_offset,
        video_frames=video_frames,
    )
    # The native adapter currently exposes the historical frame sampling
    # (offset 1, stride 2); reject silent geometry drift until its loader is
    # extended with a different temporal contract.
    if (frame_stride, frame_offset) != (2, 1):
        raise ValueError("DINOv2 native16 requires frame_stride=2 and frame_offset=1")
    runtime_spec = replace(
        DINOV2_VITL_NATIVE16_SPEC,
        latent_spatial_grid=None,
        interpolate_rope=bool(interpolate_rope),
    )
    return _attach_spec(encoder, runtime_spec)


def build_dinov3_vitl_native14(
    *,
    checkpoint: str,
    image_size: tuple[int, int],
    video_frames: int,
    checkpoint_key: str | None = None,
    image_microbatch_size: int | None = None,
    frame_stride: int = 2,
    frame_offset: int = 1,
    num_views: int | None = None,
    interpolate_rope: bool = True,
    **kwargs,
):
    del kwargs
    if checkpoint_key not in (None, "local_pretrained", "target_encoder"):
        raise ValueError(
            "DINO encoders load a Hugging Face directory directly; "
            f"checkpoint key {checkpoint_key!r} is unsupported"
        )
    image_size = tuple(image_size)
    if any(value % 16 for value in image_size):
        raise ValueError(f"DINOv3 native14 image size must be divisible by 16, got {image_size}")
    output_grid = tuple(value // 16 for value in image_size)
    encoder = load_dino_latent_encoder(
        "dinov3",
        checkpoint,
        image_microbatch_size=image_microbatch_size,
        output_grid=output_grid,
        interpolate_spatial=False,
        logical_patch_size=16,
        frame_stride=frame_stride,
        frame_offset=frame_offset,
        num_views=num_views,
        input_size=tuple(image_size),
        video_frames=video_frames,
    )
    runtime_spec = replace(
        DINOV3_VITL_NATIVE14_SPEC,
        latent_spatial_grid=None,
        temporal_stride=frame_stride,
        temporal_offset=frame_offset,
        interpolate_rope=bool(interpolate_rope),
    )
    return _attach_spec(encoder, runtime_spec)


# Historical ablation names are aliases, not separate implementations.  Keep
# their metadata names stable while exposing the canonical geometry explicitly.
DINOV2_VITL_SPEC = replace(
    DINOV2_VITL_ALIGNED14_SPEC,
    name="dinov2_vitl",
    model_name="dinov2_vitl",
)
DINOV3_VITL_SPEC = replace(
    DINOV3_VITL_NATIVE14_SPEC,
    name="dinov3_vitl",
    model_name="dinov3_vitl",
)


def build_dinov2_vitl(**kwargs):
    encoder = build_dinov2_vitl_aligned14(**kwargs)
    return _attach_spec(
        encoder,
        replace(
            DINOV2_VITL_SPEC,
            temporal_stride=encoder.spec.temporal_stride,
            temporal_offset=encoder.spec.temporal_offset,
            interpolate_rope=encoder.spec.interpolate_rope,
        ),
    )


def build_dinov3_vitl(**kwargs):
    encoder = build_dinov3_vitl_native14(**kwargs)
    return _attach_spec(
        encoder,
        replace(
            DINOV3_VITL_SPEC,
            temporal_stride=encoder.spec.temporal_stride,
            temporal_offset=encoder.spec.temporal_offset,
            interpolate_rope=encoder.spec.interpolate_rope,
        ),
    )


__all__ = [
    "DINOV2_VITL_ALIGNED14_SPEC",
    "DINOV2_VITL_NATIVE16_SPEC",
    "DINOV2_VITL_SPEC",
    "DINOV3_VITL_NATIVE14_SPEC",
    "DINOV3_VITL_SPEC",
    "build_dinov2_vitl_aligned14",
    "build_dinov2_vitl_native16",
    "build_dinov2_vitl",
    "build_dinov3_vitl_native14",
    "build_dinov3_vitl",
]
