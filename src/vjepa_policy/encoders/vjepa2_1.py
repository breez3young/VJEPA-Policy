"""V-JEPA2.1 ViT-L adapter for the encoder registry.

The upstream V-JEPA2.1 implementation has a slightly different forward
signature (the frozen target encoder is called with ``training=False``).  The
registry wrapper keeps that detail out of the canonical training entrypoint
while exposing the same geometry metadata as the V-JEPA2 adapter.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from vjepa_policy.models.vjepa2_1 import vision_transformer

from .base import EncoderSpec


VJEPA2_1_VITL_SPEC = EncoderSpec(
    name="vjepa2_1_vitl",
    family="vjepa2_1",
    model_name="vit_large",
    checkpoint_key="ema_encoder",
    input_patch_size=16,
    source_patch_size=16,
    temporal_mode="tubelet",
    temporal_stride=2,
    interpolate_rope=True,
    canonical_spatial_grid=(16, 16),
)


def _strip_checkpoint_prefix(state_dict, prefix="module.backbone."):
    stripped = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    # Some locally exported checkpoints already contain bare encoder keys.
    return stripped or dict(state_dict)


class VJEPA2_1EncoderAdapter(nn.Module):
    """Frozen V-JEPA2.1 target encoder with registry metadata."""

    def __init__(self, encoder: nn.Module, spec: EncoderSpec):
        super().__init__()
        self.encoder = encoder
        self.spec = spec
        self.embed_dim = encoder.embed_dim
        self.patch_size = encoder.patch_size
        self.tubelet_size = encoder.tubelet_size
        self._vjepa_policy_training_false = True
        self._vjepa_policy_family = spec.family
        self._vjepa_policy_model_name = spec.model_name
        self._vjepa_policy_checkpoint_key = spec.checkpoint_key

    @property
    def blocks(self):
        return self.encoder.blocks

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, clips, masks=None, *args, **kwargs):
        # ``encode_video_views`` may already pass ``training=False`` based on
        # the adapter flag; consume it here to avoid duplicate keyword errors.
        kwargs.pop("training", None)
        return self.encoder(clips, masks, *args, training=False, **kwargs)


def build_vjepa2_1_vitl(
    *,
    checkpoint: str,
    image_size: tuple[int, int],
    video_frames: int,
    tubelet_size: int = 2,
    interpolate_rope: bool = True,
    checkpoint_key: str | None = None,
    model_name: str = "vit_large",
    **kwargs,
):
    if model_name != VJEPA2_1_VITL_SPEC.model_name:
        raise ValueError(
            f"{VJEPA2_1_VITL_SPEC.name} only supports model_name='vit_large', "
            f"got {model_name!r}"
        )
    if tubelet_size != VJEPA2_1_VITL_SPEC.temporal_stride:
        raise ValueError(
            "vjepa2_1_vitl currently requires tubelet_size=2 so its pretrained "
            "temporal contract remains valid"
        )
    encoder = vision_transformer.vit_large(
        img_size=image_size,
        patch_size=VJEPA2_1_VITL_SPEC.source_patch_size or 16,
        num_frames=video_frames,
        tubelet_size=tubelet_size,
        uniform_power=False,
        use_rope=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        interpolate_rope=interpolate_rope,
        img_temporal_dim_size=1,
        modality_embedding=True,
        n_output_distillation=1,
    )
    key = checkpoint_key or VJEPA2_1_VITL_SPEC.checkpoint_key
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(payload, dict) and key in payload:
        state_dict = payload[key]
    elif (
        checkpoint_key in (None, "target_encoder")
        and isinstance(payload, dict)
        and VJEPA2_1_VITL_SPEC.checkpoint_key in payload
    ):
        key = VJEPA2_1_VITL_SPEC.checkpoint_key
        state_dict = payload[key]
    elif isinstance(payload, dict) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        state_dict = payload
        key = "direct"
    else:
        raise KeyError(
            f"checkpoint has no {key!r}; available keys: {sorted(payload)}"
        )
    encoder.load_state_dict(_strip_checkpoint_prefix(state_dict), strict=True)
    del payload
    runtime_spec = replace(
        VJEPA2_1_VITL_SPEC,
        checkpoint_key=key,
        interpolate_rope=bool(interpolate_rope),
    )
    adapter = VJEPA2_1EncoderAdapter(encoder, runtime_spec)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter


__all__ = [
    "VJEPA2_1_VITL_SPEC",
    "VJEPA2_1EncoderAdapter",
    "build_vjepa2_1_vitl",
]
