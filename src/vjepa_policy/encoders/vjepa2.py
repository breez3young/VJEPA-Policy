"""V-JEPA2 ViT-L adapter used by the canonical training entrypoint."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from vjepa_policy.models import vision_transformer

from .base import EncoderSpec


VJEPA2_VITL_SPEC = EncoderSpec(
    name="vjepa2_vitl",
    family="vjepa2",
    model_name="vit_large",
    checkpoint_key="target_encoder",
    input_patch_size=16,
    source_patch_size=16,
    temporal_mode="tubelet",
    temporal_stride=2,
    interpolate_rope=True,
    canonical_spatial_grid=(16, 16),
)


def _strip_prefix(state_dict, prefix="module.backbone."):
    stripped = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    return stripped or dict(state_dict)


class VJEPA2EncoderAdapter(nn.Module):
    """Thin metadata wrapper; token values and checkpoint keys stay unchanged."""

    def __init__(self, encoder: nn.Module, spec: EncoderSpec):
        super().__init__()
        self.encoder = encoder
        self.spec = spec
        self.embed_dim = encoder.embed_dim
        self.patch_size = encoder.patch_size
        self.tubelet_size = encoder.tubelet_size
        self._vjepa_policy_training_false = getattr(
            encoder, "_vjepa_policy_training_false", False
        )
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
        if self._vjepa_policy_training_false:
            return self.encoder(clips, masks, *args, training=False, **kwargs)
        return self.encoder(clips, masks, *args, **kwargs)


def build_vjepa2_vitl(
    *,
    checkpoint: str,
    image_size: tuple[int, int],
    video_frames: int,
    tubelet_size: int = 2,
    interpolate_rope: bool = True,
    checkpoint_key: str | None = None,
    **kwargs,
):
    del kwargs
    if tubelet_size != VJEPA2_VITL_SPEC.temporal_stride:
        raise ValueError(
            "vjepa2_vitl currently requires tubelet_size=2 so its pretrained "
            "temporal contract remains valid"
        )
    encoder = vision_transformer.vit_large(
        img_size=image_size,
        patch_size=VJEPA2_VITL_SPEC.input_patch_size,
        num_frames=video_frames,
        tubelet_size=tubelet_size,
        uniform_power=False,
        use_rope=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        interpolate_rope=interpolate_rope,
        canonical_spatial_grid=VJEPA2_VITL_SPEC.canonical_spatial_grid,
    )
    checkpoint_key = checkpoint_key or VJEPA2_VITL_SPEC.checkpoint_key
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(payload, dict) and checkpoint_key in payload:
        state_dict = payload[checkpoint_key]
    elif isinstance(payload, dict) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        # Accept a directly exported encoder state dict as well as the
        # official nested ``target_encoder`` checkpoint format.
        state_dict = payload
    else:
        raise KeyError(
            f"checkpoint has no {checkpoint_key!r}; available keys: {sorted(payload)}"
        )
    encoder.load_state_dict(_strip_prefix(state_dict), strict=True)
    del payload
    runtime_spec = replace(
        VJEPA2_VITL_SPEC,
        checkpoint_key=checkpoint_key,
        interpolate_rope=bool(interpolate_rope),
    )
    adapter = VJEPA2EncoderAdapter(encoder, runtime_spec)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter
