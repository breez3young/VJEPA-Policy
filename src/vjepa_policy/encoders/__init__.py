"""Frozen latent encoder registry for the canonical VJEPA-Policy path."""

from __future__ import annotations

from collections.abc import Callable

from .base import EncoderSpec, LatentLayout, layout_from_encoder_output


EncoderFactory = Callable[..., object]
_REGISTRY: dict[str, EncoderFactory] = {}
_SPECS: dict[str, EncoderSpec] = {}


def register_encoder(name: str, spec: EncoderSpec, factory: EncoderFactory) -> None:
    if name != spec.name:
        raise ValueError(f"registry name {name!r} does not match spec name {spec.name!r}")
    if name in _REGISTRY:
        raise ValueError(f"encoder already registered: {name}")
    _REGISTRY[name] = factory
    _SPECS[name] = spec


def encoder_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def encoder_spec(name: str) -> EncoderSpec:
    try:
        return _SPECS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown encoder {name!r}; choose one of {encoder_names()}"
        ) from error


def encoder_specs() -> dict[str, EncoderSpec]:
    """Return a shallow copy of the serializable registry metadata."""
    return dict(_SPECS)


def build_encoder(name: str, **kwargs):
    try:
        factory = _REGISTRY[name]
    except KeyError as error:
        raise ValueError(
            f"unknown encoder {name!r}; choose one of {encoder_names()}"
        ) from error
    return factory(**kwargs)


def _register_builtin_encoders() -> None:
    # Import lazily so importing the package does not construct or load any
    # large model and optional DINO/Wan dependencies remain optional.
    from .vjepa2 import build_vjepa2_vitl, VJEPA2_VITL_SPEC
    from .vjepa2_1 import build_vjepa2_1_vitl, VJEPA2_1_VITL_SPEC
    from .dino import (
        DINOV2_VITL_ALIGNED14_SPEC,
        DINOV2_VITL_NATIVE16_SPEC,
        DINOV2_VITL_SPEC,
        DINOV3_VITL_NATIVE14_SPEC,
        DINOV3_VITL_SPEC,
        build_dinov2_vitl,
        build_dinov2_vitl_aligned14,
        build_dinov2_vitl_native16,
        build_dinov3_vitl,
        build_dinov3_vitl_native14,
    )
    from .wan import (
        WAN22_STRIDE2_SPEC,
        WAN22_STRIDE2_T5_SPEC,
        WAN22_VAE38_SPEC,
        WAN2_2_SPEC,
        build_wan2_2,
        build_wan22_stride2,
        build_wan22_stride2_t5,
        build_wan22_vae38,
    )

    register_encoder(VJEPA2_VITL_SPEC.name, VJEPA2_VITL_SPEC, build_vjepa2_vitl)
    register_encoder(VJEPA2_1_VITL_SPEC.name, VJEPA2_1_VITL_SPEC, build_vjepa2_1_vitl)
    register_encoder(
        DINOV2_VITL_ALIGNED14_SPEC.name,
        DINOV2_VITL_ALIGNED14_SPEC,
        build_dinov2_vitl_aligned14,
    )
    register_encoder(
        DINOV2_VITL_NATIVE16_SPEC.name,
        DINOV2_VITL_NATIVE16_SPEC,
        build_dinov2_vitl_native16,
    )
    register_encoder(
        DINOV3_VITL_NATIVE14_SPEC.name,
        DINOV3_VITL_NATIVE14_SPEC,
        build_dinov3_vitl_native14,
    )
    register_encoder(DINOV2_VITL_SPEC.name, DINOV2_VITL_SPEC, build_dinov2_vitl)
    register_encoder(DINOV3_VITL_SPEC.name, DINOV3_VITL_SPEC, build_dinov3_vitl)
    register_encoder(WAN22_VAE38_SPEC.name, WAN22_VAE38_SPEC, build_wan22_vae38)
    register_encoder(WAN2_2_SPEC.name, WAN2_2_SPEC, build_wan2_2)
    register_encoder(WAN22_STRIDE2_SPEC.name, WAN22_STRIDE2_SPEC, build_wan22_stride2)
    register_encoder(
        WAN22_STRIDE2_T5_SPEC.name,
        WAN22_STRIDE2_T5_SPEC,
        build_wan22_stride2_t5,
    )


_register_builtin_encoders()


__all__ = [
    "EncoderSpec",
    "LatentLayout",
    "build_encoder",
    "encoder_names",
    "encoder_spec",
    "encoder_specs",
    "layout_from_encoder_output",
    "register_encoder",
]
