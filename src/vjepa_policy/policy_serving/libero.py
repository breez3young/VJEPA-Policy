"""LIBERO inference adapter for the published V-JEPA policy."""

import hashlib
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from vjepa_policy.data import CausalPatchMask
from vjepa_policy.datasets import (
    DEFAULT_PROMPT,
    ActionStateNormalizer,
    QuadrantViewCombiner,
    VideoClipTransform,
    combine_video_clips,
)
from vjepa_policy.models.wm_predictor import (
    build_world_model_predictor,
)
from vjepa_policy.models.vjepa_policy import (
    build_vjepa_policy,
    encode_video_views,
    encoded_proprio_dim,
)
from vjepa_policy.state import DEFAULT_MAX_STATE_DIM, packed_state_dim, prepare_proprio_state
from vjepa_policy.models import vision_transformer as video_vit
from vjepa_policy.models.vjepa2_1 import vision_transformer as video_vit_2_1


_TOPOLOGY_CLASS_ALIASES = {
    "FreshLargeCrossAttentionPredictor": "WorldModelPredictor",
    "LargePredictorActionCrossExpert": "ActionExpert",
    "CleanContextLargeCrossAttentionPredictor": "CleanContextWorldModelPredictor",
}

# These are the only historical family/model/checkpoint combinations for
# which a registry id cannot add any information: the original serving path
# constructed the official V-JEPA backbone directly.  An unknown id attached
# to any other combination is potentially a different latent geometry and
# must fail loudly instead of being silently reinterpreted.
_LEGACY_RAW_ENCODER_COMBINATIONS = {
    ("vjepa2", "vit_giant_xformers", "encoder"),
    ("vjepa2", "vit_giant", "encoder"),
    # V-JEPA2.1 ViT-G was supported by the historical launcher before it had
    # a registry adapter.  Its checkpoints use the target_encoder entry.
    ("vjepa2_1", "vit_giant_xformers", "target_encoder"),
}
_LEGACY_DEFAULT_CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)

logger = logging.getLogger(__name__)


def _canonicalize_topology_value(value):
    if isinstance(value, str):
        return _TOPOLOGY_CLASS_ALIASES.get(value, value)
    return value


def _canonicalize_topology_field(name, value):
    value = _canonicalize_topology_value(value)
    if name in {"camera_keys", "views"}:
        normalized = _normalize_camera_keys(value)
        return normalized
    if name == "latent_grid":
        return _normalize_int_sequence(value, name)
    if name in {"latent_layout", "encoder_spec"}:
        return _normalize_metadata(value)
    return value


def _normalize_int_sequence(value, name):
    if value is None:
        return None
    try:
        values = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer sequence") from error
    return values


def _normalize_metadata(value):
    """Make serialized list/tuple metadata comparable across torch saves."""
    if isinstance(value, dict):
        return {
            key: _normalize_metadata(_canonicalize_topology_value(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_metadata(item) for item in value)
    return value


@dataclass(frozen=True)
class PolicyServingConfig:
    precision: str = "bf16"
    # Stable registry id.  ``None`` keeps compatibility with historical
    # family/model based checkpoints and lets metadata select the adapter.
    encoder: str | None = None
    encoder_family: str = "vjepa2"
    encoder_checkpoint_key: str = "target_encoder"
    model_name: str = "vit_large"
    crop_size: int = 224
    image_resize_mode: str = "stretch"
    patch_size: int = 16
    num_frames: int = 10
    tubelet_size: int = 2
    context_tubelets: int = 1
    pred_depth: int = 24
    pred_embed_dim: int = 1024
    pred_num_heads: int = 16
    num_mask_tokens: int = 10
    lang_dim: int = 4096
    t5_len: int = 32
    action_dim: int = 7
    proprio_dim: int = 8
    max_state_dim: int | None = DEFAULT_MAX_STATE_DIM
    proprio_encoding: str = "identity"
    action_chunk_size: int = 32
    action_hidden_size: int = 512
    action_num_layers: int = 24
    action_num_inference_steps: int = 10
    condition_num_heads: int = 8
    corrected_predictor_rope: bool = True
    encoder_interpolate_rope: bool = True
    predictor_max_views: int | None = None
    view_layout: str = "independent"
    camera_keys: tuple[str, ...] | None = None
    dataset_stats: str | None = None

    @classmethod
    def from_checkpoint(cls, ckpt_path, overrides=None):
        """Recover saved serving settings, then apply explicit overrides."""
        checkpoint = torch.load(
            ckpt_path, map_location="cpu", weights_only=False, mmap=True
        )
        topology = checkpoint.get("model_topology") or {}
        saved = checkpoint.get("policy_serving") or {}
        values = {}

        train_precision = checkpoint.get("train_precision")
        if train_precision in ("bf16", "fp16"):
            values["precision"] = train_precision
        elif train_precision in ("no", "fp32"):
            values["precision"] = "fp32"

        topology_fields = {
            "encoder": "encoder",
            "encoder_adapter": "encoder",
            "encoder_family": "encoder_family",
            "encoder_model_name": "model_name",
            "encoder_checkpoint_key": "encoder_checkpoint_key",
            "predictor_image_height": "crop_size",
            "image_resize_mode": "image_resize_mode",
            "predictor_depth": "pred_depth",
            "predictor_embed_dim": "pred_embed_dim",
            "predictor_num_heads": "pred_num_heads",
            "predictor_max_views": "predictor_max_views",
            "view_layout": "view_layout",
            "camera_keys": "camera_keys",
            "action_dim": "action_dim",
            "raw_proprio_dim": "proprio_dim",
            "max_state_dim": "max_state_dim",
            "proprio_encoding": "proprio_encoding",
            "action_chunk_size": "action_chunk_size",
            "action_hidden_size": "action_hidden_size",
            "action_num_layers": "action_num_layers",
            "encoder_interpolate_rope": "encoder_interpolate_rope",
        }
        topology_encoder_ids = [
            topology.get(name)
            for name in ("encoder", "encoder_adapter")
            if topology.get(name) is not None
        ]
        if topology_encoder_ids and len(set(topology_encoder_ids)) != 1:
            raise ValueError(
                "checkpoint topology has conflicting encoder identities: "
                f"{topology_encoder_ids!r}"
            )
        for source, target in topology_fields.items():
            value = topology.get(source)
            if value is not None:
                if target == "camera_keys":
                    value = _normalize_camera_keys(value)
                    if value is None:
                        continue
                values[target] = value
        # Older canonical checkpoints used ``builtin`` as a placeholder
        # instead of a registry adapter.  Let the family/model/checkpoint-key
        # fields below select the corresponding built-in encoder.
        if values.get("encoder") == "builtin":
            values["encoder"] = None
        topology_camera_keys = _normalize_camera_keys(topology.get("camera_keys"))
        topology_views = _normalize_camera_keys(topology.get("views"))
        if (
            topology_camera_keys is not None
            and topology_views is not None
            and topology_camera_keys != topology_views
        ):
            raise ValueError(
                "checkpoint topology camera_keys and views disagree: "
                f"{topology_camera_keys!r} vs {topology_views!r}"
            )
        if topology_camera_keys is None and topology_views is not None:
            values["camera_keys"] = topology_views
        topology_spec = topology.get("encoder_spec")
        if (
            topology_encoder_ids
            and isinstance(topology_spec, dict)
            and topology_spec.get("name") is not None
            and topology_spec["name"] not in topology_encoder_ids
        ):
            raise ValueError(
                "checkpoint topology encoder and encoder_spec identities disagree: "
                f"{topology_encoder_ids!r} vs {topology_spec['name']!r}"
            )
        if (
            values.get("encoder") is None
            and isinstance(topology_spec, dict)
            and topology_spec.get("name")
        ):
            values["encoder"] = topology_spec["name"]
        if (
            topology.get("encoder_interpolate_rope") is None
            and isinstance(topology_spec, dict)
            and topology_spec.get("interpolate_rope") is not None
        ):
            values["encoder_interpolate_rope"] = bool(
                topology_spec["interpolate_rope"]
            )
        elif topology.get("encoder_interpolate_rope") is None:
            # Pre-registry checkpoints used the legacy non-interpolated
            # encoder path and did not serialize this field.  Preserve that
            # behavior while allowing policy_serving/CLI values below to
            # override it for newer or explicitly configured checkpoints.
            values.setdefault("encoder_interpolate_rope", False)
        rope_pairing = topology.get("predictor_rope_frequency_pairing")
        if rope_pairing is not None:
            values["corrected_predictor_rope"] = rope_pairing == "corrected"
        else:
            # The pre-correction predictor serialized no pairing field and
            # used the legacy ordering.  New checkpoints carry an explicit
            # value in either topology or policy_serving.config below.
            values.setdefault("corrected_predictor_rope", False)

        saved_config = saved.get("config") if isinstance(saved, dict) else None
        saved_encoder_ids = [
            saved_config.get("encoder")
            if isinstance(saved_config, dict)
            else None,
            saved.get("encoder_adapter") if isinstance(saved, dict) else None,
        ]
        saved_encoder_ids = [value for value in saved_encoder_ids if value is not None]
        if saved_encoder_ids and len(set(saved_encoder_ids)) != 1:
            raise ValueError(
                "checkpoint serving metadata has conflicting encoder identities: "
                f"{saved_encoder_ids!r}"
            )
        # ``encoder_spec.name`` is another identity source in newer
        # checkpoints.  Include both topology and serving specs in the same
        # consistency check; otherwise a spec-only serving record could
        # silently override a conflicting topology encoder below.
        topology_spec_ids = [
            topology_spec.get("name")
            if isinstance(topology_spec, dict)
            else None
        ]
        saved_spec = (
            saved_config.get("encoder_spec")
            if isinstance(saved_config, dict)
            else None
        )
        saved_spec_ids = [
            saved_spec.get("name") if isinstance(saved_spec, dict) else None
        ]
        stable_encoder_ids = (
            topology_encoder_ids
            + saved_encoder_ids
            + [value for value in topology_spec_ids + saved_spec_ids if value is not None]
        )
        override_encoder = (
            overrides.get("encoder") if isinstance(overrides, dict) else None
        )
        effective_encoder_ids = stable_encoder_ids + (
            [override_encoder] if override_encoder is not None else []
        )
        if effective_encoder_ids and len(set(effective_encoder_ids)) != 1:
            raise ValueError(
                "checkpoint metadata and overrides have conflicting encoder identities: "
                f"{effective_encoder_ids!r}"
            )
        legacy_family = (
            overrides.get("encoder_family")
            if isinstance(overrides, dict)
            and overrides.get("encoder_family") is not None
            else saved_config.get("encoder_family")
            if isinstance(saved_config, dict)
            and saved_config.get("encoder_family") is not None
            else values.get("encoder_family", "vjepa2")
        )
        if not effective_encoder_ids and legacy_family in {"vjepa2", "vjepa2_1"}:
            # The original Route 2 checkpoint had no encoder identity (and in
            # some variants no family/model/key fields at all).  Its serving
            # contract was the official V-JEPA2 ViT-G raw checkpoint.  Keep
            # this fallback local to metadata-free checkpoints; any explicit
            # fields below, including policy_serving.config, still win.
            values.setdefault("model_name", "vit_giant_xformers")
            values.setdefault(
                "encoder_checkpoint_key",
                "target_encoder"
                if legacy_family == "vjepa2_1"
                else "encoder",
            )
            values.setdefault("camera_keys", _LEGACY_DEFAULT_CAMERA_KEYS)
            if "view_layout" not in values:
                predictor_views = topology.get("predictor_num_views")
                predictor_height = topology.get("predictor_image_height")
                predictor_width = topology.get("predictor_image_width")
                if predictor_views == 1 and (predictor_height, predictor_width) == (
                    256,
                    256,
                ):
                    # Route 2 baseline used a 256px quadrant canvas while
                    # keeping one composite predictor view.
                    values["view_layout"] = "quadrant"
                elif isinstance(predictor_views, int) and predictor_views > 1:
                    values["view_layout"] = "independent"
        if isinstance(saved, dict) and saved.get("encoder_adapter") not in (None, "builtin"):
            values["encoder"] = saved["encoder_adapter"]
        if saved_config:
            valid_fields = {field.name for field in fields(cls)}
            saved_camera_keys = _normalize_camera_keys(saved_config.get("camera_keys"))
            saved_views = _normalize_camera_keys(saved_config.get("views"))
            if (
                saved_camera_keys is not None
                and saved_views is not None
                and saved_camera_keys != saved_views
            ):
                raise ValueError(
                    "checkpoint serving camera_keys and views disagree: "
                    f"{saved_camera_keys!r} vs {saved_views!r}"
                )
            for name, value in saved_config.items():
                if name not in valid_fields:
                    continue
                if value is None:
                    continue
                if name == "camera_keys":
                    value = _normalize_camera_keys(value)
                    if value is None:
                        continue
                values[name] = value
            if (
                "views" in saved_config
                and _normalize_camera_keys(saved_config.get("camera_keys")) is None
            ):
                camera_keys = _normalize_camera_keys(saved_config["views"])
                if camera_keys is not None:
                    values["camera_keys"] = camera_keys
            # New checkpoints keep the stable registry id in the serialized
            # EncoderSpec.  Older checkpoints only have family/model fields.
            spec = saved_config.get("encoder_spec")
            if (
                saved_config.get("encoder") is None
                and isinstance(spec, dict)
                and spec.get("name")
            ):
                values["encoder"] = spec["name"]
            if saved_config.get("encoder_interpolate_rope") is None:
                if isinstance(spec, dict) and spec.get("interpolate_rope") is not None:
                    values["encoder_interpolate_rope"] = bool(spec["interpolate_rope"])
                else:
                    values["encoder_interpolate_rope"] = values.get(
                        "encoder_interpolate_rope", False
                    )
        if values.get("encoder") == "builtin":
            values["encoder"] = None
        if overrides:
            values.update(overrides)
        del checkpoint
        return cls(**values)


def _normalize_camera_keys(value):
    """Normalize serialized/CLI camera names without assuming a view count."""
    if value is None:
        return None
    if isinstance(value, str):
        value = (value,)
    try:
        keys = tuple(value)
    except TypeError as error:
        raise TypeError("camera_keys/views must be a string sequence") from error
    # Empty metadata is treated as "unspecified" so an explicit CLI --views
    # override can repair old/incomplete checkpoints.
    if not keys:
        return None
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("camera_keys/views must contain non-empty names")
    if len(set(keys)) != len(keys):
        raise ValueError(f"camera_keys/views must be unique, got {keys!r}")
    return keys


def _strip_prefix(state_dict, prefix="module.backbone."):
    prefixed = {
        name[len(prefix) :]: value
        for name, value in state_dict.items()
        if name.startswith(prefix)
    }
    if prefixed:
        return prefixed
    # Some local exports already contain encoder-relative keys (and a few
    # distributed saves use only ``module.``).  Do not silently turn those
    # valid checkpoints into an empty state dict.
    module_prefixed = {
        name[len("module.") :]: value
        for name, value in state_dict.items()
        if name.startswith("module.")
    }
    return module_prefixed or dict(state_dict)


def to_env_action(action, binarize_gripper=True):
    """Convert LeRobot LIBERO actions to the robosuite action convention."""
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 1.0 - 2.0 * action[..., -1]
    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])
    action[..., :6] = np.clip(action[..., :6], -1.0, 1.0)
    return action


class VJEPAPolicyServing:
    def __init__(
        self,
        ckpt_path,
        pretrained_encoder,
        text_cache_dir,
        config=None,
        device="cuda",
        encoder_builder=None,
        predictor_builder=None,
        policy_builder=None,
    ):
        # A serving process should be reproducible from a checkpoint alone.
        # Explicit config values still win, while omitted topology (especially
        # camera views) is recovered from the saved serving metadata.
        metadata_config = None
        config_field_names = tuple(field.name for field in fields(PolicyServingConfig))
        if config is None or any(
            getattr(config, name) is None for name in config_field_names
        ):
            metadata_config = PolicyServingConfig.from_checkpoint(ckpt_path)
        if config is None:
            args = metadata_config or PolicyServingConfig()
        else:
            args = config
            inherited = {}
            if metadata_config is not None:
                inherited.update(
                    {
                        name: getattr(metadata_config, name)
                        for name in config_field_names
                        if getattr(args, name) is None
                        and getattr(metadata_config, name) is not None
                    }
                )
            if inherited:
                args = replace(args, **inherited)
            # Keep direct callers that intentionally pass a partially filled
            # config usable even when the checkpoint has no metadata.
            defaults = PolicyServingConfig()
            missing_defaults = {
                name: getattr(defaults, name)
                for name in config_field_names
                if getattr(args, name) is None
            }
            if missing_defaults:
                args = replace(args, **missing_defaults)
        # ``0`` is the explicit CLI/config sentinel for the historical
        # variable-width state contract.  Keep ``None`` available internally
        # without allowing metadata inheritance to erase that explicit choice.
        if args.max_state_dim == 0:
            args = replace(args, max_state_dim=None)
        self.device = torch.device(device)
        self.precision = args.precision
        precision_dtypes = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if self.precision not in precision_dtypes:
            raise ValueError(
                f"unsupported precision={self.precision!r}; choose bf16, fp16, or fp32"
            )
        self.dtype = precision_dtypes[self.precision]
        if self.precision == "fp32":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.crop_size = args.crop_size
        self.t5_len = args.t5_len
        self.binarize_gripper = True
        if args.view_layout not in ("quadrant", "horizontal", "independent"):
            raise ValueError(f"unsupported serving view_layout={args.view_layout!r}")
        self.view_layout = args.view_layout
        camera_keys = _normalize_camera_keys(args.camera_keys)
        if camera_keys is None:
            raise ValueError(
                "camera views are not present in checkpoint metadata; pass explicit "
                "--views (or set PolicyServingConfig.camera_keys)"
            )
        self.camera_keys = camera_keys
        if args.view_layout == "quadrant" and len(self.camera_keys) > 4:
            raise ValueError("quadrant serving supports at most four camera views")
        if args.view_layout == "horizontal" and len(self.camera_keys) < 2:
            raise ValueError("horizontal serving requires at least two camera views")
        num_views = len(self.camera_keys) if args.view_layout == "independent" else 1
        image_size = (
            (args.crop_size, args.crop_size * len(self.camera_keys))
            if args.view_layout == "horizontal"
            else (args.crop_size, args.crop_size)
        )
        self.model_image_size = image_size
        policy_factory = policy_builder or build_vjepa_policy
        # Experimental geometry/ablation builders retain their historical
        # state topology until they explicitly adopt the packed contract.
        state_contract_max = (
            args.max_state_dim if policy_factory is build_vjepa_policy else None
        )
        if state_contract_max is None:
            model_proprio_dim = encoded_proprio_dim(
                args.proprio_dim, args.proprio_encoding
            )
        else:
            if args.proprio_encoding != "identity":
                raise ValueError(
                    "fixed-width state packing requires proprio_encoding='identity'; "
                    "set max_state_dim=None for legacy sincos serving"
                )
            if args.proprio_dim > state_contract_max:
                raise ValueError(
                    f"proprio_dim={args.proprio_dim} exceeds max_state_dim={state_contract_max}"
                )
            model_proprio_dim = packed_state_dim(state_contract_max)

        encoder_spec = None
        registry_builder = None
        if encoder_builder is None and args.encoder is not None:
            from vjepa_policy.encoders import (
                build_encoder as registry_builder,
                encoder_spec as get_encoder_spec,
            )

            try:
                encoder_spec = get_encoder_spec(args.encoder)
            except ValueError as error:
                legacy_combination = (
                    args.encoder_family,
                    args.model_name,
                    args.encoder_checkpoint_key,
                )
                if legacy_combination not in _LEGACY_RAW_ENCODER_COMBINATIONS:
                    raise ValueError(
                        f"encoder adapter {args.encoder!r} is not registered and "
                        f"cannot be reconstructed from {legacy_combination!r}"
                    ) from error
                metadata_encoder = None
                if config is not None:
                    try:
                        metadata_encoder = PolicyServingConfig.from_checkpoint(
                            ckpt_path
                        ).encoder
                    except Exception:  # metadata is advisory; state loading validates the rest
                        logger.debug(
                            "Could not inspect checkpoint encoder metadata",
                            exc_info=True,
                        )
                if (
                    config is not None
                    and config.encoder is not None
                    and metadata_encoder != config.encoder
                ):
                    raise ValueError(
                        "explicit encoder adapter does not match checkpoint metadata: "
                        f"{config.encoder!r} != {metadata_encoder!r}"
                    ) from error
                logger.warning(
                    "Checkpoint names unregistered encoder adapter %r; falling back "
                    "to family/model metadata",
                    args.encoder,
                )
                args = replace(args, encoder=None)
            else:
                # A stable adapter id carries its own checkpoint contract.
                # Use that key when a caller did not override the canonical
                # V-JEPA2 target-encoder default (e.g. DINO local_pretrained).
                if (
                    args.encoder_checkpoint_key == "target_encoder"
                    and encoder_spec.checkpoint_key != "target_encoder"
                ):
                    args = replace(
                        args,
                        encoder_checkpoint_key=encoder_spec.checkpoint_key,
                    )
        if encoder_builder is not None:
            encoder = encoder_builder(args, pretrained_encoder)
        elif args.encoder is not None:
            # Registry adapters own their checkpoint format and latent geometry.
            # Keep this import lazy so historical serving environments that do
            # not install optional encoder dependencies remain usable.
            encoder = registry_builder(
                args.encoder,
                checkpoint=pretrained_encoder,
                image_size=image_size,
                video_frames=args.num_frames,
                num_views=num_views,
                tubelet_size=args.tubelet_size,
                interpolate_rope=args.encoder_interpolate_rope,
                checkpoint_key=args.encoder_checkpoint_key,
            )
        else:
            if args.encoder_family not in ("vjepa2", "vjepa2_1"):
                raise ValueError(f"unsupported encoder_family={args.encoder_family!r}")
            if args.encoder_family == "vjepa2" and args.model_name == "vit_giant":
                raise ValueError(
                    "Use vit_giant_xformers for the official 22-head V-JEPA2 ViT-G"
                )
            encoder_module = (
                video_vit if args.encoder_family == "vjepa2" else video_vit_2_1
            )
            canonical_spatial_grid = (
                (16, 16)
                if args.model_name == "vit_large"
                else None
            )
            encoder_kwargs = dict(
                img_size=image_size,
                patch_size=args.patch_size,
                num_frames=args.num_frames,
                tubelet_size=args.tubelet_size,
                uniform_power=False,
                use_rope=True,
                use_sdpa=True,
                interpolate_rope=args.encoder_interpolate_rope,
                canonical_spatial_grid=canonical_spatial_grid,
            )
            if args.encoder_family == "vjepa2_1":
                encoder_kwargs.update(
                    img_temporal_dim_size=1,
                    modality_embedding=True,
                    n_output_distillation=1,
                )
            encoder = encoder_module.__dict__[args.model_name](**encoder_kwargs)
            encoder._vjepa_policy_training_false = args.encoder_family == "vjepa2_1"
            if (
                args.model_name == "vit_giant_xformers"
                and encoder.blocks[0].attn.num_heads != 22
            ):
                raise RuntimeError(
                    "Official V-JEPA2 ViT-G must have 22 attention heads"
                )
        runtime_encoder_spec = getattr(encoder, "spec", None)
        if runtime_encoder_spec is not None:
            encoder_spec = runtime_encoder_spec
        if encoder_builder is None and encoder_spec is None and args.model_name == "vit_large":
            # Raw family/model serving remains available for historical
            # launchers, but the canonical ViT-L geometry should match the
            # registry path (including endpoint-aligned spatial RoPE).
            registry_name = {
                "vjepa2": "vjepa2_vitl",
                "vjepa2_1": "vjepa2_1_vitl",
            }.get(args.encoder_family)
            if registry_name is not None:
                try:
                    from vjepa_policy.encoders import encoder_spec as get_encoder_spec

                    encoder_spec = get_encoder_spec(registry_name)
                except (ImportError, ValueError):
                    logger.debug(
                        "Registry metadata unavailable for raw encoder %s",
                        registry_name,
                        exc_info=True,
                    )
        latent_layout = None
        if encoder_spec is not None:
            latent_layout = encoder_spec.layout_for_clip(
                video_frames=args.num_frames,
                image_size=image_size,
                num_views=num_views,
                context_steps=args.context_tubelets,
            )
        predictor_kwargs = dict(
            img_size=image_size,
            patch_size=(
                encoder_spec.input_patch_size
                if encoder_spec is not None
                else args.patch_size
            ),
            num_frames=args.num_frames,
            tubelet_size=(
                encoder_spec.temporal_stride
                if encoder_spec is not None
                else args.tubelet_size
            ),
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=args.pred_embed_dim,
            depth=args.pred_depth,
            num_heads=args.pred_num_heads,
            num_mask_tokens=args.num_mask_tokens,
            lang_dim=args.lang_dim,
            proprio_dim=model_proprio_dim,
            use_activation_checkpointing=False,
            interpolate_rope=True,
            corrected_rope_frequency_pairing=args.corrected_predictor_rope,
            num_views=num_views,
            max_views=(
                args.predictor_max_views
                if args.predictor_max_views is not None
                else num_views
            ),
        )
        if latent_layout is not None:
            predictor_kwargs.update(
                latent_grid=(
                    latent_layout.grid_depth,
                    latent_layout.grid_height,
                    latent_layout.grid_width,
                ),
                canonical_spatial_grid=encoder_spec.canonical_spatial_grid,
            )
        predictor_factory = predictor_builder or build_world_model_predictor
        predictor = predictor_factory(**predictor_kwargs)
        mask_builder = getattr(encoder, "build_policy_mask", None)
        if latent_layout is not None:
            mask = CausalPatchMask.from_layout(latent_layout)
        elif mask_builder is None:
            mask = CausalPatchMask(
                image_size=image_size,
                patch_size=args.patch_size,
                tubelet_size=args.tubelet_size,
                video_frames=args.num_frames,
                context_tubelets=args.context_tubelets,
                num_views=num_views,
            )
        else:
            mask = mask_builder(image_size=image_size, num_views=num_views)
        policy_kwargs = dict(
            action_dim=args.action_dim,
            action_chunk_size=args.action_chunk_size,
            context_len=mask.n_ctx,
            proprio_dim=args.proprio_dim,
            proprio_encoding=args.proprio_encoding,
            max_state_dim=state_contract_max,
            action_hidden_size=args.action_hidden_size,
            action_num_layers=args.action_num_layers,
            condition_num_heads=args.condition_num_heads,
            action_num_inference_steps=args.action_num_inference_steps,
        )
        if latent_layout is not None and policy_factory is build_vjepa_policy:
            policy_kwargs["latent_layout"] = latent_layout
        policy = policy_factory(encoder, predictor, **policy_kwargs)

        if encoder_builder is None and args.encoder is None:
            encoder_checkpoint = torch.load(
                pretrained_encoder, map_location="cpu", weights_only=False, mmap=True
            )
            if args.encoder_checkpoint_key not in encoder_checkpoint:
                raise KeyError(
                    f"Encoder checkpoint has no {args.encoder_checkpoint_key!r}"
                )
            print(
                "[load] encoder:",
                policy.encoder.load_state_dict(
                    _strip_prefix(encoder_checkpoint[args.encoder_checkpoint_key])
                ),
            )
            del encoder_checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        expected_precision = {
            "fp32": "no",
            "fp16": "fp16",
            "bf16": "bf16",
        }[self.precision]
        checkpoint_precision = checkpoint.get("train_precision")
        if checkpoint_precision is None:
            logger.warning(
                "Checkpoint has no train_precision metadata; relying on state-dict validation"
            )
        elif not (
            checkpoint_precision == expected_precision
            or self.precision == "fp32"
            and checkpoint_precision == "fp32"
        ):
            raise ValueError(
                f"checkpoint precision {checkpoint_precision!r} "
                f"!= serving {expected_precision!r}"
            )
        topology = checkpoint.get("model_topology", {})
        encoder_blocks = getattr(policy.encoder, "blocks", ())
        encoder_attention = (
            getattr(encoder_blocks[0], "attn", None) if encoder_blocks else None
        )
        encoder_interpolate_rope = bool(
            getattr(
                encoder_attention,
                "interpolate_rope",
                getattr(encoder_spec, "interpolate_rope", args.encoder_interpolate_rope),
            )
        )
        encoder_rope_pairing = (
            "corrected"
            if getattr(encoder_attention, "corrected_frequency_pairing", False)
            else "legacy"
        )
        encoder_family = getattr(
            policy.encoder, "_vjepa_policy_family", args.encoder_family
        )
        encoder_model_name = getattr(
            policy.encoder, "_vjepa_policy_model_name", args.model_name
        )
        encoder_checkpoint_key = getattr(
            policy.encoder,
            "_vjepa_policy_checkpoint_key",
            args.encoder_checkpoint_key,
        )
        runtime_encoder_name = (
            encoder_spec.name if encoder_spec is not None else args.encoder
        )
        runtime_latent_grid = (
            tuple(int(size) for size in policy.predictor.latent_grid)
            if getattr(policy.predictor, "latent_grid", None) is not None
            else None
        )
        runtime_latent_layout = (
            latent_layout.to_dict() if latent_layout is not None else None
        )
        runtime_encoder_spec = (
            encoder_spec.to_dict() if encoder_spec is not None else None
        )
        expected_topology = {
            "predictor_class": type(policy.predictor).__name__,
            "predictor_depth": args.pred_depth,
            "predictor_embed_dim": args.pred_embed_dim,
            "predictor_num_heads": args.pred_num_heads,
            "predictor_image_height": int(policy.predictor.img_height),
            "predictor_image_width": int(policy.predictor.img_width),
            "predictor_num_views": int(policy.predictor.num_views),
            "action_expert_class": type(policy.action_expert).__name__,
            "action_hidden_size": args.action_hidden_size,
            "action_num_layers": args.action_num_layers,
            "action_num_heads": args.pred_num_heads,
            "action_head_dim": args.pred_embed_dim // args.pred_num_heads,
            "action_dim": args.action_dim,
            "action_chunk_size": args.action_chunk_size,
            "proprio_encoding": args.proprio_encoding,
            "raw_proprio_dim": args.proprio_dim,
            "encoded_proprio_dim": encoded_proprio_dim(
                args.proprio_dim, args.proprio_encoding
            ),
            "max_state_dim": state_contract_max or 0,
            "packed_proprio_dim": model_proprio_dim,
            "predictor_interpolate_rope": True,
            "predictor_rope_frequency_pairing": (
                "corrected" if args.corrected_predictor_rope else "legacy"
            ),
            "encoder_interpolate_rope": encoder_interpolate_rope,
            "encoder_rope_frequency_pairing": encoder_rope_pairing,
            "clean_context_attention": bool(
                getattr(policy.predictor, "use_clean_context_attention", False)
            ),
            "read_full_predictor_kv": bool(
                getattr(policy.action_expert, "read_full_predictor_kv", False)
            ),
        }
        missing_topology_defaults = {
            "predictor_interpolate_rope": True,
            # from_checkpoint resolves omitted fields from checkpoint metadata
            # (legacy checkpoints) or policy_serving.config (new checkpoints).
            # Use the resolved runtime value here so a config-only field is not
            # contradicted by a hard-coded legacy fallback.
            "predictor_rope_frequency_pairing": (
                "corrected" if args.corrected_predictor_rope else "legacy"
            ),
            "encoder_interpolate_rope": bool(args.encoder_interpolate_rope),
            "encoder_rope_frequency_pairing": "legacy",
            "predictor_image_height": 256,
            "predictor_image_width": 256,
            "predictor_num_views": 1,
            "action_dim": args.action_dim,
            "action_chunk_size": args.action_chunk_size,
            "proprio_encoding": "identity",
            "raw_proprio_dim": args.proprio_dim,
            "encoded_proprio_dim": encoded_proprio_dim(
                args.proprio_dim, args.proprio_encoding
            ),
            "max_state_dim": state_contract_max or 0,
            "packed_proprio_dim": model_proprio_dim,
            "clean_context_attention": False,
            "read_full_predictor_kv": False,
        }
        if topology:
            observed_topology = {}
            for name in expected_topology:
                if name in topology:
                    observed_topology[name] = _canonicalize_topology_value(
                        topology[name]
                    )
                elif name in missing_topology_defaults:
                    observed_topology[name] = _canonicalize_topology_value(
                        missing_topology_defaults[name]
                    )
            mismatches = {
                name: (observed_topology[name], expected)
                for name, expected in expected_topology.items()
                if name in observed_topology
                if observed_topology[name] != expected
            }
            if mismatches:
                raise ValueError(f"checkpoint topology mismatch: {mismatches}")
            missing_topology = sorted(set(expected_topology) - set(observed_topology))
            if missing_topology:
                logger.warning(
                    "Checkpoint topology omits %s; relying on state-dict validation",
                    missing_topology,
                )
        else:
            logger.warning(
                "Checkpoint has no model_topology metadata; relying on state-dict validation"
            )
        optional_topology = {
            "view_layout": args.view_layout,
            "camera_keys": list(self.camera_keys),
            "views": list(self.camera_keys),
            # Trainer checkpoints historically used ``encoder`` while the
            # serving metadata used ``encoder_adapter``.  Check both aliases
            # when present so a stale or hand-edited identity cannot slip by.
            "encoder": runtime_encoder_name,
            "encoder_adapter": runtime_encoder_name,
            "encoder_family": encoder_family,
            "encoder_model_name": encoder_model_name,
            "encoder_checkpoint_key": encoder_checkpoint_key,
            "latent_grid": runtime_latent_grid,
            "latent_layout": runtime_latent_layout,
            "encoder_spec": runtime_encoder_spec,
        }
        optional_mismatches = {}
        for name, expected in optional_topology.items():
            if name not in topology or expected is None:
                continue
            observed = _canonicalize_topology_field(name, topology[name])
            # Empty camera metadata is intentionally treated as unspecified;
            # an explicit serving --views override can repair old checkpoints.
            if name in {"camera_keys", "views"} and observed is None:
                continue
            # ``None`` in historical topology dictionaries means the field
            # was not declared, rather than an affirmative incompatible value.
            if observed is None:
                continue
            canonical_expected = _canonicalize_topology_field(name, expected)
            if observed != canonical_expected:
                optional_mismatches[name] = (observed, canonical_expected)
        if optional_mismatches:
            raise ValueError(
                f"checkpoint input/encoder mismatch: {optional_mismatches}"
            )
        print(
            "[load] predictor:",
            policy.predictor.load_state_dict(checkpoint["predictor"]),
        )
        print(
            "[load] action_expert:",
            policy.action_expert.load_state_dict(checkpoint["action_expert"]),
        )
        print(
            "[load] proprio_encoder:",
            policy.proprio_encoder.load_state_dict(checkpoint["proprio_encoder"]),
        )
        print("[load] checkpoint step:", checkpoint.get("step"))
        del checkpoint

        self.encoder = policy.encoder.to(self.device).eval()
        self.predictor = policy.predictor.to(self.device).eval()
        self.action_expert = policy.action_expert.to(self.device).eval()
        self.proprio_encoder = policy.proprio_encoder.to(self.device).eval()
        self.encoder_spec = encoder_spec
        self.encoder_name = (
            encoder_spec.name if encoder_spec is not None else args.encoder
        )
        self.latent_layout = latent_layout
        self.num_views = num_views
        self.context_len = mask.n_ctx
        self.num_frames = args.num_frames
        self.action_chunk_size = args.action_chunk_size
        self.raw_proprio_dim = args.proprio_dim
        self.max_state_dim = state_contract_max
        self.proprio_encoding = args.proprio_encoding
        self.m_enc = mask.ctx_idx.unsqueeze(0).to(self.device)
        self.m_pred = mask.tgt_idx.unsqueeze(0).to(self.device)

        if args.view_layout == "independent":
            self.video_transform = VideoClipTransform(
                (args.crop_size, args.crop_size), resize_mode=args.image_resize_mode
            )
            self.view_combiner = "independent"
        elif args.view_layout == "horizontal":
            self.video_transform = VideoClipTransform(
                (args.crop_size, args.crop_size), resize_mode=args.image_resize_mode
            )
            self.view_combiner = "horizontal"
        else:
            self.video_transform = VideoClipTransform(
                (args.crop_size // 2, args.crop_size // 2),
                resize_mode=args.image_resize_mode,
            )
            self.view_combiner = QuadrantViewCombiner(
                image_size, fill_value=self.video_transform.normalized_black
            )
        self.text_cache_dir = text_cache_dir
        self._lang_cache = {}
        stats_path = args.dataset_stats or os.path.join(
            os.path.dirname(ckpt_path), "dataset_stats.json"
        )
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(
                f"action normalization stats not found: {stats_path}"
            )
        self.action_normalizer = ActionStateNormalizer.load(stats_path)

    def _load_language(self, task_text):
        if task_text in self._lang_cache:
            return self._lang_cache[task_text]
        prompt = DEFAULT_PROMPT.format(task=task_text)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = os.path.join(self.text_cache_dir, f"{digest}.t5_len{self.t5_len}.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        language = payload["context"].unsqueeze(0).to(self.device)
        language_mask = payload["mask"].unsqueeze(0).bool().to(self.device)
        self._lang_cache[task_text] = (language, language_mask)
        return language, language_mask

    @staticmethod
    def _to_uint8_hwc(image):
        if image.ndim == 4:
            image = image.squeeze(0)
        image = (image.float() * 255.0).clamp(0, 255).round().to(torch.uint8)
        return image.permute(1, 2, 0).cpu().numpy()

    @staticmethod
    def _to_state_tensor(state):
        state = state if isinstance(state, torch.Tensor) else torch.as_tensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError(
                f"observation.state must be [D] or [B,D], got {tuple(state.shape)}"
            )
        return state.float()

    def _prepare_observation(self, obs):
        views = []
        for key in self.camera_keys:
            current = self._to_uint8_hwc(obs[key])
            past_key = f"{key}_past"
            if past_key not in obs:
                raise KeyError(f"Missing past-frame view {past_key!r}")
            past = self._to_uint8_hwc(obs[past_key])
            frames = np.stack([past, current] + [current] * (self.num_frames - 2))
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
            views.append(self.video_transform(frames))
        clip = combine_video_clips(views, self.view_combiner)

        language, language_mask = self._load_language(obs["task"][0])
        state_key = "observation.state"
        if state_key not in obs:
            raise KeyError(f"Missing {state_key!r}; proprio conditioning is required")
        state = self._to_state_tensor(obs[state_key])
        return clip, language, language_mask, state

    @torch.no_grad()
    def infer_batch(self, observations):
        if not observations:
            return []
        prepared = [self._prepare_observation(obs) for obs in observations]
        clips = torch.stack([item[0] for item in prepared]).to(self.device)
        language = torch.cat([item[1] for item in prepared])
        language_mask = torch.cat([item[2] for item in prepared])
        state_key = "observation.state"
        state = torch.cat([item[3] for item in prepared])
        state = self.action_normalizer({state_key: state})[state_key].to(self.device)
        state = prepare_proprio_state(
            state,
            raw_state_dim=getattr(self, "raw_proprio_dim", state.shape[-1]),
            encoding=getattr(self, "proprio_encoding", "identity"),
            max_state_dim=getattr(self, "max_state_dim", None),
        )
        batch_size = len(observations)
        m_enc = self.m_enc.expand(batch_size, -1)
        m_pred = self.m_pred.expand(batch_size, -1)
        precision_context = (
            torch.autocast(device_type=self.device.type, dtype=self.dtype)
            if self.precision != "fp32"
            else nullcontext()
        )
        with precision_context:
            context_latents = encode_video_views(self.encoder, clips, m_enc)
            _, predictor_kv = self.predictor(
                context_latents,
                m_enc,
                m_pred,
                language=language,
                language_mask=language_mask,
                proprio=state,
            )
            actions = self.action_expert.predict_action(
                predictor_kv,
                language,
                language_mask,
                state,
                context_len=self.context_len,
            )
        actions = self.action_normalizer.unnormalize_action(actions.float().cpu())
        return [
            {
                "actions": to_env_action(
                    action.numpy(), binarize_gripper=self.binarize_gripper
                )
            }
            for action in actions
        ]

    def infer(self, obs):
        return self.infer_batch([obs])[0]
