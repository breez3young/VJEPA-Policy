"""RoboCasa GR-1 observation, action, and V-JEPA serving utilities."""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch


Array: TypeAlias = np.ndarray | torch.Tensor

GR1_VIDEO_KEY = "video.ego_view_bg_crop_pad_res256_freq20"
GR1_DATASET_VIDEO_KEY = "observation.images.ego_view"
GR1_LANGUAGE_KEY = "annotation.human.coarse_action"
GR1_LANGUAGE_PREFIX = "unlocked_waist: "
GR1_STATE_KEY = "observation.state"
GR1_ACTION_KEY = "action"
GR1_VIDEO_DELTA_INDICES = (-4, 0)
GR1_STATE_DELTA_INDICES = (0,)
GR1_ACTION_HORIZON = 16
GR1_MODEL_FRAMES = 6
GR1_IMAGE_SIZE = 256
GR1_MODEL_IMAGE_SIZE = 224
GR1_GROUP_DIMS = (
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_hand", 6),
    ("right_hand", 6),
    ("waist", 3),
)
GR1_ACTION_DIM = 29
GR1_RELATIVE_ACTION_DIM = 26
GR1_ACTION_STATE_INDICES = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    7,
    8,
    9,
    10,
    11,
    12,
    29,
    30,
    31,
    32,
    33,
    34,
    41,
    42,
    43,
)


def _validate_array(value: object, name: str) -> Array:
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError(f"{name} must be a numpy array or torch tensor")
    if value.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    return value


def _validate_compatible(reference: Array, value: Array, name: str) -> None:
    if type(value) is not type(reference):
        raise TypeError(f"{name} must use the same array type as the other groups")
    if value.dtype != reference.dtype:
        raise TypeError(f"{name} must use dtype {reference.dtype}, got {value.dtype}")
    if isinstance(value, torch.Tensor) and value.device != reference.device:
        raise ValueError(
            f"{name} must be on device {reference.device}, got {value.device}"
        )


def grouped_state_to_29d(state: Mapping[str, Array]) -> Array:
    """Pack grouped raw state in official GR-1 order along the last axis."""
    groups = []
    reference = None
    leading_shape = None
    for group_name, group_dim in GR1_GROUP_DIMS:
        if group_name not in state:
            raise KeyError(f"Missing GR-1 state group {group_name!r}")
        value = _validate_array(state[group_name], f"state[{group_name!r}]")
        if value.shape[-1] != group_dim:
            raise ValueError(
                f"state[{group_name!r}] must end in dimension {group_dim}, "
                f"got shape {tuple(value.shape)}"
            )
        if reference is None:
            reference = value
            leading_shape = value.shape[:-1]
        else:
            _validate_compatible(reference, value, f"state[{group_name!r}]")
            if value.shape[:-1] != leading_shape:
                raise ValueError(
                    f"state[{group_name!r}] has leading shape {tuple(value.shape[:-1])}, "
                    f"expected {tuple(leading_shape)}"
                )
        groups.append(value)

    if isinstance(reference, torch.Tensor):
        return torch.cat(groups, dim=-1)
    return np.concatenate(groups, axis=-1)


def relative_action_to_absolute(action: Array, raw_state: Array) -> Array:
    """Legacy conversion helper for checkpoints trained with relative actions.

    Current GR-1 serving uses absolute actions and never calls this helper.

    ``action`` accepts ``[T, 29]`` or ``[B, T, 29]``. For batched actions,
    ``raw_state`` accepts ``[29]``, ``[B, 29]``, or ``[B, S, 29]``; the last
    state timestep is used in the latter case. For unbatched actions, raw state
    accepts ``[29]`` or ``[S, 29]``.
    """
    action = _validate_array(action, "action")
    raw_state = _validate_array(raw_state, "raw_state")
    _validate_compatible(action, raw_state, "raw_state")
    if action.ndim not in (2, 3) or action.shape[-1] != GR1_ACTION_DIM:
        raise ValueError(
            f"action must have shape [T,{GR1_ACTION_DIM}] or [B,T,{GR1_ACTION_DIM}], "
            f"got {tuple(action.shape)}"
        )
    if raw_state.shape[-1] != GR1_ACTION_DIM:
        raise ValueError(
            f"raw_state must end in dimension {GR1_ACTION_DIM}, got {tuple(raw_state.shape)}"
        )

    if action.ndim == 2:
        if raw_state.ndim == 1:
            current_state = raw_state
        elif raw_state.ndim == 2:
            if raw_state.shape[0] == 0:
                raise ValueError("raw_state history must not be empty")
            current_state = raw_state[-1]
        else:
            raise ValueError(
                "raw_state for unbatched action must have shape [29] or [S,29]"
            )
        reference = current_state[None, :GR1_RELATIVE_ACTION_DIM]
    else:
        batch_size = action.shape[0]
        if raw_state.ndim == 1:
            current_state = raw_state
            reference = current_state[None, None, :GR1_RELATIVE_ACTION_DIM]
        elif raw_state.ndim == 2:
            if raw_state.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"raw_state batch {raw_state.shape[0]} does not match action batch {batch_size}"
                )
            current_state = raw_state
            reference = current_state[:, None, :GR1_RELATIVE_ACTION_DIM]
        elif raw_state.ndim == 3:
            if raw_state.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"raw_state batch {raw_state.shape[0]} does not match action batch {batch_size}"
                )
            if raw_state.shape[1] == 0:
                raise ValueError("raw_state history must not be empty")
            current_state = raw_state[:, -1]
            reference = current_state[:, None, :GR1_RELATIVE_ACTION_DIM]
        else:
            raise ValueError(
                "raw_state for batched action must have shape [29], [B,29], or [B,S,29]"
            )

    absolute = action.clone() if isinstance(action, torch.Tensor) else action.copy()
    absolute[..., :GR1_RELATIVE_ACTION_DIM] += reference
    return absolute


def action_29d_to_robocasa(action: Array) -> dict[str, Array]:
    """Split dense absolute GR-1 actions into RoboCasa ``action.*`` groups."""
    action = _validate_array(action, "action")
    if action.shape[-1] != GR1_ACTION_DIM:
        raise ValueError(
            f"action must end in dimension {GR1_ACTION_DIM}, got {tuple(action.shape)}"
        )

    result = {}
    start = 0
    for group_name, group_dim in GR1_GROUP_DIMS:
        value = action[..., start : start + group_dim]
        result[f"action.{group_name}"] = (
            value.clone() if isinstance(value, torch.Tensor) else value.copy()
        )
        start += group_dim
    return result


def strip_gr1_language_prefix(instruction: str) -> str:
    """Remove the simulator-only waist mode prefix from one instruction."""
    if not isinstance(instruction, str):
        raise TypeError(f"GR-1 instruction must be a string, got {type(instruction)}")
    if not instruction.startswith(GR1_LANGUAGE_PREFIX):
        raise ValueError(
            f"GR-1 instruction must start with {GR1_LANGUAGE_PREFIX!r}, got {instruction!r}"
        )
    instruction = instruction[len(GR1_LANGUAGE_PREFIX) :].strip()
    if not instruction:
        raise ValueError("GR-1 instruction is empty after removing its waist prefix")
    return instruction


def _language_sequence(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError("GR-1 language ndarray must not use object dtype")
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("GR-1 language batch must be a string sequence or ndarray")
    result = list(value)
    if not all(isinstance(item, str) for item in result):
        raise TypeError("Every GR-1 language batch item must be a string")
    return result


def gr1_language_batch(value: object, batch_size: int) -> list[str]:
    """Normalize and de-prefix the flat simulator language batch."""
    instructions = _language_sequence(value)
    if len(instructions) != batch_size:
        raise ValueError(
            f"GR-1 language batch has size {len(instructions)}, expected {batch_size}"
        )
    return [strip_gr1_language_prefix(instruction) for instruction in instructions]


def validate_gr1_wire_observation(observation: Mapping[str, Any]) -> int:
    """Validate the batched flat schema emitted by the official rollout client."""
    if GR1_VIDEO_KEY not in observation:
        raise KeyError(f"Missing GR-1 video key {GR1_VIDEO_KEY!r}")
    video = observation[GR1_VIDEO_KEY]
    if not isinstance(video, np.ndarray):
        raise TypeError(f"{GR1_VIDEO_KEY} must be a numpy array")
    if video.dtype != np.uint8:
        raise TypeError(f"{GR1_VIDEO_KEY} must use uint8, got {video.dtype}")
    expected_tail = (len(GR1_VIDEO_DELTA_INDICES), GR1_IMAGE_SIZE, GR1_IMAGE_SIZE, 3)
    if video.ndim != 5 or video.shape[1:] != expected_tail or video.shape[0] < 1:
        raise ValueError(
            f"{GR1_VIDEO_KEY} must be [B,{','.join(map(str, expected_tail))}] with B>=1, "
            f"got {video.shape}"
        )
    batch_size = video.shape[0]

    for group_name, group_dim in GR1_GROUP_DIMS:
        key = f"state.{group_name}"
        if key not in observation:
            raise KeyError(f"Missing GR-1 state key {key!r}")
        value = observation[key]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{key} must be a numpy array")
        if value.dtype != np.float32:
            raise TypeError(f"{key} must use float32, got {value.dtype}")
        expected_shape = (batch_size, len(GR1_STATE_DELTA_INDICES), group_dim)
        if value.shape != expected_shape:
            raise ValueError(
                f"{key} must have shape {expected_shape}, got {value.shape}"
            )

    if GR1_LANGUAGE_KEY not in observation:
        raise KeyError(f"Missing GR-1 language key {GR1_LANGUAGE_KEY!r}")
    gr1_language_batch(observation[GR1_LANGUAGE_KEY], batch_size)
    return batch_size


def flat_gr1_state_to_29d(observation: Mapping[str, Any]) -> np.ndarray:
    """Pack the five flat simulator state groups into ``[B, 1, 29]``."""
    groups = {
        group_name: observation[f"state.{group_name}"]
        for group_name, _ in GR1_GROUP_DIMS
    }
    packed = grouped_state_to_29d(groups)
    if not isinstance(packed, np.ndarray):
        raise TypeError("Flat GR-1 wire state must contain numpy arrays")
    return packed


def build_gr1_model_clip(
    video: np.ndarray,
    video_transform=None,
    model_frames: int = GR1_MODEL_FRAMES,
) -> torch.Tensor:
    """Convert two real simulator frames into a padded model clip.

    The first two model frames are the observations at offsets ``[-4, 0]``.
    Future slots are filled with the current frame because only their mask tokens,
    not their pixels, are consumed by the Predictor.
    """
    if not isinstance(model_frames, int) or model_frames < len(GR1_VIDEO_DELTA_INDICES):
        raise ValueError(
            f"model_frames must be an integer >= {len(GR1_VIDEO_DELTA_INDICES)}"
        )
    if not isinstance(video, np.ndarray) or video.dtype != np.uint8:
        raise TypeError("GR-1 video must be a uint8 numpy array")
    expected_tail = (len(GR1_VIDEO_DELTA_INDICES), GR1_IMAGE_SIZE, GR1_IMAGE_SIZE, 3)
    if video.ndim != 5 or video.shape[0] < 1 or video.shape[1:] != expected_tail:
        raise ValueError(
            f"GR-1 video must be [B,{','.join(map(str, expected_tail))}], got {video.shape}"
        )
    if video_transform is None:
        from vjepa_policy.datasets import VideoClipTransform

        video_transform = VideoClipTransform(
            (GR1_MODEL_IMAGE_SIZE, GR1_MODEL_IMAGE_SIZE)
        )

    batch_size = video.shape[0]
    frames = (
        torch.from_numpy(video)
        .permute(0, 1, 4, 2, 3)
        .reshape(
            batch_size * len(GR1_VIDEO_DELTA_INDICES), 3, GR1_IMAGE_SIZE, GR1_IMAGE_SIZE
        )
    )
    frames = video_transform(frames)
    expected_shape = (
        batch_size * len(GR1_VIDEO_DELTA_INDICES),
        3,
        GR1_MODEL_IMAGE_SIZE,
        GR1_MODEL_IMAGE_SIZE,
    )
    if frames.shape != expected_shape:
        raise ValueError(
            f"GR-1 video transform must return {expected_shape}, got {tuple(frames.shape)}"
        )
    frames = frames.reshape(
        batch_size,
        len(GR1_VIDEO_DELTA_INDICES),
        3,
        GR1_MODEL_IMAGE_SIZE,
        GR1_MODEL_IMAGE_SIZE,
    )
    current = frames[:, -1:]
    future = current.expand(-1, model_frames - frames.shape[1], -1, -1, -1)
    frames = torch.cat((frames, future), dim=1)
    return frames.permute(0, 2, 1, 3, 4).unsqueeze(1).contiguous()


@dataclass(frozen=True)
class GR1ModelInputs:
    clip: torch.Tensor
    raw_state: np.ndarray
    instructions: list[str]


def prepare_gr1_model_inputs(
    observation: Mapping[str, Any],
    video_transform=None,
    model_frames: int = GR1_MODEL_FRAMES,
) -> GR1ModelInputs:
    """Validate and convert a complete flat GR-1 wire observation."""
    batch_size = validate_gr1_wire_observation(observation)
    raw_state = flat_gr1_state_to_29d(observation)
    if raw_state.shape != (batch_size, len(GR1_STATE_DELTA_INDICES), GR1_ACTION_DIM):
        raise ValueError(f"Packed GR-1 state has invalid shape {raw_state.shape}")
    return GR1ModelInputs(
        clip=build_gr1_model_clip(
            observation[GR1_VIDEO_KEY], video_transform, model_frames=model_frames
        ),
        raw_state=raw_state,
        instructions=gr1_language_batch(observation[GR1_LANGUAGE_KEY], batch_size),
    )


def decode_gr1_action_chunk(
    absolute_action: Array,
    action_horizon: int | None = None,
) -> dict[str, Array]:
    """Validate and split one batched absolute policy action chunk."""
    absolute_action = _validate_array(absolute_action, "absolute_action")
    if action_horizon is None:
        action_horizon = absolute_action.shape[1] if absolute_action.ndim == 3 else None
    if not isinstance(action_horizon, int) or action_horizon < 1:
        raise ValueError("action_horizon must be a positive integer")
    if absolute_action.ndim != 3 or absolute_action.shape[1:] != (
        action_horizon,
        GR1_ACTION_DIM,
    ):
        raise ValueError(
            f"absolute_action must be [B,{action_horizon},{GR1_ACTION_DIM}], "
            f"got {tuple(absolute_action.shape)}"
        )
    return action_29d_to_robocasa(absolute_action)


class VJEPAGR1Serving:
    """Batched GR-1 runtime built on the published V-JEPA model loader."""

    def __init__(
        self,
        ckpt_path,
        pretrained_encoder,
        text_cache_dir,
        config,
        device="cuda",
    ):
        from vjepa_policy.policy_serving.libero import VJEPAPolicyServing

        self.core = VJEPAPolicyServing(
            ckpt_path=ckpt_path,
            pretrained_encoder=pretrained_encoder,
            text_cache_dir=text_cache_dir,
            config=config,
            device=device,
        )
        self._validate_core()

    def _validate_core(self) -> None:
        core = self.core
        if core.view_layout != "independent":
            raise ValueError("GR-1 serving requires the independent view layout")
        if core.camera_keys != (GR1_DATASET_VIDEO_KEY,):
            raise ValueError(
                f"GR-1 serving requires camera_keys={(GR1_DATASET_VIDEO_KEY,)}, "
                f"got {core.camera_keys}"
            )
        if core.crop_size != GR1_MODEL_IMAGE_SIZE:
            raise ValueError(
                f"GR-1 serving requires {GR1_MODEL_IMAGE_SIZE}x{GR1_MODEL_IMAGE_SIZE} frames"
            )
        if core.num_frames < len(GR1_VIDEO_DELTA_INDICES):
            raise ValueError("GR-1 serving needs at least the two context frames")
        tubelet_size = getattr(core.encoder, "tubelet_size", 2)
        if core.num_frames % tubelet_size:
            raise ValueError(
                f"GR-1 model frames {core.num_frames} must be divisible by tubelet size "
                f"{tubelet_size}"
            )
        if core.action_expert.action_dim != GR1_ACTION_DIM:
            raise ValueError(
                f"GR-1 Action Expert must have {GR1_ACTION_DIM} action dimensions"
            )
        if core.action_expert.action_chunk_size < 1:
            raise ValueError(
                "GR-1 Action Expert action_chunk_size must be positive"
            )
        action_horizon = core.action_expert.action_chunk_size
        if action_horizon % 4:
            raise ValueError("GR-1 action horizon must be a multiple of four")
        expected_model_frames = len(GR1_VIDEO_DELTA_INDICES) + action_horizon // 4
        if core.num_frames != expected_model_frames:
            raise ValueError(
                f"GR-1 action horizon {action_horizon} requires "
                f"{expected_model_frames} model frames, got {core.num_frames}"
            )
        if getattr(core, "proprio_encoding", "identity") != "sincos":
            raise ValueError("GR-1 serving requires sincos proprio encoding")

        normalizer = core.action_normalizer
        for key, shape in (
            (GR1_ACTION_KEY, (GR1_ACTION_DIM,)),
            (GR1_STATE_KEY, (GR1_ACTION_DIM,)),
        ):
            if (
                key not in normalizer.features
                or normalizer.features[key].shape != shape
            ):
                raise ValueError(
                    f"GR-1 normalizer feature {key!r} must have shape {shape}"
                )
        expected_codec = {
            "action_indices": list(GR1_ACTION_STATE_INDICES),
            "state_indices": list(GR1_ACTION_STATE_INDICES),
            "relative_action_indices": [],
            "action_chunk_size": self.action_horizon,
        }
        if normalizer.metadata.get("action_codec") != expected_codec:
            raise ValueError(
                "GR-1 normalizer action codec does not match the serving contract"
            )
        expected_proprio = {"type": "sincos", "raw_dim": 29, "encoded_dim": 58}
        if normalizer.metadata.get("proprio_encoding") != expected_proprio:
            raise ValueError(
                "GR-1 normalizer proprio encoding does not match the serving contract"
            )
        if normalizer.action_mode != "MIN_MAX":
            raise ValueError("GR-1 serving requires MIN_MAX action normalization")
        if normalizer.state_mode != "IDENTITY":
            raise ValueError("GR-1 serving requires IDENTITY state normalization")
        if not normalizer.clip_values:
            raise ValueError("GR-1 serving requires clipped action normalization")

    @property
    def action_horizon(self) -> int:
        return int(self.core.action_expert.action_chunk_size)

    @property
    def model_frames(self) -> int:
        return int(self.core.num_frames)

    @torch.no_grad()
    def infer_flat_observation(
        self,
        observation: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        """Infer one full absolute action chunk for every environment in a batch."""
        core = self.core
        inputs = prepare_gr1_model_inputs(
            observation,
            core.video_transform,
            model_frames=self.model_frames,
        )
        clip = inputs.clip.to(core.device)
        raw_state = torch.from_numpy(inputs.raw_state)
        state = core.action_normalizer({GR1_STATE_KEY: raw_state[:, -1]})[GR1_STATE_KEY]
        state = state.to(core.device)
        from vjepa_policy.models.vjepa_policy import encode_proprio

        state = encode_proprio(state, core.proprio_encoding)

        language_batches = [core._load_language(text) for text in inputs.instructions]
        language = torch.cat([item[0] for item in language_batches], dim=0)
        language_mask = torch.cat([item[1] for item in language_batches], dim=0)
        batch_size = clip.shape[0]
        masks_enc = core.m_enc.expand(batch_size, -1)
        masks_pred = core.m_pred.expand(batch_size, -1)
        precision_context = (
            torch.autocast(device_type=core.device.type, dtype=core.dtype)
            if core.precision == "bf16"
            else nullcontext()
        )
        with precision_context:
            from vjepa_policy.models.vjepa_policy import encode_video_views

            context_latents = encode_video_views(core.encoder, clip, masks_enc)
            _, predictor_kv = core.predictor(
                context_latents,
                masks_enc,
                masks_pred,
                language=language,
                language_mask=language_mask,
                proprio=state,
            )
            normalized_action = core.action_expert.predict_action(
                predictor_kv,
                language,
                language_mask,
                state,
                context_len=core.context_len,
            )

        absolute_action = core.action_normalizer.unnormalize_action(
            normalized_action.float().cpu()
        )
        grouped = decode_gr1_action_chunk(
            absolute_action, action_horizon=self.action_horizon
        )
        return {
            key: value.numpy().astype(np.float32, copy=False)
            for key, value in grouped.items()
        }


__all__ = [
    "GR1_ACTION_DIM",
    "GR1_ACTION_HORIZON",
    "GR1_ACTION_STATE_INDICES",
    "GR1_DATASET_VIDEO_KEY",
    "GR1_GROUP_DIMS",
    "GR1_IMAGE_SIZE",
    "GR1_LANGUAGE_KEY",
    "GR1_LANGUAGE_PREFIX",
    "GR1_MODEL_FRAMES",
    "GR1_MODEL_IMAGE_SIZE",
    "GR1_RELATIVE_ACTION_DIM",
    "GR1_STATE_DELTA_INDICES",
    "GR1_VIDEO_DELTA_INDICES",
    "GR1_VIDEO_KEY",
    "GR1ModelInputs",
    "VJEPAGR1Serving",
    "action_29d_to_robocasa",
    "build_gr1_model_clip",
    "decode_gr1_action_chunk",
    "flat_gr1_state_to_29d",
    "gr1_language_batch",
    "grouped_state_to_29d",
    "prepare_gr1_model_inputs",
    "relative_action_to_absolute",
    "strip_gr1_language_prefix",
    "validate_gr1_wire_observation",
]
