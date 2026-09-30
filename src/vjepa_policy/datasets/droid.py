"""DROID data plumbing for language-conditioned V-JEPA pre-training.

The DROID release used by this project is recorded at 15 Hz while the V-JEPA
input stream is sampled at 5 Hz.  This module keeps the sampling, instruction
canonicalisation, image preprocessing and packed T5 cache format in one place
so that the cache builder and the training dataset cannot silently disagree.

The module deliberately does not import PRTS.  Its small instruction helper is
the behaviour used by PRTS' DROID loader, copied here so PRTS remains an
unchanged dependency.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import random
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from vjepa_policy.datasets.prompts import DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------------------

DROID_ROOT = os.environ.get("DROID_DATASET_ROOT", "")
DROID_BENCH_CACHE_ROOT = os.environ.get("DROID_TEXT_CACHE", "")
DROID_REPO_ID = "lerobot/droid_1.0.1"
DROID_FPS = 15
# 15 Hz is exactly divisible by 5 Hz.  Keeping this as an explicit contract
# avoids silently falling back to a rounded 4 Hz schedule (15/4 = 3.75).
DROID_TARGET_FPS = 5
DROID_SOURCE_FRAME_STRIDE = DROID_FPS // DROID_TARGET_FPS

DROID_VIDEO_KEYS: tuple[str, str] = (
    "observation.images.exterior_1_left",
    "observation.images.wrist_left",
)
# Short aliases are useful in launch scripts and retain the exact feature-key
# spelling in one canonical constant.
DROID_VIEWS = DROID_VIDEO_KEYS
DROID_INSTRUCTION_KEYS: tuple[str, str, str] = (
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
)
DROID_INSTRUCTION_FIELDS = DROID_INSTRUCTION_KEYS
DROID_STATE_KEY = "observation.state"

# Nearest 15-Hz frame for the 5-Hz sample points [-.2, 0, .2, ..., 1.6].
# The first tubelet is context and the remaining four tubelets are future.
DROID_VIDEO_OFFSETS: tuple[int, ...] = (-3, 0, 3, 6, 9, 12, 15, 18, 21, 24)
DROID_CONTEXT_OFFSETS: tuple[int, ...] = DROID_VIDEO_OFFSETS[:2]
DROID_FUTURE_OFFSETS: tuple[int, ...] = DROID_VIDEO_OFFSETS[2:]
DROID_NUM_FRAMES = len(DROID_VIDEO_OFFSETS)
DROID_TUBELET_SIZE = 2
DROID_CONTEXT_TUBELETS = 1
DROID_FUTURE_FRAMES = DROID_NUM_FRAMES - DROID_CONTEXT_TUBELETS * DROID_TUBELET_SIZE
DROID_MAX_FUTURE_OFFSET = max(DROID_VIDEO_OFFSETS)

# The observed maximum after applying the PRTS canonicalisation and the policy
# prompt is 129 tokens on the complete DROID_v21 release.  Keeping this as a
# contract catches accidental truncation when the dataset or prompt changes.
DROID_T5_CONTEXT_LENGTH = 129
DROID_CONTEXT_LENGTH = DROID_T5_CONTEXT_LENGTH
DROID_T5_MAX_LENGTH = DROID_T5_CONTEXT_LENGTH
DROID_PROMPT_TEMPLATE = DEFAULT_PROMPT
DROID_PROMPT = DEFAULT_PROMPT


def load_droid_info(
    root: str | os.PathLike[str] = DROID_ROOT,
) -> dict[str, Any]:
    """Read DROID's ``meta/info.json`` with the release's known fps typo.

    The cached DROID_v21 metadata currently contains the literal fragment
    ``"fps": ,``.  It is the only malformed field in that file, and the
    release contract above establishes the source rate as 15 Hz.  Repair only
    that exact fragment; any other JSON error is raised to the caller instead
    of being swallowed.
    """

    path = Path(root) / "meta" / "info.json"
    text = path.read_text(encoding="utf-8")
    repaired_fps = False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        repaired, replacements = re.subn(
            r'("fps"\s*:\s*)(?=,)', rf"\g<1>{DROID_FPS}", text, count=1
        )
        if replacements != 1:
            raise
        payload = json.loads(repaired)
        repaired_fps = True
    if not isinstance(payload, dict):
        raise ValueError(f"DROID metadata must be a JSON object: {path}")
    # A valid file may carry fps explicitly; the malformed-release repair is
    # only accepted when it resolves to the source rate expected by DROID.
    fps = payload.get("fps")
    if fps is None:
        if not repaired_fps:
            raise ValueError(f"DROID metadata has no source fps: {path}")
        payload["fps"] = DROID_FPS
    elif int(fps) != DROID_FPS:
        raise ValueError(f"DROID source fps must be {DROID_FPS}, found {fps!r}")
    return payload


def make_droid_video_offsets() -> tuple[int, ...]:
    """Return the fixed nearest-frame map used by the DROID contract."""

    return DROID_VIDEO_OFFSETS


def _plain_text(value: Any) -> str | None:
    """Convert a LeRobot scalar to text without changing its contents."""

    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    # hf_transform_to_torch leaves strings as strings, but small synthetic
    # fixtures often use scalar tensors or one-element lists.
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0].item()
    elif isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        return _plain_text(value[0])
    try:
        return str(value)
    except Exception:
        return None


def canonicalize_droid_instruction(text: str) -> str:
    """Apply the exact task canonicalisation used by the PRTS DROID loader.

    PRTS performs ``task.lower().strip().rstrip('.').capitalize() + '.'``.
    In particular, this intentionally does not collapse internal whitespace or
    alter punctuation other than trailing ASCII full stops.
    """

    if not isinstance(text, str):
        raise TypeError(f"DROID instruction must be a string, got {type(text)!r}")
    return text.lower().strip().rstrip(".").capitalize() + "."


def droid_instruction_candidates(
    item: Mapping[str, Any],
    instruction_keys: Sequence[str] = DROID_INSTRUCTION_KEYS,
) -> list[str]:
    """Return non-empty instruction candidates in source-column order."""

    candidates: list[str] = []
    for key in instruction_keys:
        value = _plain_text(item.get(key))
        if value is None:
            continue
        value = value.strip()
        if value:
            candidates.append(value)
    return candidates


def sample_droid_instruction(
    item: Mapping[str, Any],
    chooser: Callable[[Sequence[str]], str] | None = None,
    instruction_keys: Sequence[str] = DROID_INSTRUCTION_KEYS,
) -> str:
    """Uniformly sample one non-empty DROID instruction and canonicalise it.

    ``chooser`` is injectable for deterministic tests and for reproducible data
    workers.  It receives the same ordered candidate list that PRTS passes to
    ``random.choice``.
    """

    candidates = droid_instruction_candidates(item, instruction_keys)
    if not candidates:
        raise ValueError(
            f"DROID item has no non-empty instruction in {tuple(instruction_keys)!r}"
        )
    selected = (random.choice if chooser is None else chooser)(candidates)
    if not isinstance(selected, str):
        selected = _plain_text(selected) or ""
    if not selected.strip():
        raise ValueError("Instruction chooser returned an empty instruction")
    return canonicalize_droid_instruction(selected)


def droid_prompt(
    instruction: str,
    prompt_template: str = DROID_PROMPT,
    *,
    canonicalize: bool = True,
) -> str:
    """Format a policy prompt using the same text seen by the T5 cache."""

    task = canonicalize_droid_instruction(instruction) if canonicalize else instruction
    return prompt_template.format(task=task)


# Readable aliases used by a few downstream launchers.
format_droid_prompt = droid_prompt
canonicalize_instruction = canonicalize_droid_instruction


def load_droid_state_quantiles(
    root: str | os.PathLike[str] = DROID_ROOT,
    state_key: str = DROID_STATE_KEY,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the q01/q99 state bounds shipped with DROID."""

    path = Path(root) / "meta" / "norm_stats.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        stats = payload[state_key]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Missing q01/q99 state statistics for {state_key!r} in {path}"
        ) from error
    if q01.ndim != 1 or q99.shape != q01.shape:
        raise ValueError(
            f"Invalid DROID quantile shapes: q01={q01.shape}, q99={q99.shape}"
        )
    return q01, q99


def quantile_normalize_droid_state(
    state: torch.Tensor | np.ndarray | Sequence[float],
    q01: torch.Tensor | np.ndarray | Sequence[float],
    q99: torch.Tensor | np.ndarray | Sequence[float],
) -> torch.Tensor:
    """Map state values using PRTS' q01/q99 ``[-1, 1]`` convention."""

    value = torch.as_tensor(state, dtype=torch.float32)
    low = torch.as_tensor(q01, dtype=value.dtype, device=value.device)
    high = torch.as_tensor(q99, dtype=value.dtype, device=value.device)
    if low.ndim != 1 or high.shape != low.shape:
        raise ValueError(
            f"Invalid q01/q99 shapes: {tuple(low.shape)}, {tuple(high.shape)}"
        )
    if value.ndim == 0 or value.shape[-1] != low.numel():
        raise ValueError(
            f"State last dimension {tuple(value.shape)} does not match quantiles "
            f"({low.numel()},)"
        )
    return (value - low) / (high - low + 1e-8) * 2.0 - 1.0


droid_state_normalize = quantile_normalize_droid_state


# ---------------------------------------------------------------------------
# Compact full-future window indexing
# ---------------------------------------------------------------------------


def build_droid_window_index(
    episode_starts: Sequence[int],
    episode_ends: Sequence[int],
    max_future_offset: int = DROID_MAX_FUTURE_OFFSET,
) -> tuple[list[int], list[int]]:
    """Build compact-index metadata while dropping incomplete future windows.

    ``episode_ends`` are exclusive, as in LeRobot's ``episode_data_index``.
    Context frames before an episode start are allowed to clamp (the first
    offset is -4); only future frames are required to remain in the episode.
    """

    starts = [int(x) for x in episode_starts]
    ends = [int(x) for x in episode_ends]
    if len(starts) != len(ends):
        raise ValueError("episode_starts and episode_ends must have equal lengths")
    if max_future_offset < 0:
        raise ValueError("max_future_offset must be non-negative")
    cumulative: list[int] = []
    running = 0
    for start, end in zip(starts, ends, strict=True):
        if end < start:
            raise ValueError(f"Episode end {end} precedes start {start}")
        running += max(0, end - start - max_future_offset)
        cumulative.append(running)
    return starts, cumulative


def compact_droid_index(
    index: int,
    episode_starts: Sequence[int],
    cumulative_sizes: Sequence[int],
) -> int:
    """Map a compact dataset index to its absolute LeRobot frame index."""

    total = int(cumulative_sizes[-1]) if len(cumulative_sizes) else 0
    if index < 0:
        index += total
    if not 0 <= index < total:
        raise IndexError(index)
    cumulative_for_bisect = (
        cumulative_sizes.tolist()
        if isinstance(cumulative_sizes, (torch.Tensor, np.ndarray))
        else cumulative_sizes
    )
    episode_number = bisect.bisect_right(cumulative_for_bisect, index)
    previous = 0 if episode_number == 0 else int(cumulative_sizes[episode_number - 1])
    return int(episode_starts[episode_number]) + index - previous


# Backwards-friendly names for callers that already use the generic wording.
build_full_window_index = build_droid_window_index
compact_window_index = compact_droid_index


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


def _as_tchw(clip: torch.Tensor | np.ndarray | Sequence[Any]) -> torch.Tensor:
    """Convert common LeRobot clip layouts to ``[T,C,H,W]`` float tensors."""

    if not torch.is_tensor(clip):
        if isinstance(clip, (list, tuple)) and clip and not hasattr(clip, "shape"):
            clip = torch.stack([torch.as_tensor(frame) for frame in clip])
        else:
            clip = torch.as_tensor(clip)
    if clip.ndim != 4:
        raise ValueError(f"Expected a 4-D video clip, got {tuple(clip.shape)}")
    # LeRobot video decoding is [T,C,H,W].  Accept [T,H,W,C] as well for
    # pyarrow/PIL fixtures and official VJEPA transform parity.
    if clip.shape[1] in (1, 3, 4):
        result = clip
    elif clip.shape[-1] in (1, 3, 4):
        result = clip.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            "Cannot infer channel dimension; expected [T,C,H,W] or [T,H,W,C], "
            f"got {tuple(clip.shape)}"
        )
    result = result.float()
    if result.numel() and float(result.max()) > 1.0:
        result = result / 255.0
    return result


def _resize_letterbox_float(
    clip: torch.Tensor,
    size: tuple[int, int],
    fill: float = 0.0,
) -> torch.Tensor:
    """Resize [T,C,H,W] into a canvas without changing pixel aspect ratio."""

    target_h, target_w = (int(size[0]), int(size[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Target size must be positive, got {size}")
    _, channels, height, width = clip.shape
    if height <= 0 or width <= 0:
        raise ValueError("Video frames must have positive height and width")
    scale = min(target_h / height, target_w / width)
    new_h = max(1, min(target_h, int(round(height * scale))))
    new_w = max(1, min(target_w, int(round(width * scale))))
    if (new_h, new_w) != (height, width):
        resized = F.interpolate(
            clip,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    else:
        resized = clip
    canvas = torch.full(
        (clip.shape[0], channels, target_h, target_w),
        float(fill),
        dtype=resized.dtype,
        device=resized.device,
    )
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[:, :, top : top + new_h, left : left + new_w] = resized
    return canvas


class DroidLetterboxTransform:
    """Deterministic 256-square DROID transform with centered black padding."""

    def __init__(
        self,
        size: int | tuple[int, int] = 256,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.size = (
            (int(size), int(size)) if isinstance(size, int) else tuple(map(int, size))
        )
        if len(self.size) != 2:
            raise ValueError(f"size must be an int or (height, width), got {size!r}")
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        if self.mean.numel() != self.std.numel() or self.mean.numel() not in (1, 3, 4):
            raise ValueError("mean and std must contain one or RGB/RGBA channel values")
        if bool(torch.any(self.std == 0)):
            raise ValueError("std values must be non-zero")

    @property
    def normalized_black(self) -> torch.Tensor:
        return (-self.mean / self.std).flatten()

    def __call__(self, clip: torch.Tensor | np.ndarray | Sequence[Any]) -> torch.Tensor:
        value = _as_tchw(clip)
        if value.shape[1] != self.mean.shape[1] and self.mean.shape[1] != 1:
            raise ValueError(
                f"Transform statistics have {self.mean.shape[1]} channels, clip has {value.shape[1]}"
            )
        value = _resize_letterbox_float(value, self.size)
        mean = self.mean.to(device=value.device, dtype=value.dtype)
        std = self.std.to(device=value.device, dtype=value.dtype)
        return ((value - mean) / std).contiguous()


def _sample_resized_crop(
    height: int,
    width: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    generator: torch.Generator | None = None,
) -> tuple[int, int, int, int]:
    """Sample torchvision/VJEPA-style crop parameters once for a whole clip."""

    if not (0 < scale[0] <= scale[1]):
        raise ValueError(f"Invalid crop scale {scale}")
    if not (0 < ratio[0] <= ratio[1]):
        raise ValueError(f"Invalid crop aspect ratio {ratio}")
    area = float(height * width)
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * float(
            torch.rand((), generator=generator) * (scale[1] - scale[0]) + scale[0]
        )
        aspect = math.exp(
            float(
                torch.rand((), generator=generator) * (log_ratio[1] - log_ratio[0])
                + log_ratio[0]
            )
        )
        crop_h = int(round(math.sqrt(target_area / aspect)))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        if 0 < crop_h <= height and 0 < crop_w <= width:
            top_max = height - crop_h
            left_max = width - crop_w
            top = (
                int(torch.randint(top_max + 1, (), generator=generator))
                if top_max
                else 0
            )
            left = (
                int(torch.randint(left_max + 1, (), generator=generator))
                if left_max
                else 0
            )
            return top, left, crop_h, crop_w
    # torchvision's fallback is a centered crop at the closest feasible ratio.
    in_ratio = width / height
    if in_ratio < ratio[0]:
        crop_w = width
        crop_h = int(round(crop_w / ratio[0]))
    elif in_ratio > ratio[1]:
        crop_h = height
        crop_w = int(round(crop_h * ratio[1]))
    else:
        crop_h = crop_w = min(height, width)
    crop_h, crop_w = min(crop_h, height), min(crop_w, width)
    return (height - crop_h) // 2, (width - crop_w) // 2, crop_h, crop_w


class VJEPA2ACVideoTransform:
    """Reference V-JEPA2-AC spatial augmentation for a complete clip.

    The parameter ranges match ``app.vjepa_droid.transforms.make_transforms``:
    random resized crop scale ``(0.3, 1.0)``, aspect ratio ``(3/4, 4/3)`` and
    50% horizontal flip.  Unlike the original implementation, this adapter
    letterboxes the sampled crop before normalization, so no image is stretched
    and the requested 256x256 canvas is guaranteed for DROID's 16:9 frames.
    Random parameters are sampled once and shared by every frame in the clip.
    """

    def __init__(
        self,
        size: int | tuple[int, int] = 256,
        random_horizontal_flip: bool = True,
        random_resize_aspect_ratio: tuple[float, float] = (3 / 4, 4 / 3),
        random_resize_scale: tuple[float, float] = (0.3, 1.0),
        normalize: tuple[Sequence[float], Sequence[float]] = (
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
        generator: torch.Generator | None = None,
    ) -> None:
        self.size = (
            (int(size), int(size)) if isinstance(size, int) else tuple(map(int, size))
        )
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.random_resize_aspect_ratio = tuple(map(float, random_resize_aspect_ratio))
        self.random_resize_scale = tuple(map(float, random_resize_scale))
        self.mean = torch.as_tensor(normalize[0], dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.as_tensor(normalize[1], dtype=torch.float32).view(1, -1, 1, 1)
        self.generator = generator

    def __call__(self, clip: torch.Tensor | np.ndarray | Sequence[Any]) -> torch.Tensor:
        value = _as_tchw(clip)
        _, channels, height, width = value.shape
        if self.mean.shape[1] not in (1, channels):
            raise ValueError(
                f"Normalization has {self.mean.shape[1]} channels, clip has {channels}"
            )
        top, left, crop_h, crop_w = _sample_resized_crop(
            height,
            width,
            self.random_resize_scale,
            self.random_resize_aspect_ratio,
            self.generator,
        )
        value = value[:, :, top : top + crop_h, left : left + crop_w]
        if self.random_horizontal_flip:
            do_flip = bool(torch.rand((), generator=self.generator) < 0.5)
            if do_flip:
                value = value.flip(-1)
        value = _resize_letterbox_float(value, self.size)
        mean = self.mean.to(device=value.device, dtype=value.dtype)
        std = self.std.to(device=value.device, dtype=value.dtype)
        return ((value - mean) / std).permute(1, 0, 2, 3).contiguous()


class PRTSCropRotateVideoTransform:
    """Apply the PRTS crop and rotation policy to one complete video clip.

    PRTS samples a crop covering 95--100% of the original image area while
    preserving its aspect ratio, then applies a small ``(-3, 3)`` degree
    rotation.  Both random parameters are sampled once for the whole clip;
    frames therefore remain temporally aligned.  The transformed crop is
    letterboxed (rather than stretched) to the requested square before the
    ImageNet normalization used by the V-JEPA2 encoder.
    """

    def __init__(
        self,
        size: int | tuple[int, int] = 256,
        crop_scale: tuple[float, float] = (0.95, 1.0),
        rotation_degrees: tuple[float, float] = (-3.0, 3.0),
        normalize: tuple[Sequence[float], Sequence[float]] = (
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
        generator: torch.Generator | None = None,
    ) -> None:
        self.size = (
            (int(size), int(size)) if isinstance(size, int) else tuple(map(int, size))
        )
        if len(self.size) != 2:
            raise ValueError(f"size must be an int or (height, width), got {size!r}")
        self.crop_scale = tuple(map(float, crop_scale))
        self.rotation_degrees = tuple(map(float, rotation_degrees))
        if not (0 < self.crop_scale[0] <= self.crop_scale[1]):
            raise ValueError(f"Invalid PRTS crop scale {self.crop_scale}")
        if not self.rotation_degrees[0] <= self.rotation_degrees[1]:
            raise ValueError(
                f"Invalid PRTS rotation range {self.rotation_degrees}"
            )
        self.mean = torch.as_tensor(normalize[0], dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.as_tensor(normalize[1], dtype=torch.float32).view(1, -1, 1, 1)
        if self.mean.numel() != self.std.numel() or self.mean.numel() not in (1, 3, 4):
            raise ValueError("mean and std must contain one or RGB/RGBA channel values")
        if bool(torch.any(self.std == 0)):
            raise ValueError("std values must be non-zero")
        self.generator = generator

    def _sample_crop(self, height: int, width: int) -> tuple[int, int, int, int]:
        """Reproduce PRTS ``RandomScaleCrop`` sampling for one clip."""

        area = float(height * width)
        target_area = area * float(
            torch.rand((), generator=self.generator)
            * (self.crop_scale[1] - self.crop_scale[0])
            + self.crop_scale[0]
        )
        aspect_ratio = width / height
        crop_h = max(1, int(round(math.sqrt(target_area / aspect_ratio))))
        crop_w = max(1, int(round(target_area / crop_h)))
        crop_w = min(crop_w, width)
        crop_h = min(crop_h, height)
        top_max = height - crop_h
        left_max = width - crop_w
        top = (
            int(torch.randint(top_max + 1, (), generator=self.generator))
            if top_max
            else 0
        )
        left = (
            int(torch.randint(left_max + 1, (), generator=self.generator))
            if left_max
            else 0
        )
        return top, left, crop_h, crop_w

    def __call__(self, clip: torch.Tensor | np.ndarray | Sequence[Any]) -> torch.Tensor:
        value = _as_tchw(clip)
        _, channels, height, width = value.shape
        if self.mean.shape[1] not in (1, channels):
            raise ValueError(
                f"Normalization has {self.mean.shape[1]} channels, clip has {channels}"
            )
        top, left, crop_h, crop_w = self._sample_crop(height, width)
        value = value[:, :, top : top + crop_h, left : left + crop_w]

        angle = float(
            torch.rand((), generator=self.generator)
            * (self.rotation_degrees[1] - self.rotation_degrees[0])
            + self.rotation_degrees[0]
        )
        # PRTS uses torchvision v2.RandomRotation defaults (nearest
        # interpolation, no expansion, black fill).  Import lazily so the
        # rest of the DROID metadata/cache helpers remain usable without a
        # torchvision import at module load time.
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import rotate

        value = rotate(
            value,
            angle,
            interpolation=InterpolationMode.NEAREST,
            expand=False,
            fill=0,
        )
        value = _resize_letterbox_float(value, self.size)
        mean = self.mean.to(device=value.device, dtype=value.dtype)
        std = self.std.to(device=value.device, dtype=value.dtype)
        return ((value - mean) / std).permute(1, 0, 2, 3).contiguous()


# Explicit aliases make the augmentation choice obvious in config files.
OfficialVJEPA2ACTransform = VJEPA2ACVideoTransform
VJEPA2ACTransform = VJEPA2ACVideoTransform
DroidVJEPA2ACTransform = VJEPA2ACVideoTransform
PRTSCropRotateTransform = PRTSCropRotateVideoTransform


def build_droid_video_transform(
    augmentation: str = "letterbox",
    **kwargs: Any,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Construct one of the supported DROID clip transforms."""

    name = augmentation.lower().replace("-", "_")
    if name in {"letterbox", "resize", "current", "default"}:
        return DroidLetterboxTransform(**kwargs)
    if name in {"vjepa2_ac", "vjepa_ac", "official", "official_vjepa2_ac"}:
        # Values from vjepa2/configs/train/vitg16/droid-256px-8f.yaml.  The
        # upstream transform stretches its sampled crop to a square; this
        # adapter keeps the sampled crop's pixel aspect ratio via letterboxing.
        kwargs.setdefault("random_horizontal_flip", False)
        kwargs.setdefault("random_resize_aspect_ratio", (0.75, 1.35))
        kwargs.setdefault("random_resize_scale", (1.777, 1.777))
        return VJEPA2ACVideoTransform(**kwargs)
    if name in {"prts", "prts_crop_rotate", "crop_rotate", "prts_crop_and_rotate"}:
        kwargs.setdefault("crop_scale", (0.95, 1.0))
        kwargs.setdefault("rotation_degrees", (-3.0, 3.0))
        return PRTSCropRotateVideoTransform(**kwargs)
    raise ValueError(f"Unknown DROID augmentation {augmentation!r}")


def build_vjepa2_ac_transform(**kwargs: Any) -> VJEPA2ACVideoTransform:
    """Build the official-configured, aspect-preserving AC reference transform."""

    return build_droid_video_transform("vjepa2_ac", **kwargs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Packed T5 cache
# ---------------------------------------------------------------------------


class PackedT5EmbeddingCache:
    """Read a prompt-indexed, contiguous T5 embedding cache.

    ``embeddings.bin`` stores rows in ``[num_prompts, context_length, dim]``
    order.  ``index.json`` maps the complete prompt string directly to a row;
    no digest is involved, so a cache hit can never silently alias two prompts.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        context_length: int = DROID_T5_CONTEXT_LENGTH,
        *,
        context_len: int | None = None,
        mmap: bool = True,
    ) -> None:
        if context_len is not None:
            if context_length != DROID_T5_CONTEXT_LENGTH and context_length != context_len:
                raise ValueError("context_length and context_len disagree")
            context_length = int(context_len)
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        index_path = self.cache_dir / "index.json"
        if not manifest_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"DROID T5 cache requires {manifest_path} and {index_path}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_shape = self.manifest.get("shape")
        manifest_context = self.manifest.get("context_length", self.manifest.get("context_len"))
        if manifest_context is None and isinstance(manifest_shape, (list, tuple)) and len(manifest_shape) == 3:
            manifest_context = manifest_shape[1]
        if int(manifest_context if manifest_context is not None else -1) != int(context_length):
            raise ValueError(
                f"Cache context length {manifest_context!r} does not "
                f"match requested {context_length}"
            )
        self.context_length = int(context_length)
        self.context_len = self.context_length
        manifest_dim = self.manifest.get("embedding_dim")
        if manifest_dim is None and isinstance(manifest_shape, (list, tuple)) and len(manifest_shape) == 3:
            manifest_dim = manifest_shape[-1]
        self.embedding_dim = int(manifest_dim or 0)
        if self.embedding_dim <= 0:
            raise ValueError("Cache manifest has an invalid embedding_dim")
        self.dtype = np.dtype(self.manifest.get("dtype", "float16"))
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        raw_index = index_payload.get("prompts", index_payload)
        if not isinstance(raw_index, dict):
            raise ValueError("DROID cache index must map prompts to rows")
        self._index: dict[str, tuple[int, int]] = {}
        for prompt, record in raw_index.items():
            if isinstance(record, int):
                row, length = record, self.context_length
            elif isinstance(record, Mapping):
                row = int(record.get("row", -1))
                length = int(record.get("length", record.get("valid_length", -1)))
            else:
                raise ValueError(f"Invalid cache index record for prompt {prompt!r}")
            if not 0 <= row < int(self.manifest.get("num_prompts", len(raw_index))):
                raise ValueError(
                    f"Cache row {row} for prompt {prompt!r} is out of range"
                )
            if not 0 <= length <= self.context_length:
                raise ValueError(f"Invalid token length {length} for prompt {prompt!r}")
            self._index[str(prompt)] = (row, length)
        if len(self._index) != int(self.manifest.get("num_prompts", len(self._index))):
            raise ValueError("Cache manifest/index prompt counts disagree")
        embedding_name = self.manifest.get("embedding_file", "embeddings.bin")
        embedding_path = self.cache_dir / embedding_name
        expected_shape = (len(self._index), self.context_length, self.embedding_dim)
        expected_bytes = int(np.prod(expected_shape)) * self.dtype.itemsize
        if (
            not embedding_path.is_file()
            or embedding_path.stat().st_size != expected_bytes
        ):
            raise ValueError(
                f"Invalid packed embedding file {embedding_path}: expected {expected_bytes} bytes"
            )
        mode = "r" if mmap else "r+"
        self._embeddings = np.memmap(
            embedding_path,
            mode=mode,
            dtype=self.dtype,
            shape=expected_shape,
            order="C",
        )

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, prompt: str) -> bool:
        return prompt in self._index

    def load(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            row, length = self._index[prompt]
        except KeyError as error:
            raise KeyError(
                f"Prompt is absent from DROID T5 cache: {prompt!r}"
            ) from error
        # Clone because np.memmap is read-only and the training collator may
        # move/ cast the returned tensor in place.
        context = torch.from_numpy(np.array(self._embeddings[row], copy=True))
        mask = torch.zeros(self.context_length, dtype=torch.bool)
        mask[:length] = True
        return context, mask

    __getitem__ = load

    get = load

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # np.memmap keeps an open file descriptor and is not portable across
        # DataLoader spawn workers; workers reopen the same read-only mapping.
        state.pop("_embeddings", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        cache_dir = state["cache_dir"]
        context_length = int(state["context_length"])
        self.__init__(cache_dir, context_length=context_length)

    def close(self) -> None:
        mmap_obj = getattr(self._embeddings, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()


# ---------------------------------------------------------------------------
# LeRobot-backed dataset
# ---------------------------------------------------------------------------


def _camera_key(key: str) -> str:
    return key if key.startswith("observation.images.") else f"observation.images.{key}"


class DroidPretrainingDataset(Dataset):
    """Direct, episode-oriented DROID loader for predictor-only pre-training.

    The generic LeRobot dataset eagerly concatenates every parquet file and
    validates all 17.9M timestamps during construction.  That is an avoidable
    multi-minute startup and also makes a worker decode DROID's unused third
    camera.  This loader reads the small metadata/index files up front, then
    loads one episode parquet table and two video decoders per worker as needed.
    """

    def __init__(
        self,
        root: str | os.PathLike[str] = DROID_ROOT,
        *,
        repo_id: str | None = None,
        video_keys: Sequence[str] | None = None,
        views: Sequence[str] | None = None,
        state_key: str = DROID_STATE_KEY,
        episodes: list[int] | None = None,
        video_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        augmentation: str = "prts_crop_rotate",
        text_cache: PackedT5EmbeddingCache | str | os.PathLike[str] | None = None,
        prompt_template: str = DROID_PROMPT,
        instruction_chooser: Callable[[Sequence[str]], str] | None = None,
        normalize_state: bool = True,
        q01: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        q99: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        video_backend: str | None = None,
        frame_decoder: Callable[[Path, Sequence[int]], torch.Tensor] | None = None,
    ) -> None:
        root_path = Path(root)
        self.root = root_path
        self.repo_id = repo_id or root_path.name
        if video_keys is not None and views is not None and tuple(video_keys) != tuple(views):
            raise ValueError("video_keys and views disagree")
        selected_views = video_keys if video_keys is not None else (views or DROID_VIDEO_KEYS)
        self.video_keys = tuple(_camera_key(key) for key in selected_views)
        if len(self.video_keys) != 2:
            raise ValueError(
                f"DROID pretraining expects exactly two views, got {self.video_keys}"
            )
        self.state_key = state_key
        self._sampling_num_frames = DROID_NUM_FRAMES
        self._video_offsets = DROID_VIDEO_OFFSETS
        self._state_offsets = (0,)
        self.instruction_chooser = instruction_chooser
        self.prompt_template = prompt_template
        self.normalize_state = bool(normalize_state)
        self.video_transform = video_transform or build_droid_video_transform(
            augmentation
        )
        if isinstance(text_cache, (str, os.PathLike)):
            text_cache = PackedT5EmbeddingCache(text_cache)
        self.text_cache = text_cache
        self.video_backend = (video_backend or "auto").lower()
        if self.video_backend not in {"auto", "torchcodec", "pyav"}:
            raise ValueError(
                f"Unsupported DROID video backend {video_backend!r}; choose auto, torchcodec or pyav"
            )
        self.frame_decoder = frame_decoder

        info_path = root_path / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing DROID metadata: {info_path}")
        self.info = load_droid_info(root_path)
        self.fps = int(self.info.get("fps", -1))
        if self.fps != DROID_FPS:
            raise ValueError(
                f"DROID pretraining expects {DROID_FPS} fps, found {self.fps}"
            )
        self.features = self.info.get("features", {})
        for key in (*self.video_keys, state_key, *DROID_INSTRUCTION_KEYS):
            if key not in self.features:
                raise KeyError(f"Feature {key!r} not found in {root_path}")
        state_shape = tuple(self.features[state_key].get("shape", ()))
        if len(state_shape) != 1:
            raise ValueError(
                f"Expected flat DROID state feature, got shape {state_shape}"
            )
        self.state_dim = int(state_shape[0])
        if self.normalize_state and (q01 is None or q99 is None):
            loaded_q01, loaded_q99 = load_droid_state_quantiles(root_path, state_key)
            q01 = loaded_q01 if q01 is None else q01
            q99 = loaded_q99 if q99 is None else q99
        elif not self.normalize_state:
            # Quantiles are not consulted in identity-state mode; still keep
            # shape-compatible buffers for callers that inspect the dataset.
            q01 = np.zeros(self.state_dim, dtype=np.float32) if q01 is None else q01
            q99 = np.ones(self.state_dim, dtype=np.float32) if q99 is None else q99
        self.q01 = torch.as_tensor(q01, dtype=torch.float32).flatten()
        self.q99 = torch.as_tensor(q99, dtype=torch.float32).flatten()
        if self.q01.numel() != self.state_dim or self.q99.shape != self.q01.shape:
            raise ValueError(
                f"State quantiles have shape {tuple(self.q01.shape)}, expected ({self.state_dim},)"
            )

        # Episode metadata is tiny (~7 MB) compared with loading all parquet
        # rows.  Preserve caller episode order, matching LeRobot's semantics.
        episode_records: dict[int, dict[str, Any]] = {}
        episodes_path = root_path / "meta" / "episodes.jsonl"
        with episodes_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                episode_records[int(record["episode_index"])] = record
        selected_ids = (
            list(episode_records) if episodes is None else [int(ep) for ep in episodes]
        )
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("episodes must not contain duplicates")
        missing = [ep for ep in selected_ids if ep not in episode_records]
        if missing:
            raise KeyError(f"Unknown DROID episode(s): {missing[:5]}")
        self.episode_ids = selected_ids
        self._episode_position = {
            episode_id: position for position, episode_id in enumerate(selected_ids)
        }
        self.episode_lengths = [
            int(episode_records[ep]["length"]) for ep in selected_ids
        ]
        local_starts: list[int] = []
        local_ends: list[int] = []
        running = 0
        for length in self.episode_lengths:
            local_starts.append(running)
            running += length
            local_ends.append(running)
        self.episode_data_index = {
            "from": torch.tensor(local_starts, dtype=torch.long),
            "to": torch.tensor(local_ends, dtype=torch.long),
        }
        self._full_window_episode_starts = local_starts
        self._full_window_cumulative_sizes: list[int] = []
        self._build_full_window_index(local_ends=local_ends)
        # A light metadata object keeps familiar ``dataset.meta.features`` and
        # ``dataset.meta.fps`` access available without constructing LeRobot.
        self.meta = SimpleNamespace(
            features=self.features, fps=self.fps, episodes=episode_records
        )
        self._data_path_template = self.info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        self._video_path_template = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        self._chunk_size = int(self.info.get("chunks_size", 1000))
        self._table_columns = [
            *DROID_INSTRUCTION_KEYS,
            state_key,
            "episode_index",
            "timestamp",
        ]
        self._current_episode: int | None = None
        self._current_table: Any | None = None
        self._av_containers: dict[str, Any] = {}
        self._torchcodec_decoders: dict[str, Any] = {}
        self._torchcodec_disabled = False

    @property
    def video_offsets(self) -> tuple[int, ...]:
        return self._video_offsets

    @property
    def sampling_num_frames(self) -> int:
        return self._sampling_num_frames

    @property
    def context_length(self) -> int:
        return DROID_T5_CONTEXT_LENGTH

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ids)

    def _frame_index(self, index: int) -> int:
        return compact_droid_index(
            index,
            self._full_window_episode_starts,
            self._full_window_cumulative_sizes,
        )

    def _build_full_window_index(
        self,
        *,
        local_ends: Sequence[int] | None = None,
    ) -> None:
        """Rebuild compact indexing from the episode data-index metadata."""

        starts_raw = self.episode_data_index["from"]
        ends_raw = self.episode_data_index["to"]
        starts = starts_raw.tolist() if hasattr(starts_raw, "tolist") else list(starts_raw)
        ends = (
            ends_raw.tolist() if hasattr(ends_raw, "tolist") else list(ends_raw)
            if local_ends is None
            else [int(value) for value in local_ends]
        )
        self._full_window_episode_starts, self._full_window_cumulative_sizes = (
            build_droid_window_index(starts, ends, DROID_MAX_FUTURE_OFFSET)
        )

    def _episode_frame(self, index: int) -> tuple[int, int]:
        absolute = self._frame_index(index)
        position = bisect.bisect_right(self._full_window_episode_starts, absolute) - 1
        if position < 0:
            raise RuntimeError(f"Could not locate frame {absolute} in episode index")
        return self.episode_ids[position], absolute - self._full_window_episode_starts[
            position
        ]

    def __len__(self) -> int:
        return (
            self._full_window_cumulative_sizes[-1]
            if self._full_window_cumulative_sizes
            else 0
        )

    def _episode_data_path(self, episode_id: int) -> Path:
        return self.root / self._data_path_template.format(
            episode_chunk=episode_id // self._chunk_size,
            episode_index=episode_id,
        )

    def _episode_video_path(self, episode_id: int, key: str) -> Path:
        return self.root / self._video_path_template.format(
            episode_chunk=episode_id // self._chunk_size,
            episode_index=episode_id,
            video_key=key,
        )

    def _load_episode_table(self, episode_id: int) -> Any:
        if self._current_episode == episode_id and self._current_table is not None:
            return self._current_table
        # A worker changes episodes over time; release decoder handles before
        # replacing the table to avoid exhausting file descriptors.
        self._close_video_handles()
        import pyarrow.parquet as parquet

        path = self._episode_data_path(episode_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing DROID episode parquet: {path}")
        self._current_table = parquet.read_table(path, columns=self._table_columns)
        self._current_episode = episode_id
        return self._current_table

    @staticmethod
    def _table_value(table: Any, key: str, row: int) -> Any:
        return table[key][row].as_py()

    @staticmethod
    def _normalise_decoded_frames(frames: Any, expected: int) -> torch.Tensor:
        value = torch.as_tensor(frames)
        if value.ndim != 4:
            raise ValueError(
                f"Video decoder returned {tuple(value.shape)}, expected 4 dimensions"
            )
        if value.shape[0] != expected:
            raise ValueError(
                f"Video decoder returned {value.shape[0]} frames, expected {expected}"
            )
        if value.shape[1] in (1, 3, 4):
            result = value
        elif value.shape[-1] in (1, 3, 4):
            result = value.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"Cannot infer decoder channel layout {tuple(value.shape)}"
            )
        result = result.float()
        if result.numel() and float(result.max()) > 1.0:
            result = result / 255.0
        return result.contiguous()

    def _decode_with_torchcodec(
        self, path: Path, indices: Sequence[int]
    ) -> torch.Tensor | None:
        if self._torchcodec_disabled or self.video_backend == "pyav":
            return None
        try:
            from torchcodec.decoders import VideoDecoder

            key = str(path)
            decoder = self._torchcodec_decoders.get(key)
            if decoder is None:
                decoder = VideoDecoder(str(path), device="cpu", seek_mode="approximate")
                self._torchcodec_decoders[key] = decoder
            frames = decoder.get_frames_at(indices=[int(i) for i in indices]).data
            return self._normalise_decoded_frames(frames, len(indices))
        except Exception:
            # Torchcodec is optional and can be importable while its FFmpeg
            # shared library is unavailable.  Disable it for this worker and
            # use the deterministic PyAV path below.
            self._torchcodec_disabled = True
            self._torchcodec_decoders.clear()
            return None

    def _decode_with_pyav(self, path: Path, indices: Sequence[int]) -> torch.Tensor:
        import av

        requested = [max(0, int(i)) for i in indices]
        if not requested:
            return torch.empty((0, 3, 0, 0), dtype=torch.float32)
        key = str(path)
        container = self._av_containers.get(key)
        try:
            if container is None:
                container = av.open(str(path))
                self._av_containers[key] = container
            stream = container.streams.video[0]
            fps = float(stream.average_rate or DROID_FPS)
            first, last = min(requested), max(requested)
            # Seek to a preceding keyframe.  PyAV accepts stream time-base
            # units for ``offset`` when a stream is provided.
            seek_pts = int(max(0.0, first / fps) / float(stream.time_base))
            container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
            found: dict[int, torch.Tensor] = {}
            for frame in container.decode(stream):
                frame_index = int(round(float(frame.time) * fps))
                image = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(
                    2, 0, 1
                )
                image = image.float() / 255.0
                if frame_index in requested:
                    found.setdefault(frame_index, image)
                if len(found) == len(set(requested)) and frame_index >= last:
                    break
                if frame_index > last + 2:
                    break
            if len(found) < len(set(requested)):
                # A seek around a sparse/keyframe-only stream may begin after
                # the requested point.  Reopen and perform one bounded decode
                # from the beginning rather than returning a wrong frame.
                container.close()
                container = av.open(str(path))
                self._av_containers[key] = container
                stream = container.streams.video[0]
                all_frames: dict[int, torch.Tensor] = {}
                for frame in container.decode(stream):
                    frame_index = int(round(float(frame.time) * fps))
                    if frame_index in requested:
                        all_frames.setdefault(
                            frame_index,
                            torch.from_numpy(frame.to_ndarray(format="rgb24"))
                            .permute(2, 0, 1)
                            .float()
                            / 255.0,
                        )
                    if frame_index > last:
                        break
                found.update(all_frames)
            if len(found) < len(set(requested)):
                raise RuntimeError(
                    f"Could not decode requested frames {requested} from {path}; found {sorted(found)}"
                )
            return torch.stack([found[i] for i in requested])
        except Exception:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            self._av_containers.pop(key, None)
            raise

    def _decode_frames(self, path: Path, indices: Sequence[int]) -> torch.Tensor:
        if self.frame_decoder is not None:
            return self._normalise_decoded_frames(
                self.frame_decoder(path, indices), len(indices)
            )
        decoded = self._decode_with_torchcodec(path, indices)
        return decoded if decoded is not None else self._decode_with_pyav(path, indices)

    def _close_video_handles(self) -> None:
        for container in self._av_containers.values():
            try:
                container.close()
            except Exception:
                pass
        self._av_containers.clear()
        self._torchcodec_decoders.clear()

    def __del__(self) -> None:
        try:
            self._close_video_handles()
        except Exception:
            pass

    def close(self) -> None:
        """Release the worker-local video decoder handles."""

        self._close_video_handles()

    def __getstate__(self) -> dict[str, Any]:
        """Keep DataLoader worker pickling free of live FFmpeg handles."""

        state = dict(self.__dict__)
        state["_current_episode"] = None
        state["_current_table"] = None
        state["_av_containers"] = {}
        state["_torchcodec_decoders"] = {}
        return state

    @staticmethod
    def _as_cthw(value: torch.Tensor, expected_frames: int) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(
                f"Expected transformed clip with 4 dimensions, got {tuple(value.shape)}"
            )
        if value.shape[0] == expected_frames and value.shape[1] in (1, 3, 4):
            return value.permute(1, 0, 2, 3).contiguous()
        if value.shape[0] in (1, 3, 4) and value.shape[1] == expected_frames:
            return value.contiguous()
        if value.shape[-1] in (1, 3, 4) and value.shape[0] == expected_frames:
            return value.permute(3, 0, 1, 2).contiguous()
        raise ValueError(
            f"Cannot infer transformed clip layout for {tuple(value.shape)}; "
            f"expected {expected_frames} frames"
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_id, frame_offset = self._episode_frame(index)
        table = self._load_episode_table(episode_id)
        length = self.episode_lengths[self._episode_position[episode_id]]
        raw_indices = [frame_offset + offset for offset in self.video_offsets]
        clamped_indices = [min(length - 1, max(0, value)) for value in raw_indices]
        clips: list[torch.Tensor] = []
        video_pad = torch.tensor(
            [
                [value < 0 or value >= length for value in raw_indices]
                for _ in self.video_keys
            ],
            dtype=torch.bool,
        )
        for key in self.video_keys:
            path = self._episode_video_path(episode_id, key)
            clip = self._decode_frames(path, clamped_indices)
            clips.append(clip)

        # Sample spatial augmentation once for the *paired* multi-view clip.
        # Concatenating views along the temporal axis lets a clip transform
        # share one crop/rotation (and any other random parameters) across all
        # frames and both cameras.  Split only after the transform so the
        # predictor still receives [views, C, T, H, W].
        if self.video_transform is not None:
            first_shape = tuple(clips[0].shape[1:])
            if any(tuple(clip.shape[1:]) != first_shape for clip in clips[1:]):
                raise ValueError(
                    "DROID views must have matching channel/height/width for "
                    "paired clip augmentation"
                )
            transformed = self.video_transform(torch.cat(clips, dim=0))
            transformed_cthw = self._as_cthw(
                transformed, len(self.video_offsets) * len(self.video_keys)
            )
            channels, _, height, width = transformed_cthw.shape
            transformed_cthw = transformed_cthw.reshape(
                channels, len(self.video_keys), len(self.video_offsets), height, width
            )
            video = transformed_cthw.permute(1, 0, 2, 3, 4).contiguous()
        else:
            video = torch.stack(
                [self._as_cthw(clip, len(self.video_offsets)) for clip in clips],
                dim=0,
            )

        state = torch.as_tensor(
            self._table_value(table, self.state_key, frame_offset), dtype=torch.float32
        ).reshape(-1)
        if self.normalize_state:
            state = quantile_normalize_droid_state(state, self.q01, self.q99)
        row_item = {
            key: self._table_value(table, key, frame_offset)
            for key in DROID_INSTRUCTION_KEYS
        }
        instruction = sample_droid_instruction(
            row_item, chooser=self.instruction_chooser
        )
        prompt = droid_prompt(instruction, self.prompt_template, canonicalize=False)
        result: dict[str, Any] = {
            "video": video,
            "proprio": state,
            "instruction": instruction,
            "task": instruction,
            "prompt": prompt,
            "index": int(index),
            "absolute_index": int(self._frame_index(index)),
            "frame_index": int(frame_offset),
            "episode_index": int(episode_id),
            "video_is_pad": video_pad,
            "image_is_pad": video_pad,
            "proprio_is_pad": torch.tensor(False),
        }
        result.update(row_item)
        timestamp = self._table_value(table, "timestamp", frame_offset)
        if timestamp is not None:
            result["timestamp"] = float(timestamp)
        if self.text_cache is not None:
            result["context"], result["context_mask"] = self.text_cache.load(prompt)
        return result


__all__ = [
    "DROID_ROOT",
    "DROID_BENCH_CACHE_ROOT",
    "DROID_REPO_ID",
    "DROID_FPS",
    "DROID_TARGET_FPS",
    "DROID_SOURCE_FRAME_STRIDE",
    "DROID_VIDEO_KEYS",
    "DROID_VIEWS",
    "DROID_INSTRUCTION_KEYS",
    "DROID_INSTRUCTION_FIELDS",
    "DROID_STATE_KEY",
    "DROID_VIDEO_OFFSETS",
    "DROID_CONTEXT_OFFSETS",
    "DROID_FUTURE_OFFSETS",
    "DROID_NUM_FRAMES",
    "DROID_TUBELET_SIZE",
    "DROID_CONTEXT_TUBELETS",
    "DROID_FUTURE_FRAMES",
    "DROID_MAX_FUTURE_OFFSET",
    "DROID_T5_CONTEXT_LENGTH",
    "DROID_T5_MAX_LENGTH",
    "DROID_CONTEXT_LENGTH",
    "DROID_PROMPT_TEMPLATE",
    "DROID_PROMPT",
    "make_droid_video_offsets",
    "load_droid_info",
    "canonicalize_droid_instruction",
    "canonicalize_instruction",
    "droid_instruction_candidates",
    "sample_droid_instruction",
    "droid_prompt",
    "format_droid_prompt",
    "load_droid_state_quantiles",
    "quantile_normalize_droid_state",
    "droid_state_normalize",
    "build_droid_window_index",
    "build_full_window_index",
    "compact_droid_index",
    "compact_window_index",
    "DroidLetterboxTransform",
    "VJEPA2ACVideoTransform",
    "OfficialVJEPA2ACTransform",
    "VJEPA2ACTransform",
    "DroidVJEPA2ACTransform",
    "PRTSCropRotateVideoTransform",
    "PRTSCropRotateTransform",
    "build_droid_video_transform",
    "build_vjepa2_ac_transform",
    "PackedT5EmbeddingCache",
    "DroidPretrainingDataset",
]
