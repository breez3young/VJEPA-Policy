from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F


class VideoClipTransform:
    """Resize and normalize a complete ``[T, C, H, W]`` video clip.

    Because the temporal dimension is treated as the batch dimension, every
    frame receives the same deterministic spatial transform. Callers can pass
    any callable with the same input/output contract to the dataset instead.
    """

    def __init__(
        self,
        size: tuple[int, int],
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
        resize_mode: str = "stretch",
    ) -> None:
        self.size = tuple(size)
        if len(self.size) != 2 or any(side <= 0 for side in self.size):
            raise ValueError(f"Resize target must contain two positive values, got {size}")
        if resize_mode not in ("stretch", "letterbox"):
            raise ValueError(f"Unsupported resize mode: {resize_mode!r}")
        self.resize_mode = resize_mode
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)

    @property
    def normalized_black(self) -> torch.Tensor:
        """Per-channel value of an RGB-black pixel after normalization."""
        return (-self.mean / self.std).flatten()

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 4:
            raise ValueError(f"Expected clip [T, C, H, W], got {tuple(clip.shape)}")
        clip = clip.float()
        if clip.numel() and clip.max() > 1.0:
            clip = clip / 255.0
        resize_size = self.size
        if self.resize_mode == "letterbox":
            source_height, source_width = clip.shape[-2:]
            target_height, target_width = self.size
            scale = min(target_height / source_height, target_width / source_width)
            resize_size = (
                min(target_height, max(1, round(source_height * scale))),
                min(target_width, max(1, round(source_width * scale))),
            )
        clip = F.interpolate(
            clip,
            size=resize_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        if self.resize_mode == "letterbox":
            pad_height = self.size[0] - resize_size[0]
            pad_width = self.size[1] - resize_size[1]
            clip = F.pad(
                clip,
                (
                    pad_width // 2,
                    pad_width - pad_width // 2,
                    pad_height // 2,
                    pad_height - pad_height // 2,
                ),
                value=0.0,
            )
        mean = self.mean.to(device=clip.device, dtype=clip.dtype)
        std = self.std.to(device=clip.device, dtype=clip.dtype)
        return (clip - mean) / std


class QuadrantViewCombiner:
    """Place up to four transformed views in a fixed 2x2 canvas.

    Views are assigned to top-left, top-right, bottom-left, and bottom-right in
    input order. Each input is resized to one quadrant; unoccupied quadrants
    use the transformed value of an RGB-black pixel.
    """

    def __init__(
        self,
        size: tuple[int, int],
        fill_value: float | Sequence[float] = -1.0,
    ) -> None:
        if len(size) != 2 or any(side <= 0 or side % 2 for side in size):
            raise ValueError(f"Quadrant canvas size must contain two positive even values, got {size}")
        self.size = tuple(size)
        self.fill_value = torch.as_tensor(fill_value, dtype=torch.float32).flatten()

    def __call__(self, clips: Sequence[torch.Tensor]) -> torch.Tensor:
        if not 1 <= len(clips) <= 4:
            raise ValueError(f"Quadrant composition requires 1 to 4 views, got {len(clips)}")
        first = clips[0]
        if first.ndim != 4:
            raise ValueError(f"Expected view [T,C,H,W], got {tuple(first.shape)}")
        time, channels = first.shape[:2]
        if self.fill_value.numel() not in (1, channels):
            raise ValueError(
                f"fill_value has {self.fill_value.numel()} channels, but views have {channels}"
            )

        height, width = self.size
        half_height, half_width = height // 2, width // 2
        fill = self.fill_value.to(device=first.device, dtype=first.dtype)
        if fill.numel() == 1:
            fill = fill.expand(channels)
        canvas = fill.view(1, channels, 1, 1).expand(time, -1, height, width).clone()
        slots = (
            (0, 0),
            (0, half_width),
            (half_height, 0),
            (half_height, half_width),
        )
        for clip, (top, left) in zip(clips, slots[: len(clips)], strict=True):
            if clip.ndim != 4 or clip.shape[:2] != (time, channels):
                raise ValueError(
                    "All views must share [T,C]; "
                    f"expected {(time, channels)}, got {tuple(clip.shape[:2])}"
                )
            if clip.shape[-2:] != (half_height, half_width):
                clip = F.interpolate(
                    clip,
                    size=(half_height, half_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            canvas[:, :, top : top + half_height, left : left + half_width] = clip
        return canvas


def combine_video_clips(
    clips: Sequence[torch.Tensor],
    mode: str | Callable[[Sequence[torch.Tensor]], torch.Tensor] = "independent",
) -> torch.Tensor:
    """Combine transformed views into a canvas or an independent-view tensor.

    Canvas layouts return ``[C,T,H,W]``. The ``independent`` layout preserves
    each view and returns ``[V,C,T,H,W]``.
    """
    if not clips:
        raise ValueError("At least one video clip is required")
    if callable(mode):
        combined = mode(clips)
    elif mode == "independent":
        first_shape = clips[0].shape
        if any(clip.ndim != 4 or clip.shape != first_shape for clip in clips):
            raise ValueError("Independent views must share the same [T,C,H,W] shape")
        return torch.stack(list(clips)).permute(0, 2, 1, 3, 4).contiguous()
    elif len(clips) == 1:
        combined = clips[0]
    elif mode == "horizontal":
        combined = torch.cat(list(clips), dim=-1)
    elif mode == "vertical":
        combined = torch.cat(list(clips), dim=-2)
    else:
        raise ValueError(f"Unsupported multi-view combine mode: {mode!r}")

    if combined.ndim != 4:
        raise ValueError(f"View combiner must return [T,C,H,W], got {tuple(combined.shape)}")
    return combined.permute(1, 0, 2, 3).contiguous()


def build_patch_valid_mask(
    crop_size: int,
    patch_size: int,
    tubelet_size: int,
    video_frames: int,
    num_views: int,
) -> torch.Tensor:
    """Mark real-view tokens in a temporal-major 2x2 quadrant token grid."""
    if not 1 <= num_views <= 4:
        raise ValueError(f"num_views must be in [1, 4], got {num_views}")
    if crop_size <= 0 or patch_size <= 0 or crop_size % (2 * patch_size):
        raise ValueError("crop_size must be divisible by 2 * patch_size")
    if tubelet_size <= 0 or video_frames <= 0 or video_frames % tubelet_size:
        raise ValueError("video_frames must be positive and divisible by tubelet_size")

    grid = crop_size // patch_size
    rows = torch.arange(grid).unsqueeze(1)
    columns = torch.arange(grid).unsqueeze(0)
    quadrants = (rows // (grid // 2)) * 2 + columns // (grid // 2)
    per_tubelet = (quadrants < num_views).flatten()
    return per_tubelet.repeat(video_frames // tubelet_size)
