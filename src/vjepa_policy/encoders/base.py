"""Common contracts for frozen latent encoders used by VJEPA-Policy.

The policy consumes a compact logical token lattice.  An encoder may use a
different RGB patch size or temporal compression, but it must expose the
resulting layout explicitly instead of relying on a padded/fake lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class LatentLayout:
    """A fixed token layout for one training/serving configuration.

    Token ids use ``(time, view, height, width)`` order.  ``context_steps``
    and ``target_steps`` refer to the compact temporal axis; physical frame
    offsets, when relevant, are kept in ``temporal_positions`` for metadata.
    """

    grid_depth: int
    grid_height: int
    grid_width: int
    num_views: int
    context_steps: tuple[int, ...] = (0,)
    target_steps: tuple[int, ...] | None = None
    temporal_positions: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        dimensions = (
            self.grid_depth,
            self.grid_height,
            self.grid_width,
            self.num_views,
        )
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError(f"layout dimensions must be positive integers: {dimensions}")
        context = tuple(int(value) for value in self.context_steps)
        target = (
            tuple(index for index in range(self.grid_depth) if index not in context)
            if self.target_steps is None
            else tuple(int(value) for value in self.target_steps)
        )
        if not context or not target:
            raise ValueError("layout must contain non-empty context and target steps")
        if len(set(context)) != len(context) or len(set(target)) != len(target):
            raise ValueError("context/target steps must not contain duplicates")
        if set(context) & set(target):
            raise ValueError("context and target steps must be disjoint")
        if any(value < 0 or value >= self.grid_depth for value in (*context, *target)):
            raise ValueError("context/target step is outside grid_depth")
        if set(context) | set(target) != set(range(self.grid_depth)):
            raise ValueError(
                "context_steps and target_steps must cover the compact temporal layout"
            )
        positions = self.temporal_positions
        if positions is not None:
            positions = tuple(int(value) for value in positions)
            if len(positions) != self.grid_depth:
                raise ValueError(
                    "temporal_positions must have one entry per compact temporal step"
                )
        object.__setattr__(self, "context_steps", context)
        object.__setattr__(self, "target_steps", target)
        object.__setattr__(self, "temporal_positions", positions)

    @property
    def patches_per_step(self) -> int:
        return self.num_views * self.grid_height * self.grid_width

    @property
    def total_tokens(self) -> int:
        return self.grid_depth * self.patches_per_step

    @property
    def context_tokens(self) -> int:
        return len(self.context_steps) * self.patches_per_step

    @property
    def target_tokens(self) -> int:
        return len(self.target_steps) * self.patches_per_step

    def token_ids(self, steps: tuple[int, ...], *, device=None) -> torch.Tensor:
        """Return flattened global ids for the supplied temporal steps."""
        all_ids = torch.arange(self.total_tokens, dtype=torch.long, device=device)
        all_ids = all_ids.reshape(
            self.grid_depth,
            self.num_views,
            self.grid_height,
            self.grid_width,
        )
        return all_ids[list(steps)].reshape(-1)

    def masks(self, *, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.token_ids(self.context_steps, device=device),
            self.token_ids(self.target_steps, device=device),
        )

    def validate_mask(self, mask: torch.Tensor, *, target: bool) -> None:
        """Validate a batched global mask against this compact token lattice."""
        if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
            raise ValueError("mask must be a [B,K] tensor")
        if mask.dtype == torch.bool or mask.is_floating_point():
            raise TypeError("mask indices must use an integer dtype")
        expected = self.target_tokens if target else self.context_tokens
        if mask.shape[1] != expected:
            kind = "target" if target else "context"
            raise ValueError(
                f"{kind} mask selects {mask.shape[1]} tokens; layout requires {expected}"
            )
        if mask.numel() and (mask.min() < 0 or mask.max() >= self.total_tokens):
            raise ValueError("mask contains an id outside the compact token lattice")
        steps = self.target_steps if target else self.context_steps
        expected_ids = self.token_ids(steps, device=mask.device).sort().values
        if not torch.equal(mask.sort(dim=1).values, expected_ids.expand(mask.shape[0], -1)):
            kind = "target" if target else "context"
            raise ValueError(f"{kind} mask ids do not match the declared layout")

    def to_dict(self) -> dict:
        return {
            "grid": [self.grid_depth, self.grid_height, self.grid_width],
            "num_views": self.num_views,
            "context_steps": list(self.context_steps),
            "target_steps": list(self.target_steps),
            "temporal_positions": (
                list(self.temporal_positions)
                if self.temporal_positions is not None
                else None
            ),
        }


@dataclass(frozen=True)
class EncoderSpec:
    """Serializable construction and geometry metadata for one encoder."""

    name: str
    family: str
    model_name: str
    checkpoint_key: str
    input_patch_size: int
    # Patch size used by the frozen backbone itself.  ``input_patch_size``
    # remains the logical policy-grid patch size for compatibility with the
    # predictor and mask builders; adapters such as DINOv2 native16 have
    # different source and logical patch sizes.
    source_patch_size: int | None = None
    temporal_mode: str = "tubelet"
    temporal_stride: int = 2
    temporal_offset: int = 0
    latent_spatial_grid: tuple[int, int] | None = None
    interpolate_rope: bool = True
    canonical_spatial_grid: tuple[int, int] | None = None

    def spatial_grid(self, image_size: tuple[int, int]) -> tuple[int, int]:
        if self.latent_spatial_grid is not None:
            return tuple(self.latent_spatial_grid)
        height, width = image_size
        if height % self.input_patch_size or width % self.input_patch_size:
            raise ValueError(
                f"image_size={image_size} is not divisible by input patch size "
                f"{self.input_patch_size} for {self.name}"
            )
        return height // self.input_patch_size, width // self.input_patch_size

    def temporal_grid(self, video_frames: int) -> int:
        if video_frames <= 0:
            raise ValueError("video_frames must be positive")
        if self.temporal_mode == "tubelet":
            if video_frames % self.temporal_stride:
                raise ValueError(
                    f"video_frames={video_frames} is not divisible by tubelet "
                    f"size {self.temporal_stride} for {self.name}"
                )
            return video_frames // self.temporal_stride
        if self.temporal_mode == "frame":
            return video_frames
        if self.temporal_mode in {"sampled_frame", "frame_stride"}:
            if self.temporal_stride <= 0:
                raise ValueError("temporal_stride must be positive")
            if self.temporal_offset < 0:
                raise ValueError("temporal_offset must be non-negative")
            effective_offset = min(self.temporal_offset, video_frames - 1)
            return 1 + (video_frames - 1 - effective_offset) // self.temporal_stride
        if self.temporal_mode == "causal_stride":
            if self.temporal_stride <= 0:
                raise ValueError("temporal_stride must be positive")
            return 1 + (video_frames - 1) // self.temporal_stride
        raise ValueError(f"unsupported temporal_mode={self.temporal_mode!r}")

    def layout_for_clip(
        self,
        *,
        video_frames: int,
        image_size: tuple[int, int],
        num_views: int,
        context_steps: int = 1,
    ) -> LatentLayout:
        depth = self.temporal_grid(video_frames)
        height, width = self.spatial_grid(image_size)
        if not 0 < context_steps < depth:
            raise ValueError(
                f"context_steps={context_steps} invalid for latent depth {depth}"
            )
        physical_positions = None
        if self.temporal_mode in {"causal_stride", "sampled_frame", "frame_stride"}:
            offset = min(self.temporal_offset, video_frames - 1)
            physical_positions = tuple(
                offset + index * self.temporal_stride
                for index in range(depth)
            )
        return LatentLayout(
            grid_depth=depth,
            grid_height=height,
            grid_width=width,
            num_views=num_views,
            context_steps=tuple(range(context_steps)),
            target_steps=tuple(range(context_steps, depth)),
            temporal_positions=physical_positions,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "model_name": self.model_name,
            "checkpoint_key": self.checkpoint_key,
            "input_patch_size": self.input_patch_size,
            "source_patch_size": (
                self.source_patch_size
                if self.source_patch_size is not None
                else self.input_patch_size
            ),
            "temporal_mode": self.temporal_mode,
            "temporal_stride": self.temporal_stride,
            "temporal_offset": self.temporal_offset,
            "latent_spatial_grid": (
                list(self.latent_spatial_grid)
                if self.latent_spatial_grid is not None
                else None
            ),
            "interpolate_rope": self.interpolate_rope,
            "canonical_spatial_grid": (
                list(self.canonical_spatial_grid)
                if self.canonical_spatial_grid is not None
                else None
            ),
        }


def layout_from_encoder_output(
    output: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    num_views: int,
    context_steps: int = 1,
) -> LatentLayout:
    """Infer a compact temporal layout from an encoder token tensor.

    This is intentionally a validation/helper path: adapters may expose a
    more precise ``EncoderSpec``, but the actual token count remains the final
    authority and cannot silently be padded into nonexistent time steps.
    """
    if output.ndim != 3:
        raise ValueError(f"encoder output must be [B,N,D], got {tuple(output.shape)}")
    tokens_per_step = num_views * grid_height * grid_width
    if tokens_per_step <= 0 or output.shape[1] % tokens_per_step:
        raise ValueError(
            f"encoder token count {output.shape[1]} is not divisible by per-step "
            f"count {tokens_per_step}"
        )
    depth = output.shape[1] // tokens_per_step
    if not 0 < context_steps < depth:
        raise ValueError(f"context_steps={context_steps} invalid for output depth {depth}")
    return LatentLayout(
        grid_depth=depth,
        grid_height=grid_height,
        grid_width=grid_width,
        num_views=num_views,
        context_steps=tuple(range(context_steps)),
        target_steps=tuple(range(context_steps, depth)),
    )
