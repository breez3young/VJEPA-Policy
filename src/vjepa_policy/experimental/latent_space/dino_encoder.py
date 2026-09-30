"""Frozen DINO image encoders adapted to the V-JEPA policy token geometry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoLatentEncoder(nn.Module):
    """Frozen DINO image adapter with an explicit, compact token lattice.

    DINO is an image encoder, so temporal tokens are obtained by sampling one
    image every ``frame_stride`` frames.  The output grid can either be the
    native backbone grid (for example DINOv2's 16x16 at 224px) or an explicit
    endpoint-aligned resized grid (the historical 14x14 path).  ``num_views``
    is optional and only validates a configured topology; a 6D clip may carry
    any positive number of views when it is left unset.
    """

    embed_dim = 1024
    patch_size = 16
    tubelet_size = 2
    num_frames = 10
    num_views = None
    temporal_mode = "sampled_frame"
    grid_depth = 5
    grid_height = 14
    grid_width = 14
    patches_per_step = grid_height * grid_width
    num_patches_per_view = grid_depth * patches_per_step
    num_patches = None

    _FAMILIES = {
        "dinov2": {"native_patch_size": 14, "prefix_tokens": 1, "register_tokens": 0},
        "dinov3": {"native_patch_size": 16, "prefix_tokens": 5, "register_tokens": 4},
    }

    def __init__(
        self,
        backbone: nn.Module,
        family: str,
        *,
        image_microbatch_size: int | None = None,
        output_grid: tuple[int, int] | None = None,
        interpolate_spatial: bool | None = None,
        logical_patch_size: int | None = None,
        frame_stride: int = 2,
        frame_offset: int = 1,
        num_views: int | None = None,
        input_size: tuple[int, int] = (224, 224),
        video_frames: int = 10,
    ) -> None:
        super().__init__()
        if family not in self._FAMILIES:
            raise ValueError(f"unsupported DINO family {family!r}")
        if image_microbatch_size is not None and image_microbatch_size <= 0:
            raise ValueError("image_microbatch_size must be positive")
        self.family = family
        self.encoder_family = family
        self.image_microbatch_size = image_microbatch_size
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if frame_offset < 0:
            raise ValueError("frame_offset must be non-negative")
        if num_views is not None and (not isinstance(num_views, int) or num_views <= 0):
            raise ValueError("num_views must be a positive integer when configured")
        self.frame_stride = int(frame_stride)
        self.frame_offset = int(frame_offset)
        if not isinstance(video_frames, int) or video_frames <= 0:
            raise ValueError("video_frames must be a positive integer")
        self.num_frames = int(video_frames)
        self.input_size = tuple(int(value) for value in input_size)
        if len(self.input_size) != 2 or any(value <= 0 for value in self.input_size):
            raise ValueError("input_size must contain two positive dimensions")
        self.num_views = num_views
        self.backbone = backbone
        self._vjepa_policy_family = family
        self._vjepa_policy_model_name = f"{family}_vitl"
        self._vjepa_policy_checkpoint_key = "local_pretrained"
        self.blocks = (
            SimpleNamespace(
                attn=SimpleNamespace(
                    interpolate_rope=False,
                    corrected_frequency_pairing=False,
                )
            ),
        )

        expected = self._FAMILIES[family]
        config = getattr(backbone, "config", None)
        if config is None:
            raise ValueError("DINO backbone must expose a Hugging Face-style config")
        required = {
            "hidden_size": self.embed_dim,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "patch_size": expected["native_patch_size"],
        }
        mismatches = {
            name: (getattr(config, name, None), value)
            for name, value in required.items()
            if getattr(config, name, None) != value
        }
        registers = int(getattr(config, "num_register_tokens", 0))
        if registers != expected["register_tokens"]:
            mismatches["num_register_tokens"] = (
                registers,
                expected["register_tokens"],
            )
        if mismatches:
            raise ValueError(f"DINO ViT-L configuration mismatch: {mismatches}")

        self.native_patch_size = expected["native_patch_size"]
        self.num_prefix_tokens = expected["prefix_tokens"]
        native_grid = tuple(value // self.native_patch_size for value in self.input_size)
        if any(value <= 0 for value in native_grid):
            raise ValueError(
                f"input_size={self.input_size} is too small for DINO patch size "
                f"{self.native_patch_size}"
            )
        if output_grid is None:
            output_grid = (14, 14) if family == "dinov2" else native_grid
        output_grid = tuple(int(value) for value in output_grid)
        if len(output_grid) != 2 or any(value <= 0 for value in output_grid):
            raise ValueError("output_grid must contain two positive dimensions")
        if interpolate_spatial is None:
            interpolate_spatial = output_grid != native_grid
        if not interpolate_spatial and output_grid != native_grid:
            raise ValueError(
                f"{family} output_grid={output_grid} differs from native grid "
                f"{native_grid}; enable interpolate_spatial to resize"
            )
        self.native_grid = native_grid
        self.output_grid = output_grid
        self.interpolate_spatial = bool(interpolate_spatial)
        self.logical_patch_size = int(
            logical_patch_size
            if logical_patch_size is not None
            else self.input_size[0] // output_grid[0]
        )
        if self.logical_patch_size <= 0:
            raise ValueError("logical_patch_size must be positive")
        # ``patch_size`` is consumed by encode_video_views to localize policy
        # masks, hence it describes the logical output grid rather than the
        # source DINO patch size.
        self.patch_size = self.logical_patch_size
        self.tubelet_size = self.frame_stride
        self.grid_height, self.grid_width = self.output_grid
        self.patches_per_step = self.grid_height * self.grid_width
        self.grid_depth = len(self._frame_indices_for(self.num_frames))
        self.num_patches_per_view = self.grid_depth * self.patches_per_step
        self.num_patches = (
            None
            if self.num_views is None
            else self.num_views * self.num_patches_per_view
        )
        self._vjepa_policy_latent_grid = (
            self.grid_depth,
            self.grid_height,
            self.grid_width,
        )
        self.register_buffer(
            "frame_indices",
            self._frame_indices_for(self.num_frames),
            persistent=False,
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    def _frame_indices_for(self, frame_count: int) -> torch.Tensor:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        indices = list(range(self.frame_offset, frame_count, self.frame_stride))
        # A short clip may not reach the configured offset.  In that case use
        # the first available frame rather than inventing/padding a time step.
        if not indices:
            indices = list(range(0, frame_count, self.frame_stride))
        return torch.tensor(indices, dtype=torch.long)

    def train(self, mode: bool = True) -> DinoLatentEncoder:
        """Keep the frozen backbone in eval mode, including under parent.train()."""
        super().train(False)
        return self

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        rgb = images.mul(0.5).add(0.5)
        mean = self.image_mean.to(dtype=rgb.dtype)
        std = self.image_std.to(dtype=rgb.dtype)
        return (rgb - mean) / std

    @staticmethod
    def _last_hidden_state(output: Any) -> torch.Tensor:
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, dict) and "last_hidden_state" in output:
            return output["last_hidden_state"]
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise TypeError("DINO backbone output has no last_hidden_state")

    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        output = self.backbone(pixel_values=self._normalize_images(images))
        tokens = self._last_hidden_state(output)
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"DINO last_hidden_state must be [B,N,1024], got {tuple(tokens.shape)}"
            )

        native_grid = (
            images.shape[-2] // self.native_patch_size,
            images.shape[-1] // self.native_patch_size,
        )
        expected_patches = native_grid[0] * native_grid[1]
        expected_tokens = self.num_prefix_tokens + expected_patches
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"DINO returned {tokens.shape[1]} tokens, expected {expected_tokens}"
            )
        patches = tokens[:, self.num_prefix_tokens :]

        if patches.shape[1] != native_grid[0] * native_grid[1]:
            raise ValueError(
                f"DINO patch count does not match native grid {native_grid}: "
                f"{patches.shape[1]}"
            )
        if self.interpolate_spatial:
            patches = patches.reshape(
                -1, native_grid[0], native_grid[1], self.embed_dim
            ).permute(0, 3, 1, 2)
            patches = F.interpolate(
                patches,
                size=self.output_grid,
                mode="bilinear",
                align_corners=True,
            )
            patches = patches.permute(0, 2, 3, 1).reshape(
                -1, self.patches_per_step, self.embed_dim
            )
        elif patches.shape[1] != self.patches_per_step:
            raise ValueError(
                f"DINO output grid {native_grid} does not match requested "
                f"{self.output_grid}"
            )
        return patches.contiguous()

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        microbatch = self.image_microbatch_size
        if microbatch is None or images.shape[0] <= microbatch:
            return self._encode_image_batch(images)
        return torch.cat(
            [
                self._encode_image_batch(images[start : start + microbatch])
                for start in range(0, images.shape[0], microbatch)
            ],
            dim=0,
        )

    def _validate_5d(self, clips: torch.Tensor) -> None:
        if clips.ndim != 5:
            raise ValueError(f"clips must be [B,C,T,H,W], got {tuple(clips.shape)}")
        _, channels, frames, height, width = clips.shape
        if (channels, height, width) != (3, *self.input_size):
            raise ValueError(
                f"DINO clips must be [B,3,T,{self.input_size[0]},{self.input_size[1]}], "
                f"got {tuple(clips.shape)}"
            )
        if frames <= 0:
            raise ValueError("DINO clips must contain at least one frame")
        if not torch.is_floating_point(clips):
            raise TypeError("DINO clips must be floating point")

    def _canonical_masks(
        self,
        masks: torch.Tensor | Sequence[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        mask_list = [masks] if torch.is_tensor(masks) else list(masks)
        if not mask_list:
            raise ValueError("masks must not be empty")
        token_counts = set()
        canonical = []
        for mask in mask_list:
            if not torch.is_tensor(mask) or mask.ndim != 2:
                raise ValueError("each mask must be a [B,K] tensor")
            if mask.shape[0] != batch_size or mask.shape[1] == 0:
                raise ValueError(
                    f"mask batch {mask.shape[0]} does not match clips batch {batch_size}"
                )
            if mask.dtype == torch.bool or torch.is_floating_point(mask):
                raise TypeError("mask indices must use an integer dtype")
            token_counts.add(mask.shape[1])
            mask = mask.to(device=device, dtype=torch.long)
            max_tokens = self.grid_depth_for_frames(self._current_frame_count)
            if mask.numel() and (
                mask.min() < 0 or mask.max() >= max_tokens * self.patches_per_step
            ):
                raise ValueError("DINO mask index is outside the compact latent lattice")
            canonical.append(mask)
        if len(token_counts) != 1:
            raise ValueError("all masks must select the same number of tokens")
        return canonical

    def grid_depth_for_frames(self, frame_count: int) -> int:
        return int(self._frame_indices_for(frame_count).numel())

    def _forward_5d(
        self,
        clips: torch.Tensor,
        masks: torch.Tensor | Sequence[torch.Tensor] | None,
    ) -> torch.Tensor:
        self._validate_5d(clips)
        batch_size = clips.shape[0]
        self._current_frame_count = clips.shape[2]
        frame_indices = self._frame_indices_for(self._current_frame_count).to(clips.device)
        grid_depth = int(frame_indices.numel())
        self.grid_depth = grid_depth
        self.num_patches_per_view = grid_depth * self.patches_per_step
        self._vjepa_policy_latent_grid = (grid_depth, self.grid_height, self.grid_width)

        if masks is None:
            temporal_ids = torch.arange(grid_depth, device=clips.device)
            mask_list = None
        else:
            mask_list = self._canonical_masks(masks, batch_size, clips.device)
            all_ids = torch.cat([mask.reshape(-1) for mask in mask_list])
            temporal_ids = torch.unique(
                torch.div(all_ids, self.patches_per_step, rounding_mode="floor"),
                sorted=True,
            )
            if temporal_ids.numel() and temporal_ids.max() >= grid_depth:
                raise ValueError(
                    f"DINO mask requests temporal step {int(temporal_ids.max())}, "
                    f"but clip provides only {grid_depth} sampled steps"
                )

        frame_ids = frame_indices.index_select(0, temporal_ids)
        images = clips.index_select(2, frame_ids)
        images = images.permute(0, 2, 1, 3, 4).reshape(
            batch_size * temporal_ids.numel(),
            3,
            self.input_size[0],
            self.input_size[1],
        )
        patches = self._encode_images(images).reshape(
            batch_size,
            temporal_ids.numel(),
            self.patches_per_step,
            self.embed_dim,
        )

        if mask_list is None:
            return patches.flatten(1, 2)

        batch_ids = torch.arange(batch_size, device=clips.device).unsqueeze(1)
        selected = []
        for mask in mask_list:
            mask_temporal = torch.div(
                mask, self.patches_per_step, rounding_mode="floor"
            )
            mask_spatial = mask.remainder(self.patches_per_step)
            encoded_temporal = torch.searchsorted(temporal_ids, mask_temporal)
            selected.append(patches[batch_ids, encoded_temporal, mask_spatial])
        return torch.cat(selected, dim=0)

    def forward(
        self,
        clips: torch.Tensor,
        masks: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if clips.ndim == 5:
            return self._forward_5d(clips, masks)
        if clips.ndim != 6:
            raise ValueError(
                f"clips must be [B,C,T,H,W] or [B,V,C,T,H,W], got {tuple(clips.shape)}"
            )
        if masks is not None:
            raise ValueError(
                "masked 6D clips must use vjepa_policy.models.vjepa_policy."
                "encode_video_views so global view masks are localized first"
            )

        batch_size, num_views, channels, frames, height, width = clips.shape
        if self.num_views is not None and num_views != self.num_views:
            raise ValueError(
                f"configured DINO adapter expects {self.num_views} views, got {num_views}"
            )
        per_view = self._forward_5d(
            clips.reshape(batch_size * num_views, channels, frames, height, width),
            None,
        )
        per_view_depth = per_view.shape[1] // self.patches_per_step
        self.grid_depth = per_view_depth
        self.num_patches_per_view = per_view_depth * self.patches_per_step
        self.num_patches = num_views * self.num_patches_per_view
        self._vjepa_policy_latent_grid = (
            per_view_depth,
            self.grid_height,
            self.grid_width,
        )
        return (
            per_view.reshape(
                batch_size,
                num_views,
                per_view_depth,
                self.patches_per_step,
                self.embed_dim,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size, self.num_patches, self.embed_dim)
        )

    @classmethod
    def from_pretrained(
        cls,
        family: str,
        pretrained_path: str | Path,
        *,
        image_microbatch_size: int | None = None,
    ) -> DinoLatentEncoder:
        return load_dino_latent_encoder(
            family,
            pretrained_path,
            image_microbatch_size=image_microbatch_size,
        )


def load_dino_latent_encoder(
    family: str,
    pretrained_path: str | Path,
    *,
    image_microbatch_size: int | None = None,
    output_grid: tuple[int, int] | None = None,
    interpolate_spatial: bool | None = None,
    logical_patch_size: int | None = None,
    frame_stride: int = 2,
    frame_offset: int = 1,
    num_views: int | None = None,
    input_size: tuple[int, int] = (224, 224),
    video_frames: int = 10,
) -> DinoLatentEncoder:
    """Load an official DINO ViT-L from an explicit local HF directory."""
    if family not in DinoLatentEncoder._FAMILIES:
        raise ValueError(f"unsupported DINO family {family!r}")
    path = Path(pretrained_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"pretrained_path must be an existing local Hugging Face directory: {path}"
        )
    path = path.resolve()

    try:
        from transformers import DINOv3ViTModel, Dinov2Model
    except ImportError as error:
        raise ImportError(
            "loading DINO encoders requires a transformers build with DINOv3 support"
        ) from error

    model_class = Dinov2Model if family == "dinov2" else DINOv3ViTModel
    backbone = model_class.from_pretrained(str(path), local_files_only=True)
    return DinoLatentEncoder(
        backbone,
        family,
        image_microbatch_size=image_microbatch_size,
        output_grid=output_grid,
        interpolate_spatial=interpolate_spatial,
        logical_patch_size=logical_patch_size,
        frame_stride=frame_stride,
        frame_offset=frame_offset,
        num_views=num_views,
        input_size=input_size,
        video_frames=video_frames,
    )
