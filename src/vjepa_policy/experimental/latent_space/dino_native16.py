"""DINOv2 native token-grid adapter for the geometry ablation.

The regular :class:`DinoLatentEncoder` intentionally reproduces the existing
latent ablation and resizes DINOv2's native 16x16 grid to 14x14.  This module
keeps that path intact and exposes a separate adapter which returns the native
grid.  Its ``patch_size`` is the *image-side* DINOv2 patch size (14), so the
shared multi-view encoder helper derives the correct 16x16 local mask grid
from a 224px crop.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from vjepa_policy.data import CausalPatchMask
from vjepa_policy.experimental.latent_space.dino_encoder import DinoLatentEncoder


DINOV2_NATIVE16_PATCH_SIZE = 14
DINOV2_NATIVE16_GRID_HEIGHT = 16
DINOV2_NATIVE16_GRID_WIDTH = 16
DINOV2_NATIVE16_GRID_DEPTH = 5
DINOV2_NATIVE16_IMAGE_SIZE = (224, 224)
DINOV2_NATIVE16_LOGICAL_IMAGE_SIZE = (256, 256)


class DinoV2Native16LatentEncoder(DinoLatentEncoder):
    """Frozen DINOv2 ViT-L/14 returning ``5 x 16 x 16`` latent tokens.

    DINOv2 consumes five endpoint images from the same ten-frame policy clip
    as the existing adapter.  No spatial interpolation or learned projection
    is applied; the 256 native patch tokens from each image are returned in
    row-major order.
    """

    # ``encode_video_views`` uses this value to localize global masks against
    # the actual 224px input.  224 / 14 = 16, which matches the native output.
    patch_size = DINOV2_NATIVE16_PATCH_SIZE
    num_views = None
    grid_depth = DINOV2_NATIVE16_GRID_DEPTH
    grid_height = DINOV2_NATIVE16_GRID_HEIGHT
    grid_width = DINOV2_NATIVE16_GRID_WIDTH
    patches_per_step = grid_height * grid_width
    num_patches_per_view = grid_depth * patches_per_step
    num_patches = None

    def __init__(
        self,
        backbone: nn.Module,
        *,
        image_microbatch_size: int | None = None,
        num_views: int | None = None,
        input_size: tuple[int, int] = DINOV2_NATIVE16_IMAGE_SIZE,
        frame_stride: int = 2,
        frame_offset: int = 1,
        video_frames: int = 10,
    ) -> None:
        input_size = tuple(input_size)
        if any(value % DINOV2_NATIVE16_PATCH_SIZE for value in input_size):
            raise ValueError(
                f"DINOv2 native input size {input_size} must be divisible by "
                f"patch size {DINOV2_NATIVE16_PATCH_SIZE}"
            )
        output_grid = tuple(
            value // DINOV2_NATIVE16_PATCH_SIZE for value in input_size
        )
        super().__init__(
            backbone,
            "dinov2",
            image_microbatch_size=image_microbatch_size,
            output_grid=output_grid,
            interpolate_spatial=False,
            logical_patch_size=DINOV2_NATIVE16_PATCH_SIZE,
            frame_stride=frame_stride,
            frame_offset=frame_offset,
            num_views=num_views,
            input_size=input_size,
            video_frames=video_frames,
        )
        # Keep geometry explicit on the instance for builders and run metadata.
        self.patch_size = DINOV2_NATIVE16_PATCH_SIZE
        self.grid_depth = self.grid_depth_for_frames(video_frames)
        self.grid_height, self.grid_width = output_grid
        self.patches_per_step = self.grid_height * self.grid_width
        self.num_patches_per_view = self.grid_depth * self.patches_per_step
        self.num_patches = (
            None
            if self.num_views is None
            else self.num_views * self.num_patches_per_view
        )
        self._vjepa_policy_model_name = "dinov2_vitl_native16"
        self._vjepa_policy_latent_grid = (
            self.grid_depth,
            self.grid_height,
            self.grid_width,
        )
        self._vjepa_policy_native_grid = True

    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images and strip prefix/register tokens without resizing."""
        self.backbone.eval()
        output = self.backbone(pixel_values=self._normalize_images(images))
        tokens = self._last_hidden_state(output)
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"DINO last_hidden_state must be [B,N,1024], got {tuple(tokens.shape)}"
            )

        native_height = images.shape[-2] // self.native_patch_size
        native_width = images.shape[-1] // self.native_patch_size
        native_grid = (native_height, native_width)
        if native_grid != self.output_grid:
            raise ValueError(
                "DINOv2 native16 adapter output grid must match its configured "
                f"native grid {self.output_grid}, "
                f"got image={tuple(images.shape[-2:])} grid={native_grid}"
            )
        expected_tokens = self.num_prefix_tokens + native_height * native_width
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"DINO returned {tokens.shape[1]} tokens, expected {expected_tokens}"
            )
        patches = tokens[:, self.num_prefix_tokens :]
        if patches.shape[1] != self.patches_per_step:
            raise ValueError(
                "DINOv2 native16 backbone output does not contain 16x16 patch tokens: "
                f"{patches.shape[1]}"
            )
        return patches.reshape(-1, self.patches_per_step, self.embed_dim).contiguous()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        *,
        image_microbatch_size: int | None = None,
    ) -> "DinoV2Native16LatentEncoder":
        return load_dinov2_native16_latent_encoder(
            pretrained_path,
            image_microbatch_size=image_microbatch_size,
        )


def load_dinov2_native16_latent_encoder(
    pretrained_path: str | Path,
    *,
    image_microbatch_size: int | None = None,
    num_views: int | None = None,
    input_size: tuple[int, int] = DINOV2_NATIVE16_IMAGE_SIZE,
    frame_stride: int = 2,
    frame_offset: int = 1,
    video_frames: int = 10,
) -> DinoV2Native16LatentEncoder:
    """Load DINOv2 ViT-L/14 from an explicit local Hugging Face directory."""
    path = Path(pretrained_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"pretrained_path must be an existing local Hugging Face directory: {path}"
        )
    path = path.resolve()
    try:
        from transformers import Dinov2Model
    except ImportError as error:
        raise ImportError(
            "loading DINOv2 requires a transformers installation with Dinov2Model"
        ) from error
    backbone = Dinov2Model.from_pretrained(str(path), local_files_only=True)
    return DinoV2Native16LatentEncoder(
        backbone,
        image_microbatch_size=image_microbatch_size,
        num_views=num_views,
        input_size=input_size,
        frame_stride=frame_stride,
        frame_offset=frame_offset,
        video_frames=video_frames,
    )


def build_dinov2_native16_policy_mask(
    *,
    num_views: int = 1,
    image_size: tuple[int, int] = DINOV2_NATIVE16_LOGICAL_IMAGE_SIZE,
    patch_size: int = 16,
    tubelet_size: int = 2,
    video_frames: int = 10,
    context_tubelets: int = 1,
) -> CausalPatchMask:
    """Build the logical 5x16x16 causal mask for native DINOv2 tokens.

    ``image_size`` is a logical predictor geometry, not the 224px crop sent
    to DINOv2.  Keeping this distinction local prevents the old 14x14 masks
    from being reused accidentally.
    """
    if tuple(image_size) != DINOV2_NATIVE16_LOGICAL_IMAGE_SIZE:
        raise ValueError(
            "native16 policy mask requires logical image_size=(256, 256), "
            f"got {tuple(image_size)}"
        )
    resolved_grid = (
        image_size[0] // patch_size,
        image_size[1] // patch_size,
    )
    if resolved_grid != (
        DINOV2_NATIVE16_GRID_HEIGHT,
        DINOV2_NATIVE16_GRID_WIDTH,
    ):
        raise ValueError(
            "native16 policy mask must resolve to a 16x16 spatial grid"
        )
    if video_frames // tubelet_size != DINOV2_NATIVE16_GRID_DEPTH:
        raise ValueError(
            "native16 policy mask must resolve to five temporal slots"
        )
    return CausalPatchMask(
        image_size=image_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        video_frames=video_frames,
        context_tubelets=context_tubelets,
        num_views=num_views,
    )


__all__ = [
    "DINOV2_NATIVE16_GRID_DEPTH",
    "DINOV2_NATIVE16_GRID_HEIGHT",
    "DINOV2_NATIVE16_GRID_WIDTH",
    "DINOV2_NATIVE16_IMAGE_SIZE",
    "DINOV2_NATIVE16_LOGICAL_IMAGE_SIZE",
    "DINOV2_NATIVE16_PATCH_SIZE",
    "DinoV2Native16LatentEncoder",
    "build_dinov2_native16_policy_mask",
    "load_dinov2_native16_latent_encoder",
]
