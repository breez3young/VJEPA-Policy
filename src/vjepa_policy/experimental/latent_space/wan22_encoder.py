"""Frozen Wan2.2 VAE38 latent encoder for the LIBERO latent-space ablation."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


WAN22_POLICY_FRAMES = 10
WAN22_VAE_INPUT_FRAMES = 9
WAN22_LATENT_FRAMES = 3
WAN22_LATENT_CHANNELS = 48
WAN22_GRID_SIZE = 14
WAN22_PATCH_SIZE = 16
WAN22_PREDICTOR_DEPTH = 5
WAN22_LOGICAL_TEMPORAL_IDS = (0, 2, 4)
WAN22_NUM_VIEWS = 2
WAN22_PATCHES_PER_STEP = WAN22_GRID_SIZE * WAN22_GRID_SIZE

_FASTWAM_SRC = os.environ.get("FASTWAM_SRC_DIR")
DEFAULT_FASTWAM_SRC_DIR = Path(_FASTWAM_SRC) if _FASTWAM_SRC else None

_WAN22_LATENT_MEAN = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)

_WAN22_LATENT_STD = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)

_WAN22_DIFFUSERS_CONFIG = {
    "base_dim": 160,
    "z_dim": WAN22_LATENT_CHANNELS,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "is_residual": True,
    "in_channels": 12,
    "patch_size": 2,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
}

_DIFFUSERS_CHECKPOINT = "diffusers"
_FASTWAM_CHECKPOINT = "fastwam"


def build_wan22_predictor_token_ids(
    num_views: int = WAN22_NUM_VIEWS,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return context t0 and future t2/t4 IDs in the original 5-step lattice."""
    if not isinstance(num_views, int) or isinstance(num_views, bool) or num_views <= 0:
        raise ValueError("num_views must be a positive integer")

    token_ids = torch.arange(
        WAN22_PREDICTOR_DEPTH * num_views * WAN22_GRID_SIZE * WAN22_GRID_SIZE,
        dtype=torch.long,
        device=device,
    ).reshape(WAN22_PREDICTOR_DEPTH, num_views, WAN22_GRID_SIZE, WAN22_GRID_SIZE)
    context_ids = token_ids[WAN22_LOGICAL_TEMPORAL_IDS[0]].reshape(-1)
    future_ids = token_ids[list(WAN22_LOGICAL_TEMPORAL_IDS[1:])].reshape(-1)
    return context_ids, future_ids


def build_wan22_policy_mask(
    *,
    image_size: tuple[int, int],
    num_views: int = WAN22_NUM_VIEWS,
):
    """Build sparse t0 -> {t2,t4} masks on the baseline T=5 lattice."""
    if tuple(image_size) != (224, 224):
        raise ValueError(f"Wan2.2 ablation requires 224x224 views, got {image_size}")
    if num_views != WAN22_NUM_VIEWS:
        raise ValueError(
            f"Wan2.2 ablation requires exactly {WAN22_NUM_VIEWS} views, got {num_views}"
        )
    context_ids, future_ids = build_wan22_predictor_token_ids(num_views)
    return SimpleNamespace(
        ctx_idx=context_ids,
        tgt_idx=future_ids,
        n_ctx=context_ids.numel(),
    )


class Wan22LatentEncoder(nn.Module):
    """Adapt policy clips to deterministic Wan2.2 VAE38 tokens.

    Inputs must already use the policy's ``[-1, 1]`` pixel normalization. A
    10-frame policy clip is ``[-4, 0, 4, ..., 32]``; Wan receives only
    ``[0, 4, ..., 32]`` so its first-frame-plus-4n contract retains both
    physical endpoints.
    """

    embed_dim = WAN22_LATENT_CHANNELS
    patch_size = WAN22_PATCH_SIZE
    num_frames = WAN22_POLICY_FRAMES
    latent_frames = WAN22_LATENT_FRAMES
    grid_size = WAN22_GRID_SIZE
    logical_grid_depth = WAN22_PREDICTOR_DEPTH
    logical_temporal_ids = WAN22_LOGICAL_TEMPORAL_IDS
    tubelet_size = 2
    num_views = None
    patches_per_step = WAN22_PATCHES_PER_STEP
    num_patches_per_view = logical_grid_depth * patches_per_step
    num_patches = None

    def __init__(
        self,
        encoder_core: nn.Module,
        *,
        video_microbatch_size: int | None = None,
        num_views: int | None = None,
    ):
        super().__init__()
        if video_microbatch_size is not None and video_microbatch_size <= 0:
            raise ValueError("video_microbatch_size must be positive")
        self.encoder_core = encoder_core.eval().requires_grad_(False)
        self.video_microbatch_size = video_microbatch_size
        if num_views is not None and (not isinstance(num_views, int) or num_views <= 0):
            raise ValueError("num_views must be a positive integer when configured")
        self.num_views = num_views
        self._vjepa_policy_family = "wan2_2"
        self._vjepa_policy_model_name = "vae38"
        self._vjepa_policy_checkpoint_key = "encoder_quant_conv"
        self.blocks = (
            SimpleNamespace(
                attn=SimpleNamespace(
                    interpolate_rope=False,
                    corrected_frequency_pairing=False,
                )
            ),
        )

    def train(self, mode: bool = True) -> Wan22LatentEncoder:
        super().train(False)
        return self

    def build_policy_mask(self, *, image_size, num_views):
        return build_wan22_policy_mask(
            image_size=tuple(image_size),
            num_views=num_views,
        )

    @staticmethod
    def _validate_clips(clips: torch.Tensor) -> None:
        if clips.ndim not in (5, 6):
            raise ValueError(
                "Wan2.2 clips must be [B,C,T,H,W] or [B,V,C,T,H,W], "
                f"got {tuple(clips.shape)}"
            )
        channels, frames, height, width = clips.shape[-4:]
        if (channels, frames, height, width) != (3, WAN22_POLICY_FRAMES, 224, 224):
            raise ValueError(
                "Wan2.2 clips must have trailing shape [3,10,224,224], "
                f"got {tuple(clips.shape[-4:])}"
            )
        if not torch.is_floating_point(clips):
            raise TypeError(
                "Wan2.2 clips must be floating point and normalized to [-1, 1]"
            )

    @staticmethod
    def _validate_latents(
        latents: torch.Tensor,
        expected_batch: int,
        expected_frames: int,
    ) -> None:
        expected = (
            expected_batch,
            WAN22_LATENT_CHANNELS,
            expected_frames,
            WAN22_GRID_SIZE,
            WAN22_GRID_SIZE,
        )
        if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != expected:
            observed = (
                tuple(latents.shape)
                if isinstance(latents, torch.Tensor)
                else type(latents)
            )
            raise ValueError(
                f"Wan2.2 VAE38 encoder must return {expected}, got {observed}"
            )

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
                raise ValueError("each mask must contain tokens for every clip")
            if mask.dtype == torch.bool or torch.is_floating_point(mask):
                raise TypeError("mask indices must use an integer dtype")
            mask = mask.to(device=device, dtype=torch.long)
            if mask.min() < 0 or mask.max() >= self.num_patches_per_view:
                raise ValueError("Wan2.2 mask index is outside the logical T=5 lattice")
            temporal_ids = torch.div(
                mask,
                self.patches_per_step,
                rounding_mode="floor",
            )
            available = torch.zeros_like(temporal_ids, dtype=torch.bool)
            for temporal_id in self.logical_temporal_ids:
                available |= temporal_ids == temporal_id
            if not torch.all(available):
                invalid = torch.unique(temporal_ids[~available]).tolist()
                raise ValueError(
                    f"Wan2.2 has no latents at logical temporal IDs {invalid}"
                )
            token_counts.add(mask.shape[1])
            canonical.append(mask)
        if len(token_counts) != 1:
            raise ValueError("all masks must select the same number of tokens")
        return canonical

    @torch.no_grad()
    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        microbatch = self.video_microbatch_size
        if microbatch is None or video.shape[0] <= microbatch:
            return self.encoder_core(video)
        return torch.cat(
            [
                self.encoder_core(video[start : start + microbatch])
                for start in range(0, video.shape[0], microbatch)
            ],
            dim=0,
        )

    def _forward_5d(
        self,
        clips: torch.Tensor,
        masks: torch.Tensor | Sequence[torch.Tensor] | None,
    ) -> torch.Tensor:
        self._validate_clips(clips)
        if clips.ndim != 5:
            raise ValueError(f"clips must be [B,C,T,H,W], got {tuple(clips.shape)}")
        batch_size = clips.shape[0]
        if masks is None:
            mask_list = None
            max_logical_temporal = self.logical_temporal_ids[-1]
        else:
            mask_list = self._canonical_masks(masks, batch_size, clips.device)
            max_logical_temporal = max(
                int(mask.max().item()) // self.patches_per_step for mask in mask_list
            )

        vae_frames = 1 + 4 * (max_logical_temporal // 2)
        vae_input = clips[:, :, 1 : 1 + vae_frames].contiguous()
        latents = self._encode_video(vae_input)
        expected_latent_frames = 1 + max_logical_temporal // 2
        self._validate_latents(
            latents,
            batch_size,
            expected_latent_frames,
        )
        patches = latents.permute(0, 2, 3, 4, 1).reshape(
            batch_size,
            expected_latent_frames,
            self.patches_per_step,
            self.embed_dim,
        )

        if mask_list is None:
            logical = patches.new_zeros(
                batch_size,
                self.logical_grid_depth,
                self.patches_per_step,
                self.embed_dim,
            )
            logical[:, list(self.logical_temporal_ids)] = patches
            return logical.flatten(1, 2)

        batch_ids = torch.arange(batch_size, device=clips.device).unsqueeze(1)
        selected = []
        for mask in mask_list:
            logical_temporal = torch.div(
                mask,
                self.patches_per_step,
                rounding_mode="floor",
            )
            physical_temporal = torch.div(
                logical_temporal,
                2,
                rounding_mode="floor",
            )
            spatial = mask.remainder(self.patches_per_step)
            selected.append(patches[batch_ids, physical_temporal, spatial])
        return torch.cat(selected, dim=0)

    def forward(
        self,
        clips: torch.Tensor,
        masks: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if clips.ndim == 5:
            return self._forward_5d(clips, masks)
        self._validate_clips(clips)
        if masks is not None:
            raise ValueError(
                "masked 6D clips must use vjepa_policy.models.vjepa_policy."
                "encode_video_views so global view masks are localized first"
            )
        batch_size, num_views, channels, frames, height, width = clips.shape
        if self.num_views is not None and num_views != self.num_views:
            raise ValueError(
                f"configured Wan2.2 adapter expects {self.num_views} views, got {num_views}"
            )
        per_view = self._forward_5d(
            clips.reshape(batch_size * num_views, channels, frames, height, width),
            None,
        )
        self.num_patches = num_views * self.num_patches_per_view
        return (
            per_view.reshape(
                batch_size,
                num_views,
                self.logical_grid_depth,
                self.patches_per_step,
                self.embed_dim,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size, self.num_patches, self.embed_dim)
        )


def build_wan22_compact_policy_mask(
    *,
    image_size: tuple[int, int],
    num_views: int = 1,
    grid_depth: int,
    context_steps: int = 1,
):
    """Build a causal mask on the *actual* Wan temporal lattice.

    Unlike :func:`build_wan22_policy_mask`, this helper never allocates
    placeholder slots for unavailable VAE times.  ``grid_depth`` is obtained
    from the VAE output (or its declared temporal contract) by the caller.
    """
    if tuple(image_size) != (224, 224):
        raise ValueError(f"Wan2.2 compact adapter requires 224x224 views, got {image_size}")
    if not isinstance(num_views, int) or num_views <= 0:
        raise ValueError("num_views must be a positive integer")
    if not isinstance(grid_depth, int) or grid_depth <= 1:
        raise ValueError("grid_depth must be an integer greater than one")
    if not 0 < context_steps < grid_depth:
        raise ValueError(
            f"context_steps={context_steps} invalid for compact depth {grid_depth}"
        )
    token_ids = torch.arange(
        grid_depth * num_views * WAN22_PATCHES_PER_STEP,
        dtype=torch.long,
    ).reshape(grid_depth, num_views, WAN22_GRID_SIZE, WAN22_GRID_SIZE)
    context_ids = token_ids[:context_steps].reshape(-1)
    target_ids = token_ids[context_steps:].reshape(-1)
    return SimpleNamespace(
        ctx_idx=context_ids,
        tgt_idx=target_ids,
        n_ctx=context_ids.numel(),
        grid_depth=grid_depth,
        num_views=num_views,
    )


class Wan22CompactLatentEncoder(nn.Module):
    """Wan VAE adapter that exposes the native temporal output compactly.

    The VAE consumes a first frame followed by groups of four frames.  For a
    clip with ``T`` RGB frames the adapter chooses the largest supported
    ``1 + 4*n`` suffix and returns exactly ``n + 1`` latent steps.  No logical
    t=1/t=3 placeholders are inserted when the VAE only produced t=0..t=2.
    Any number of camera views is accepted; direct 6D output is ordered as
    ``(time, view, height, width)``.
    """

    embed_dim = WAN22_LATENT_CHANNELS
    patch_size = 16  # 224 / 16 = the native 14x14 VAE grid
    # The policy helper uses this only to validate raw clip divisibility when
    # localizing masks.  Wan's actual temporal compression is declared by the
    # registry spec (stride 4), and raw clips such as the 17-frame stride-2
    # experiment are intentionally not divisible by that VAE ratio.
    tubelet_size = 1
    temporal_mode = "causal_stride"
    num_views = None
    patches_per_step = WAN22_PATCHES_PER_STEP

    def __init__(
        self,
        encoder_core: nn.Module,
        *,
        video_microbatch_size: int | None = None,
        num_views: int | None = None,
    ):
        super().__init__()
        if video_microbatch_size is not None and video_microbatch_size <= 0:
            raise ValueError("video_microbatch_size must be positive")
        if num_views is not None and (not isinstance(num_views, int) or num_views <= 0):
            raise ValueError("num_views must be a positive integer when configured")
        self.encoder_core = encoder_core.eval().requires_grad_(False)
        self.video_microbatch_size = video_microbatch_size
        self.num_views = num_views
        self._vjepa_policy_family = "wan2_2"
        self._vjepa_policy_model_name = "vae38_compact"
        self._vjepa_policy_checkpoint_key = "encoder_quant_conv"
        self.blocks = (
            SimpleNamespace(
                attn=SimpleNamespace(
                    interpolate_rope=False,
                    corrected_frequency_pairing=False,
                )
            ),
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder_core.eval()
        return self

    @property
    def max_supported_vae_frames(self) -> int | None:
        if getattr(self.encoder_core, "_supports_dynamic_temporal", False):
            return None
        supported = getattr(self.encoder_core, "_SUPPORTED_INPUT_FRAMES", None)
        if supported is None:
            supported = (1, 5, WAN22_VAE_INPUT_FRAMES)
        return max(int(value) for value in supported)

    @staticmethod
    def _validate_clips(clips: torch.Tensor) -> None:
        if clips.ndim not in (5, 6):
            raise ValueError(
                "Wan2.2 compact clips must be [B,C,T,H,W] or [B,V,C,T,H,W], "
                f"got {tuple(clips.shape)}"
            )
        channels, frames, height, width = clips.shape[-4:]
        if channels != 3 or frames <= 0 or (height, width) != (224, 224):
            raise ValueError(
                "Wan2.2 compact clips must have trailing shape [3,T,224,224], "
                f"got {tuple(clips.shape[-4:])}"
            )
        if not torch.is_floating_point(clips):
            raise TypeError("Wan2.2 compact clips must be floating point")

    @staticmethod
    def _validate_latents(latents: torch.Tensor, batch_size: int) -> int:
        if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
            raise ValueError(
                "Wan2.2 compact VAE must return [B,48,T,14,14], "
                f"got {tuple(latents.shape) if isinstance(latents, torch.Tensor) else type(latents)}"
            )
        expected_prefix = (batch_size, WAN22_LATENT_CHANNELS)
        if tuple(latents.shape[:2]) != expected_prefix or tuple(latents.shape[-2:]) != (
            WAN22_GRID_SIZE,
            WAN22_GRID_SIZE,
        ):
            raise ValueError(
                "Wan2.2 compact VAE must return [B,48,T,14,14], "
                f"got {tuple(latents.shape)}"
            )
        if latents.shape[2] <= 0:
            raise ValueError("Wan2.2 compact VAE returned no temporal latents")
        return int(latents.shape[2])

    def _canonical_masks(self, masks, batch_size: int, device: torch.device):
        mask_list = [masks] if torch.is_tensor(masks) else list(masks)
        if not mask_list:
            raise ValueError("masks must not be empty")
        counts = set()
        result = []
        for mask in mask_list:
            if not torch.is_tensor(mask) or mask.ndim != 2:
                raise ValueError("each mask must be a [B,K] tensor")
            if mask.shape[0] != batch_size or mask.shape[1] == 0:
                raise ValueError("each mask must contain tokens for every clip")
            if mask.dtype == torch.bool or torch.is_floating_point(mask):
                raise TypeError("mask indices must use an integer dtype")
            mask = mask.to(device=device, dtype=torch.long)
            if mask.min() < 0:
                raise ValueError("Wan2.2 compact mask indices must be non-negative")
            counts.add(mask.shape[1])
            result.append(mask)
        if len(counts) != 1:
            raise ValueError("all masks must select the same number of tokens")
        return result

    @torch.no_grad()
    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        microbatch = self.video_microbatch_size
        if microbatch is None or video.shape[0] <= microbatch:
            return self.encoder_core(video)
        return torch.cat(
            [
                self.encoder_core(video[start : start + microbatch])
                for start in range(0, video.shape[0], microbatch)
            ],
            dim=0,
        )

    def _usable_frames(self, frame_count: int) -> int:
        max_supported = self.max_supported_vae_frames
        candidate = 1 + 4 * ((frame_count - 1) // 4)
        if max_supported is not None:
            candidate = min(candidate, max_supported)
        if candidate < 1:
            raise ValueError("clip does not contain a usable Wan VAE frame")
        return candidate

    def _forward_5d(self, clips: torch.Tensor, masks=None) -> torch.Tensor:
        self._validate_clips(clips)
        if clips.ndim != 5:
            raise ValueError(f"clips must be [B,C,T,H,W], got {tuple(clips.shape)}")
        batch_size, _, frame_count, _, _ = clips.shape
        usable_frames = self._usable_frames(frame_count)
        full_depth = (usable_frames - 1) // 4 + 1
        mask_list = None if masks is None else self._canonical_masks(
            masks, batch_size, clips.device
        )
        if mask_list is None:
            needed_latent_steps = full_depth
        else:
            max_token = max(int(mask.max().item()) for mask in mask_list)
            needed_latent_steps = max_token // self.patches_per_step + 1
            if needed_latent_steps <= 0:
                raise ValueError("mask selects no temporal latent")
            if needed_latent_steps > full_depth:
                raise ValueError(
                    f"mask requests {needed_latent_steps} temporal latents, but clip "
                    f"provides only {full_depth}"
                )
        vae_frames = 1 + 4 * (needed_latent_steps - 1)
        # Keep the same causal window anchor for context-only and target
        # encodes.  Slicing from ``frame_count - vae_frames`` would make a
        # context mask read the final future frame instead of the current t0.
        start = frame_count - usable_frames
        vae_input = clips[:, :, start : start + vae_frames].contiguous()
        latents = self._encode_video(vae_input)
        actual_steps = self._validate_latents(latents, batch_size)
        if actual_steps < needed_latent_steps:
            raise ValueError(
                f"Wan2.2 VAE returned {actual_steps} steps, needed {needed_latent_steps}"
            )
        self.grid_depth = full_depth
        self.num_patches_per_view = full_depth * self.patches_per_step
        self._vjepa_policy_latent_grid = (
            full_depth,
            WAN22_GRID_SIZE,
            WAN22_GRID_SIZE,
        )
        patches = latents.permute(0, 2, 3, 4, 1).reshape(
            batch_size,
            actual_steps,
            self.patches_per_step,
            self.embed_dim,
        )
        if mask_list is None:
            return patches.flatten(1, 2)
        batch_ids = torch.arange(batch_size, device=clips.device).unsqueeze(1)
        selected = []
        for mask in mask_list:
            temporal = torch.div(mask, self.patches_per_step, rounding_mode="floor")
            spatial = mask.remainder(self.patches_per_step)
            if temporal.max() >= actual_steps:
                raise ValueError("mask requests a temporal latent absent from Wan output")
            selected.append(patches[batch_ids, temporal, spatial])
        return torch.cat(selected, dim=0)

    def forward(self, clips: torch.Tensor, masks=None) -> torch.Tensor:
        if clips.ndim == 5:
            return self._forward_5d(clips, masks)
        self._validate_clips(clips)
        if masks is not None:
            raise ValueError(
                "masked 6D clips must be localized to per-view masks before calling "
                "Wan22CompactLatentEncoder"
            )
        batch_size, num_views, channels, frames, height, width = clips.shape
        if self.num_views is not None and num_views != self.num_views:
            raise ValueError(
                f"configured Wan adapter expects {self.num_views} views, got {num_views}"
            )
        per_view = self._forward_5d(
            clips.reshape(batch_size * num_views, channels, frames, height, width),
            None,
        )
        depth = per_view.shape[1] // self.patches_per_step
        self.num_patches_per_view = depth * self.patches_per_step
        self.num_patches = num_views * self.num_patches_per_view
        return (
            per_view.reshape(
                batch_size,
                num_views,
                depth,
                self.patches_per_step,
                self.embed_dim,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size, depth * num_views * self.patches_per_step, self.embed_dim)
        )


class _Wan22VAE38EncoderCore(nn.Module):
    """Encoder and quantization head only; the decoder is never constructed."""

    _supports_dynamic_temporal = True

    def __init__(self, implementation, *, checkpoint_format: str):
        super().__init__()
        if checkpoint_format == _DIFFUSERS_CHECKPOINT:
            config = _WAN22_DIFFUSERS_CONFIG
            self.encoder = implementation.WanEncoder3d(
                in_channels=config["in_channels"],
                dim=config["base_dim"],
                z_dim=config["z_dim"] * 2,
                dim_mult=config["dim_mult"],
                num_res_blocks=config["num_res_blocks"],
                attn_scales=config["attn_scales"],
                temperal_downsample=config["temperal_downsample"],
                dropout=config["dropout"],
                is_residual=config["is_residual"],
            )
            self.quant_conv = implementation.WanCausalConv3d(
                WAN22_LATENT_CHANNELS * 2,
                WAN22_LATENT_CHANNELS * 2,
                1,
            )
            causal_conv_type = implementation.WanCausalConv3d
        elif checkpoint_format == _FASTWAM_CHECKPOINT:
            self.encoder = implementation.Encoder3d_38(
                dim=160,
                z_dim=WAN22_LATENT_CHANNELS * 2,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[False, True, True],
                dropout=0.0,
            )
            self.conv1 = implementation.CausalConv3d(
                WAN22_LATENT_CHANNELS * 2,
                WAN22_LATENT_CHANNELS * 2,
                1,
            )
            causal_conv_type = implementation.CausalConv3d
        else:
            raise ValueError(
                f"unsupported Wan2.2 checkpoint format: {checkpoint_format}"
            )

        self._checkpoint_format = checkpoint_format
        self._patchify = implementation.patchify
        self._causal_conv_type = causal_conv_type
        self.register_buffer(
            "latent_mean",
            torch.tensor(_WAN22_LATENT_MEAN).reshape(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latent_inv_std",
            torch.tensor(_WAN22_LATENT_STD).reciprocal().reshape(1, -1, 1, 1, 1),
            persistent=False,
        )

    def materialize_scale_buffers(self) -> None:
        reference = next(self.parameters())
        self.latent_mean = torch.tensor(
            _WAN22_LATENT_MEAN,
            dtype=reference.dtype,
            device=reference.device,
        ).reshape(1, -1, 1, 1, 1)
        self.latent_inv_std = (
            torch.tensor(
                _WAN22_LATENT_STD,
                dtype=reference.dtype,
                device=reference.device,
            )
            .reciprocal()
            .reshape(1, -1, 1, 1, 1)
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if (
            video.ndim != 5
            or video.shape[1] != 3
            or video.shape[2] <= 0
            or (video.shape[2] - 1) % 4
        ):
            raise ValueError(
                "Wan2.2 encoder core expects [B,3,T,H,W] with T=1+4n, "
                f"got {tuple(video.shape)}"
            )

        video = self._patchify(video, patch_size=2)
        feature_cache = [None] * sum(
            isinstance(module, self._causal_conv_type)
            for module in self.encoder.modules()
        )
        encoded_chunks = []
        chunks = [(0, 1)] + [
            (start, min(start + 4, video.shape[2]))
            for start in range(1, video.shape[2], 4)
        ]
        for start, end in chunks:
            feature_index = [0]
            encoded = self.encoder(
                video[:, :, start:end],
                feat_cache=feature_cache,
                feat_idx=feature_index,
            )
            if self._checkpoint_format == _FASTWAM_CHECKPOINT:
                encoded, feature_cache, _ = encoded
            encoded_chunks.append(encoded)

        encoded = torch.cat(encoded_chunks, dim=2)
        moments = (
            self.quant_conv(encoded)
            if self._checkpoint_format == _DIFFUSERS_CHECKPOINT
            else self.conv1(encoded)
        )
        mean, _ = moments.chunk(2, dim=1)
        latent_mean = self.latent_mean.to(device=mean.device, dtype=mean.dtype)
        latent_inv_std = self.latent_inv_std.to(device=mean.device, dtype=mean.dtype)
        return (mean - latent_mean) * latent_inv_std


def _load_fastwam_vae_module(fastwam_src_dir: str | Path | None):
    if fastwam_src_dir is None:
        raise ValueError("WAN2.2 latent encoding requires FASTWAM_SRC_DIR")
    implementation_path = (
        Path(fastwam_src_dir) / "fastwam" / "models" / "wan22" / "wan_video_vae.py"
    )
    if not implementation_path.is_file():
        raise FileNotFoundError(
            f"FastWAM Wan2.2 VAE implementation not found: {implementation_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "_vjepa_policy_fastwam_wan_video_vae",
        implementation_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import FastWAM VAE implementation: {implementation_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_diffusers_vae_module():
    try:
        import diffusers
        from diffusers.models.autoencoders import autoencoder_kl_wan
    except ImportError as error:
        raise ImportError(
            "diffusers==0.38.0 is required for official Wan2.2 VAE checkpoints"
        ) from error

    if diffusers.__version__ != "0.38.0":
        raise RuntimeError(
            "official Wan2.2 VAE loading requires diffusers==0.38.0, "
            f"got {diffusers.__version__}"
        )
    return autoencoder_kl_wan


def _load_encoder_only_state(
    checkpoint_path: Path,
) -> tuple[str, dict[str, torch.Tensor]]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError(
            "safetensors is required to load the Wan2.2 VAE38 encoder"
        ) from error

    state_dict = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = [
            (file_key, file_key.removeprefix("model."))
            for file_key in checkpoint.keys()
        ]
        has_quant_conv = any(
            key.startswith("quant_conv.") for _, key in checkpoint_keys
        )
        has_conv1 = any(key.startswith("conv1.") for _, key in checkpoint_keys)
        if has_quant_conv and has_conv1:
            raise ValueError(
                "checkpoint mixes Diffusers quant_conv and FastWAM conv1 weights"
            )
        if has_quant_conv:
            checkpoint_format = _DIFFUSERS_CHECKPOINT
            prefixes = ("encoder.", "quant_conv.")
        elif has_conv1:
            checkpoint_format = _FASTWAM_CHECKPOINT
            prefixes = ("encoder.", "conv1.")
        else:
            raise ValueError(
                "checkpoint contains neither Diffusers quant_conv nor FastWAM conv1 "
                f"weights: {checkpoint_path}"
            )

        for file_key, key in checkpoint_keys:
            if key.startswith(prefixes):
                if key in state_dict:
                    raise ValueError(
                        f"duplicate Wan2.2 encoder key after prefix removal: {key}"
                    )
                state_dict[key] = checkpoint.get_tensor(file_key)
    return checkpoint_format, state_dict


def _load_wan22_encoder_core(
    checkpoint_path: str | Path,
    *,
    fastwam_src_dir: str | Path = DEFAULT_FASTWAM_SRC_DIR,
) -> nn.Module:
    """Load the shared encoder-only VAE core used by legacy and compact paths."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError(
            "Wan2.2 VAE38 checkpoint must be an explicit .safetensors file"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Wan2.2 VAE38 checkpoint not found: {checkpoint_path}")
    checkpoint_format, state_dict = _load_encoder_only_state(checkpoint_path)
    implementation = (
        _load_diffusers_vae_module()
        if checkpoint_format == _DIFFUSERS_CHECKPOINT
        else _load_fastwam_vae_module(fastwam_src_dir)
    )
    with torch.device("meta"):
        encoder_core = _Wan22VAE38EncoderCore(
            implementation,
            checkpoint_format=checkpoint_format,
        )
    encoder_core.load_state_dict(state_dict, strict=True, assign=True)
    encoder_core.materialize_scale_buffers()
    return encoder_core


def load_wan22_latent_encoder(
    checkpoint_path: str | Path,
    *,
    fastwam_src_dir: str | Path = DEFAULT_FASTWAM_SRC_DIR,
    video_microbatch_size: int | None = None,
) -> Wan22LatentEncoder:
    """Load a local VAE38 safetensors file without downloads or decoder weights."""
    encoder_core = _load_wan22_encoder_core(
        checkpoint_path,
        fastwam_src_dir=fastwam_src_dir,
    )
    return Wan22LatentEncoder(
        encoder_core,
        video_microbatch_size=video_microbatch_size,
    )


def load_wan22_compact_latent_encoder(
    checkpoint_path: str | Path,
    *,
    fastwam_src_dir: str | Path = DEFAULT_FASTWAM_SRC_DIR,
    video_microbatch_size: int | None = None,
    num_views: int | None = None,
) -> Wan22CompactLatentEncoder:
    """Load a VAE38 core and expose its native temporal length compactly."""
    encoder_core = _load_wan22_encoder_core(
        checkpoint_path,
        fastwam_src_dir=fastwam_src_dir,
    )
    return Wan22CompactLatentEncoder(
        encoder_core,
        video_microbatch_size=video_microbatch_size,
        num_views=num_views,
    )
