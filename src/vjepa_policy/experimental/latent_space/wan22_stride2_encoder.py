"""Experimental Wan2.2 VAE38 adapter for stride-2 video sampling.

The original Wan2.2 latent ablation keeps the policy clip's 10 sampled frames
and therefore exposes only three native VAE time steps.  This module is an
isolated geometry variant: it consumes ``[0, 2, ..., 32]`` (17 RGB frames),
which produces five native temporal latents and maps them to logical slots
``t0 .. t4``.  The existing :mod:`wan22_encoder` path is intentionally left
unchanged so previously completed runs remain reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from vjepa_policy.experimental.latent_space.wan22_encoder import (
    DEFAULT_FASTWAM_SRC_DIR,
    WAN22_GRID_SIZE,
    WAN22_LATENT_CHANNELS,
    WAN22_PATCH_SIZE,
    _DIFFUSERS_CHECKPOINT,
    _FASTWAM_CHECKPOINT,
    _Wan22VAE38EncoderCore,
    Wan22CompactLatentEncoder,
    _load_diffusers_vae_module,
    _load_encoder_only_state,
    _load_fastwam_vae_module,
)


WAN22_STRIDE2_POLICY_FRAMES = 17
WAN22_STRIDE2_VAE_INPUT_FRAMES = 17
WAN22_STRIDE2_LATENT_FRAMES = 5
WAN22_STRIDE2_LOGICAL_TEMPORAL_IDS = tuple(range(WAN22_STRIDE2_LATENT_FRAMES))
WAN22_STRIDE2_POLICY_FRAME_STRIDE = 2
WAN22_STRIDE2_PREDICTOR_DEPTH = WAN22_STRIDE2_LATENT_FRAMES
WAN22_STRIDE2_NUM_VIEWS = 2
WAN22_STRIDE2_PATCHES_PER_STEP = WAN22_GRID_SIZE * WAN22_GRID_SIZE
WAN22_STRIDE2_NUM_PATCHES_PER_VIEW = (
    WAN22_STRIDE2_PREDICTOR_DEPTH * WAN22_STRIDE2_PATCHES_PER_STEP
)
WAN22_STRIDE2_SUPPORTED_RAW_CLIP_FRAMES = (17, 18, 19)


def make_wan22_stride2_video_offsets(
    *, num_frames: int = 33,
) -> tuple[int, ...]:
    """Return the exact RGB offsets used by the stride-2 Wan experiment.

    ``num_frames`` describes the contiguous action window, so the default
    produces ``(0, 2, ..., 32)``.  A non-default value is accepted only when
    its final offset is still aligned to the VAE's four-frame temporal ratio;
    this prevents silently changing the five-latent geometry.
    """
    if not isinstance(num_frames, int) or isinstance(num_frames, bool):
        raise ValueError("num_frames must be an integer")
    if num_frames != 33:
        raise ValueError(
            "the Wan2.2 stride-2 T=5 experiment requires num_frames=33"
        )
    return tuple(range(0, WAN22_STRIDE2_VAE_INPUT_FRAMES * 2, 2))


def build_wan22_stride2_predictor_token_ids(
    num_views: int = WAN22_STRIDE2_NUM_VIEWS,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return context ``t0`` IDs and future ``t1..t4`` IDs."""
    if not isinstance(num_views, int) or isinstance(num_views, bool) or num_views <= 0:
        raise ValueError("num_views must be a positive integer")
    token_ids = torch.arange(
        WAN22_STRIDE2_PREDICTOR_DEPTH
        * num_views
        * WAN22_STRIDE2_PATCHES_PER_STEP,
        dtype=torch.long,
        device=device,
    ).reshape(
        WAN22_STRIDE2_PREDICTOR_DEPTH,
        num_views,
        WAN22_GRID_SIZE,
        WAN22_GRID_SIZE,
    )
    context_ids = token_ids[0].reshape(-1)
    future_ids = token_ids[1:].reshape(-1)
    return context_ids, future_ids


def build_wan22_stride2_policy_mask(
    *,
    image_size: tuple[int, int],
    num_views: int = WAN22_STRIDE2_NUM_VIEWS,
):
    """Build a dense ``t0 -> {t1,t2,t3,t4}`` world-prediction mask."""
    if tuple(image_size) != (224, 224):
        raise ValueError(
            f"Wan2.2 stride-2 ablation requires 224x224 views, got {image_size}"
        )
    if num_views != WAN22_STRIDE2_NUM_VIEWS:
        raise ValueError(
            "Wan2.2 stride-2 ablation requires exactly "
            f"{WAN22_STRIDE2_NUM_VIEWS} views, got {num_views}"
        )
    context_ids, future_ids = build_wan22_stride2_predictor_token_ids(num_views)
    return SimpleNamespace(
        ctx_idx=context_ids,
        tgt_idx=future_ids,
        n_ctx=context_ids.numel(),
    )


class _Wan22VAE38Stride2EncoderCore(_Wan22VAE38EncoderCore):
    """Wan VAE core whose temporal validator also permits a 17-frame input.

    The upstream implementation processes a first frame followed by groups of
    four frames.  Its original adapter validates only ``1/5/9`` frames; the
    same chunking naturally yields ``1/2/3/4/5`` latents for
    ``1/5/9/13/17`` frames, so only the validation lives here.
    """

    _SUPPORTED_INPUT_FRAMES = (1, 5, 9, 13, 17)
    # The chunking implementation is valid for every 1+4n input.  Keep the
    # tuple for historical validation/documentation, while registry compact
    # adapters use the dynamic contract just like the base VAE core.
    _supports_dynamic_temporal = True

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if (
            video.ndim == 5
            and video.shape[1] == 3
            and video.shape[2] > 0
            and (video.shape[2] - 1) % 4 == 0
            and video.shape[2] not in self._SUPPORTED_INPUT_FRAMES
        ):
            return super().forward(video)
        if (
            video.ndim != 5
            or video.shape[1] != 3
            or video.shape[2] not in self._SUPPORTED_INPUT_FRAMES
        ):
            raise ValueError(
                "Wan2.2 stride-2 encoder core expects [B,3,T,H,W] with T in "
                f"{self._SUPPORTED_INPUT_FRAMES}, got {tuple(video.shape)}"
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


class Wan22Stride2T5LatentEncoder(nn.Module):
    """Frozen Wan2.2 VAE tokens on the native five-step temporal lattice.

    The adapter accepts a 17-frame clip ``[0,2,...,32]``.  For convenience it
    also accepts 18/19-frame clips with one/two leading history frames and
    always takes the final 17 frames, preserving the physical ``0..32``
    window.  A local mask selects logical ``t0..t4`` tokens; the VAE input is
    truncated to ``1 + 4 * max(mask_time)`` frames for efficiency.
    """

    embed_dim = WAN22_LATENT_CHANNELS
    patch_size = WAN22_PATCH_SIZE
    num_frames = WAN22_STRIDE2_POLICY_FRAMES
    latent_frames = WAN22_STRIDE2_LATENT_FRAMES
    grid_size = WAN22_GRID_SIZE
    logical_grid_depth = WAN22_STRIDE2_PREDICTOR_DEPTH
    logical_temporal_ids = WAN22_STRIDE2_LOGICAL_TEMPORAL_IDS
    # Raw RGB sampling is frame-wise; temporal tubelets are VAE latents.
    tubelet_size = 1
    num_views = WAN22_STRIDE2_NUM_VIEWS
    patches_per_step = WAN22_STRIDE2_PATCHES_PER_STEP
    num_patches_per_view = WAN22_STRIDE2_NUM_PATCHES_PER_VIEW
    num_patches = None

    def __init__(
        self,
        encoder_core: nn.Module,
        *,
        video_microbatch_size: int | None = None,
        num_views: int | None = WAN22_STRIDE2_NUM_VIEWS,
    ):
        super().__init__()
        if video_microbatch_size is not None and video_microbatch_size <= 0:
            raise ValueError("video_microbatch_size must be positive")
        self.encoder_core = encoder_core.eval().requires_grad_(False)
        self.video_microbatch_size = video_microbatch_size
        if num_views is not None and (not isinstance(num_views, int) or num_views <= 0):
            raise ValueError("num_views must be a positive integer when configured")
        self.num_views = num_views
        self._vjepa_policy_family = "wan2_2_stride2_t5"
        self._vjepa_policy_model_name = "vae38"
        self._vjepa_policy_checkpoint_key = "encoder_quant_conv"
        self._vjepa_policy_raw_frame_stride = WAN22_STRIDE2_POLICY_FRAME_STRIDE
        self._vjepa_policy_latent_frames = WAN22_STRIDE2_LATENT_FRAMES
        self.blocks = (
            SimpleNamespace(
                attn=SimpleNamespace(
                    interpolate_rope=False,
                    corrected_frequency_pairing=False,
                )
            ),
        )

    def train(self, mode: bool = True) -> "Wan22Stride2T5LatentEncoder":
        super().train(False)
        return self

    def build_policy_mask(self, *, image_size, num_views):
        return build_wan22_stride2_policy_mask(
            image_size=tuple(image_size),
            num_views=num_views,
        )

    @staticmethod
    def _validate_clips(clips: torch.Tensor) -> None:
        if clips.ndim not in (5, 6):
            raise ValueError(
                "Wan2.2 stride-2 clips must be [B,C,T,H,W] or [B,V,C,T,H,W], "
                f"got {tuple(clips.shape)}"
            )
        channels, frames, height, width = clips.shape[-4:]
        if (
            channels,
            frames,
            height,
            width,
        ) not in tuple(
            (3, raw_frames, 224, 224)
            for raw_frames in WAN22_STRIDE2_SUPPORTED_RAW_CLIP_FRAMES
        ):
            raise ValueError(
                "Wan2.2 stride-2 clips must have trailing shape "
                f"[3,T,224,224] with T in {WAN22_STRIDE2_SUPPORTED_RAW_CLIP_FRAMES}, "
                f"got {tuple(clips.shape[-4:])}"
            )
        if not torch.is_floating_point(clips):
            raise TypeError(
                "Wan2.2 stride-2 clips must be floating point and normalized to [-1, 1]"
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
                f"Wan2.2 stride-2 VAE38 encoder must return {expected}, got {observed}"
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
                raise ValueError(
                    "Wan2.2 stride-2 mask index is outside the logical T=5 lattice"
                )
            temporal_ids = torch.div(
                mask,
                self.patches_per_step,
                rounding_mode="floor",
            )
            if torch.any(temporal_ids < 0) or torch.any(
                temporal_ids >= self.logical_grid_depth
            ):
                raise ValueError("Wan2.2 stride-2 mask time is outside t0..t4")
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
            raise ValueError(
                f"clips must be [B,C,T,H,W], got {tuple(clips.shape)}"
            )
        batch_size, _, raw_frames, _, _ = clips.shape
        current_index = raw_frames - WAN22_STRIDE2_VAE_INPUT_FRAMES
        if masks is None:
            mask_list = None
            max_logical_temporal = WAN22_STRIDE2_LATENT_FRAMES - 1
        else:
            mask_list = self._canonical_masks(masks, batch_size, clips.device)
            max_logical_temporal = max(
                int(mask.max().item()) // self.patches_per_step
                for mask in mask_list
            )

        vae_frames = 1 + 4 * max_logical_temporal
        vae_input = clips[
            :, :, current_index : current_index + vae_frames
        ].contiguous()
        latents = self._encode_video(vae_input)
        expected_latent_frames = max_logical_temporal + 1
        self._validate_latents(latents, batch_size, expected_latent_frames)
        patches = latents.permute(0, 2, 3, 4, 1).reshape(
            batch_size,
            expected_latent_frames,
            self.patches_per_step,
            self.embed_dim,
        )

        if mask_list is None:
            return patches.flatten(1, 2)

        batch_ids = torch.arange(batch_size, device=clips.device).unsqueeze(1)
        selected = []
        for mask in mask_list:
            temporal = torch.div(
                mask,
                self.patches_per_step,
                rounding_mode="floor",
            )
            spatial = mask.remainder(self.patches_per_step)
            selected.append(patches[batch_ids, temporal, spatial])
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
                "masked 6D clips must use a view-localized mask before calling "
                "Wan22Stride2T5LatentEncoder"
            )
        batch_size, num_views, channels, frames, height, width = clips.shape
        if self.num_views is not None and num_views != self.num_views:
            raise ValueError(
                f"configured Wan2.2 stride-2 adapter expects {self.num_views} views, "
                f"got {num_views}"
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


def load_wan22_stride2_t5_latent_encoder(
    checkpoint_path: str | Path,
    *,
    fastwam_src_dir: str | Path = DEFAULT_FASTWAM_SRC_DIR,
    video_microbatch_size: int | None = None,
) -> Wan22Stride2T5LatentEncoder:
    """Load only the Wan2.2 VAE38 encoder weights for the stride-2 variant."""
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
        encoder_core = _Wan22VAE38Stride2EncoderCore(
            implementation,
            checkpoint_format=checkpoint_format,
        )
    encoder_core.load_state_dict(state_dict, strict=True, assign=True)
    encoder_core.materialize_scale_buffers()
    return Wan22Stride2T5LatentEncoder(
        encoder_core,
        video_microbatch_size=video_microbatch_size,
    )


def load_wan22_stride2_compact_latent_encoder(
    checkpoint_path: str | Path,
    *,
    fastwam_src_dir: str | Path = DEFAULT_FASTWAM_SRC_DIR,
    video_microbatch_size: int | None = None,
    num_views: int | None = None,
) -> Wan22CompactLatentEncoder:
    """Load the stride-2 VAE core while preserving its native T dynamically.

    This registry path intentionally returns the compact adapter instead of
    :class:`Wan22Stride2T5LatentEncoder`: the latter remains available for the
    historical fixed-T=5 geometry experiment.
    """
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
        encoder_core = _Wan22VAE38Stride2EncoderCore(
            implementation,
            checkpoint_format=checkpoint_format,
        )
    encoder_core.load_state_dict(state_dict, strict=True, assign=True)
    encoder_core.materialize_scale_buffers()
    adapter = Wan22CompactLatentEncoder(
        encoder_core,
        video_microbatch_size=video_microbatch_size,
        num_views=num_views,
    )
    adapter._vjepa_policy_family = "wan2_2_stride2"
    adapter._vjepa_policy_model_name = "vae38_compact"
    adapter._vjepa_policy_checkpoint_key = "encoder_quant_conv"
    return adapter
