"""Predictor-only pretraining with a frozen V-JEPA encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_policy.models.backbone.masks import apply_masks
from vjepa_policy.models.vjepa_policy import (
    encode_video_views,
    encoded_proprio_dim,
)
from vjepa_policy.state import packed_state_dim, prepare_proprio_state


class PredictorPretraining(nn.Module):
    """Train the language-conditioned world predictor without an Action Expert."""

    def __init__(
        self,
        encoder,
        predictor,
        *,
        proprio_dim,
        proprio_encoding="identity",
        max_state_dim=None,
        loss_exp=1.0,
        patch_valid=None,
        latent_layout=None,
    ):
        super().__init__()
        state_dim = encoded_proprio_dim(proprio_dim, proprio_encoding)
        if max_state_dim is not None and state_dim > int(max_state_dim):
            raise ValueError(
                f"encoded state dimension {state_dim} exceeds max_state_dim={max_state_dim}"
            )
        prepared_state_dim = (
            state_dim
            if max_state_dim is None
            else packed_state_dim(max_state_dim)
        )
        if predictor.proprio_dim != prepared_state_dim:
            raise ValueError(
                f"predictor proprio_dim={predictor.proprio_dim} does not match "
                f"prepared state dimension {prepared_state_dim}"
            )
        self.encoder = encoder
        self.predictor = predictor
        self.raw_proprio_dim = proprio_dim
        self.proprio_encoding = proprio_encoding
        self.encoded_proprio_dim = state_dim
        self.max_state_dim = (
            None if max_state_dim is None else int(max_state_dim)
        )
        self.packed_proprio_dim = prepared_state_dim
        self.loss_exp = loss_exp
        self.latent_layout = latent_layout
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if patch_valid is None:
            self.patch_valid = None
        else:
            self.register_buffer("patch_valid", patch_valid.bool(), persistent=False)

    def forward(self, batch):
        clips, language, language_mask, masks_enc, masks_pred, state = batch
        if clips.ndim == 6 and clips.shape[1] != getattr(
            self.predictor, "num_views", 1
        ):
            raise ValueError(
                f"clip view count {clips.shape[1]} does not match Predictor "
                f"num_views={getattr(self.predictor, 'num_views', 1)}"
            )
        if state.ndim != 2 or state.shape[-1] != self.raw_proprio_dim:
            raise ValueError(
                f"state must be [B,{self.raw_proprio_dim}], got {tuple(state.shape)}"
            )
        state = prepare_proprio_state(
            state,
            raw_state_dim=self.raw_proprio_dim,
            encoding=self.proprio_encoding,
            max_state_dim=self.max_state_dim,
        )

        with torch.no_grad():
            target_latents = encode_video_views(self.encoder, clips)
            target_latents = F.layer_norm(target_latents, (target_latents.size(-1),))
        if self.latent_layout is not None:
            expected_tokens = self.latent_layout.total_tokens
            if target_latents.shape[1] != expected_tokens:
                raise ValueError(
                    f"encoder returned {target_latents.shape[1]} compact tokens, "
                    f"but layout requires {expected_tokens}"
                )

        loss_world = clips.new_zeros(())
        cosine = clips.new_zeros(())
        for mask_enc, mask_pred in zip(masks_enc, masks_pred):
            if self.latent_layout is not None:
                self.latent_layout.validate_mask(mask_enc, target=False)
                self.latent_layout.validate_mask(mask_pred, target=True)
            with torch.no_grad():
                context_latents = encode_video_views(self.encoder, clips, mask_enc)
            prediction, _ = self.predictor(
                context_latents,
                mask_enc,
                mask_pred,
                language=language,
                language_mask=language_mask,
                proprio=state,
            )
            target = apply_masks(target_latents, [mask_pred])
            per_token = (
                torch.mean(torch.abs(prediction - target) ** self.loss_exp, dim=-1)
                / self.loss_exp
            )
            per_token_cosine = F.cosine_similarity(
                prediction.float(), target.float(), dim=-1
            )
            if self.patch_valid is None:
                loss_world = loss_world + per_token.mean()
                cosine = cosine + per_token_cosine.mean()
            else:
                valid_tokens = self.patch_valid[mask_pred].to(per_token.dtype)
                denominator = valid_tokens.sum().clamp_min(1.0)
                loss_world = loss_world + (per_token * valid_tokens).sum() / denominator
                cosine = cosine + (per_token_cosine * valid_tokens).sum() / denominator
        mask_count = len(masks_enc)
        loss_world = loss_world / mask_count
        return loss_world, {
            "loss": loss_world.detach(),
            "loss_world": loss_world.detach(),
            "cos_sim": (cosine / mask_count).detach(),
        }


def build_predictor_pretraining(encoder, predictor, **kwargs):
    return PredictorPretraining(encoder, predictor, **kwargs)
