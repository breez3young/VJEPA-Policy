"""V-JEPA policy with latent world modeling and flow-matching actions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_policy.models.backbone.masks import apply_masks
from vjepa_policy.models.action_expert import ActionExpert
from vjepa_policy.state import packed_state_dim, prepare_proprio_state


def encoded_proprio_dim(proprio_dim, encoding="identity"):
    if proprio_dim <= 0:
        raise ValueError("proprio_dim must be positive")
    if encoding == "identity":
        return proprio_dim
    if encoding == "sincos":
        return 2 * proprio_dim
    raise ValueError(f"unsupported proprio encoding {encoding!r}")


def encode_proprio(state, encoding="identity"):
    if encoding == "identity":
        return state
    if encoding == "sincos":
        return torch.cat((torch.sin(state), torch.cos(state)), dim=-1)
    raise ValueError(f"unsupported proprio encoding {encoding!r}")


def _local_view_masks(mask, *, num_views, patches_per_view):
    """Convert global t-view-patch indices to per-view t-patch indices."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be [B,K], got {tuple(mask.shape)}")
    tokens_per_step = num_views * patches_per_view
    temporal = mask // tokens_per_step
    within_step = mask % tokens_per_step
    view = within_step // patches_per_view
    spatial = within_step % patches_per_view
    local = temporal * patches_per_view + spatial

    if mask.shape[1] % num_views:
        raise ValueError("mask token count must be divisible by num_views")
    expected_count = mask.shape[1] // num_views
    view_counts = torch.stack(
        [(view == view_index).sum(dim=1) for view_index in range(num_views)],
        dim=1,
    )
    if not torch.all(view_counts == expected_count):
        raise ValueError("each sample and view must contribute the same number of mask tokens")

    token_positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    per_view = []
    for view_index in range(num_views):
        positions = torch.where(
            view == view_index,
            token_positions,
            mask.shape[1],
        )
        positions = positions.sort(dim=1).values[:, :expected_count]
        per_view.append(torch.gather(local, 1, positions))
    return torch.stack(per_view, dim=1).reshape(mask.shape[0] * num_views, -1)


def encode_video_views(encoder, clips, mask=None):
    """Encode one composite clip or independently encode a stack of camera views."""
    def run_encoder(inputs, selection):
        if getattr(encoder, "_vjepa_policy_training_false", False):
            return encoder(inputs, selection, training=False)
        return encoder(inputs, selection)

    if clips.ndim != 6:
        return run_encoder(clips, mask)

    batch_size, num_views, channels, frames, height, width = clips.shape
    patch_size = int(getattr(encoder, "patch_size", 1))
    if patch_size <= 0:
        raise ValueError("encoder.patch_size must be positive")
    if height % patch_size or width % patch_size:
        raise ValueError("view height and width must be divisible by encoder.patch_size")
    tubelet_size = int(getattr(encoder, "tubelet_size", 1))
    if tubelet_size <= 0:
        raise ValueError("encoder.tubelet_size must be positive")
    # V-JEPA tubelets require complete temporal groups.  Frame-sampled and
    # causal-compressed adapters use this field as a stride hint only and can
    # validly consume a final partial stride; their adapter output determines
    # the compact temporal depth.
    encoder_spec = getattr(encoder, "spec", None)
    temporal_mode = getattr(
        encoder_spec,
        "temporal_mode",
        getattr(encoder, "temporal_mode", "tubelet"),
    )
    if frames % tubelet_size and temporal_mode == "tubelet":
        raise ValueError("frame count must be divisible by encoder.tubelet_size")

    # ``patch_size`` describes the logical policy grid exposed by an adapter.
    # Some image encoders (DINO) use a different native patch size internally,
    # while still returning this policy grid.
    patches_per_view = (height // patch_size) * (width // patch_size)
    flat_clips = clips.reshape(
        batch_size * num_views, channels, frames, height, width
    )
    local_mask = None
    if mask is not None:
        local_mask = _local_view_masks(
            mask,
            num_views=num_views,
            patches_per_view=patches_per_view,
        )

    latents = run_encoder(flat_clips, local_mask)
    if latents.ndim != 3:
        raise ValueError(
            f"multi-view encoder output must be [B,N,D], got {tuple(latents.shape)}"
        )
    tokens_per_view = latents.shape[1]
    if tokens_per_view % patches_per_view:
        raise ValueError("multi-view encoder output must contain complete spatial grids")
    selected_steps = tokens_per_view // patches_per_view
    if selected_steps <= 0:
        raise ValueError(
            f"encoder returned no tokens per view (patches_per_view={patches_per_view})"
        )

    feature_dim = latents.shape[-1]
    return (
        latents.reshape(
            batch_size,
            num_views,
            selected_steps,
            patches_per_view,
            feature_dim,
        )
        .permute(0, 2, 1, 3, 4)
        .reshape(batch_size, selected_steps * num_views * patches_per_view, feature_dim)
    )


class VJEPAPolicy(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_expert,
        *,
        context_len,
        proprio_dim,
        proprio_encoding="identity",
        max_state_dim=None,
        loss_exp=1.0,
        world_loss_weight=1.0,
        action_loss_weight=1.0,
        patch_valid=None,
        latent_layout=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_expert = action_expert
        self.raw_proprio_dim = proprio_dim
        self.proprio_encoding = proprio_encoding
        self.encoded_proprio_dim = encoded_proprio_dim(
            proprio_dim, proprio_encoding
        )
        self.max_state_dim = (
            None if max_state_dim is None else int(max_state_dim)
        )
        if (
            self.max_state_dim is not None
            and self.encoded_proprio_dim > self.max_state_dim
        ):
            raise ValueError(
                f"encoded state dimension {self.encoded_proprio_dim} exceeds "
                f"max_state_dim={self.max_state_dim}"
            )
        self.packed_proprio_dim = (
            self.encoded_proprio_dim
            if self.max_state_dim is None
            else packed_state_dim(self.max_state_dim)
        )
        # Retain the published initialization sequence for exact seeded
        # reproducibility; state conditioning is implemented inside both experts.
        self.proprio_encoder = nn.Linear(proprio_dim, predictor.lang_dim)
        self.context_len = context_len
        self.loss_exp = loss_exp
        self.world_loss_weight = world_loss_weight
        self.action_loss_weight = action_loss_weight
        self.latent_layout = latent_layout
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if patch_valid is None:
            self.patch_valid = None
        else:
            self.register_buffer("patch_valid", patch_valid.bool(), persistent=False)
        self.proprio_encoder = nn.Identity()

    def forward(self, batch):
        clips, language, language_mask, masks_enc, masks_pred, action, action_pad, state = batch
        if clips.ndim == 6 and clips.shape[1] != getattr(self.predictor, "num_views", 1):
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

        # Encode the complete compact lattice once.  The target encoder must
        # see the same temporal/spatial context as the pretrained V-JEPA
        # target path; masks only select which compact tokens contribute to the
        # loss.  Dynamic adapters (for example Wan) expose their actual T, so
        # this does not create placeholder time slots.
        with torch.no_grad():
            target_latents = encode_video_views(self.encoder, clips)
            target_latents = F.layer_norm(
                target_latents, (target_latents.size(-1),)
            )
        if self.latent_layout is not None:
            expected_tokens = self.latent_layout.total_tokens
            if target_latents.shape[1] != expected_tokens:
                raise ValueError(
                    f"encoder returned {target_latents.shape[1]} compact tokens, "
                    f"but layout requires {expected_tokens}"
                )

        loss_world = clips.new_zeros(())
        cosine = clips.new_zeros(())
        predictor_kv = None
        for mask_enc, mask_pred in zip(masks_enc, masks_pred):
            if self.latent_layout is not None:
                self.latent_layout.validate_mask(mask_enc, target=False)
                self.latent_layout.validate_mask(mask_pred, target=True)
            with torch.no_grad():
                context_latents = encode_video_views(self.encoder, clips, mask_enc)
            prediction, predictor_kv = self.predictor(
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

        dof_mask = (~action_pad).unsqueeze(-1).expand_as(action).float()
        predicted_velocity, target_velocity = self.action_expert(
            predictor_kv,
            action,
            language,
            language_mask,
            state,
            context_len=self.context_len,
            dof_mask=dof_mask,
        )
        valid_actions = (~action_pad).float()
        per_action = F.mse_loss(
            predicted_velocity, target_velocity, reduction="none"
        ).mean(dim=-1)
        loss_action = (
            (per_action * valid_actions).sum()
            / valid_actions.sum().clamp_min(1.0)
        )
        loss = (
            self.world_loss_weight * loss_world
            + self.action_loss_weight * loss_action
        )
        self.forward_dtypes = {
            "clips": clips.dtype,
            "encoder": target_latents.dtype,
            "predictor_k": predictor_kv[0][0].dtype,
            "predictor_v": predictor_kv[0][1].dtype,
            "action": predicted_velocity.dtype,
            "loss": loss.dtype,
        }
        return loss, {
            "loss": loss.detach(),
            "loss_world": loss_world.detach(),
            "loss_action": loss_action.detach(),
            "cos_sim": (cosine / mask_count).detach(),
        }


def build_vjepa_policy(
    encoder,
    predictor,
    *,
    action_dim,
    action_chunk_size,
    context_len,
    proprio_dim,
    proprio_encoding="identity",
    max_state_dim=None,
    action_hidden_size=512,
    action_num_layers=24,
    condition_num_heads=8,
    action_num_inference_steps=10,
    loss_exp=1.0,
    world_loss_weight=1.0,
    action_loss_weight=1.0,
    patch_valid=None,
    latent_layout=None,
):
    if len(predictor.predictor_blocks) != action_num_layers:
        raise ValueError("predictor depth must equal action_num_layers")
    predictor_attention = predictor.predictor_blocks[0].self_attn
    encoded_state_dim = encoded_proprio_dim(proprio_dim, proprio_encoding)
    if max_state_dim is not None and encoded_state_dim > int(max_state_dim):
        raise ValueError(
            f"encoded state dimension {encoded_state_dim} exceeds "
            f"max_state_dim={max_state_dim}"
        )
    state_dim = (
        encoded_state_dim
        if max_state_dim is None
        else packed_state_dim(max_state_dim)
    )
    if predictor.proprio_dim != state_dim:
        raise ValueError(
            f"predictor proprio_dim={predictor.proprio_dim} does not match encoded "
            f"state dimension {state_dim}"
        )
    action_expert = ActionExpert(
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        num_layers=action_num_layers,
        predictor_num_heads=predictor_attention.num_heads,
        predictor_head_dim=predictor_attention.head_dim,
        state_dim=state_dim,
        hidden_size=action_hidden_size,
        lang_dim=predictor.lang_dim,
        condition_num_heads=condition_num_heads,
        num_inference_steps=action_num_inference_steps,
    )
    return VJEPAPolicy(
        encoder,
        predictor,
        action_expert,
        context_len=context_len,
        proprio_dim=proprio_dim,
        proprio_encoding=proprio_encoding,
        max_state_dim=max_state_dim,
        loss_exp=loss_exp,
        world_loss_weight=world_loss_weight,
        action_loss_weight=action_loss_weight,
        patch_valid=patch_valid,
        latent_layout=latent_layout,
    )
