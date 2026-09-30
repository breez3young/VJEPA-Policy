"""Language- and proprio-conditioned predictor for future V-JEPA latents."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_policy.models.backbone.masks import apply_masks
from vjepa_policy.models.backbone.modules import DropPath, MLP, RoPEAttention
from vjepa_policy.models.backbone.tensors import trunc_normal_


class ConditionCrossAttention(nn.Module):
    """Cross-attention over projected T5 tokens and one proprio token."""

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, queries, condition, condition_valid):
        batch_size, query_length, dim = queries.shape
        condition_length = condition.shape[1]
        q = (
            self.q(queries)
            .view(batch_size, query_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k(condition)
            .view(batch_size, condition_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v(condition)
            .view(batch_size, condition_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=condition_valid[:, None, None, :],
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch_size, query_length, dim)
        return self.proj_drop(self.proj(output))


class WorldModelPredictorBlock(nn.Module):
    """3D-RoPE self-attention, condition cross-attention, then FFN."""

    def __init__(
        self,
        dim,
        num_heads,
        *,
        grid_size,
        patch_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        interpolate_rope=True,
        corrected_rope_frequency_pairing=False,
        canonical_spatial_grid=None,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = RoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            grid_size=grid_size,
            interpolate_rope=interpolate_rope,
            patch_size=patch_size,
            corrected_frequency_pairing=corrected_rope_frequency_pairing,
            canonical_spatial_grid=canonical_spatial_grid,
        )
        self.cross_norm = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = ConditionCrossAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = MLP(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x,
        condition,
        condition_valid,
        position_ids,
        *,
        T,
        H_patches,
        W_patches,
        return_kv=False,
    ):
        attention_output = self.self_attn(
            self.self_norm(x),
            mask=position_ids,
            T=T,
            H_patches=H_patches,
            W_patches=W_patches,
            return_kv=return_kv,
        )
        if return_kv:
            attention_output, key, value = attention_output
        x = x + self.drop_path(attention_output)
        x = x + self.drop_path(
            self.cross_attn(self.cross_norm(x), condition, condition_valid)
        )
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        if return_kv:
            return x, key, value
        return x


class WorldModelPredictor(nn.Module):
    """Predict future frozen-encoder latents with a multimodal transformer.

    Context and future-mask tokens share unrestricted self-attention, matching
    the strongest existing policy. Every block explicitly returns its post-RoPE
    key/value tensors for a layer-aligned Action Expert.
    """

    def __init__(
        self,
        img_size=(256, 256),
        patch_size=16,
        num_frames=10,
        tubelet_size=2,
        embed_dim=1408,
        predictor_embed_dim=1024,
        out_embed_dim=None,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_mask_tokens=10,
        lang_dim=4096,
        proprio_dim=8,
        use_activation_checkpointing=False,
        activation_checkpointing_blocks=None,
        interpolate_rope=True,
        corrected_rope_frequency_pairing=False,
        num_views=1,
        max_views=None,
        latent_grid=None,
        canonical_spatial_grid=None,
        init_std=0.02,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if len(img_size) != 2 or any(size <= 0 for size in img_size):
            raise ValueError("img_size must contain two positive dimensions")
        if patch_size <= 0 or tubelet_size <= 0:
            raise ValueError("patch_size and tubelet_size must be positive")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if latent_grid is not None:
            if len(latent_grid) != 3 or any(
                not isinstance(size, int) or size <= 0 for size in latent_grid
            ):
                raise ValueError("latent_grid must be a positive (T, H, W) tuple")
            latent_grid = tuple(int(size) for size in latent_grid)
            grid_depth, grid_height, grid_width = latent_grid
        else:
            if img_size[0] % patch_size or img_size[1] % patch_size:
                raise ValueError("img_size must be divisible by patch_size")
            if num_frames % tubelet_size:
                raise ValueError("num_frames must be divisible by tubelet_size")
            grid_depth = num_frames // tubelet_size
            grid_height = img_size[0] // patch_size
            grid_width = img_size[1] // patch_size
        if predictor_embed_dim % num_heads:
            raise ValueError("predictor_embed_dim must be divisible by num_heads")
        if num_mask_tokens <= 0:
            raise ValueError("num_mask_tokens must be positive")
        if not isinstance(num_views, int) or num_views <= 0:
            raise ValueError("num_views must be a positive integer")
        if max_views is None:
            max_views = num_views
        if not isinstance(max_views, int) or max_views < num_views:
            raise ValueError("max_views must be an integer greater than or equal to num_views")

        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_depth = grid_depth
        self.latent_grid = (grid_depth, grid_height, grid_width)
        self.num_views = num_views
        self.max_views = max_views
        self.num_patches_per_view = self.grid_depth * self.grid_height * self.grid_width
        self.num_patches = self.num_views * self.num_patches_per_view
        self.predictor_embed_dim = predictor_embed_dim
        self.lang_dim = lang_dim
        self.proprio_dim = proprio_dim
        if activation_checkpointing_blocks is None:
            activation_checkpointing_blocks = depth if use_activation_checkpointing else 0
        if not 0 <= activation_checkpointing_blocks <= depth:
            raise ValueError(
                "activation_checkpointing_blocks must be between 0 and depth"
            )
        self.activation_checkpointing_blocks = activation_checkpointing_blocks
        self.use_activation_checkpointing = activation_checkpointing_blocks > 0
        self.init_std = init_std

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        # Preserve the exact single-view state dict used by existing checkpoints.
        self.view_embedding = (
            nn.Embedding(max_views, predictor_embed_dim) if max_views > 1 else None
        )
        self.mask_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for _ in range(num_mask_tokens)
            ]
        )
        self.num_mask_tokens = num_mask_tokens

        self.language_norm = nn.LayerNorm(lang_dim, eps=1e-6)
        self.language_projection = nn.Linear(lang_dim, predictor_embed_dim)
        self.proprio_norm = nn.LayerNorm(proprio_dim, eps=1e-6)
        self.proprio_projection = nn.Linear(proprio_dim, predictor_embed_dim)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.predictor_blocks = nn.ModuleList(
            [
                WorldModelPredictorBlock(
                    predictor_embed_dim,
                    num_heads,
                    grid_size=self.grid_height,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[index],
                    interpolate_rope=interpolate_rope,
                    corrected_rope_frequency_pairing=corrected_rope_frequency_pairing,
                    canonical_spatial_grid=canonical_spatial_grid,
                )
                for index in range(depth)
            ]
        )
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim,
            embed_dim if out_embed_dim is None else out_embed_dim,
        )

        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            trunc_normal_(module.weight, std=self.init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _rescale_blocks(self):
        for layer_id, block in enumerate(self.predictor_blocks, start=1):
            scale = math.sqrt(3.0 * layer_id)
            block.self_attn.proj.weight.data.div_(scale)
            block.cross_attn.proj.weight.data.div_(scale)
            block.ffn.fc2.weight.data.div_(scale)

    def _condition_tokens(self, language, language_mask, proprio):
        if language.ndim != 3 or language.shape[-1] != self.lang_dim:
            raise ValueError(
                f"language must be [B,S,{self.lang_dim}], got {tuple(language.shape)}"
            )
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"proprio must be [B,{self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        if language.shape[0] != proprio.shape[0]:
            raise ValueError("language and proprio batch sizes must match")
        if language_mask is None:
            language_mask = torch.ones(
                language.shape[:2], dtype=torch.bool, device=language.device
            )
        elif language_mask.shape != language.shape[:2]:
            raise ValueError(
                f"language_mask must be {tuple(language.shape[:2])}, "
                f"got {tuple(language_mask.shape)}"
            )
        language_tokens = self.language_projection(self.language_norm(language))
        proprio_token = self.proprio_projection(self.proprio_norm(proprio)).unsqueeze(1)
        condition = torch.cat((language_tokens, proprio_token), dim=1)
        proprio_valid = torch.ones(
            (language.shape[0], 1), dtype=torch.bool, device=language.device
        )
        condition_valid = torch.cat((language_mask.bool(), proprio_valid), dim=1)
        return condition, condition_valid

    @staticmethod
    def _match_batch(tensor, batch_size):
        if tensor.shape[0] == batch_size:
            return tensor
        if batch_size % tensor.shape[0]:
            raise ValueError(
                f"cannot expand condition batch {tensor.shape[0]} to {batch_size}"
            )
        repeats = batch_size // tensor.shape[0]
        return tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))

    def _selection_to_view_ids(self, selection_ids):
        """Return camera IDs for global IDs laid out as ``t-view-h-w``."""
        patches_per_frame = self.grid_height * self.grid_width
        return torch.div(
            selection_ids, patches_per_frame, rounding_mode="floor"
        ).remainder(self.num_views)

    def _selection_to_rope_ids(self, selection_ids):
        """Map global selection IDs to shared per-view ``t-h-w`` RoPE IDs."""
        patches_per_frame = self.grid_height * self.grid_width
        temporal_ids = torch.div(
            selection_ids,
            self.num_views * patches_per_frame,
            rounding_mode="floor",
        )
        spatial_ids = selection_ids.remainder(patches_per_frame)
        return temporal_ids * patches_per_frame + spatial_ids

    def _add_view_embeddings(self, tokens, selection_ids):
        if self.view_embedding is None:
            return tokens
        if tokens.shape[:-1] != selection_ids.shape:
            raise ValueError(
                "tokens and selection_ids must have matching batch/token dimensions"
            )
        return tokens + self.view_embedding(self._selection_to_view_ids(selection_ids))

    def forward(
        self,
        context,
        masks_x,
        masks_y,
        *,
        language,
        language_mask,
        proprio,
        mask_index=1,
        return_prediction=True,
    ):
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]
        if len(masks_x) != len(masks_y):
            raise ValueError(
                "masks_x and masks_y must contain the same number of masks"
            )
        if not masks_x:
            raise ValueError("at least one context/future mask pair is required")

        condition, condition_valid = self._condition_tokens(
            language, language_mask, proprio
        )
        context = self.predictor_embed(context)
        batch_size = context.shape[0] // len(masks_x)
        context_length = context.shape[1]

        mask_index %= self.num_mask_tokens
        future = self.mask_tokens[mask_index].expand(batch_size, self.num_patches, -1)
        future = apply_masks(future, masks_y)
        context = context.repeat(len(masks_x), 1, 1)

        masks_x = torch.cat(masks_x, dim=0)
        masks_y = torch.cat(masks_y, dim=0)
        context = self._add_view_embeddings(context, masks_x)
        future = self._add_view_embeddings(future, masks_y)
        tokens = torch.cat((context, future), dim=1)

        selection_ids = torch.cat((masks_x, masks_y), dim=1)
        argsort = torch.argsort(selection_ids, dim=1)
        selection_ids = torch.gather(selection_ids, 1, argsort)
        tokens = torch.gather(
            tokens,
            1,
            argsort.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        position_ids = self._selection_to_rope_ids(selection_ids)
        condition = self._match_batch(condition, tokens.shape[0])
        condition_valid = self._match_batch(condition_valid, tokens.shape[0])

        predictor_kv = []
        for block_index, block in enumerate(self.predictor_blocks):
            if block_index < self.activation_checkpointing_blocks and self.training:
                tokens, key, value = torch.utils.checkpoint.checkpoint(
                    block,
                    tokens,
                    condition,
                    condition_valid,
                    position_ids,
                    T=self.grid_depth,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    return_kv=True,
                    use_reentrant=False,
                )
            else:
                tokens, key, value = block(
                    tokens,
                    condition,
                    condition_valid,
                    position_ids,
                    T=self.grid_depth,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    return_kv=True,
                )
            predictor_kv.append(
                (
                    key[:, :, :context_length].contiguous(),
                    value[:, :, :context_length].contiguous(),
                )
            )

        if not return_prediction:
            return None, tuple(predictor_kv)

        tokens = self.predictor_norm(tokens)
        reverse_argsort = torch.argsort(argsort, dim=1)
        tokens = torch.gather(
            tokens,
            1,
            reverse_argsort.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        prediction = self.predictor_proj(tokens[:, context_length:])
        return prediction, tuple(predictor_kv)


def build_world_model_predictor(**kwargs):
    """Build the language- and proprio-conditioned latent predictor."""
    return WorldModelPredictor(**kwargs)
