"""Cross-attention flow-matching Action Expert used by the best policy."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timestep):
        half = self.dim // 2
        exponent = -torch.arange(
            half, dtype=torch.float32, device=timestep.device
        ) * (math.log(10000.0) / half)
        frequencies = timestep.float().unsqueeze(-1) * exponent.exp()
        return torch.cat(
            (torch.sin(frequencies), torch.cos(frequencies)), dim=-1
        ).to(timestep.dtype)


class AdaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(dim, dim * 3)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, condition):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normalized = (x * torch.rsqrt(variance + self.eps)).to(x.dtype)
        scale, shift, gate = self.modulation(condition).chunk(3, dim=-1)
        normalized = normalized * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return normalized, gate.unsqueeze(1)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PredictorKVActionAttention(nn.Module):
    """Action queries attend to Predictor K/V and the action chunk's own K/V."""

    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, inner_dim)
        self.k_proj = nn.Linear(hidden_size, inner_dim)
        self.v_proj = nn.Linear(hidden_size, inner_dim)
        self.o_proj = nn.Linear(inner_dim, hidden_size)

    def forward(self, action_hidden, context_key, context_value):
        batch_size, action_length, _ = action_hidden.shape
        query = self.q_proj(action_hidden).view(
            batch_size, action_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        action_key = self.k_proj(action_hidden).view(
            batch_size, action_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        action_value = self.v_proj(action_hidden).view(
            batch_size, action_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = torch.cat((context_key, action_key), dim=2)
        value = torch.cat((context_value, action_value), dim=2)
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, action_length, -1
        )
        return self.o_proj(output)


class _ReferenceConditionEncoder(nn.Module):
    """Project T5 tokens and one proprio token without pooling them together."""

    def __init__(self, lang_dim, state_dim, hidden_size, max_language_tokens=32):
        super().__init__()
        if max_language_tokens <= 0:
            raise ValueError("max_language_tokens must be positive")
        self.lang_dim = lang_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.max_language_tokens = max_language_tokens
        self.lang_norm = nn.LayerNorm(lang_dim)
        self.lang_projection = nn.Linear(lang_dim, hidden_size)
        self.state_projection = nn.Linear(state_dim, hidden_size)
        self.language_position = nn.Embedding(max_language_tokens, hidden_size)
        self.token_type = nn.Parameter(torch.zeros(2, hidden_size))
        nn.init.normal_(self.language_position.weight, std=0.02)
        nn.init.normal_(self.token_type, std=0.02)

    def forward(self, lang, lang_mask, state):
        if lang.ndim != 3 or lang.shape[-1] != self.lang_dim:
            raise ValueError(f"lang must be [B,S,{self.lang_dim}], got {tuple(lang.shape)}")
        if state.ndim != 2 or state.shape != (lang.shape[0], self.state_dim):
            raise ValueError(
                f"state must be [B,{self.state_dim}], got {tuple(state.shape)}"
            )
        if lang.shape[1] > self.max_language_tokens:
            raise ValueError(
                f"language length {lang.shape[1]} exceeds max {self.max_language_tokens}"
            )
        if lang_mask is None:
            lang_mask = torch.ones(
                lang.shape[:2], dtype=torch.bool, device=lang.device
            )
        elif lang_mask.shape != lang.shape[:2]:
            raise ValueError(
                f"lang_mask must be {tuple(lang.shape[:2])}, got {tuple(lang_mask.shape)}"
            )

        positions = torch.arange(lang.shape[1], device=lang.device)
        language_tokens = self.lang_projection(self.lang_norm(lang))
        language_tokens = (
            language_tokens
            + self.language_position(positions).unsqueeze(0)
            + self.token_type[1]
        )
        state_token = (
            self.state_projection(state) + self.token_type[0]
        ).unsqueeze(1)
        tokens = torch.cat((state_token, language_tokens), dim=1)
        valid = torch.cat(
            (
                torch.ones((lang.shape[0], 1), dtype=torch.bool, device=lang.device),
                lang_mask.bool(),
            ),
            dim=1,
        )
        return tokens, valid


class ConditionCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by condition_num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_projection = nn.Linear(hidden_size, hidden_size)
        self.k_projection = nn.Linear(hidden_size, hidden_size)
        self.v_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, queries, condition_tokens, condition_valid):
        batch_size, query_length, hidden_size = queries.shape
        condition_length = condition_tokens.shape[1]
        q = self.q_projection(queries).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_projection(condition_tokens).view(
            batch_size, condition_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_projection(condition_tokens).view(
            batch_size, condition_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attention_mask = condition_valid[:, None, None, :]
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_length, hidden_size
        )
        return self.output_projection(output)


class ActionExpertBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        predictor_num_heads,
        predictor_head_dim,
        condition_num_heads,
        ffn_mult=4,
    ):
        super().__init__()
        self.condition_norm = nn.RMSNorm(hidden_size)
        self.condition_attention = ConditionCrossAttention(
            hidden_size, condition_num_heads
        )
        self.pre_attn_norm = AdaRMSNorm(hidden_size)
        self.attn = PredictorKVActionAttention(
            hidden_size, predictor_num_heads, predictor_head_dim
        )
        self.pre_ffn_norm = AdaRMSNorm(hidden_size)
        self.ffn = SwiGLUFeedForward(hidden_size, hidden_size * ffn_mult)

    def forward(
        self,
        x,
        ctx_k,
        ctx_v,
        t_cond,
        condition_tokens,
        condition_valid,
    ):
        normed, attention_gate = self.pre_attn_norm(x, t_cond)
        x = x + self.attn(normed, ctx_k, ctx_v) * attention_gate
        x = x + self.condition_attention(
            self.condition_norm(x), condition_tokens, condition_valid
        )
        normed, ffn_gate = self.pre_ffn_norm(x, t_cond)
        return x + self.ffn(normed) * ffn_gate


class _ActionExpertBase(nn.Module):
    """Common flow-matching implementation used by the published Action Expert."""

    def __init__(
        self,
        *,
        action_dim,
        action_chunk_size,
        num_layers,
        predictor_num_heads,
        predictor_head_dim,
        state_dim,
        hidden_size=384,
        ffn_mult=4,
        lang_dim=4096,
        condition_num_heads=8,
        max_language_tokens=32,
        num_inference_steps=10,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        num_timestep_buckets=1000,
    ):
        super().__init__()
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for sinusoidal time embeddings")
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.num_layers = num_layers
        self.num_inference_steps = num_inference_steps
        self.num_timestep_buckets = num_timestep_buckets
        self._beta_alpha = noise_beta_alpha
        self._beta_beta = noise_beta_beta

        self.action_in_proj = nn.Linear(action_dim * 2, hidden_size)
        self.time_sinusoidal = SinusoidalTimeEmbedding(hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.position_embedding = nn.Embedding(action_chunk_size, hidden_size)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        self.conditioner = _ReferenceConditionEncoder(
            lang_dim=lang_dim,
            state_dim=state_dim,
            hidden_size=hidden_size,
            max_language_tokens=max_language_tokens,
        )
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    hidden_size=hidden_size,
                    predictor_num_heads=predictor_num_heads,
                    predictor_head_dim=predictor_head_dim,
                    condition_num_heads=condition_num_heads,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = AdaRMSNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, action_dim)

    def sample_time(self, batch_size, device, dtype):
        beta_dist = Beta(self._beta_alpha, self._beta_beta)
        sample = beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=0.999)
        return (0.999 - sample) / 0.999

    def _time_cond(self, timestep):
        return self.time_mlp(self.time_sinusoidal(timestep.float()))

    def _encode_actions(self, actions, dof_mask):
        dof_mask = dof_mask if dof_mask is not None else torch.ones_like(actions)
        tokens = self.action_in_proj(
            torch.cat((actions, dof_mask.to(actions.dtype)), dim=-1)
        )
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return tokens + self.position_embedding(positions).unsqueeze(0)

    def _run_blocks(
        self,
        tokens,
        predictor_kv,
        context_len,
        t_cond,
        condition_tokens,
        condition_valid,
    ):
        if len(predictor_kv) != self.num_layers:
            raise ValueError(
                f"expert has {self.num_layers} blocks, predictor supplied {len(predictor_kv)}"
            )
        hidden = tokens
        for index, block in enumerate(self.blocks):
            ctx_k, ctx_v = predictor_kv[index]
            if context_len is not None:
                ctx_k = ctx_k[:, :, :context_len]
                ctx_v = ctx_v[:, :, :context_len]
            hidden = block(
                hidden,
                ctx_k,
                ctx_v,
                t_cond,
                condition_tokens,
                condition_valid,
            )
        hidden, _ = self.final_norm(hidden, t_cond)
        return self.out_proj(hidden)

    def forward(
        self,
        predictor_kv,
        actions,
        lang,
        lang_mask,
        state,
        context_len=None,
        dof_mask=None,
    ):
        noise = torch.randn_like(actions)
        time = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * actions
        target_velocity = actions - noise
        timestep = (time * self.num_timestep_buckets).long()
        condition_tokens, condition_valid = self.conditioner(lang, lang_mask, state)
        predicted_velocity = self._run_blocks(
            self._encode_actions(noisy, dof_mask),
            predictor_kv,
            context_len,
            self._time_cond(timestep),
            condition_tokens,
            condition_valid,
        )
        return predicted_velocity, target_velocity

    @torch.no_grad()
    def predict_action(
        self,
        predictor_kv,
        lang,
        lang_mask,
        state,
        context_len=None,
        dof_mask=None,
    ):
        first_key = predictor_kv[0][0]
        actions = torch.randn(
            (first_key.shape[0], self.action_chunk_size, self.action_dim),
            device=first_key.device,
            dtype=first_key.dtype,
        )
        condition_tokens, condition_valid = self.conditioner(lang, lang_mask, state)
        step_size = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            timestep = torch.full(
                (actions.shape[0],),
                int(step / self.num_inference_steps * self.num_timestep_buckets),
                device=actions.device,
                dtype=torch.long,
            )
            velocity = self._run_blocks(
                self._encode_actions(actions, dof_mask),
                predictor_kv,
                context_len,
                self._time_cond(timestep),
                condition_tokens,
                condition_valid,
            )
            actions = actions + step_size * velocity
        return actions


class ActionConditionEncoder(nn.Module):
    """Project the same ``[language tokens, state token]`` used by the Predictor."""

    def __init__(self, lang_dim, state_dim, hidden_size):
        super().__init__()
        self.lang_dim = lang_dim
        self.state_dim = state_dim
        self.language_norm = nn.LayerNorm(lang_dim, eps=1e-6)
        self.language_projection = nn.Linear(lang_dim, hidden_size)
        self.state_norm = nn.LayerNorm(state_dim, eps=1e-6)
        self.state_projection = nn.Linear(state_dim, hidden_size)

    def forward(self, language, language_mask, state):
        if language.ndim != 3 or language.shape[-1] != self.lang_dim:
            raise ValueError(
                f"language must be [B,S,{self.lang_dim}], got {tuple(language.shape)}"
            )
        if state.ndim != 2 or state.shape != (language.shape[0], self.state_dim):
            raise ValueError(
                f"state must be [B,{self.state_dim}], got {tuple(state.shape)}"
            )
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
        state_token = self.state_projection(self.state_norm(state)).unsqueeze(1)
        condition = torch.cat((language_tokens, state_token), dim=1)
        state_valid = torch.ones(
            (language.shape[0], 1), dtype=torch.bool, device=language.device
        )
        condition_valid = torch.cat((language_mask.bool(), state_valid), dim=1)
        return condition, condition_valid


class ActionExpert(_ActionExpertBase):
    """Flow-matching DiT conditioned on Predictor K/V, language, and state."""

    def __init__(
        self,
        *,
        action_dim,
        action_chunk_size,
        num_layers,
        predictor_num_heads,
        predictor_head_dim,
        state_dim,
        hidden_size=512,
        ffn_mult=4,
        lang_dim=4096,
        condition_num_heads=8,
        num_inference_steps=10,
    ):
        super().__init__(
            action_dim=action_dim,
            action_chunk_size=action_chunk_size,
            num_layers=num_layers,
            predictor_num_heads=predictor_num_heads,
            predictor_head_dim=predictor_head_dim,
            state_dim=state_dim,
            hidden_size=hidden_size,
            ffn_mult=ffn_mult,
            lang_dim=lang_dim,
            condition_num_heads=condition_num_heads,
            num_inference_steps=num_inference_steps,
        )
        # Retain the replacement order used by the published checkpoint so a
        # fixed seed reproduces its parameter initialization exactly.
        self.conditioner = ActionConditionEncoder(
            lang_dim=lang_dim,
            state_dim=state_dim,
            hidden_size=hidden_size,
        )
        self.num_heads = predictor_num_heads
        self.head_dim = predictor_head_dim
        self.read_full_predictor_kv = False
