"""Common proprioceptive-state packing used by training and serving.

The policy consumes a fixed-width state contract so a predictor checkpoint can
be reused by robots whose native state vectors have different lengths.  A
normalized native state is zero-padded to ``max_state_dim`` and concatenated
with a binary validity vector of the same width.
"""

from __future__ import annotations

import torch


DEFAULT_MAX_STATE_DIM = 48


def packed_state_dim(max_state_dim: int) -> int:
    """Return the width of ``[padded_state, valid_mask]``."""

    max_state_dim = int(max_state_dim)
    if max_state_dim <= 0:
        raise ValueError(f"max_state_dim must be positive, got {max_state_dim}")
    return 2 * max_state_dim


def pack_normalized_state(
    state: torch.Tensor,
    max_state_dim: int = DEFAULT_MAX_STATE_DIM,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack a normalized state and its per-dimension validity mask.

    ``state`` may have any leading batch/time dimensions and must end in a
    native state dimension no larger than ``max_state_dim``.  When no mask is
    supplied, every supplied native dimension is marked valid.  Padding is
    always exactly zero and therefore cannot accidentally acquire a dataset
    statistic or a stale value.

    The result has shape ``state.shape[:-1] + (2 * max_state_dim,)`` and is
    ordered as ``[padded_normalized_state, valid_mask]``.
    """

    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    if state.ndim == 0:
        raise ValueError("state must have at least one dimension")
    max_state_dim = int(max_state_dim)
    if max_state_dim <= 0:
        raise ValueError(f"max_state_dim must be positive, got {max_state_dim}")
    native_dim = int(state.shape[-1])
    if native_dim <= 0 or native_dim > max_state_dim:
        raise ValueError(
            "state last dimension must be in "
            f"[1, max_state_dim={max_state_dim}], got {native_dim}"
        )

    padded = state.new_zeros((*state.shape[:-1], max_state_dim))
    padded[..., :native_dim] = state

    if valid_mask is None:
        mask = state.new_zeros((*state.shape[:-1], max_state_dim))
        mask[..., :native_dim] = 1
    else:
        if not torch.is_tensor(valid_mask):
            valid_mask = torch.as_tensor(valid_mask, device=state.device)
        if valid_mask.shape[:-1] != state.shape[:-1]:
            raise ValueError(
                "valid_mask leading dimensions must match state: "
                f"{tuple(valid_mask.shape)} vs {tuple(state.shape)}"
            )
        mask_dim = int(valid_mask.shape[-1])
        if mask_dim not in (native_dim, max_state_dim):
            raise ValueError(
                "valid_mask last dimension must match native state or max_state_dim, "
                f"got {mask_dim}; state={native_dim}, max={max_state_dim}"
            )
        mask = state.new_zeros((*state.shape[:-1], max_state_dim))
        mask[..., :mask_dim] = valid_mask.to(
            device=state.device, dtype=state.dtype
        )
        if mask_dim == max_state_dim:
            # A full-width mask may only describe supplied state dimensions;
            # values beyond the native vector are never valid.
            mask[..., native_dim:] = 0

    return torch.cat((padded, mask), dim=-1)


def prepare_proprio_state(
    state: torch.Tensor,
    *,
    raw_state_dim: int,
    encoding: str = "identity",
    max_state_dim: int | None = DEFAULT_MAX_STATE_DIM,
) -> torch.Tensor:
    """Validate and prepare state conditioning for a policy module.

    ``max_state_dim=None`` selects the historical variable-width path.  The
    fixed-width contract intentionally supports only identity encoding: the
    validity mask describes normalized native dimensions, and applying sin/cos
    to that packed representation would change its meaning.  Callers using
    the historical ``sincos`` representation can explicitly pass
    ``max_state_dim=None``.
    """

    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    raw_state_dim = int(raw_state_dim)
    if state.ndim < 1 or state.shape[-1] != raw_state_dim:
        raise ValueError(
            f"state must end in dimension {raw_state_dim}, got {tuple(state.shape)}"
        )
    if max_state_dim is None:
        if encoding == "identity":
            return state
        if encoding == "sincos":
            return torch.cat((torch.sin(state), torch.cos(state)), dim=-1)
        raise ValueError(f"unsupported proprio encoding {encoding!r}")
    if encoding != "identity":
        raise ValueError(
            "fixed-width state packing requires proprio_encoding='identity'; "
            "pass max_state_dim=None to use the legacy sincos representation"
        )
    return pack_normalized_state(state, max_state_dim)


__all__ = [
    "DEFAULT_MAX_STATE_DIM",
    "packed_state_dim",
    "pack_normalized_state",
    "prepare_proprio_state",
]
