"""Small dataset contracts used by the public training entry points.

An adapter keeps dataset-specific decoding out of the model and trainer.  A
factory is imported from ``module:function`` and receives the parsed argparse
namespace.  The returned dataset must yield the keys documented in
``README.md``; the stock collators add masks and stack tensors.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

import torch

from vjepa_policy.data import CausalPatchMask


def import_factory(path: str) -> Callable[..., Any]:
    """Import ``module:function`` without adding a plugin framework."""
    try:
        module_name, function_name = path.split(":", 1)
    except ValueError as error:
        raise ValueError(f"Factory must use module:function syntax: {path!r}") from error
    if not module_name or not function_name:
        raise ValueError(f"Factory must use module:function syntax: {path!r}")
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise TypeError(f"Dataset factory is not callable: {path!r}")
    return factory


def build_dataset(factory_path: str, args: Any, *, include_action: bool):
    """Build a custom dataset and optional collator from a user adapter.

    A factory may return a Dataset, ``(dataset, collator)``, or a mapping with
    ``dataset`` and ``collator`` entries.  A custom collator is useful when a
    dataset has a non-standard storage format; otherwise the stock collator is
    selected by the caller.
    """
    factory = import_factory(factory_path)
    try:
        result = factory(args, include_action=include_action)
    except TypeError as first_error:
        try:
            result = factory(args)
        except TypeError:
            raise first_error
    if isinstance(result, tuple) and len(result) == 2:
        return result
    if isinstance(result, dict) and "dataset" in result:
        return result["dataset"], result.get("collator")
    return result, None


class PredictorBatchCollator:
    """Collate generic predictor samples into the model batch contract."""

    def __init__(self, layout, *, include_action: bool = False):
        self.layout = layout
        self.include_action = include_action
        self._context, self._target = layout.masks()

    def __call__(self, batch: Sequence[dict[str, Any]]):
        clips = torch.stack([sample["video"] for sample in batch])
        language = torch.stack([sample["context"] for sample in batch])
        language_mask = torch.stack([sample["context_mask"] for sample in batch])
        state = torch.stack([sample["proprio"] for sample in batch])
        masks_enc = [
            torch.as_tensor(self._context, dtype=torch.long)
            .unsqueeze(0)
            .expand(len(batch), -1)
            .clone()
        ]
        masks_pred = [
            torch.as_tensor(self._target, dtype=torch.long)
            .unsqueeze(0)
            .expand(len(batch), -1)
            .clone()
        ]
        if not self.include_action:
            return clips, language, language_mask, masks_enc, masks_pred, state
        action = torch.stack([sample["action"] for sample in batch])
        action_pad = torch.stack([sample["action_is_pad"] for sample in batch])
        return (
            clips,
            language,
            language_mask,
            masks_enc,
            masks_pred,
            action,
            action_pad,
            state,
        )


__all__ = ["PredictorBatchCollator", "build_dataset", "import_factory"]
