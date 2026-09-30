import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import dataset_to_policy_features


NORMALIZATION_MODES = {"MIN_MAX", "MEAN_STD", "IDENTITY", "QUANTILE"}


def _mode(value: str | NormalizationMode) -> str:
    mode = value.value if isinstance(value, NormalizationMode) else value.upper()
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unsupported normalization mode {value!r}; choose from {NORMALIZATION_MODES}")
    return mode


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return np.asarray(value)
    return value


def _aggregate_quantiles(metadata, key: str) -> dict[str, np.ndarray]:
    """Conservatively combine per-dataset quantiles as in FastWAM."""
    quantiles = {}
    for stat_name, reducer in (("q01", np.min), ("q99", np.max)):
        values = []
        for meta in metadata:
            if stat_name not in meta.stats[key]:
                return {}
            values.append(np.asarray(meta.stats[key][stat_name]))
        quantiles[stat_name] = reducer(np.stack(values), axis=0)
    return quantiles


def _select_stats(
    stats: dict[str, Any],
    indices: tuple[int, ...],
    original_size: int,
) -> dict[str, Any]:
    selected = {}
    for name, value in stats.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[-1] == original_size:
            selected[name] = np.take(array, indices, axis=-1)
        else:
            selected[name] = value
    return selected


class ActionStateNormalizer:
    """Shared LeRobot normalizer for action and current robot state."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        stats: dict[str, dict[str, np.ndarray | torch.Tensor]],
        action_mode: str | NormalizationMode = NormalizationMode.MIN_MAX,
        state_mode: str | NormalizationMode = NormalizationMode.MIN_MAX,
        clip_values: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.features = features
        self.stats = stats
        self.action_mode = _mode(action_mode)
        self.state_mode = _mode(state_mode)
        self.clip_values = bool(clip_values)
        self.metadata = dict(metadata or {})
        self.norm_map = {
            FeatureType.ACTION: self.action_mode,
            FeatureType.STATE: self.state_mode,
        }
        required_stats = {
            "MEAN_STD": ("mean", "std"),
            "MIN_MAX": ("min", "max"),
            "QUANTILE": ("q01", "q99"),
            "IDENTITY": (),
        }
        for key, feature in features.items():
            mode = self.norm_map.get(feature.type, "IDENTITY")
            missing = [name for name in required_stats[mode] if name not in stats[key]]
            if missing:
                raise ValueError(
                    f"Normalization stats for {key!r} do not contain {missing} required by {mode}. "
                    "Precompute shared quantile stats before using QUANTILE."
                )

    @classmethod
    def from_metadata(
        cls,
        metadata,
        action_key: str,
        state_key: str,
        action_mode: str | NormalizationMode = NormalizationMode.MIN_MAX,
        state_mode: str | NormalizationMode = NormalizationMode.MIN_MAX,
        action_indices: tuple[int, ...] | None = None,
        state_indices: tuple[int, ...] | None = None,
        clip_values: bool = False,
    ) -> "ActionStateNormalizer":
        first_features = metadata[0].features
        for meta in metadata[1:]:
            for key in (action_key, state_key):
                if meta.features[key]["shape"] != first_features[key]["shape"]:
                    raise ValueError(
                        f"Feature {key!r} shape mismatch: "
                        f"{first_features[key]['shape']} vs {meta.features[key]['shape']}"
                    )
        features = dataset_to_policy_features(
            {key: first_features[key] for key in (action_key, state_key)}
        )
        stats_inputs = []
        for meta in metadata:
            dataset_stats = {}
            for key in (action_key, state_key):
                feature_stats = {
                    name: np.asarray(value) for name, value in meta.stats[key].items()
                }
                feature_stats.setdefault("count", np.asarray([meta.total_frames]))
                dataset_stats[key] = feature_stats
            stats_inputs.append(dataset_stats)
        stats = aggregate_stats(stats_inputs)
        for key in (action_key, state_key):
            stats[key].update(_aggregate_quantiles(metadata, key))

        selected_indices = {
            action_key: action_indices,
            state_key: state_indices,
        }
        for key, indices in selected_indices.items():
            if indices is None:
                continue
            original_size = int(first_features[key]["shape"][0])
            stats[key] = _select_stats(stats[key], indices, original_size)
            features[key] = PolicyFeature(
                type=features[key].type,
                shape=(len(indices),),
            )
        return cls(
            features,
            stats,
            action_mode=action_mode,
            state_mode=state_mode,
            clip_values=clip_values,
        )

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._transform(batch, inverse=False)

    def unnormalize(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._transform(batch, inverse=True)

    def unnormalize_action(self, action: torch.Tensor, action_key: str = "action") -> torch.Tensor:
        return self.unnormalize({action_key: action})[action_key]

    def _transform(
        self,
        batch: dict[str, torch.Tensor],
        inverse: bool,
    ) -> dict[str, torch.Tensor]:
        result = dict(batch)
        for key, feature in self.features.items():
            if key not in result:
                continue
            mode = self.norm_map.get(feature.type, "IDENTITY")
            if mode == "IDENTITY":
                continue
            value = result[key]
            if inverse and self.clip_values:
                value = value.clamp(-1.0, 1.0)
            stats = {
                name: torch.as_tensor(stat, device=value.device, dtype=value.dtype)
                for name, stat in self.stats[key].items()
                if name in {"mean", "std", "min", "max", "q01", "q99"}
            }
            if mode == "MEAN_STD":
                if inverse:
                    value = value * stats["std"] + stats["mean"]
                else:
                    value = (value - stats["mean"]) / (stats["std"] + 1e-8)
            else:
                low_key, high_key = ("q01", "q99") if mode == "QUANTILE" else ("min", "max")
                low, high = stats[low_key], stats[high_key]
                value_range = high - low
                constant = value_range.abs() < 1e-4
                if inverse:
                    scaled = (value + 1.0) / 2.0 * value_range + low
                    value = torch.where(constant, low, scaled)
                else:
                    scaled = (value - low) / (value_range + 1e-8) * 2.0 - 1.0
                    value = torch.where(constant, value - low, scaled)
            if not inverse and self.clip_values:
                value = value.clamp(-1.0, 1.0)
            result[key] = value
        return result

    def save(self, path: str | os.PathLike[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "features": {
                key: {"type": feature.type.value, "shape": list(feature.shape)}
                for key, feature in self.features.items()
            },
            "normalization": {
                "action": self.action_mode,
                "state": self.state_mode,
                "clip_values": self.clip_values,
            },
            "stats": _jsonable(self.stats),
            "metadata": _jsonable(self.metadata),
        }
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ActionStateNormalizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported normalization stats version in {path}")
        features = {
            key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
            for key, value in payload["features"].items()
        }
        return cls(
            features=features,
            stats=_arrays(payload["stats"]),
            action_mode=payload["normalization"]["action"],
            state_mode=payload["normalization"]["state"],
            clip_values=payload["normalization"].get("clip_values", False),
            metadata=payload.get("metadata"),
        )
