"""Compute shared GR-1 absolute-action min/max statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.configs.types import FeatureType, PolicyFeature

from scripts.gr1.validate_dataset import (
    GR1_DATASET_REVISION,
    build_dataset_manifest,
)
from vjepa_policy.datasets import ActionStateNormalizer


GR1_ACTION_STATE_INDICES = (
    *range(0, 7),
    *range(22, 29),
    *range(7, 13),
    *range(29, 35),
    *range(41, 44),
)


def _read_feature_stats(root: Path, key: str, stat: str) -> np.ndarray:
    with (root / "meta" / "stats.json").open(encoding="utf-8") as handle:
        stats = json.load(handle)
    return np.asarray(stats[key][stat], dtype=np.float32)[
        list(GR1_ACTION_STATE_INDICES)
    ]


def compute_stats(
    data_root: Path,
    output: Path,
    action_chunk_size: int = 16,
    expected_tasks: int = 24,
    expected_episodes: int = 1000,
    revision: str = GR1_DATASET_REVISION,
) -> None:
    dataset_manifest = build_dataset_manifest(
        data_root,
        expected_tasks=expected_tasks,
        expected_episodes=expected_episodes,
        revision=revision,
    )
    roots = sorted(path.parent.parent for path in data_root.glob("*/meta/info.json"))

    if action_chunk_size <= 0 or action_chunk_size % 4:
        raise ValueError("action_chunk_size must be a positive multiple of 4")

    action_min = [_read_feature_stats(root, "action", "min") for root in roots]
    action_max = [_read_feature_stats(root, "action", "max") for root in roots]

    stats = {
        "action": {
            "min": np.min(np.stack(action_min), axis=0),
            "max": np.max(np.stack(action_max), axis=0),
        },
        "observation.state": {},
    }
    features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(29,)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(29,)),
    }
    normalizer = ActionStateNormalizer(
        features=features,
        stats=stats,
        action_mode="MIN_MAX",
        state_mode="IDENTITY",
        clip_values=True,
        metadata={
            "dataset": dataset_manifest,
            "action_codec": {
                "action_indices": list(GR1_ACTION_STATE_INDICES),
                "state_indices": list(GR1_ACTION_STATE_INDICES),
                "relative_action_indices": [],
                "action_chunk_size": action_chunk_size,
            },
            "proprio_encoding": {
                "type": "sincos",
                "raw_dim": 29,
                "encoded_dim": 58,
            },
        },
    )
    normalizer.save(output)
    print(f"Saved GR-1 dataset stats to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--expected-tasks", type=int, default=24)
    parser.add_argument("--expected-episodes", type=int, default=1000)
    parser.add_argument("--revision", default=GR1_DATASET_REVISION)
    args = parser.parse_args(argv)
    compute_stats(
        args.data_root,
        args.output,
        action_chunk_size=args.action_chunk_size,
        expected_tasks=args.expected_tasks,
        expected_episodes=args.expected_episodes,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
