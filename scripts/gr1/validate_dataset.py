"""Validate the local GR-1 LeRobot dataset layout."""

import argparse
import json
import re
from pathlib import Path

from scripts.gr1.dataset_manifest import REPO_REVISION, TASK_NAMES

GR1_DATASET_REVISION = REPO_REVISION
REQUIRED_META_FILES = (
    "episodes.jsonl",
    "info.json",
    "modality.json",
    "stats.json",
    "tasks.jsonl",
)
EPISODE_FILE_PATTERN = re.compile(r"episode_(\d+)\.(?:parquet|mp4)$")
GR1_FPS = 20.0
GR1_RAW_DIM = 44
GR1_VIDEO_KEY = "observation.images.ego_view"
GR1_VIDEO_SHAPE = [256, 256, 3]
GR1_CODEC_MODALITY_RANGES = {
    "left_arm": (0, 7),
    "right_arm": (22, 29),
    "left_hand": (7, 13),
    "right_hand": (29, 35),
    "waist": (41, 44),
}


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
    return records


def _episode_indices(paths: list[Path], *, root: Path, kind: str) -> set[int]:
    indices = []
    for path in paths:
        match = EPISODE_FILE_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Unexpected {kind} episode filename in {root}: {path}")
        indices.append(int(match.group(1)))
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate {kind} episode indices in {root}")
    return set(indices)


def _validate_info_schema(info: dict, root: Path) -> None:
    if float(info.get("fps", -1)) != GR1_FPS:
        raise ValueError(f"{root} must use fps={GR1_FPS:g}")

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{root} info.json is missing features")
    for key in ("observation.state", "action"):
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("shape") != [GR1_RAW_DIM]:
            raise ValueError(f"{root} feature {key!r} must have shape [{GR1_RAW_DIM}]")

    video_keys = {
        key for key, feature in features.items() if feature.get("dtype") == "video"
    }
    if video_keys != {GR1_VIDEO_KEY}:
        raise ValueError(
            f"{root} video features must be exactly [{GR1_VIDEO_KEY!r}], "
            f"found {sorted(video_keys)}"
        )
    video = features[GR1_VIDEO_KEY]
    if video.get("shape") != GR1_VIDEO_SHAPE:
        raise ValueError(
            f"{root} feature {GR1_VIDEO_KEY!r} must have shape {GR1_VIDEO_SHAPE}"
        )
    if float(video.get("video_info", {}).get("video.fps", -1)) != GR1_FPS:
        raise ValueError(f"{root} feature {GR1_VIDEO_KEY!r} must use fps={GR1_FPS:g}")


def _validate_modality_schema(modality: dict, root: Path) -> None:
    for section, original_key in (
        ("state", "observation.state"),
        ("action", "action"),
    ):
        groups = modality.get(section)
        if not isinstance(groups, dict):
            raise ValueError(f"{root} modality.json is missing {section!r}")
        for group, (start, end) in GR1_CODEC_MODALITY_RANGES.items():
            expected = {"original_key": original_key, "start": start, "end": end}
            if groups.get(group) != expected:
                raise ValueError(
                    f"{root} modality {section}.{group} must equal {expected}, "
                    f"found {groups.get(group)}"
                )

    expected_video = {"original_key": GR1_VIDEO_KEY}
    if modality.get("video", {}).get("ego_view") != expected_video:
        raise ValueError(f"{root} modality video.ego_view must equal {expected_video}")


def build_dataset_manifest(
    data_root: Path,
    *,
    expected_tasks: int = 24,
    expected_episodes: int = 1000,
    revision: str = GR1_DATASET_REVISION,
) -> dict:
    data_root = data_root.resolve()
    roots = sorted(path.parent.parent for path in data_root.glob("*/meta/info.json"))
    if len(roots) != expected_tasks:
        raise ValueError(
            f"Expected {expected_tasks} GR-1 task roots in {data_root}, found {len(roots)}"
        )
    task_names = [root.name for root in roots]
    if expected_tasks == len(TASK_NAMES) and set(task_names) != set(TASK_NAMES):
        missing = sorted(set(TASK_NAMES) - set(task_names))
        unexpected = sorted(set(task_names) - set(TASK_NAMES))
        raise ValueError(
            "GR-1 task directory names do not match ABot-M0: "
            f"missing={missing}, unexpected={unexpected}"
        )

    expected_indices = set(range(expected_episodes))
    for root in roots:
        meta_paths = [root / "meta" / name for name in REQUIRED_META_FILES]
        missing = [path.name for path in meta_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing metadata in {root}: {missing}")

        for path in meta_paths:
            if path.suffix == ".json":
                _load_json(path)
            elif path.suffix == ".jsonl":
                _load_jsonl(path)

        info = _load_json(root / "meta" / "info.json")
        _validate_info_schema(info, root)
        _validate_modality_schema(_load_json(root / "meta" / "modality.json"), root)
        if int(info.get("total_episodes", -1)) != expected_episodes:
            raise ValueError(
                f"{root} declares total_episodes={info.get('total_episodes')}, "
                f"expected {expected_episodes}"
            )
        episodes = _load_jsonl(root / "meta" / "episodes.jsonl")
        if len(episodes) != expected_episodes:
            raise ValueError(
                f"Expected {expected_episodes} episode records in {root}, found {len(episodes)}"
            )
        episode_indices = {int(record["episode_index"]) for record in episodes}
        if episode_indices != expected_indices:
            raise ValueError(f"Episode metadata indices are incomplete in {root}")
        missing_remarks = [
            int(record["episode_index"])
            for record in episodes
            if not isinstance(record.get("remarks"), str)
            or not record["remarks"].strip()
        ]
        if missing_remarks:
            raise ValueError(
                f"Episodes with missing remarks in {root}: {missing_remarks[:10]}"
            )

        parquet_files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
        video_files = sorted(root.glob("videos/chunk-*/*/episode_*.mp4"))
        for kind, paths in (("parquet", parquet_files), ("video", video_files)):
            if len(paths) != expected_episodes:
                raise ValueError(
                    f"Expected {expected_episodes} {kind} files in {root}, found {len(paths)}"
                )
            if _episode_indices(paths, root=root, kind=kind) != expected_indices:
                raise ValueError(
                    f"{kind.capitalize()} episode indices are incomplete in {root}"
                )
    return {
        "revision": revision,
        "task_roots": task_names,
        "episodes_per_task": expected_episodes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=24)
    parser.add_argument("--expected-episodes", type=int, default=1000)
    parser.add_argument("--revision", default=GR1_DATASET_REVISION)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = build_dataset_manifest(
        args.data_root,
        expected_tasks=args.expected_tasks,
        expected_episodes=args.expected_episodes,
        revision=args.revision,
    )
    if args.stats is not None:
        with args.stats.open(encoding="utf-8") as handle:
            saved = json.load(handle).get("metadata", {}).get("dataset")
        if saved != manifest:
            raise ValueError(
                f"Dataset stats manifest does not match local data: saved={saved}, current={manifest}"
            )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
