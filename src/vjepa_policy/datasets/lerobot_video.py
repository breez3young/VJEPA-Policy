import bisect
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset as BaseLeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import Dataset

from vjepa_policy.datasets.normalization import ActionStateNormalizer
from vjepa_policy.datasets.prompts import DEFAULT_PROMPT
from vjepa_policy.datasets.transforms import combine_video_clips


def make_video_offsets(
    num_frames: int = 33,
    past_frames: int = 4,
    video_stride: int = 4,
) -> tuple[int, ...]:
    """Derive downsampled video offsets from a contiguous action window."""
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if past_frames < 0:
        raise ValueError("past_frames must be non-negative")
    if video_stride <= 0:
        raise ValueError("video_stride must be positive")
    if past_frames % video_stride:
        raise ValueError("past_frames must be divisible by video_stride so offset 0 is sampled")
    if (num_frames - 1) % video_stride:
        raise ValueError(
            "num_frames - 1 must be divisible by video_stride so the final frame is sampled"
        )
    return tuple(range(-past_frames, num_frames, video_stride))


def _camera_key(key: str) -> str:
    return key if key.startswith("observation.images") else f"observation.images.{key}"


def discover_video_keys(
    dataset_dirs: Sequence[str | os.PathLike[str]],
) -> list[str]:
    """Return a stable camera ordering shared by all LeRobot roots."""
    if not dataset_dirs:
        raise ValueError("At least one dataset directory is required")

    expected = None
    for dataset_dir in dataset_dirs:
        root = Path(dataset_dir)
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
        with info_path.open(encoding="utf-8") as handle:
            features = json.load(handle).get("features")
        if not isinstance(features, dict):
            raise ValueError(f"Invalid or missing features in {info_path}")
        keys = sorted(
            key
            for key, feature in features.items()
            if feature.get("dtype") in {"image", "video"}
        )
        if not keys:
            raise ValueError(f"No image or video features found in {root}")
        if expected is None:
            expected = keys
        elif keys != expected:
            raise ValueError(
                "All dataset roots must expose the same ordered camera features; "
                f"expected {expected}, got {keys} in {root}"
            )
    return expected


def read_instructions(
    dataset_dir: str | os.PathLike[str],
    instruction_field: str = "task",
) -> dict[int, str]:
    """Read instructions keyed by task or episode index from LeRobot metadata."""
    root = Path(dataset_dir)
    if instruction_field == "task":
        path = root / "meta" / "tasks.jsonl"
        index_key = "task_index"
    elif instruction_field == "remarks":
        path = root / "meta" / "episodes.jsonl"
        index_key = "episode_index"
    else:
        raise ValueError(
            f"Unsupported instruction_field={instruction_field!r}; choose 'task' or 'remarks'"
        )

    instructions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            instruction = record.get(instruction_field)
            if not isinstance(instruction, str) or not instruction.strip():
                continue
            instructions[int(record[index_key])] = instruction.strip()
    if not instructions:
        raise ValueError(f"No non-empty {instruction_field!r} instructions found in {path}")
    return instructions


def _validate_feature_indices(
    indices: Sequence[int] | None,
    feature_size: int,
    feature_key: str,
) -> tuple[int, ...]:
    if indices is None:
        return tuple(range(feature_size))
    selected = tuple(int(index) for index in indices)
    if not selected:
        raise ValueError(f"{feature_key} indices must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{feature_key} indices must not contain duplicates: {selected}")
    invalid = [index for index in selected if not 0 <= index < feature_size]
    if invalid:
        raise ValueError(
            f"{feature_key} indices {invalid} exceed feature size {feature_size}"
        )
    return selected


def _make_relative_action(
    action: torch.Tensor,
    state: torch.Tensor,
    relative_indices: torch.Tensor,
) -> torch.Tensor:
    result = action.clone()
    result[..., relative_indices] -= state[-1, relative_indices]
    return result


class TextEmbeddingCache:
    def __init__(self, cache_dir: str | os.PathLike[str], context_len: int) -> None:
        self.cache_dir = Path(cache_dir)
        self.context_len = context_len
        self._memory: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def load(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt in self._memory:
            return self._memory[prompt]
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{digest}.t5_len{self.context_len}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing text embedding cache: {path}. Run "
                "`python -m vjepa_policy.text_embeddings` first."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        context = payload["context"]
        mask = payload["mask"].bool()
        if context.ndim != 2 or context.shape[0] != self.context_len:
            raise ValueError(f"Invalid cached context shape {tuple(context.shape)} in {path}")
        if mask.shape != (self.context_len,):
            raise ValueError(f"Invalid cached mask shape {tuple(mask.shape)} in {path}")
        self._memory[prompt] = (context, mask)
        return context, mask


class LeRobotClipDataset(BaseLeRobotDataset):
    """Thin LeRobot subclass that returns video views and one action chunk.

    Sampling and video decoding remain entirely in LeRobot. This subclass only
    derives delta timestamps from a contiguous ``num_frames`` action window,
    downsamples its video frames, applies a whole-clip transform, and maps the
    result into VJEPA-Policy's batch schema. Canvas view combiners return
    ``[C,T,H,W]``; ``independent`` returns ``[V,C,T,H,W]``.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        video_keys: Sequence[str] | None = None,
        num_frames: int = 33,
        past_frames: int = 4,
        video_stride: int = 4,
        action_key: str = "action",
        state_key: str = "observation.state",
        action_indices: Sequence[int] | None = None,
        state_indices: Sequence[int] | None = None,
        relative_action_indices: Sequence[int] | None = None,
        video_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        view_combiner: str | Callable[[Sequence[torch.Tensor]], torch.Tensor] = "independent",
        text_cache: TextEmbeddingCache | None = None,
        prompt_template: str = DEFAULT_PROMPT,
        instruction_field: str = "task",
        repo_id: str | None = None,
        episodes: list[int] | None = None,
        video_backend: str | None = None,
        require_full_future_window: bool = False,
    ) -> None:
        root = Path(root)
        repo_id = repo_id or root.name
        metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        if video_keys is None:
            video_keys = discover_video_keys([root])
        self.video_keys = [_camera_key(key) for key in video_keys]
        for key in [*self.video_keys, action_key, state_key]:
            if key not in metadata.features:
                raise KeyError(f"Feature {key!r} not found in {root}")
        self._sampling_num_frames = num_frames
        self._past_frames = past_frames
        self._video_stride = video_stride
        self._video_offsets = make_video_offsets(num_frames, past_frames, video_stride)
        self._action_offsets = tuple(range(num_frames - 1))
        self._state_offsets = (0,)
        self.require_full_future_window = bool(require_full_future_window)
        self._full_window_episode_starts: list[int] | None = None
        self._full_window_cumulative_sizes: list[int] | None = None
        self.action_key = action_key
        self.state_key = state_key
        self.action_indices = _validate_feature_indices(
            action_indices,
            int(metadata.features[action_key]["shape"][0]),
            action_key,
        )
        self.state_indices = _validate_feature_indices(
            state_indices,
            int(metadata.features[state_key]["shape"][0]),
            state_key,
        )
        self._action_index = torch.tensor(self.action_indices, dtype=torch.long)
        self._state_index = torch.tensor(self.state_indices, dtype=torch.long)
        self.relative_action_indices = _validate_feature_indices(
            relative_action_indices,
            len(self.action_indices),
            "relative action",
        ) if relative_action_indices is not None else ()
        if self.relative_action_indices and self.action_indices != self.state_indices:
            raise ValueError(
                "Relative action conversion requires aligned action and state indices"
            )
        self._relative_action_index = torch.tensor(
            self.relative_action_indices, dtype=torch.long
        )
        self.video_transform = video_transform
        self.view_combiner = view_combiner
        self.text_cache = text_cache
        self.prompt_template = prompt_template
        self.instruction_field = instruction_field
        self._episode_instructions = (
            read_instructions(root, "remarks") if instruction_field == "remarks" else None
        )
        self.action_state_normalizer: ActionStateNormalizer | None = None

        delta_timestamps = {
            key: [offset / metadata.fps for offset in self.video_offsets]
            for key in self.video_keys
        }
        delta_timestamps[action_key] = [offset / metadata.fps for offset in self.action_offsets]
        delta_timestamps[state_key] = [offset / metadata.fps for offset in self.state_offsets]
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=None,
            delta_timestamps=delta_timestamps,
            download_videos=True,
            video_backend=video_backend,
        )
        if self.require_full_future_window:
            self._build_full_window_index()

    def _build_full_window_index(self) -> None:
        max_future_offset = max(
            0,
            *self.video_offsets,
            *self.action_offsets,
            *self.state_offsets,
        )
        episode_starts = self.episode_data_index["from"].tolist()
        episode_ends = self.episode_data_index["to"].tolist()
        cumulative_sizes = []
        running = 0
        for start, end in zip(episode_starts, episode_ends, strict=True):
            running += max(0, end - start - max_future_offset)
            cumulative_sizes.append(running)
        self._full_window_episode_starts = episode_starts
        self._full_window_cumulative_sizes = cumulative_sizes

    def _frame_index(self, index: int) -> int:
        if self._full_window_cumulative_sizes is None:
            return index
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(
            self._full_window_cumulative_sizes, index
        )
        episode_offset = (
            0
            if episode_index == 0
            else self._full_window_cumulative_sizes[episode_index - 1]
        )
        return self._full_window_episode_starts[episode_index] + index - episode_offset

    def __len__(self) -> int:
        if self._full_window_cumulative_sizes is None:
            return super().__len__()
        return self._full_window_cumulative_sizes[-1] if self._full_window_cumulative_sizes else 0

    @property
    def video_offsets(self) -> tuple[int, ...]:
        return self._video_offsets

    @property
    def sampling_num_frames(self) -> int:
        return self._sampling_num_frames

    @property
    def past_frames(self) -> int:
        return self._past_frames

    @property
    def video_stride(self) -> int:
        return self._video_stride

    @property
    def action_offsets(self) -> tuple[int, ...]:
        return self._action_offsets

    @property
    def state_offsets(self) -> tuple[int, ...]:
        return self._state_offsets

    @property
    def action_dim(self) -> int:
        return len(self.action_indices)

    @property
    def state_dim(self) -> int:
        return len(self.state_indices)

    def set_normalizer(self, normalizer: ActionStateNormalizer) -> None:
        self.action_state_normalizer = normalizer

    def _get_query_timestamps(self, current_ts, query_indices=None):
        timestamps = super()._get_query_timestamps(current_ts, query_indices)
        return {key: timestamps[key] for key in self.video_keys}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        item = super().__getitem__(self._frame_index(index))
        clips = []
        for key in self.video_keys:
            clip = item[key]
            if clip.ndim != 4 or clip.shape[0] != len(self.video_offsets):
                raise ValueError(
                    f"Expected {key} clip [{len(self.video_offsets)},C,H,W], got {tuple(clip.shape)}"
                )
            if self.video_transform is not None:
                clip = self.video_transform(clip)
            clips.append(clip)
        video = combine_video_clips(clips, self.view_combiner)

        action = item[self.action_key].float().index_select(-1, self._action_index)
        state = item[self.state_key].float().index_select(-1, self._state_index)
        if self.relative_action_indices:
            action = _make_relative_action(action, state, self._relative_action_index)
        action_state = {self.action_key: action, self.state_key: state}
        if self.action_state_normalizer is not None:
            action_state = self.action_state_normalizer(action_state)
        state = action_state[self.state_key]
        if state.shape[0] == 1:
            state = state[0]

        task_id = item["task"]
        if self._episode_instructions is None:
            instruction = task_id
        else:
            episode_index = int(item["episode_index"])
            try:
                instruction = self._episode_instructions[episode_index]
            except KeyError as error:
                raise KeyError(
                    f"Episode {episode_index} has no non-empty remarks instruction in {self.root}"
                ) from error
        prompt = self.prompt_template.format(task=instruction)
        result = {
            "video": video,
            "action": action_state[self.action_key],
            "proprio": state,
            "prompt": prompt,
            "task": instruction,
            "task_id": task_id,
            "instruction": instruction,
            "action_is_pad": item[f"{self.action_key}_is_pad"].bool(),
            "proprio_is_pad": item[f"{self.state_key}_is_pad"].bool(),
            "image_is_pad": torch.stack(
                [item[f"{key}_is_pad"].bool() for key in self.video_keys]
            ).any(dim=0),
            "index": index,
        }
        if self.text_cache is not None:
            result["context"], result["context_mask"] = self.text_cache.load(prompt)
        return result


class LeRobotVideoDataset(Dataset):
    """Combine local LeRobot roots with shared sampling and normalization."""

    def __init__(
        self,
        dataset_dirs: Sequence[str | os.PathLike[str]],
        video_keys: Sequence[str] | None = None,
        num_frames: int = 33,
        past_frames: int = 4,
        video_stride: int = 4,
        action_key: str = "action",
        state_key: str = "observation.state",
        action_indices: Sequence[int] | None = None,
        state_indices: Sequence[int] | None = None,
        relative_action_indices: Sequence[int] | None = None,
        video_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        view_combiner: str | Callable[[Sequence[torch.Tensor]], torch.Tensor] = "independent",
        text_cache_dir: str | os.PathLike[str] | None = None,
        context_len: int = 32,
        prompt_template: str = DEFAULT_PROMPT,
        instruction_field: str = "task",
        action_normalization: str = "MIN_MAX",
        state_normalization: str = "MIN_MAX",
        clip_normalized: bool = False,
        normalization_stats_path: str | os.PathLike[str] | None = None,
        stats_output_path: str | os.PathLike[str] | None = None,
        episodes: dict[str, list[int]] | None = None,
        video_backend: str | None = None,
        require_full_future_window: bool = False,
    ) -> None:
        if not dataset_dirs:
            raise ValueError("At least one dataset directory is required")
        video_keys = (
            discover_video_keys(dataset_dirs) if video_keys is None else list(video_keys)
        )
        text_cache = TextEmbeddingCache(text_cache_dir, context_len) if text_cache_dir else None
        self.datasets = []
        for dataset_dir in dataset_dirs:
            root = Path(dataset_dir)
            selected_episodes = episodes.get(str(root)) if episodes else None
            self.datasets.append(
                LeRobotClipDataset(
                    root=root,
                    video_keys=video_keys,
                    num_frames=num_frames,
                    past_frames=past_frames,
                    video_stride=video_stride,
                    action_key=action_key,
                    state_key=state_key,
                    action_indices=action_indices,
                    state_indices=state_indices,
                    relative_action_indices=relative_action_indices,
                    video_transform=video_transform,
                    view_combiner=view_combiner,
                    text_cache=text_cache,
                    prompt_template=prompt_template,
                    instruction_field=instruction_field,
                    episodes=selected_episodes,
                    video_backend=video_backend,
                    require_full_future_window=require_full_future_window,
                )
            )

        self.action_key = action_key
        self.state_key = state_key
        self.action_indices = self.datasets[0].action_indices
        self.state_indices = self.datasets[0].state_indices
        self.relative_action_indices = self.datasets[0].relative_action_indices
        self.instruction_field = instruction_field
        self.video_keys = self.datasets[0].video_keys
        self._sampling_num_frames = num_frames
        self._past_frames = past_frames
        self._video_stride = video_stride
        self._video_offsets = self.datasets[0].video_offsets
        self._action_offsets = self.datasets[0].action_offsets
        self._state_offsets = self.datasets[0].state_offsets
        self.cumulative_sizes = []
        running = 0
        for dataset in self.datasets:
            if dataset.action_dim != self.datasets[0].action_dim:
                raise ValueError("All datasets must have the same action dimension")
            if dataset.state_dim != self.datasets[0].state_dim:
                raise ValueError("All datasets must have the same state dimension")
            if dataset.action_indices != self.action_indices:
                raise ValueError("All datasets must use the same action indices")
            if dataset.state_indices != self.state_indices:
                raise ValueError("All datasets must use the same state indices")
            if dataset.relative_action_indices != self.relative_action_indices:
                raise ValueError("All datasets must use the same relative action indices")
            running += len(dataset)
            self.cumulative_sizes.append(running)

        if self.relative_action_indices and not normalization_stats_path:
            raise ValueError(
                "Relative actions require a projected relative-action dataset stats file"
            )
        if normalization_stats_path:
            self.normalizer = ActionStateNormalizer.load(normalization_stats_path)
            requested_modes = {
                "action": getattr(action_normalization, "value", action_normalization).upper(),
                "state": getattr(state_normalization, "value", state_normalization).upper(),
            }
            loaded_modes = {
                "action": self.normalizer.action_mode,
                "state": self.normalizer.state_mode,
            }
            if loaded_modes != requested_modes:
                raise ValueError(
                    f"Normalization modes in stats {loaded_modes} do not match "
                    f"requested modes {requested_modes}"
                )
            if self.normalizer.clip_values != bool(clip_normalized):
                raise ValueError(
                    f"Normalization stats clip_values={self.normalizer.clip_values} does not "
                    f"match clip_normalized={bool(clip_normalized)}"
                )
            for key in (action_key, state_key):
                if key not in self.normalizer.features:
                    raise KeyError(f"Normalization stats do not contain feature {key!r}")
            expected_shapes = {
                action_key: (self.action_dim,),
                state_key: (self.state_dim,),
            }
            for key, expected_shape in expected_shapes.items():
                if tuple(self.normalizer.features[key].shape) != expected_shape:
                    raise ValueError(
                        f"Normalization feature {key!r} has shape "
                        f"{self.normalizer.features[key].shape}, expected {expected_shape}"
                    )
            expected_codec = {
                "action_indices": list(self.action_indices),
                "state_indices": list(self.state_indices),
                "relative_action_indices": list(self.relative_action_indices),
                "action_chunk_size": len(self.action_offsets),
            }
            actual_codec = self.normalizer.metadata.get("action_codec")
            if actual_codec is not None and actual_codec != expected_codec:
                raise ValueError(
                    "Action stats codec does not match the dataset: "
                    f"expected {expected_codec}, got {actual_codec}"
                )
        else:
            self.normalizer = ActionStateNormalizer.from_metadata(
                [dataset.meta for dataset in self.datasets],
                action_key=action_key,
                state_key=state_key,
                action_mode=action_normalization,
                state_mode=state_normalization,
                action_indices=self.action_indices,
                state_indices=self.state_indices,
                clip_values=clip_normalized,
            )
        for dataset in self.datasets:
            dataset.set_normalizer(self.normalizer)

        if stats_output_path and int(os.environ.get("RANK", "0")) == 0:
            self.normalizer.save(stats_output_path)

    @property
    def action_dim(self) -> int:
        return self.datasets[0].action_dim

    @property
    def video_offsets(self) -> tuple[int, ...]:
        return self._video_offsets

    @property
    def sampling_num_frames(self) -> int:
        return self._sampling_num_frames

    @property
    def past_frames(self) -> int:
        return self._past_frames

    @property
    def video_stride(self) -> int:
        return self._video_stride

    @property
    def action_offsets(self) -> tuple[int, ...]:
        return self._action_offsets

    @property
    def state_offsets(self) -> tuple[int, ...]:
        return self._state_offsets

    @property
    def state_dim(self) -> int:
        return self.datasets[0].state_dim

    @property
    def num_episodes(self) -> int:
        return sum(dataset.num_episodes for dataset in self.datasets)

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.normalizer.unnormalize_action(action, self.action_key)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        start = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        result = self.datasets[dataset_index][index - start]
        result["dataset_index"] = dataset_index
        return result
