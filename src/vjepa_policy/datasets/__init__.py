"""Dataset utilities with optional LeRobot loading."""

from vjepa_policy.datasets.prompts import DEFAULT_PROMPT
from vjepa_policy.datasets.transforms import (
    QuadrantViewCombiner,
    VideoClipTransform,
    build_patch_valid_mask,
    combine_video_clips,
)

__all__ = [
    "DEFAULT_PROMPT",
    "QuadrantViewCombiner",
    "VideoClipTransform",
    "build_patch_valid_mask",
    "combine_video_clips",
]


def __getattr__(name):
    """Load the optional LeRobot adapter only when it is requested."""
    if name in {
        "ActionStateNormalizer",
        "LeRobotClipDataset",
        "LeRobotVideoDataset",
        "TextEmbeddingCache",
        "discover_video_keys",
        "make_video_offsets",
        "read_instructions",
    }:
        if name == "ActionStateNormalizer":
            from vjepa_policy.datasets.normalization import ActionStateNormalizer

            value = ActionStateNormalizer
        else:
            from vjepa_policy.datasets import lerobot_video

            value = getattr(lerobot_video, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
