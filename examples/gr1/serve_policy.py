"""Serve a V-JEPA GR-1 policy with the official Isaac-GR00T protocol."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

import numpy as np

from gr00t.data.types import ModalityConfig
from gr00t.policy import BasePolicy
from gr00t.policy.server_client import PolicyServer

from vjepa_policy.policy_serving.gr1 import (
    GR1_ACTION_HORIZON,
    GR1_DATASET_VIDEO_KEY,
    GR1_GROUP_DIMS,
    GR1_LANGUAGE_KEY,
    GR1_MODEL_FRAMES,
    GR1_MODEL_IMAGE_SIZE,
    GR1_STATE_DELTA_INDICES,
    GR1_VIDEO_DELTA_INDICES,
    GR1_VIDEO_KEY,
    VJEPAGR1Serving,
    validate_gr1_wire_observation,
)
from vjepa_policy.policy_serving.libero import PolicyServingConfig


def gr1_modality_config(
    action_horizon: int = GR1_ACTION_HORIZON,
) -> dict[str, ModalityConfig]:
    """Return the horizon contract consumed by the official rollout client."""
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    group_names = [name for name, _ in GR1_GROUP_DIMS]
    return {
        "video": ModalityConfig(
            delta_indices=list(GR1_VIDEO_DELTA_INDICES),
            modality_keys=[GR1_VIDEO_KEY.removeprefix("video.")],
        ),
        "state": ModalityConfig(
            delta_indices=list(GR1_STATE_DELTA_INDICES),
            modality_keys=group_names,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=group_names,
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[GR1_LANGUAGE_KEY],
        ),
    }


class VJEPAGR1Policy(BasePolicy):
    """Official BasePolicy adapter around the batched V-JEPA GR-1 runtime."""

    def __init__(self, runtime: Any, *, strict: bool = True):
        super().__init__(strict=strict)
        self.runtime = runtime
        self.action_horizon = int(runtime.action_horizon)
        self.modality_config = gr1_modality_config(self.action_horizon)

    def check_observation(self, observation: dict[str, Any]) -> None:
        validate_gr1_wire_observation(observation)

    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        return self.runtime.infer_flat_observation(observation), {}

    def check_action(self, action: dict[str, Any]) -> None:
        expected_keys = [f"action.{name}" for name, _ in GR1_GROUP_DIMS]
        if list(action) != expected_keys:
            raise ValueError(
                f"GR-1 action keys must be {expected_keys}, got {list(action)}"
            )
        batch_size = None
        for (group_name, group_dim), key in zip(
            GR1_GROUP_DIMS, expected_keys, strict=True
        ):
            value = action[key]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                raise TypeError(f"action.{group_name} must be a float32 numpy array")
            if value.ndim != 3 or value.shape[1:] != (self.action_horizon, group_dim):
                raise ValueError(
                    f"action.{group_name} must be [B,{self.action_horizon},{group_dim}], "
                    f"got {value.shape}"
                )
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("All GR-1 action groups must have the same batch size")

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_config

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        del options
        return {}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-encoder", default="vitl.pt")
    parser.add_argument("--text-cache-dir", required=True)
    parser.add_argument("--t5-len", type=int, default=48)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--encoder-family", choices=("vjepa2", "vjepa2_1"), default="vjepa2"
    )
    parser.add_argument("--encoder-model-name", default="vit_large")
    parser.add_argument("--encoder-checkpoint-key", default="target_encoder")
    parser.add_argument("--pred-depth", type=int, default=24)
    parser.add_argument("--pred-embed-dim", type=int, default=1024)
    parser.add_argument("--pred-num-heads", type=int, default=16)
    parser.add_argument("--action-hidden-size", type=int, default=512)
    parser.add_argument("--action-chunk-size", type=int, default=GR1_ACTION_HORIZON)
    parser.add_argument("--num-frames", type=int, default=GR1_MODEL_FRAMES)
    parser.add_argument("--action-num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--encoder-interpolate-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--corrected-predictor-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.t5_len <= 0:
        raise ValueError("t5_len must be positive")
    if args.action_chunk_size < 1 or args.action_chunk_size % 4:
        raise ValueError("action_chunk_size must be a positive multiple of four")
    expected_num_frames = len(GR1_VIDEO_DELTA_INDICES) + args.action_chunk_size // 4
    if args.num_frames != expected_num_frames:
        raise ValueError(
            f"action_chunk_size={args.action_chunk_size} requires "
            f"num_frames={expected_num_frames}, got {args.num_frames}"
        )
    config = replace(
        PolicyServingConfig(),
        precision=args.precision,
        encoder_family=args.encoder_family,
        encoder_checkpoint_key=args.encoder_checkpoint_key,
        model_name=args.encoder_model_name,
        t5_len=args.t5_len,
        crop_size=GR1_MODEL_IMAGE_SIZE,
        num_frames=args.num_frames,
        pred_depth=args.pred_depth,
        pred_embed_dim=args.pred_embed_dim,
        pred_num_heads=args.pred_num_heads,
        action_dim=29,
        proprio_dim=29,
        # 0 explicitly selects the legacy sincos state contract.  A missing
        # value would be inherited as the new fixed-width default (48).
        max_state_dim=0,
        proprio_encoding="sincos",
        action_chunk_size=args.action_chunk_size,
        action_hidden_size=args.action_hidden_size,
        action_num_layers=args.pred_depth,
        action_num_inference_steps=args.action_num_inference_steps,
        corrected_predictor_rope=args.corrected_predictor_rope,
        encoder_interpolate_rope=args.encoder_interpolate_rope,
        view_layout="independent",
        camera_keys=(GR1_DATASET_VIDEO_KEY,),
        dataset_stats=args.dataset_stats,
    )
    runtime = VJEPAGR1Serving(
        ckpt_path=args.checkpoint,
        pretrained_encoder=args.pretrained_encoder,
        text_cache_dir=args.text_cache_dir,
        config=config,
        device=args.device,
    )
    policy = VJEPAGR1Policy(runtime, strict=not args.no_strict)
    with PolicyServer(policy=policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
