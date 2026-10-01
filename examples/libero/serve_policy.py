"""Serve a trained VJEPA-Policy checkpoint over the openpi websocket protocol."""

import dataclasses
import logging
import socket
from typing import Literal

import numpy as np
import torch
import tyro

from vjepa_policy.policy_serving import PolicyServingConfig, VJEPAPolicyServing
from vjepa_policy.policy_serving.websocket import WebsocketPolicyServer


@dataclasses.dataclass
class Args:
    ckpt: str
    pretrained_encoder: str
    text_cache_dir: str
    dataset_stats: str | None = None
    host: str = "0.0.0.0"
    port: int = 10000
    seed: int = 7
    precision: Literal["bf16", "fp16", "fp32"] | None = None
    device: str = "cuda"
    encoder: str | None = None
    encoder_family: Literal["vjepa2", "vjepa2_1"] | None = None
    encoder_checkpoint_key: str | None = None
    model_name: str | None = None
    crop_size: int | None = None
    image_resize_mode: Literal["stretch", "letterbox"] | None = None
    patch_size: int | None = None
    num_frames: int | None = None
    tubelet_size: int | None = None
    context_tubelets: int | None = None
    pred_depth: int | None = None
    pred_embed_dim: int | None = None
    pred_num_heads: int | None = None
    num_mask_tokens: int | None = None
    lang_dim: int | None = None
    t5_len: int | None = None
    action_dim: int | None = None
    proprio_dim: int | None = None
    max_state_dim: int | None = None
    action_chunk_size: int | None = None
    action_hidden_size: int | None = None
    action_num_layers: int | None = None
    action_num_inference_steps: int | None = None
    condition_num_heads: int | None = None
    corrected_predictor_rope: bool | None = None
    encoder_interpolate_rope: bool | None = None
    view_layout: Literal["quadrant", "horizontal", "independent"] | None = None
    views: tuple[str, ...] | None = None
    # Backward-compatible alias for existing launch commands. New commands
    # should use --views, matching the canonical training CLI.
    camera_keys: tuple[str, ...] | None = None


def policy_config_from_args(args) -> PolicyServingConfig:
    """Load checkpoint topology and apply only explicit CLI overrides."""
    views = getattr(args, "views", None)
    camera_arg = getattr(args, "camera_keys", None)
    if views and camera_arg and tuple(views) != tuple(camera_arg):
        raise ValueError("--views and --camera-keys disagree")
    camera_keys = views or camera_arg
    config_fields = {field.name for field in dataclasses.fields(PolicyServingConfig)}
    overrides = {
        name: getattr(args, name)
        for name in config_fields
        if hasattr(args, name) and getattr(args, name) is not None
    }
    if camera_keys is not None:
        overrides["camera_keys"] = tuple(camera_keys)
    return PolicyServingConfig.from_checkpoint(args.ckpt, overrides=overrides)


def main(args, encoder_builder=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = policy_config_from_args(args)
    policy = VJEPAPolicyServing(
        ckpt_path=args.ckpt,
        pretrained_encoder=args.pretrained_encoder,
        text_cache_dir=args.text_cache_dir,
        config=config,
        device=args.device,
        encoder_builder=encoder_builder,
    )
    hostname = socket.gethostname()
    logging.info(
        "Starting policy server (host=%s, ip=%s, port=%d)",
        hostname,
        socket.gethostbyname(hostname),
        args.port,
    )
    WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={},
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
