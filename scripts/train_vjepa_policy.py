"""Train the language-conditioned V-JEPA world model and Action Expert."""

import argparse

import torch

from vjepa_policy.models.wm_predictor import (
    build_world_model_predictor,
)
from vjepa_policy.models.vjepa_policy import (
    build_vjepa_policy,
    encoded_proprio_dim,
)
from vjepa_policy.models import vision_transformer as video_vit
from vjepa_policy.models.vjepa2_1 import vision_transformer as video_vit_2_1
from vjepa_policy.encoders import EncoderSpec, build_encoder
from vjepa_policy.state import packed_state_dim
from vjepa_policy.train_common import (
    _strip_prefix,
    add_common_args,
    build_dataset_collator,
    build_patch_valid,
    get_model_image_size,
    get_model_video_frames,
    get_camera_keys,
    make_device,
    run_training,
    seed_process,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train V-JEPA latent prediction and flow-matching action generation."
    )
    add_common_args(parser)
    parser.set_defaults(
        pred_depth=24,
        pred_embed_dim=1024,
        pred_num_heads=16,
        action_normalization="QUANTILE",
        state_normalization="QUANTILE",
        batch_size=32,
        num_workers=8,
        prefetch_factor=2,
        weight_decay=0.01,
        num_epochs=10,
        max_steps=21360,
        mixed_precision="bf16",
        seed=7,
    )
    parser.add_argument("--action-chunk-size", type=int, default=32)
    parser.add_argument("--action-hidden-size", type=int, default=512)
    parser.add_argument("--action-num-layers", type=int, default=24)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-num-inference-steps", type=int, default=10)
    parser.add_argument("--condition-num-heads", type=int, default=8)
    parser.add_argument("--activation-checkpointing-blocks", type=int, default=12)
    parser.add_argument(
        "--predictor-rope-frequency-pairing",
        choices=("corrected", "legacy"),
        default="corrected",
    )
    parser.add_argument(
        "--encoder-interpolate-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--predictor-view-map",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Target-to-source row map for predictor view embeddings. All other "
            "predictor parameters still load strictly."
        ),
    )
    return parser.parse_args(argv)


def _implicit_encoder_name(args):
    """Resolve only the canonical registry combinations when --encoder is absent."""
    if getattr(args, "encoder", None):
        return args.encoder
    key = getattr(args, "encoder_checkpoint_key", None)
    combination = (
        getattr(args, "encoder_family", None),
        getattr(args, "model_name", None),
        key,
    )
    return {
        ("vjepa2", "vit_large", "target_encoder"): "vjepa2_vitl",
        ("vjepa2_1", "vit_large", "ema_encoder"): "vjepa2_1_vitl",
    }.get(combination)


def build_frozen_encoder(args):
    encoder_name = _implicit_encoder_name(args)
    if encoder_name:
        return build_encoder(
            encoder_name,
            checkpoint=args.checkpoint,
            image_size=get_model_image_size(args),
            video_frames=get_model_video_frames(args),
            tubelet_size=args.tubelet_size,
            interpolate_rope=args.encoder_interpolate_rope,
            checkpoint_key=args.encoder_checkpoint_key,
        )
    if args.encoder_family == "vjepa2" and args.model_name == "vit_giant":
        raise ValueError("Use vit_giant_xformers for the official 22-head V-JEPA2 ViT-G")
    encoder_module = video_vit if args.encoder_family == "vjepa2" else video_vit_2_1
    encoder_kwargs = dict(
        img_size=get_model_image_size(args),
        patch_size=args.patch_size,
        num_frames=get_model_video_frames(args),
        tubelet_size=args.tubelet_size,
        uniform_power=False,
        use_rope=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        interpolate_rope=args.encoder_interpolate_rope,
    )
    if args.encoder_family == "vjepa2_1":
        encoder_kwargs.update(
            img_temporal_dim_size=1,
            modality_embedding=True,
            n_output_distillation=1,
        )
    encoder = encoder_module.__dict__[args.model_name](**encoder_kwargs)
    encoder._vjepa_policy_training_false = args.encoder_family == "vjepa2_1"
    if args.model_name == "vit_giant_xformers" and encoder.blocks[0].attn.num_heads != 22:
        raise RuntimeError("Official ViT-G must have 22 attention heads")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if args.encoder_checkpoint_key not in checkpoint:
        raise KeyError(
            f"Checkpoint has no {args.encoder_checkpoint_key!r}; available keys: "
            f"{sorted(checkpoint)}"
        )
    print(
        f"[load] {args.encoder_family}/{args.model_name} "
        f"from {args.encoder_checkpoint_key}:",
        encoder.load_state_dict(
            _strip_prefix(checkpoint[args.encoder_checkpoint_key]), strict=True
        ),
    )
    del checkpoint
    return encoder


def parameter_summary(model):
    groups = {
        "encoder_frozen": model.encoder,
        "predictor": model.predictor,
        "action_expert": model.action_expert,
        "proprio_encoder": model.proprio_encoder,
    }
    for name, module in groups.items():
        count = sum(parameter.numel() for parameter in module.parameters())
        trainable = sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        )
        print(f"[parameters] {name}={count:,} ({count / 1e6:.2f}M), trainable={trainable:,}")


def _load_predictor_initialization(predictor, checkpoint_path, view_map=None):
    """Load only predictor weights; do not restore policy training progress."""
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if isinstance(payload, dict) and "predictor" in payload:
        state = payload["predictor"]
        metadata = payload.get("model_topology") or {}
    elif isinstance(payload, dict) and all(
        torch.is_tensor(value) for value in payload.values()
    ):
        state = payload
        metadata = {}
    else:
        raise ValueError(
            f"{checkpoint_path} is not a predictor state dict or policy checkpoint"
        )
    if state and all(name.startswith("module.") for name in state):
        state = {name[len("module.") :]: value for name, value in state.items()}
    metadata = dict(metadata)
    if view_map is not None:
        key = "view_embedding.weight"
        if key not in state or predictor.view_embedding is None:
            raise ValueError(
                "--predictor-view-map requires view embeddings in both source and target"
            )
        source = state[key]
        target_rows = predictor.view_embedding.num_embeddings
        if len(view_map) != target_rows:
            raise ValueError(
                f"predictor view map needs {target_rows} entries, got {len(view_map)}"
            )
        invalid = [index for index in view_map if not 0 <= index < source.shape[0]]
        if invalid:
            raise ValueError(
                f"predictor view map indices {invalid} exceed source rows={source.shape[0]}"
            )
        state = dict(state)
        state[key] = source.index_select(0, torch.tensor(view_map, dtype=torch.long))
        metadata["view_embedding_initialization"] = {
            "source_rows": int(source.shape[0]),
            "target_rows": int(target_rows),
            "target_to_source_map": list(view_map),
        }
    try:
        result = predictor.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Predictor initialization checkpoint is incompatible with the "
            f"requested policy predictor ({checkpoint_path}): {error}"
        ) from error
    print(f"[load] predictor initialization from {checkpoint_path}: {result}")
    return metadata


def _predictor_source_max_views(predictor_init):
    if not predictor_init:
        return None
    payload = torch.load(
        predictor_init, map_location="cpu", weights_only=False, mmap=True
    )
    topology = payload.get("model_topology", {}) if isinstance(payload, dict) else {}
    max_views = topology.get("predictor_max_views")
    if max_views is None and isinstance(payload, dict):
        state = payload.get("predictor", payload)
        view_embedding = (
            state.get("view_embedding.weight") if isinstance(state, dict) else None
        )
        if torch.is_tensor(view_embedding):
            max_views = int(view_embedding.shape[0])
    return max_views


def _predictor_max_views(requested, num_views):
    """Build the policy predictor with its native input view count."""
    max_views = num_views if requested is None else requested
    if not isinstance(max_views, int) or max_views < num_views:
        raise ValueError(
            f"predictor-max-views must be an integer >= policy views ({num_views}), "
            f"got {max_views!r}"
        )
    return max_views


def main(
    policy_builder=build_vjepa_policy,
    encoder_builder=build_frozen_encoder,
    predictor_builder=build_world_model_predictor,
    dataset_collator_builder=build_dataset_collator,
    args=None,
):
    args = parse_args() if args is None else args
    if args.action_chunk_size != args.num_frames - 1:
        raise ValueError("action_chunk_size must equal num_frames - 1")
    if args.pred_embed_dim % args.pred_num_heads:
        raise ValueError("pred_embed_dim must be divisible by pred_num_heads")
    if args.action_num_layers != args.pred_depth:
        raise ValueError("action_num_layers must equal pred_depth for direct K/V reuse")
    if args.action_hidden_size % args.condition_num_heads:
        raise ValueError("action_hidden_size must be divisible by condition_num_heads")
    predictor_init = getattr(args, "predictor_init", None)
    resume = getattr(args, "resume", None)
    if predictor_init and resume:
        raise ValueError("--predictor-init cannot be combined with --resume")

    seed_process(args.seed)
    device = make_device()
    if encoder_builder is build_frozen_encoder and not getattr(args, "encoder", None):
        args.encoder = _implicit_encoder_name(args)
    camera_keys = get_camera_keys(args)
    num_views = len(camera_keys) if args.view_layout == "independent" else 1
    predictor_max_views = _predictor_max_views(None, num_views)
    predictor_view_map = getattr(args, "predictor_view_map", None)
    source_max_views = _predictor_source_max_views(predictor_init)
    if predictor_init and predictor_view_map is None and source_max_views is not None:
        if source_max_views > num_views:
            predictor_view_map = list(range(num_views))
    if predictor_view_map is not None and len(predictor_view_map) != predictor_max_views:
        raise ValueError(
            f"predictor view map needs {predictor_max_views} entries, "
            f"got {len(predictor_view_map)}"
        )
    encoder = encoder_builder(args)
    declared_encoder_spec = getattr(encoder, "spec", None)
    encoder_spec = declared_encoder_spec
    if encoder_spec is None:
        encoder_spec = EncoderSpec(
            name=(
                getattr(args, "encoder", None)
                if encoder_builder is build_frozen_encoder
                else None
            )
            or getattr(encoder, "_vjepa_policy_model_name", None)
            or f"{args.encoder_family}_{args.model_name}",
            family=args.encoder_family,
            model_name=args.model_name,
            checkpoint_key=args.encoder_checkpoint_key,
            input_patch_size=args.patch_size,
            temporal_mode="tubelet",
            temporal_stride=args.tubelet_size,
            interpolate_rope=args.encoder_interpolate_rope,
        )
    layout = encoder_spec.layout_for_clip(
        video_frames=get_model_video_frames(args),
        image_size=get_model_image_size(args),
        num_views=num_views,
        context_steps=args.context_tubelets,
    )
    use_layout_contract = dataset_collator_builder is build_dataset_collator and (
        declared_encoder_spec is not None or encoder_builder is build_frozen_encoder
    )
    # The published cross-attention policy is the canonical fixed-width path.
    # Experimental builders retain their historical state contract until they
    # explicitly adopt the new packed-state interface.
    max_state_dim = int(getattr(args, "max_state_dim", 0) or 0)
    if max_state_dim < 0:
        raise ValueError("max_state_dim must be positive or 0")
    use_fixed_state = bool(max_state_dim) and policy_builder is build_vjepa_policy
    if use_fixed_state:
        if args.proprio_encoding != "identity":
            raise ValueError(
                "--max-state-dim requires --proprio-encoding identity; "
                "pass --max-state-dim 0 for legacy variable-width conditioning"
            )
        if args.proprio_dim > max_state_dim:
            raise ValueError(
                f"proprio_dim={args.proprio_dim} exceeds max_state_dim={max_state_dim}"
            )
        model_proprio_dim = packed_state_dim(max_state_dim)
    else:
        model_proprio_dim = encoded_proprio_dim(
            args.proprio_dim, args.proprio_encoding
        )
    predictor_kwargs = dict(
        img_size=get_model_image_size(args),
        patch_size=encoder_spec.input_patch_size,
        num_frames=get_model_video_frames(args),
        tubelet_size=encoder_spec.temporal_stride,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=args.pred_embed_dim,
        depth=args.pred_depth,
        num_heads=args.pred_num_heads,
        num_mask_tokens=args.num_mask_tokens,
        lang_dim=args.lang_dim,
        proprio_dim=model_proprio_dim,
        use_activation_checkpointing=args.use_activation_checkpointing,
        activation_checkpointing_blocks=args.activation_checkpointing_blocks,
        interpolate_rope=True,
        corrected_rope_frequency_pairing=(
            args.predictor_rope_frequency_pairing == "corrected"
        ),
        num_views=num_views,
        max_views=predictor_max_views,
    )
    if use_layout_contract:
        predictor_kwargs.update(
            latent_grid=(layout.grid_depth, layout.grid_height, layout.grid_width),
            canonical_spatial_grid=encoder_spec.canonical_spatial_grid,
        )
    predictor = predictor_builder(**predictor_kwargs)
    policy_kwargs = dict(
        action_dim=args.action_dim,
        action_chunk_size=args.action_chunk_size,
        context_len=layout.context_tokens,
        proprio_dim=args.proprio_dim,
        proprio_encoding=args.proprio_encoding,
        max_state_dim=max_state_dim if use_fixed_state else None,
        action_hidden_size=args.action_hidden_size,
        action_num_layers=args.action_num_layers,
        condition_num_heads=args.condition_num_heads,
        action_num_inference_steps=args.action_num_inference_steps,
        loss_exp=args.loss_exp,
        action_loss_weight=args.action_loss_weight,
        patch_valid=build_patch_valid(
            args, layout=layout if use_layout_contract else None
        ),
        latent_layout=layout if use_layout_contract else None,
    )
    if policy_builder is build_vjepa_policy:
        model = policy_builder(encoder, predictor, **policy_kwargs)
    else:
        policy_kwargs.pop("latent_layout", None)
        model = policy_builder(encoder, predictor, **policy_kwargs)
    predictor_init_topology = None
    if predictor_init:
        if policy_builder is not build_vjepa_policy:
            raise ValueError(
                "--predictor-init is supported only by the canonical policy builder"
            )
        predictor_init_topology = _load_predictor_initialization(
            model.predictor, predictor_init, view_map=predictor_view_map
        )
    model.camera_keys = tuple(camera_keys)
    model.view_layout = args.view_layout
    model.encoder_family = getattr(
        encoder, "_vjepa_policy_family", encoder_spec.family
    )
    model.encoder_model_name = getattr(
        encoder, "_vjepa_policy_model_name", encoder_spec.model_name
    )
    model.encoder_checkpoint_key = getattr(
        encoder, "_vjepa_policy_checkpoint_key", encoder_spec.checkpoint_key
    )
    model.encoder_name = encoder_spec.name
    model.encoder_spec = encoder_spec.to_dict() if use_layout_contract else None
    model.action_indices = tuple(args.action_indices) if args.action_indices else None
    model.state_indices = tuple(args.state_indices) if args.state_indices else None
    model.relative_action_indices = (
        tuple(args.relative_action_indices) if args.relative_action_indices else None
    )
    model.instruction_field = args.instruction_field
    model.policy_serving = {
        "config": {
            "precision": "fp32"
            if args.mixed_precision == "no"
            else args.mixed_precision,
            "encoder_family": model.encoder_family,
            "encoder_checkpoint_key": model.encoder_checkpoint_key,
            "model_name": model.encoder_model_name,
            "encoder": encoder_spec.name,
            "crop_size": args.crop_size,
            "patch_size": encoder_spec.input_patch_size,
            "num_frames": get_model_video_frames(args),
            "tubelet_size": encoder_spec.temporal_stride,
            "context_tubelets": args.context_tubelets,
            "pred_depth": args.pred_depth,
            "pred_embed_dim": args.pred_embed_dim,
            "pred_num_heads": args.pred_num_heads,
            "num_mask_tokens": args.num_mask_tokens,
            "lang_dim": args.lang_dim,
            "t5_len": args.context_len,
            "action_dim": args.action_dim,
            "proprio_dim": args.proprio_dim,
            # 0 is serialized as the explicit legacy variable-width sentinel;
            # omitting the field would make serving assume the new 48-wide
            # contract for this checkpoint.
            "max_state_dim": max_state_dim if use_fixed_state else 0,
            "packed_proprio_dim": model_proprio_dim,
            "proprio_encoding": args.proprio_encoding,
            "action_chunk_size": args.action_chunk_size,
            "action_hidden_size": args.action_hidden_size,
            "action_num_layers": args.action_num_layers,
            "action_num_inference_steps": args.action_num_inference_steps,
            "condition_num_heads": args.condition_num_heads,
            "corrected_predictor_rope": (
                args.predictor_rope_frequency_pairing == "corrected"
            ),
            "encoder_interpolate_rope": args.encoder_interpolate_rope,
            "view_layout": args.view_layout,
            "camera_keys": list(camera_keys),
            "encoder_spec": encoder_spec.to_dict() if use_layout_contract else None,
            "latent_layout": layout.to_dict() if use_layout_contract else None,
        },
        "encoder_adapter": encoder_spec.name,
    }
    if predictor_init_topology is not None:
        model.predictor_initialization = {
            "path": predictor_init,
            "source_topology": predictor_init_topology,
        }
    parameter_summary(model)
    model = model.to(device)
    if dataset_collator_builder is build_dataset_collator:
        dataset, collator = dataset_collator_builder(
            args, action_chunk_size=args.action_chunk_size, layout=layout
        )
    else:
        dataset, collator = dataset_collator_builder(
            args, action_chunk_size=args.action_chunk_size
        )
    run_training(args, model, dataset, collator)


if __name__ == "__main__":
    main()
