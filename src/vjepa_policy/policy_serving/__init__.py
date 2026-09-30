"""Deployment adapters for V-JEPA policies."""

from vjepa_policy.policy_serving.gr1 import (
    VJEPAGR1Serving,
    action_29d_to_robocasa,
    build_gr1_model_clip,
    decode_gr1_action_chunk,
    flat_gr1_state_to_29d,
    gr1_language_batch,
    grouped_state_to_29d,
    prepare_gr1_model_inputs,
    relative_action_to_absolute,
    validate_gr1_wire_observation,
)


def __getattr__(name):
    if name in {"PolicyServingConfig", "VJEPAPolicyServing"}:
        from vjepa_policy.policy_serving.libero import (
            PolicyServingConfig,
            VJEPAPolicyServing,
        )

        return {
            "PolicyServingConfig": PolicyServingConfig,
            "VJEPAPolicyServing": VJEPAPolicyServing,
        }[name]
    raise AttributeError(name)


__all__ = [
    "PolicyServingConfig",
    "VJEPAGR1Serving",
    "VJEPAPolicyServing",
    "action_29d_to_robocasa",
    "build_gr1_model_clip",
    "decode_gr1_action_chunk",
    "flat_gr1_state_to_29d",
    "gr1_language_batch",
    "grouped_state_to_29d",
    "prepare_gr1_model_inputs",
    "relative_action_to_absolute",
    "validate_gr1_wire_observation",
]
