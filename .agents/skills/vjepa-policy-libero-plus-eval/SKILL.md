---
name: vjepa-policy-libero-plus-eval
description: Prepare and evaluate VJEPA-Policy on native LIBERO-Plus using the shared LIBERO client, pinned simulator, task manifest, T5 cache, and per-episode coverage checks. Use for robustness scoring across the seven perturbation axes.
---

# V-JEPA Policy LIBERO-Plus evaluation

Read [the Plus guide](../../../examples/libero_plus/README.md). Run commands from the repository root. The public workflow uses `examples/libero_plus/prepare.py`, `cache_text.py`, and `evaluate_policy.sh`, plus `configs/libero_plus_eval.env`. It shares `examples/libero/run_libero_client.py` and the ordinary policy server. No private harness adapter or harness-specific launch flags are part of this flow.

## Environment and artifacts

Use the pinned Plus checkout from the guide in its own simulator environment: LIBERO and Plus install the same package/suite names. Keep `LIBERO_CONFIG_PATH` separate too. `prepare.py` checks the revision and classification checksum, writes the isolated config, and verifies the imported benchmark's location. Simulator state loading is tied to the documented Torch version; inspect upstream `weights_only` behavior before upgrading it. Preserve MagickWand for noise tasks.

Keep the policy environment separate. Use the LIBERO-trained checkpoint and its matching encoder and normalization statistics. Do not fine-tune on Plus tasks. Generate the full task manifest in the simulator environment, then the text cache in the policy environment, using the run's T5 snapshot and context length 128.

## Protocol contracts

- `protocol.py` defines the pinned revision, classification hash, and suite counts: 2,402 / 2,518 / 2,591 / 2,519, totaling 10,030 episodes.
- Use task order 0, one episode per task, seed 7, and ten settling steps.
- Non-language shifts use canonical instructions, including removal of the Goal `_moved` suffix. Language shifts retain the rewritten BDDL instruction. Both cache generation and rollout use the manifest's actual instructions.
- The cache records the manifest hash, T5 identity, context length, and files. Its prompt count is derived; do not assume a historical cache is interchangeable.
- Preserve current/past frames, 180-degree rotation of both cameras, frame stride 4, and execution horizon 16. Server geometry comes from checkpoint metadata.

## Execute and accept

The wrapper starts one server/client pair per suite and waits for `/healthz`. EGL is the default renderer; `LIBERO_MUJOCO_GL=osmesa` selects CPU simulation. Videos are opt-in. Use a fresh output directory; the example has no sharding or automatic resume. Use `EVAL_SERIAL=1 EVAL_GPUS=0` for a full single-GPU run.

The full score requires all workers to exit successfully and 10,030 unique, manifest-matched task outcomes. Errors abort instead of counting as failed episodes. Inspect `results.json`, per-suite `episodes.jsonl`, logs, and copied manifests. The validator reports overall episode-weighted success, category denominators, and `complete`. A partial run must stay labeled partial.

For a one-task diagnostic, keep the full manifest/cache and set `MAX_TASKS=1` with `ALLOW_INCOMPLETE=1`, one suite, and one GPU. Preserve logs on failure and stop only processes created by the run. Report which checks actually executed; static checks and a healthy server do not establish successful GPU rollouts.
