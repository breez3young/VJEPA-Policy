---
name: vjepa-policy-libero-eval
description: Run and validate the four-suite LIBERO success-rate evaluation for a VJEPA-Policy checkpoint. Use for ordinary LIBERO scoring or matched encoder comparisons; distinguish full coverage from smoke runs.
---

# V-JEPA Policy LIBERO evaluation

Run from the repository root. Read [the LIBERO guide](../../../examples/libero/README.md) and use `examples/libero/evaluate_policy.sh RUN_DIR [CHECKPOINT]`. Copy `configs/libero_eval.env` to an ignored local file for machine paths.

## Protocol and configuration

The reference protocol is four suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`), ten tasks each, 50 trials/task: 2,000 episodes. Seed is 7, replan steps 16, frame stride 4, render resolution 256. Preserve the checkpoint's encoder, ordered views, preprocessing, action and state contracts. Use the [inference skill](../vjepa-policy-inference/SKILL.md) for loading issues.

Resolve `RUN_DIR`, `CHECKPOINT`, `DATASET_STATS`, `VJEPA2_ENCODER_CHECKPOINT`, `TEXT_EMBEDDING_CACHE`, `POLICY_PYTHON`, and `LIBERO_PYTHON`. The encoder variable name is historical; its value must match the actual checkpoint's encoder. Keep topology overrides unset unless justified by checkpoint metadata or the original run configuration. Do not source an unrelated training recipe before evaluation, since exported encoder settings become explicit overrides.

## Launch and observe

- Choose a fresh `EVAL_DIR`; the launcher refuses nonempty outputs. It starts one server/client pair per suite using `EVAL_GPUS` in suite order and consecutive ports from `BASE_PORT`. It has no simulator-GPU list or sharding option.
- Use `EVAL_SERIAL=1 EVAL_GPUS=0` to evaluate all four suites sequentially on one GPU in one result directory. `EVAL_SUITES` selects a subset for diagnostics.
- A smoke run with one suite and `NUM_TRIALS_PER_TASK=1` covers ten episodes. Use the direct client `--args.max-tasks 1` for exactly one task. Honor requested smoke-only scope; do not automatically expand it into a full benchmark.
- Save the command, revision, relevant environment values, artifact identities, and renderer. Monitor logs, worker state, and completed episodes while active. A live server or high GPU utilization is not evidence of benchmark progress.

## Acceptance

Inspect `summary.txt`, each `.status`, and each suite's `*_eval_results.txt`. The launcher validates IDs 0–9, trials per task, totals, and success counts. For the full protocol require all four workers to exit zero, 500 episodes/suite, and 2,000 overall. Compute rate as total successes / total episodes; do not average partial suites into a full score. Keep diagnostic results separate.

A failed worker, missing coverage, or stale result is incomplete. Preserve logs, use a new directory for retries, and stop only this run's processes. Report the actual protocol and completion count; do not use historical private scores as pass/fail thresholds.
