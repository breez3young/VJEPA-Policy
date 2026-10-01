# Working in V-JEPA Policy

Run commands from the repository root (the directory with `pyproject.toml`). Use the user's existing checkout and artifact locations; no host-specific paths or environment names are part of the public interface.

## Task routing

Read only the skill relevant to the requested task:

| Task | Skill |
| --- | --- |
| Prepare data/caches, train, resume, transfer a predictor | [.agents/skills/vjepa-policy-training/SKILL.md](.agents/skills/vjepa-policy-training/SKILL.md) |
| Serve a checkpoint or check an action request | [.agents/skills/vjepa-policy-inference/SKILL.md](.agents/skills/vjepa-policy-inference/SKILL.md) |
| LIBERO success rate | [.agents/skills/vjepa-policy-libero-eval/SKILL.md](.agents/skills/vjepa-policy-libero-eval/SKILL.md) |
| LIBERO-Plus robustness | [.agents/skills/vjepa-policy-libero-plus-eval/SKILL.md](.agents/skills/vjepa-policy-libero-plus-eval/SKILL.md) |
| RoboCasa-GR1 | [.agents/skills/vjepa-policy-gr1/SKILL.md](.agents/skills/vjepa-policy-gr1/SKILL.md) |

## Source of truth

- Setup and artifact locations: [docs/setup.md](docs/setup.md).
- Training recipe variables and CLI precedence: [docs/training.md](docs/training.md).
- Evaluation guides: [LIBERO](examples/libero/README.md), [LIBERO-Plus](examples/libero_plus/README.md), and [RoboCasa-GR1](examples/gr1/README.md).
- Dataset and encoder interfaces: [docs/extensions.md](docs/extensions.md).
- Shared training defaults: `configs/recipes/`; lightweight checks and environment reports: [docs/development.md](docs/development.md).
- Serving metadata resolution: `PolicyServingConfig.from_checkpoint` in `src/vjepa_policy/policy_serving/libero.py`. Prefer checkpoint metadata over guesses based on a run name. GR-1 has its own explicit serving arguments.

## Working conventions

Write each Markdown prose paragraph on one source line; preserve code blocks, tables, and list structure. Use conda for the documented environment setup.

Keep machine paths in ignored `configs/*.local.env` files. Keep generated artifacts in the user's chosen output directory (`runs/` is ignored). Preserve existing experiments; use a fresh directory for a different evaluation.

Update the relevant guide and skill when an entry point changes. A skill should record decisions and non-obvious contracts, not duplicate a launcher's internals. Use the paper's variant names **From Scratch** and **Pretrained Predictor**. The latter uses predictor-only pretraining on DROID without action labels, followed by downstream joint training with a freshly initialized action expert. Do not infer a released checkpoint, supported benchmark, or verified score from a historical experiment name. Label smoke runs and partial results explicitly.

For changes to launchers, inspect their CLI and result-validation contracts. GPU training and simulator rollouts require their external assets; respect the user's execution constraints and report which checks actually ran. Public LIBERO-Plus evaluation uses the native example, its pinned simulator, and generated task/cache manifests. Do not route it through a private harness adapter.
