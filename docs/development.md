# Development checks

Run from the repository root. The baseline suite uses only the Python standard library and Bash; it does not install the training package or download weights.

```bash
python scripts/check_environment.py --profile source
python -m unittest discover -s tests -v
```

[GitHub Actions](../.github/workflows/checks.yml) runs this suite on Linux with Python 3.10 and 3.12, plus Python/Bash syntax checks. Tests exercise full and partial LIBERO-Plus result coverage, duplicate/mismatched episodes, cache provenance, language perturbation handling, and LIBERO aggregate counts. Training launchers run against a fake executable to check actual forwarded arguments, global batch validation, and the distinction between fresh and predictor-initialized runs.

These checks do not validate model numerics, GPU memory requirements, MuJoCo rendering, or published success rates. A passing workflow means the lightweight contracts passed; it does not establish benchmark reproduction.

## Inspect an execution environment

Use the interpreter that will run each part of the experiment:

```bash
"$POLICY_PYTHON" scripts/check_environment.py --profile policy --require-cuda
"$LIBERO_PYTHON" scripts/check_environment.py --profile libero
"$LIBERO_PLUS_PYTHON" scripts/check_environment.py --profile libero-plus
```

The checker lists installed distribution versions and checks Python requirements and dependency consistency (`pip check`). `--require-cuda` additionally imports PyTorch and checks CUDA availability. The Plus profile also checks the exact pins in its simulator dependency recipe. The Plus task-preparation command separately verifies the simulator checkout and classification checksum. None of these checks downloads dependencies or starts a simulator. Save the JSON output with the run.

## Recipe precedence

Paper recipe defaults live in [configs/recipes](../configs/recipes). Training launchers and editable environment templates source the same files. Existing environment values override defaults; applicable Python CLI arguments are appended last. For the LIBERO launcher, set batch size and accumulation through `BATCH_SIZE` and `GRAD_ACCUM`, not duplicate CLI flags, so the batch check covers the actual run. Source a local template explicitly to provide artifact paths. Launchers do not search for `.local.env` files. Start a clean shell when switching recipes.

Global batch is GPUs × per-device batch × accumulation. A changed hardware topology must still match `GLOBAL_BATCH_SIZE`, or explicitly change it for a new experiment. For LIBERO, for example, four GPUs with batch 32 and accumulation 1 preserve global batch 128. GR1 defaults to global batch 256 with no accumulation; DROID's reference launcher retains its eight-GPU requirement.
