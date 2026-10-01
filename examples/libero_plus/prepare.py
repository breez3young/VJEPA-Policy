"""Configure an isolated LIBERO-Plus checkout and enumerate the policy prompts."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.libero_plus.protocol import (
    CLASSIFICATION_SHA256, REVISION, SUITE_COUNTS, digest_file, load_manifest, task_instruction,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-plus-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.libero_plus_root.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error(f"Expected LIBERO-Plus revision {REVISION}; got {revision}")
    base = root / "libero/libero"
    classification = base / "benchmark/task_classification.json"
    if digest_file(classification) != CLASSIFICATION_SHA256:
        parser.error("The official task classification has changed")
    if not (base / "assets/textures").is_dir():
        parser.error("Extract the official assets.zip into LIBERO-plus/libero/libero first")
    config = {"benchmark_root": str(base), "bddl_files": str(base / "bddl_files"),
              "init_states": str(base / "init_files"), "datasets": str(root / "libero/datasets"),
              "assets": str(base / "assets")}
    config_dir = args.config_dir.resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    # JSON is valid YAML; no import of LIBERO until its isolated config exists.
    if config_file.exists():
        import yaml
        if yaml.safe_load(config_file.read_text()) != config:
            parser.error("Existing LIBERO config differs; choose a new --config-dir")
    else:
        config_file.write_text(json.dumps(config, indent=2) + "\n")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    from libero.libero import benchmark
    if Path(benchmark.__file__).resolve().parent != base / "benchmark":
        parser.error("The selected interpreter imports another LIBERO installation")
    raw = json.loads(classification.read_text())
    tasks = []
    for suite, count in SUITE_COUNTS.items():
        task_suite = benchmark.get_benchmark_dict()[suite](task_order_index=0)
        classified = {row["name"]: row for row in raw[suite]}
        if task_suite.n_tasks != count or len(classified) != count:
            parser.error(f"Wrong task coverage in {suite}")
        for task_id in range(count):
            task = task_suite.get_task(task_id)
            entry = classified[task.name]
            tasks.append({"suite": suite, "task_id": task_id, "name": task.name,
                          "category": entry["category"], "difficulty_level": entry["difficulty_level"],
                          "instruction": task_instruction(task.name, entry["category"], task.language)})
    manifest = {"schema_version": 1, "benchmark_revision": revision,
                "classification_sha256": CLASSIFICATION_SHA256,
                "canonical_task_language": True, "tasks": tasks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != encoded:
        parser.error("Output exists with different contents; select a new manifest path")
    args.output.write_text(encoded)
    load_manifest(args.output)
    print(f"Prepared {len(tasks)} tasks, {len({task['instruction'] for task in tasks})} unique instructions")
    print(f"Use LIBERO_CONFIG_PATH={config_dir}")


if __name__ == "__main__":
    main()
