"""LIBERO-Plus task and result contracts. Uses only the standard library."""

import hashlib
import json
from pathlib import Path
import re

REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
CLASSIFICATION_SHA256 = "faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6"
SUITE_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
CATEGORIES = {
    "Objects Layout", "Camera Viewpoints", "Robot Initial States",
    "Language Instructions", "Light Conditions", "Background Textures", "Sensor Noise",
}


def canonical_instruction(name):
    """Remove perturbation suffixes, retaining words such as 'table_center'."""
    base = re.split(r"_(?:table|tb|view|light|language|add|noise)_-?\d|_level\d", name, maxsplit=1)[0]
    base = base.removesuffix("_moved")
    base = re.sub(r"^[A-Z_]+_SCENE\d+_", "", base)
    return base.replace("_", " ")


def task_instruction(name, category, language):
    # Upstream non-language task.language includes perturbation suffixes.
    # Only the language axis changes the policy's instruction in this protocol.
    if category == "Language Instructions":
        if not isinstance(language, str) or not language.strip():
            raise ValueError(f"Missing language instruction for {name}")
        return language.strip()
    return canonical_instruction(name)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if (manifest.get("schema_version") != 1
            or manifest.get("benchmark_revision") != REVISION
            or manifest.get("classification_sha256") != CLASSIFICATION_SHA256
            or manifest.get("canonical_task_language") is not True):
        raise ValueError("Manifest does not describe the pinned LIBERO-Plus protocol")
    tasks = manifest.get("tasks", [])
    seen = set()
    names = set()
    for row in tasks:
        suite, task_id = row.get("suite"), row.get("task_id")
        if (suite not in SUITE_COUNTS or type(task_id) is not int
                or not 0 <= task_id < SUITE_COUNTS[suite]
                or (suite, task_id) in seen or (suite, row.get("name")) in names
                or not isinstance(row.get("name"), str)
                or row.get("category") not in CATEGORIES
                or not isinstance(row.get("instruction"), str) or not row["instruction"].strip()):
            raise ValueError(f"Invalid or duplicate manifest task: {row}")
        seen.add((suite, task_id))
        names.add((suite, row["name"]))
    expected = {(suite, index) for suite, count in SUITE_COUNTS.items() for index in range(count)}
    if seen != expected:
        raise ValueError(f"Manifest has {len(seen)} tasks; expected all {len(expected)}")
    return manifest


def validate_cache(directory, manifest_path, context_length=128):
    manifest = load_manifest(manifest_path)
    directory = Path(directory)
    cache = json.loads((directory / "libero_plus_cache.json").read_text())
    files = cache.get("files", [])
    unique = len({row["instruction"] for row in manifest["tasks"]})
    if (cache.get("manifest_sha256") != digest_file(manifest_path)
            or cache.get("context_length") != context_length or cache.get("complete") is not True
            or cache.get("tasks") != len(manifest["tasks"]) or cache.get("unique_prompts") != unique
            or len(files) != unique or len(set(files)) != unique):
        raise ValueError("Cache is incomplete or has different task/text-length provenance")
    for name in files:
        if not re.fullmatch(rf"[a-f0-9]{{64}}\.t5_len{context_length}\.pt", name):
            raise ValueError(f"Unexpected cache filename: {name}")
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
            raise ValueError(f"Missing text cache entry: {name}")
    return cache


def validate_results(directory, manifest_path, suites=None, allow_incomplete=False):
    """Require exact task identity, provenance, and one boolean outcome per task."""
    manifest = load_manifest(manifest_path)
    manifest_sha = digest_file(manifest_path)
    suites = list(SUITE_COUNTS if suites is None else suites)
    if not suites or len(set(suites)) != len(suites) or not set(suites) <= SUITE_COUNTS.keys():
        raise ValueError("Select unique, known LIBERO-Plus suites")
    expected = {(row["suite"], row["task_id"]): row for row in manifest["tasks"] if row["suite"] in suites}
    seen, counts, successes = set(), {}, 0
    categories = {}
    for suite in suites:
        file = Path(directory) / suite / "episodes.jsonl"
        counts[suite] = 0
        for line in file.read_text().splitlines():
            row = json.loads(line)
            task_id = row.get("task_id")
            if type(task_id) is not int:
                raise ValueError("Invalid episode task ID")
            key = (suite, task_id)
            task = expected.get(key)
            if (key in seen or task is None or row.get("suite") != suite
                    or type(row.get("episode_id")) is not int or row.get("episode_id") != 0
                    or type(row.get("success")) is not bool
                    or row.get("manifest_sha256") != manifest_sha
                    or any(row.get(field) != task[field] for field in ("name", "category", "instruction"))):
                raise ValueError(f"Invalid, duplicate, or mismatched episode: {key}")
            seen.add(key)
            counts[suite] += 1
            successes += row["success"]
            axis = categories.setdefault(task["category"], {"episodes": 0, "successes": 0})
            axis["episodes"] += 1
            axis["successes"] += row["success"]
    if not seen or (not allow_incomplete and seen != expected.keys()):
        raise ValueError(f"Incomplete evaluation: {len(seen)}/{len(expected)} selected tasks")
    for axis in categories.values():
        axis["success_rate"] = axis["successes"] / axis["episodes"]
    return {
        "episodes": len(seen), "successes": successes,
        "success_rate": successes / len(seen), "suites": counts, "categories": categories,
        "selected_complete": seen == expected.keys(),
        "complete": len(seen) == sum(SUITE_COUNTS.values()),
        "manifest_sha256": manifest_sha,
    }
