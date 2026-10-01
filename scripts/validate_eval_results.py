"""Validate evaluation coverage before reporting or merging a success rate.

This utility uses only the Python standard library; it needs no model or simulator.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.libero_plus.protocol import validate_results, validate_cache


def validate_libero(path, suite, trials, task_count=10):
    content = Path(path).read_text()
    if trials < 1 or task_count < 1:
        raise ValueError("Trials and task count must be positive")

    def field(name):
        matches = re.findall(rf"^{re.escape(name)}: (.+)$", content, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {name!r} field in {path}")
        return matches[0]

    if field("Task suite name") != suite:
        raise ValueError(f"Result belongs to a different suite: {path}")
    rows = re.findall(r"^  \[(\d+)\] (\d+)/(\d+) \(", content, re.MULTILINE)
    if len(rows) != task_count or {int(row[0]) for row in rows} != set(range(task_count)):
        raise ValueError(f"Incomplete or duplicate task IDs in {path}")
    successes = 0
    for _, success, episodes in rows:
        success, episodes = int(success), int(episodes)
        if episodes != trials or not 0 <= success <= episodes:
            raise ValueError(f"Invalid per-task counts in {path}")
        successes += success
    total = task_count * trials
    if int(field("Total episodes")) != total or int(field("Total success")) != successes:
        raise ValueError(f"Aggregate counts disagree with task counts in {path}")
    rate = float(field("Total success rate"))
    if not math.isfinite(rate) or abs(rate - successes / total) > 1e-6:
        raise ValueError(f"Aggregate success rate disagrees with counts in {path}")
    return successes, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="benchmark", required=True)
    libero = subparsers.add_parser("libero")
    libero.add_argument("path", type=Path)
    libero.add_argument("--suite", required=True)
    libero.add_argument("--trials", type=int, required=True)
    libero.add_argument("--task-count", type=int, default=10)
    plus = subparsers.add_parser("libero-plus")
    plus.add_argument("path", type=Path)
    plus.add_argument("--manifest", type=Path, required=True)
    plus.add_argument("--suites", nargs="+")
    plus.add_argument("--allow-incomplete", action="store_true")
    cache = subparsers.add_parser("cache-plus")
    cache.add_argument("path", type=Path)
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--context-length", type=int, default=128)
    args = parser.parse_args()
    try:
        if args.benchmark == "libero":
            print(*validate_libero(args.path, args.suite, args.trials, args.task_count))
        elif args.benchmark == "libero-plus":
            print(json.dumps(validate_results(args.path, args.manifest, args.suites, args.allow_incomplete), indent=2))
        else:
            result = validate_cache(args.path, args.manifest, args.context_length)
            print(f"Validated {result['unique_prompts']} cached prompts")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Invalid evaluation results: {error}\n")


if __name__ == "__main__":
    main()
