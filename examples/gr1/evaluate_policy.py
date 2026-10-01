"""Evaluate a served V-JEPA policy on the 24 official GR-1 tabletop tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


GR1_TASKS = (
    "PnPBottleToCabinetClose",
    "PnPCanToDrawerClose",
    "PnPCupToDrawerClose",
    "PnPMilkToMicrowaveClose",
    "PnPPotatoToMicrowaveClose",
    "PnPWineToCabinetClose",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA",
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
    "PosttrainPnPNovelFromPlacematToBasketSplitA",
    "PosttrainPnPNovelFromPlacematToBowlSplitA",
    "PosttrainPnPNovelFromPlacematToPlateSplitA",
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
    "PosttrainPnPNovelFromPlateToBowlSplitA",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA",
    "PosttrainPnPNovelFromPlateToPanSplitA",
    "PosttrainPnPNovelFromPlateToPlateSplitA",
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "PosttrainPnPNovelFromTrayToPlateSplitA",
    "PosttrainPnPNovelFromTrayToPotSplitA",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA",
)
GR1_ENV_SUFFIX = "_GR1ArmsAndWaistFourierHands_Env"
GR1_EXECUTION_HORIZON = 16
GR1_WIRE_VIDEO_KEY = "video.ego_view_bg_crop_pad_res256_freq20"


def env_id(task: str) -> str:
    if task not in GR1_TASKS:
        raise ValueError(f"Unknown GR-1 task {task!r}")
    return f"gr1_unified/{task}{GR1_ENV_SUFFIX}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_report(
    path: Path,
    records: list[dict[str, Any]],
    execution_horizon: int,
) -> None:
    success_rates = [record["success_rate"] for record in records]
    payload = {
        "execution_horizon": execution_horizon,
        "completed_tasks": len(records),
        "macro_success_rate": float(np.mean(success_rates)) if success_rates else None,
        "tasks": records,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _truncate_episode_results(successes, episode_info, n_episodes):
    successes = [bool(value) for value in successes]
    if len(successes) < n_episodes:
        raise ValueError(
            f"Rollout returned {len(successes)} episodes, expected at least {n_episodes}"
        )
    truncated_info = {}
    for key, values in episode_info.items():
        if len(values) != len(successes):
            raise ValueError(
                f"Episode info {key!r} has {len(values)} entries for "
                f"{len(successes)} successes"
            )
        truncated_info[key] = values[:n_episodes]
    return successes[:n_episodes], truncated_info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--task", choices=GR1_TASKS, action="append")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=GR1_EXECUTION_HORIZON,
        help="Number of actions executed before requesting a new policy chunk.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/gr1_eval/results.json")
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Enable rollout video recording in this directory; omitted disables recording.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    from gr00t.eval import rollout_policy

    args = parse_args(argv)
    if args.execution_horizon < 1:
        raise ValueError("execution_horizon must be positive")
    tasks = args.task or GR1_TASKS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        rollout_policy.ROBOCASA_RECORD_VIDEO_KEYS_BY_PREFIX["gr1_unified"] = (
            GR1_WIRE_VIDEO_KEY,
        )
    records = []
    for task in tasks:
        task_env_id = env_id(task)
        results = rollout_policy.run_gr00t_sim_policy(
            env_name=task_env_id,
            n_episodes=args.n_episodes,
            max_episode_steps=args.max_episode_steps,
            policy_client_host=args.host,
            policy_client_port=args.port,
            n_envs=args.n_envs,
            n_action_steps=args.execution_horizon,
            video_dir=str(args.video_dir / task)
            if args.video_dir is not None
            else None,
            seed=args.seed,
        )
        successes, episode_info = _truncate_episode_results(
            results[1], results[2], args.n_episodes
        )
        records.append(
            {
                "task": task,
                "env_id": task_env_id,
                "successes": successes,
                "success_rate": float(np.mean(successes)),
                "episode_info": _jsonable(episode_info),
            }
        )
        _write_report(args.output, records, args.execution_horizon)
        print(f"{task}: {sum(successes)}/{len(successes)} = {np.mean(successes):.3%}")

    print(
        f"Macro success rate: {np.mean([item['success_rate'] for item in records]):.3%}"
    )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
