# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# LIBERO simulation client for
# vjepa_policy.policy_serving.VJEPAPolicyServing, adapted from PRTS's
# examples/libero/main.py (same openpi_client.websocket_client_policy +
# LIBERO OffScreenRenderEnv loop). Two differences from the PRTS original:
#
#   1. `replan_steps` defaults to 16 (not 5): the policy was trained with a
#      32-step action chunk (--action-chunk-size 32), so this executes the
#      first half of each predicted chunk before re-planning.
#   2. The observation dict carries 2 EXTRA keys, `observation/image_past` /
#      `observation/wrist_image_past` -- the frame from `frame_stride` raw env
#      steps ago. The server only gets called once per CHUNK (every
#      `replan_steps` env steps), so it cannot reconstruct "frame_stride steps
#      ago" itself; this client sees every raw step, so it tracks a short
#      rolling history and sends both frames explicitly, matching training's
#      2-frame [obs(t-frame_stride), obs(t)] context exactly regardless of
#      replan_steps. The server's generic `observation/<name>` conversion
#      handles the extra keys with no changes to the shared transport code.
#
# Use the simulator environment selected in examples/libero/README.md or
# examples/libero_plus/README.md. The policy server runs separately.

import collections
import dataclasses
import logging
import json
import sys
import math
import pathlib
from typing import Literal, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from examples.libero_plus.protocol import (
    CLASSIFICATION_SHA256, digest_file, load_manifest, task_instruction,
)

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from termcolor import cprint

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 10000
    resize_size: int = 256  # Native LIBERO render resolution. The server independently
                             # resizes each camera view to the policy's 224x224 input grid.
    replan_steps: int = 16  # ck=32 trained chunk -> execute the first half before re-planning.
    frame_stride: int = 4   # Match training's --past-frames for the two context frames.

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    benchmark: Literal["libero", "libero-plus"] = "libero"
    manifest: Optional[str] = None  # Required for LIBERO-Plus.
    record_video: bool = True
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: Optional[int] = None  # Optional smoke-test limit; None evaluates the full suite.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    if min(args.replan_steps, args.num_trials_per_task, args.resize_size) <= 0 or args.frame_stride < 0 or args.num_steps_wait < 0:
        raise ValueError("Invalid rollout geometry or episode count")
    plus_tasks = {}
    manifest_sha = None
    if args.benchmark == "libero-plus":
        if not args.manifest or args.num_trials_per_task != 1:
            raise ValueError("LIBERO-Plus requires --args.manifest and one trial per task")
        manifest = load_manifest(args.manifest)
        manifest_sha = digest_file(args.manifest)
        plus_tasks = {row["task_id"]: row for row in manifest["tasks"] if row["suite"] == args.task_suite_name}
        classification = pathlib.Path(benchmark.__file__).parent / "task_classification.json"
        if digest_file(classification) != CLASSIFICATION_SHA256:
            raise ValueError("The simulator is not the pinned LIBERO-Plus installation")
    elif args.manifest:
        raise ValueError("--args.manifest is only used for LIBERO-Plus")
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.benchmark == "libero-plus" and len(plus_tasks) != num_tasks_in_suite:
        raise ValueError("Manifest and simulator task counts differ")
    if args.max_tasks is not None:
        if args.max_tasks <= 0:
            raise ValueError(f"max_tasks must be positive, got {args.max_tasks}")
        num_tasks_in_suite = min(num_tasks_in_suite, args.max_tasks)
    cprint(f"Task suite: {args.task_suite_name}", "green")
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    episodes_path = pathlib.Path(args.video_out_path) / "episodes.jsonl"
    if episodes_path.exists():
        raise ValueError("Use a fresh output directory for each evaluation")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        plus_task = plus_tasks.get(task_id)
        if plus_task is not None:
            instruction = task_instruction(task.name, plus_task["category"], task.language)
            if plus_task["name"] != task.name or plus_task["instruction"] != instruction:
                raise ValueError("Manifest and simulator task identity/instruction differ")
            task_description = instruction

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            # Rolling history of (main_img, wrist_img) for the past-frame context;
            # bounded to what _get_past_frame ever needs (frame_stride + 1 entries).
            frame_hist = collections.deque(maxlen=args.frame_stride + 2)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    frame_hist.append((img, wrist_img))

                    # Save preprocessed image for replay video
                    if args.record_video:
                        replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk.
                        # Past frame = frame_stride raw steps ago (matches training's
                        # [obs(t-frame_stride), obs(t)] context); falls back to the
                        # earliest available frame at episode start.
                        if len(frame_hist) > args.frame_stride:
                            past_img, past_wrist = frame_hist[-1 - args.frame_stride]
                        else:
                            past_img, past_wrist = frame_hist[0]

                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/image_past": past_img,
                            "observation/wrist_image_past": past_wrist,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = np.asarray(client.infer(element)["actions"])
                        if (action_chunk.ndim != 2 or action_chunk.shape[1] != 7
                                or len(action_chunk) < args.replan_steps or not np.isfinite(action_chunk).all()):
                            raise ValueError("Policy must return finite [chunk >= replan_steps, 7] actions")
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception:
                    logging.exception("Rollout failed")
                    raise

            task_episodes += 1
            total_episodes += 1

            row = {"suite": args.task_suite_name, "task_id": task_id,
                   "name": task.name, "episode_id": episode_idx,
                   "instruction": task_description, "success": bool(done)}
            if plus_task is not None:
                row.update(category=plus_task["category"], manifest_sha256=manifest_sha)
            with episodes_path.open("a", encoding="utf-8") as result_file:
                result_file.write(json.dumps(row) + "\n")
            if args.record_video and replay_images:
                suffix = "success" if done else "failure"
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"task{task_id:04d}_episode{episode_idx:03d}_{suffix}.mp4",
                    replay_images, fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        task_results.append(
            {
                "task_id": task_id,
                "task": task_description,
                "successes": task_successes,
                "episodes": task_episodes,
                "success_rate": float(task_successes) / float(task_episodes),
            }
        )
        env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # Save the test results to a txt file
    results_txt_path = pathlib.Path(args.video_out_path) / f"{args.task_suite_name}_eval_results.txt"
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Task suite name: {args.task_suite_name}\n")
        f.write(f"Total success rate: {float(total_successes) / float(total_episodes)}\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Total success: {total_successes}\n")
        f.write("Task results:\n")
        for result in task_results:
            f.write(
                f"  [{result['task_id']:02d}] {result['successes']}/{result['episodes']} "
                f"({result['success_rate']:.4f}) - {result['task']}\n"
            )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # Plus parses camera/noise suffixes with string operations before opening BDDL.
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
