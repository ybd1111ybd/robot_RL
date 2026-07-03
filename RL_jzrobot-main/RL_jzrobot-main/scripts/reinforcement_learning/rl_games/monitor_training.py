"""Summarize the latest RL-Games TensorBoard scalars for a JZ reach run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def _resolve_default_log_root() -> Path:
    isaaclab_path = os.environ.get("ISAACLAB_PATH")
    if isaaclab_path:
        return Path(isaaclab_path).expanduser().resolve() / "logs" / "rl_games" / "jz_bi_reach"
    return Path.cwd() / "logs" / "rl_games" / "jz_bi_reach"


DEFAULT_LOG_ROOT = _resolve_default_log_root()
DEFAULT_TAGS = (
    "rewards/iter",
    "shaped_rewards/iter",
    "episode_lengths/iter",
    "Episode/Metrics/left_ee_pose/position_error",
    "Episode/Metrics/right_ee_pose/position_error",
    "Episode/Episode_Reward/left_end_effector_position_tracking",
    "Episode/Episode_Reward/right_end_effector_position_tracking",
    "Episode/Episode_Reward/left_end_effector_position_tracking_fine_grained",
    "Episode/Episode_Reward/right_end_effector_position_tracking_fine_grained",
    "Episode/Episode_Reward/action_rate",
    "Episode/Episode_Reward/action_max_abs_penalty",
    "Episode/Episode_Reward/left_joint_vel",
    "Episode/Episode_Reward/right_joint_vel",
    "Episode/Episode_Reward/left_end_effector_stable_goal_bonus",
    "Episode/Episode_Reward/right_end_effector_stable_goal_bonus",
    "Episode/Episode_Reward/left_tcp_speed_near_goal",
    "Episode/Episode_Reward/right_tcp_speed_near_goal",
    "Episode/Episode_Reward/left_action_rate_near_goal",
    "Episode/Episode_Reward/right_action_rate_near_goal",
)


def _resolve_run_dir(log_root: Path, run_name: str | None) -> Path:
    if run_name:
        run_dir = log_root / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    runs = sorted((path for path in log_root.iterdir() if path.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run directories found under: {log_root}")
    return runs[0]


def _latest_event_file(run_dir: Path) -> Path:
    events = sorted((run_dir / "summaries").glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not events:
        raise FileNotFoundError(f"No TensorBoard event files found under: {run_dir / 'summaries'}")
    return events[0]


def _print_tag_summary(ea: event_accumulator.EventAccumulator, tag: str, lookback: int) -> None:
    if tag not in ea.Tags().get("scalars", []):
        print(f"{tag}: NO_DATA")
        return

    values = ea.Scalars(tag)
    if not values:
        print(f"{tag}: NO_DATA")
        return

    last = values[-1]
    prev_index = max(0, len(values) - 1 - lookback)
    prev = values[prev_index]
    delta = last.value - prev.value
    print(f"{tag}: step={last.step} last={last.value:.6f} delta{lookback}={delta:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the latest training scalars for a JZ RL-Games run.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Root directory containing run folders.")
    parser.add_argument("--run", type=str, default=None, help="Specific run directory name. Defaults to the latest run.")
    parser.add_argument("--lookback", type=int, default=5, help="How many scalar points back to compare against.")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.log_root, args.run)
    event_file = _latest_event_file(run_dir)

    ea = event_accumulator.EventAccumulator(str(event_file), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    print(f"run: {run_dir.name}")
    print(f"event: {event_file}")
    print(f"event_last_write: {event_file.stat().st_mtime:.0f}")
    print(f"scalar_tags: {len(ea.Tags().get('scalars', []))}")
    for tag in DEFAULT_TAGS:
        _print_tag_summary(ea, tag, args.lookback)


if __name__ == "__main__":
    main()
