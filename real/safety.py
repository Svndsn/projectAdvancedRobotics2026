from __future__ import annotations

from pathlib import Path

import numpy as np
import spatialmath as sm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MAX_CARTESIAN_STEP_M = 0.02
MAX_START_DISTANCE_M = 0.10


def _as_pose_list(poses: list[sm.SE3] | tuple[sm.SE3, ...]) -> list[sm.SE3]:
    pose_list = list(poses)
    if not pose_list:
        raise ValueError("poses must not be empty")
    for pose in pose_list:
        if not isinstance(pose, sm.SE3):
            raise TypeError(f"Expected spatialmath.SE3 poses, got {type(pose)!r}")
    return pose_list


def check_workspace_path(path: str | Path, workspace_root: Path = WORKSPACE_ROOT) -> Path:
    resolved = Path(path).expanduser().resolve()
    workspace_root = workspace_root.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    if workspace_root not in resolved.parents and resolved != workspace_root:
        raise ValueError(f"Path is outside the workspace root: {resolved}")
    return resolved


def check_cartesian_steps(
    poses: list[sm.SE3] | tuple[sm.SE3, ...],
    max_cartesian_step_m: float = MAX_CARTESIAN_STEP_M,
) -> None:
    pose_list = _as_pose_list(poses)
    if max_cartesian_step_m <= 0:
        raise ValueError("max_cartesian_step_m must be greater than 0")

    for idx in range(1, len(pose_list)):
        step_m = float(np.linalg.norm(pose_list[idx].t - pose_list[idx - 1].t))
        if step_m > max_cartesian_step_m:
            raise ValueError(
                f"Cartesian step at index {idx} is {step_m:.6f} m, "
                f"which exceeds the limit of {max_cartesian_step_m:.6f} m."
            )


def check_start_distance(
    poses: list[sm.SE3] | tuple[sm.SE3, ...],
    start_pose: sm.SE3,
    max_start_distance_m: float = MAX_START_DISTANCE_M,
) -> None:
    pose_list = _as_pose_list(poses)
    if not isinstance(start_pose, sm.SE3):
        raise TypeError(f"Expected start_pose to be spatialmath.SE3, got {type(start_pose)!r}")
    if max_start_distance_m <= 0:
        raise ValueError("max_start_distance_m must be greater than 0")

    start_distance_m = float(np.linalg.norm(pose_list[0].t - start_pose.t))
    if start_distance_m > max_start_distance_m:
        raise ValueError(
            f"Start pose distance is {start_distance_m:.6f} m, "
            f"which exceeds the limit of {max_start_distance_m:.6f} m."
        )


__all__ = [
    "WORKSPACE_ROOT",
    "MAX_CARTESIAN_STEP_M",
    "MAX_START_DISTANCE_M",
    "check_workspace_path",
    "check_cartesian_steps",
    "check_start_distance",
]