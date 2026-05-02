from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import spatialmath as sm

from ctrl.dmp_cartesian.dmp_cartesian import DMPCartesian
from real.robot import Robot
from utils.math import npq2np
from utils.sm import make_tf


def _pose_row_to_se3(row: pd.Series, prefix: str) -> sm.SE3:
    position = np.array([row[f"{prefix}_{i}"] for i in range(3)], dtype=float)
    rotvec = np.array([row[f"{prefix}_{i}"] for i in range(3, 6)], dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle > 1e-12:
        rotation = sm.SO3.AngVec(angle, rotvec / angle).R
    else:
        rotation = np.eye(3)
    return sm.SE3.Rt(rotation, position, check=False)


def _detect_pose_prefix(df: pd.DataFrame) -> str:
    if all(f"actual_TCP_pose_{i}" in df.columns for i in range(6)):
        return "actual_TCP_pose"
    if all(f"target_TCP_pose_{i}" in df.columns for i in range(6)):
        return "target_TCP_pose"
    raise ValueError(
        "Could not find pose columns. Expected actual_TCP_pose_[0..5] or target_TCP_pose_[0..5]."
    )


@dataclass(slots=True)
class DMPPlan:
    source_path: Path
    dmp: DMPCartesian
    demo_poses: list[sm.SE3]
    prefix: str
    rollout_poses: list[sm.SE3] | None = None


def train_dmp_from_csv(path: str | Path) -> DMPPlan:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    prefix = _detect_pose_prefix(df)
    demo_poses = [_pose_row_to_se3(row, prefix) for _, row in df.iterrows()]
    if len(demo_poses) < 2:
        raise ValueError("CSV must contain at least two pose samples to train a DMP.")

    dmp = DMPCartesian(roto_dilatation=True)
    dmp.load(demo_poses)
    return DMPPlan(
        source_path=csv_path,
        dmp=dmp,
        demo_poses=demo_poses,
        prefix=prefix,
    )


def rollout_dmp(plan: DMPPlan, servo_dt: float, tau_scale: float) -> list[sm.SE3]:
    if servo_dt <= 0:
        raise ValueError("servo_dt must be greater than 0")
    if tau_scale <= 0:
        raise ValueError("tau_scale must be greater than 0")

    rollout_tau = plan.dmp.tau * tau_scale
    n_steps = max(2, int(np.ceil(rollout_tau / servo_dt)) + 1)
    rollout_ts = np.linspace(0.0, rollout_tau, n_steps)
    p, _, _, q, _, _ = plan.dmp.rollout(rollout_ts, rollout_tau)
    poses = [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]
    plan.rollout_poses = poses
    return poses


def execute_poses_servoL(
    robot: Robot,
    poses: list[sm.SE3],
    dt: float,
    speed: float,
    acc: float,
    lookahead_time: float,
    gain: int,
) -> None:
    try:
        for pose in poses:
            robot.servoL(
                pose,
                speed=speed,
                acc=acc,
                dt=dt,
                lookahead_time=lookahead_time,
                gain=gain,
            )
            if dt > 0:
                time.sleep(dt)
    finally:
        robot.servo_stop()


__all__ = [
    "DMPPlan",
    "execute_poses_servoL",
    "rollout_dmp",
    "train_dmp_from_csv",
]