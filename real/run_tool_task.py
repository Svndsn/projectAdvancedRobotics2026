"""Approach-grasp-task DMP executor for UR5e with tool manipulation.

Primary runner for real hardware.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import spatialmath as sm

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from ctrl.dmp_cartesian.dmp_cartesian import DMPCartesian
from real.config import ROBOT_IP, T_TCP_GRASPED_TOOL_TCP, T_TCP_GRIPPER_TCP
from real.robot import Robot
from real.slip_correction import (
    compensate_tool_slip,
    load_correction_provider,
    validate_and_clip_correction,
)
from real.wsg50_gripper import WSG50CommandError, WSG50Gripper
from real.config import (
    WSG50_CLOSE_WIDTH_MM,
    WSG50_GRIP_FORCE_N,
    WSG50_GRIP_SPEED_MM_S,
    WSG50_GRIP_WIDTH_MM,
    WSG50_IP,
    WSG50_OPEN_WIDTH_MM,
    WSG50_PORT,
)
from utils.math import npq2np
from utils.sm import make_tf

_logger = logging.getLogger(__name__)


def _pose_row_to_se3(row: pd.Series, prefix: str) -> sm.SE3:
    position = np.array([row[f"{prefix}_{i}"] for i in range(3)], dtype=float)
    rotvec = np.array([row[f"{prefix}_{i}"] for i in range(3, 6)], dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle > 1e-12:
        rotation = sm.SO3.AngVec(angle, rotvec / angle).R
    else:
        rotation = np.eye(3)
    return sm.SE3.Rt(rotation, position, check=False)


def _unwrap_rotvec_sequence(rotvecs: np.ndarray) -> np.ndarray:
    """Make rotvec samples temporally continuous across ±pi branch flips.

    Equivalent axis-angle representations can jump by ~2*pi when crossing the
    principal rotvec branch. This picks the sign/2*pi-shifted representation
    closest to the previous sample.
    """
    if len(rotvecs) <= 1:
        return rotvecs.copy()

    out = rotvecs.astype(float, copy=True)
    two_pi = 2.0 * np.pi
    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]
        norm = float(np.linalg.norm(cur))
        if norm < 1e-12:
            continue
        axis = cur / norm
        candidates = np.vstack(
            [
                cur,
                -cur,
                axis * (norm + two_pi),
                axis * (norm - two_pi),
                -axis * (norm + two_pi),
                -axis * (norm - two_pi),
            ]
        )
        dists = np.linalg.norm(candidates - prev[None, :], axis=1)
        out[i] = candidates[int(np.argmin(dists))]
    return out


def _detect_pose_prefix(df: pd.DataFrame) -> str:
    if all(f"actual_TCP_pose_{i}" in df.columns for i in range(6)):
        return "actual_TCP_pose"
    if all(f"target_TCP_pose_{i}" in df.columns for i in range(6)):
        return "target_TCP_pose"
    raise ValueError(
        "Could not find pose columns. Expected actual_TCP_pose_[0..5] or target_TCP_pose_[0..5]."
    )


def _load_demo_data(demo_csv: str | Path) -> tuple[list[sm.SE3], Optional[np.ndarray], str]:
    df = pd.read_csv(demo_csv)
    prefix = _detect_pose_prefix(df)
    pos = df[[f"{prefix}_{i}" for i in range(3)]].to_numpy(dtype=float)
    rotvec = df[[f"{prefix}_{i}" for i in range(3, 6)]].to_numpy(dtype=float)
    rotvec = _unwrap_rotvec_sequence(rotvec)
    poses = []
    for p, r in zip(pos, rotvec):
        angle = float(np.linalg.norm(r))
        if angle > 1e-12:
            rotation = sm.SO3.AngVec(angle, r / angle).R
        else:
            rotation = np.eye(3)
        poses.append(sm.SE3.Rt(rotation, p, check=False))

    ts: Optional[np.ndarray] = None
    if "t" in df.columns and len(df) >= 2:
        raw_t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(raw_t).all() and np.all(np.diff(raw_t) > 0.0):
            ts = raw_t - raw_t[0]

    return poses, ts, prefix


def _parse_pose_str(s: str) -> sm.SE3:
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 6:
        raise ValueError("Pose must be 6 comma-separated floats: x,y,z,rx,ry,rz")
    position = np.array(parts[:3], dtype=float)
    rotvec = np.array(parts[3:], dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle > 1e-12:
        rotation = sm.SO3.AngVec(angle, rotvec / angle).R
    else:
        rotation = np.eye(3)
    return sm.SE3.Rt(rotation, position, check=False)


def _format_pose6(pose: sm.SE3) -> str:
    t = pose.t
    angle, axis = pose.angvec()
    rotvec = np.array(axis, dtype=float) * float(angle)
    return ",".join(f"{v:.6f}" for v in [t[0], t[1], t[2], rotvec[0], rotvec[1], rotvec[2]])


def _load_pose_from_csv(csv_path: str | Path, row_index: int) -> sm.SE3:
    df = pd.read_csv(csv_path)
    prefix = _detect_pose_prefix(df)
    row = df.iloc[row_index]
    return _pose_row_to_se3(row, prefix)


def _resolve_pose(pose_input: Optional[str], default: sm.SE3, row_index: int) -> sm.SE3:
    if pose_input is None:
        return default
    p = Path(pose_input)
    if p.exists():
        return _load_pose_from_csv(p, row_index=row_index)
    return _parse_pose_str(pose_input)


def rollout_dmp(
    dmp: DMPCartesian,
    tau_scale: float,
) -> list[sm.SE3]:
    if tau_scale <= 0:
        raise ValueError("tau_scale must be greater than 0")

    rollout_tau = dmp.tau * tau_scale
    n_steps = max(2, int(round(len(dmp.ts) * tau_scale)))
    rollout_ts = np.linspace(0.0, rollout_tau, n_steps)
    p, _, _, q, _, _ = dmp.rollout(rollout_ts, rollout_tau)
    return [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]

def _all_poses_valid(poses: list[sm.SE3]) -> bool:
    return all(_is_pose_valid(p) for p in poses)


def _pose_jump(a: sm.SE3, b: sm.SE3) -> tuple[float, float]:
    delta = a.inv() @ b
    translation = float(np.linalg.norm(delta.t))
    angle, _ = delta.angvec()
    return translation, abs(float(angle))


def _is_pose_valid(pose: sm.SE3) -> bool:
    return not (np.isnan(pose.A).any() or np.isinf(pose.A).any())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute approach-grasp-task DMPs on UR5e with gripper."
    )
    parser.add_argument("--approach-demo", required=True, help="Path to approach demo CSV file.")
    parser.add_argument("--task1-demo", required=True, help="Path to task1 demo CSV file.")
    parser.add_argument("--task2-demo", required=True, help="Path to task2 demo CSV file.")
    parser.add_argument("--robot-ip", default=ROBOT_IP, help=f"UR5e controller IP (default: {ROBOT_IP}).")
    parser.add_argument("--servo-rate", type=float, default=400.0, help="Servo loop frequency in Hz.")
    parser.add_argument("--approach-tau-scale", type=float, default=1.0)
    parser.add_argument("--task1-tau-scale", type=float, default=None)
    parser.add_argument("--task2-tau-scale", type=float, default=None)
    parser.add_argument("--task-tau-scale", type=float, default=1.0)
    parser.add_argument("--tau-scale", type=float, default=None, help="Legacy shared tau scale; overrides both when set.")
    parser.add_argument(
        "--approach-start-pose",
        type=str,
        default=None,
        help="Approach start pose: file path (first row) or CSV 'x,y,z,rx,ry,rz'.",
    )
    parser.add_argument(
        "--tool-pose",
        type=str,
        default=None,
        help="Tool contact pose: file path (first row) or CSV 'x,y,z,rx,ry,rz'.",
    )
    parser.add_argument(
        "--task-goal-pose",
        type=str,
        default=None,
        help="Task goal pose: file path (last row) or CSV 'x,y,z,rx,ry,rz'.",
    )
    parser.add_argument(
        "--grasped-tool-tcp",
        type=str,
        default=None,
        help="Flange->TCP transform to apply after gripping the tool: file path (first row) or CSV 'x,y,z,rx,ry,rz'.",
    )
    parser.add_argument(
        "--correction-provider",
        type=str,
        default=None,
        help="Optional slip correction provider in 'module:ClassName' format. Output is interpreted as TCP delta.",
    )
    parser.add_argument("--max-translation-correction-m", type=float, default=0.01)
    parser.add_argument("--max-rotation-correction-rad", type=float, default=0.1)
    parser.add_argument("--max-tcp-drift-translation-m", type=float, default=0.03)
    parser.add_argument("--max-tcp-drift-rotation-rad", type=float, default=0.35)
    parser.add_argument("--max-phase-jump-m", type=float, default=0.02)
    parser.add_argument("--max-phase-jump-rad", type=float, default=0.35)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--acc", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class ToolTaskExecutor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.servo_dt = 1.0 / args.servo_rate
        if args.tau_scale is not None:
            self.approach_tau_scale = args.tau_scale
            self.task1_tau_scale = args.tau_scale
            self.task2_tau_scale = args.tau_scale
        else:
            self.approach_tau_scale = args.approach_tau_scale
            shared_task_tau = args.task_tau_scale
            self.task1_tau_scale = (
                args.task1_tau_scale if args.task1_tau_scale is not None else shared_task_tau
            )
            self.task2_tau_scale = (
                args.task2_tau_scale if args.task2_tau_scale is not None else shared_task_tau
            )

        _logger.info(
            "Loading demos from %s, %s and %s",
            args.approach_demo,
            args.task1_demo,
            args.task2_demo,
        )
        self.approach_demo_poses, self.approach_ts, _ = _load_demo_data(args.approach_demo)
        self.task1_demo_poses, self.task1_ts, _ = _load_demo_data(args.task1_demo)
        self.task2_demo_poses, self.task2_ts, _ = _load_demo_data(args.task2_demo)

        self.approach_dmp = DMPCartesian(roto_dilatation=True)
        self.task1_dmp = DMPCartesian(roto_dilatation=True)
        self.task2_dmp = DMPCartesian(roto_dilatation=True)
        self._approach_loaded_with_time = False
        self._task1_loaded_with_time = False
        self._task2_loaded_with_time = False
        self._train_dmp(self.approach_dmp, self.approach_demo_poses, self.approach_ts, "approach")
        self._train_dmp(self.task1_dmp, self.task1_demo_poses, self.task1_ts, "task1")
        self._train_dmp(self.task2_dmp, self.task2_demo_poses, self.task2_ts, "task2")

        self.approach_start_pose: sm.SE3 = self.approach_demo_poses[0]
        self.tool_contact_pose: sm.SE3 = self.approach_demo_poses[-1]
        self.task_mid_pose: sm.SE3 = self.task1_demo_poses[-1]
        self.task_goal_pose: sm.SE3 = self.task2_demo_poses[-1]

        self.approach_poses: list[sm.SE3] = []
        self.task1_poses: list[sm.SE3] = []
        self.task2_poses: list[sm.SE3] = []
        self.pregrasp_tcp: sm.SE3 = T_TCP_GRIPPER_TCP
        self.grasped_tool_tcp: sm.SE3 = _resolve_pose(
            args.grasped_tool_tcp, T_TCP_GRASPED_TOOL_TCP, row_index=0
        )
        self.active_tcp: sm.SE3 = self.pregrasp_tcp
        self.correction_provider = load_correction_provider(args.correction_provider)
        self._t0 = time.monotonic()
        self.robot: Optional[Robot] = None
        self.gripper: Optional[WSG50Gripper] = None

    def _train_dmp(
        self,
        dmp: DMPCartesian,
        poses: list[sm.SE3],
        ts: Optional[np.ndarray],
        name: str,
    ) -> None:
        if ts is not None:
            dmp.load_with_time(poses, ts)
            if name == "approach":
                self._approach_loaded_with_time = True
            elif name == "task1":
                self._task1_loaded_with_time = True
            elif name == "task2":
                self._task2_loaded_with_time = True
            duration = float(ts[-1])
            dt_mean = float(np.mean(np.diff(ts)))
            hz = (1.0 / dt_mean) if dt_mean > 0 else float("nan")
            _logger.info(
                "Loaded %s DMP with timestamps: samples=%d duration=%.3fs mean_dt=%.4fs (~%.1fHz)",
                name,
                len(poses),
                duration,
                dt_mean,
                hz,
            )
            return

        dmp.load(poses)
        if name == "approach":
            self._approach_loaded_with_time = False
        elif name == "task1":
            self._task1_loaded_with_time = False
        elif name == "task2":
            self._task2_loaded_with_time = False
        _logger.info(
            "Loaded %s DMP without timestamps: samples=%d duration=%.3fs (fallback fixed dt)",
            name,
            len(poses),
            dmp.tau,
        )

    def _setup_rollouts(self) -> None:
        self.approach_start_pose = _resolve_pose(
            self.args.approach_start_pose,
            self.approach_demo_poses[0],
            row_index=0,
        )
        self.tool_contact_pose = _resolve_pose(
            self.args.tool_pose,
            self.approach_demo_poses[-1],
            row_index=0,
        )
        self.task_goal_pose = _resolve_pose(
            self.args.task_goal_pose,
            self.task2_demo_poses[-1],
            row_index=-1,
        )

        self.approach_dmp.set_start_goal(self.approach_start_pose, self.tool_contact_pose)
        self.task1_dmp.set_start_goal(self.tool_contact_pose, self.task_mid_pose)
        self.task2_dmp.set_start_goal(self.task_mid_pose, self.task_goal_pose)

        self.approach_poses = rollout_dmp(self.approach_dmp, self.approach_tau_scale)
        self.task1_poses = rollout_dmp(self.task1_dmp, self.task1_tau_scale)
        self.task2_poses = rollout_dmp(self.task2_dmp, self.task2_tau_scale)
        self._fallback_to_fixed_dt_if_invalid()

        _logger.info(
            "Rollout durations: approach=%.3fs task1=%.3fs task2=%.3fs",
            self.approach_dmp.tau * self.approach_tau_scale,
            self.task1_dmp.tau * self.task1_tau_scale,
            self.task2_dmp.tau * self.task2_tau_scale,
        )

    def _fallback_to_fixed_dt_if_invalid(self) -> None:
        """Retry with fixed-dt DMP training when timestamp rollouts become invalid."""
        approach_invalid = not _all_poses_valid(self.approach_poses)
        task1_invalid = not _all_poses_valid(self.task1_poses)
        task2_invalid = not _all_poses_valid(self.task2_poses)
        if not approach_invalid and not task1_invalid and not task2_invalid:
            return

        if approach_invalid and self._approach_loaded_with_time:
            _logger.warning(
                "Approach rollout invalid after timestamp training; retrying with fixed-dt DMP load()."
            )
            self.approach_dmp.load(self.approach_demo_poses)
            self._approach_loaded_with_time = False

        if task1_invalid and self._task1_loaded_with_time:
            _logger.warning(
                "Task1 rollout invalid after timestamp training; retrying with fixed-dt DMP load()."
            )
            self.task1_dmp.load(self.task1_demo_poses)
            self._task1_loaded_with_time = False
        if task2_invalid and self._task2_loaded_with_time:
            _logger.warning(
                "Task2 rollout invalid after timestamp training; retrying with fixed-dt DMP load()."
            )
            self.task2_dmp.load(self.task2_demo_poses)
            self._task2_loaded_with_time = False

        self.approach_dmp.set_start_goal(self.approach_start_pose, self.tool_contact_pose)
        self.task1_dmp.set_start_goal(self.tool_contact_pose, self.task_mid_pose)
        self.task2_dmp.set_start_goal(self.task_mid_pose, self.task_goal_pose)
        self.approach_poses = rollout_dmp(self.approach_dmp, self.approach_tau_scale)
        self.task1_poses = rollout_dmp(self.task1_dmp, self.task1_tau_scale)
        self.task2_poses = rollout_dmp(self.task2_dmp, self.task2_tau_scale)

        # Final stability guard: if DMP rollout is still invalid, use raw demos.
        if not _all_poses_valid(self.approach_poses):
            _logger.warning(
                "Approach rollout still invalid after fixed-dt fallback; using raw approach demo poses."
            )
            self.approach_poses = list(self.approach_demo_poses)
        if not _all_poses_valid(self.task1_poses):
            _logger.warning(
                "Task1 rollout still invalid after fixed-dt fallback; using raw task1 demo poses."
            )
            self.task1_poses = list(self.task1_demo_poses)
        if not _all_poses_valid(self.task2_poses):
            _logger.warning(
                "Task2 rollout still invalid after fixed-dt fallback; using raw task2 demo poses."
            )
            self.task2_poses = list(self.task2_demo_poses)

    def _check_phase_continuity(self) -> bool:
        if not self.approach_poses or not self.task1_poses or not self.task2_poses:
            return False

        checks = [
            ("approach_end -> retreat_start", self.approach_poses[-1], self.approach_poses[-1]),
            ("tool_contact -> task1_start", self.tool_contact_pose, self.task1_poses[0]),
            ("task1_end -> task2_start", self.task1_poses[-1], self.task2_poses[0]),
        ]
        ok = True
        for name, a, b in checks:
            d_m, d_rad = _pose_jump(a, b)
            _logger.info("Continuity %s: d=%.4fm, angle=%.4frad", name, d_m, d_rad)
            if d_m > self.args.max_phase_jump_m or d_rad > self.args.max_phase_jump_rad:
                _logger.error(
                    "Continuity violation for %s (limits %.4fm / %.4frad)",
                    name,
                    self.args.max_phase_jump_m,
                    self.args.max_phase_jump_rad,
                )
                ok = False
        return ok

    def _safety_check(self) -> bool:
        all_valid = True
        for i, pose in enumerate(self.approach_poses):
            if not _is_pose_valid(pose):
                _logger.error("Invalid approach pose at index %d", i)
                all_valid = False
        for i, pose in enumerate(self.task1_poses):
            if not _is_pose_valid(pose):
                _logger.error("Invalid task1 pose at index %d", i)
                all_valid = False
        for i, pose in enumerate(self.task2_poses):
            if not _is_pose_valid(pose):
                _logger.error("Invalid task2 pose at index %d", i)
                all_valid = False
        return all_valid

    def _print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("TOOL TASK EXECUTION SUMMARY")
        print("=" * 70)
        print("\nConfiguration:")
        print(f"  Approach demo:      {self.args.approach_demo}")
        print(f"  Task1 demo:         {self.args.task1_demo}")
        print(f"  Task2 demo:         {self.args.task2_demo}")
        print(f"  Servo rate:         {self.args.servo_rate:.1f} Hz")
        print(f"  Approach tau scale: {self.approach_tau_scale:.2f}x")
        print(f"  Task1 tau scale:    {self.task1_tau_scale:.2f}x")
        print(f"  Task2 tau scale:    {self.task2_tau_scale:.2f}x")
        print(f"  Dry-run mode:       {self.args.dry_run}")
        print(f"  Pre-grasp TCP:      {self.pregrasp_tcp}")
        print(f"  Grasped tool TCP:   {self.grasped_tool_tcp}")

        print("\nCalibration Checklist:")
        print("  1) Measure flange->tool TCP and set T_TCP_GRIPPER_TCP in real/config.py")
        print("  2) Use same TCP for recording (real/record_demo.py) and execution")
        print("  3) Verify UR Remote Control mode + payload/TCP setup on robot")

        print("\nApproach Phase:")
        print(f"  Demo samples:       {len(self.approach_dmp.ts)}")
        print(f"  Demo duration:      {self.approach_dmp.tau:.3f} s")
        print(f"  Rollout poses:      {len(self.approach_poses)}")
        print(f"  Rollout duration:   {self.approach_dmp.tau * self.approach_tau_scale:.3f} s")

        print("\nTask1 Phase:")
        print(f"  Demo samples:       {len(self.task1_dmp.ts)}")
        print(f"  Demo duration:      {self.task1_dmp.tau:.3f} s")
        print(f"  Rollout poses:      {len(self.task1_poses)}")
        print(f"  Rollout duration:   {self.task1_dmp.tau * self.task1_tau_scale:.3f} s")

        print("\nTask2 Phase:")
        print(f"  Demo samples:       {len(self.task2_dmp.ts)}")
        print(f"  Demo duration:      {self.task2_dmp.tau:.3f} s")
        print(f"  Rollout poses:      {len(self.task2_poses)}")
        print(f"  Rollout duration:   {self.task2_dmp.tau * self.task2_tau_scale:.3f} s")
        print("\n" + "=" * 70 + "\n")

    def execute_dry_run(self) -> None:
        self._setup_rollouts()
        if not self._safety_check():
            raise RuntimeError("Dry-run failed safety check")
        if not self._check_phase_continuity():
            raise RuntimeError("Dry-run failed continuity checks")
        self._print_summary()

    def _connect_robot(self) -> None:
        _logger.info("Connecting to robot at %s", self.args.robot_ip)
        self.robot = Robot(robot_ip=self.args.robot_ip, T_tcp_gripper_tcp=self.pregrasp_tcp)
        self.active_tcp = self.pregrasp_tcp
        self.gripper = WSG50Gripper(
            host=WSG50_IP,
            port=WSG50_PORT,
            default_speed_mm_s=WSG50_GRIP_SPEED_MM_S,
        )
        self.gripper.connect()
        _logger.info("Connected to robot and WSG50 (%s:%d)", WSG50_IP, WSG50_PORT)

    def _disconnect_robot(self) -> None:
        if self.robot is not None:
            try:
                self.robot.servo_stop()
                self.robot.stop_script()
            except Exception as exc:
                _logger.warning("Error stopping robot: %s", exc)

        if self.gripper is not None:
            try:
                self.gripper.disconnect()
            except Exception as exc:
                _logger.warning("Error disconnecting gripper: %s", exc)

    def _move_to_pose(self, pose: sm.SE3) -> None:
        if self.robot is None:
            return
        self.robot.moveL(pose, speed=self.args.speed, acc=self.args.acc)

    def _finish_servo_segment(self) -> None:
        if self.robot is None:
            return
        self.robot.servo_stop()
        time.sleep(0.05)

    def _update_active_tcp(self, new_tcp: sm.SE3, reason: str) -> None:
        if self.robot is None:
            return
        self.robot.set_tool_tcp(new_tcp)
        self.active_tcp = new_tcp
        _logger.info("Active TCP updated (%s): %s", reason, _format_pose6(new_tcp))

    def _reanchor_task_rollouts_from_current_pose(self) -> None:
        if self.robot is None:
            return
        current_pose = self.robot.T_base_tcp
        self.task1_dmp.set_start_goal(current_pose, self.task_mid_pose)
        self.task2_dmp.set_start_goal(self.task_mid_pose, self.task_goal_pose)
        self.task1_poses = rollout_dmp(self.task1_dmp, self.task1_tau_scale)
        self.task2_poses = rollout_dmp(self.task2_dmp, self.task2_tau_scale)
        if not _all_poses_valid(self.task1_poses):
            _logger.warning("Task1 rollout invalid after re-anchor; using raw task1 demo poses.")
            self.task1_poses = list(self.task1_demo_poses)
        if not _all_poses_valid(self.task2_poses):
            _logger.warning("Task2 rollout invalid after re-anchor; using raw task2 demo poses.")
            self.task2_poses = list(self.task2_demo_poses)

    def _compute_slip_corrected_pose(self, target_pose: sm.SE3, phase: str) -> sm.SE3:
        if self.robot is None:
            return target_pose
        now = time.monotonic() - self._t0
        actual_pose = self.robot.T_base_tcp
        raw_delta = self.correction_provider.get_correction(
            actual_pose=actual_pose,
            target_pose=target_pose,
            phase=phase,
            t=now,
        )
        delta = validate_and_clip_correction(
            raw_delta,
            self.args.max_translation_correction_m,
            self.args.max_rotation_correction_rad,
        )
        return compensate_tool_slip(target_pose, delta)

    def _servo_phase(self, phase: str, poses: list[sm.SE3], apply_tcp_correction: bool = False) -> None:
        assert self.robot is not None
        _logger.info("Executing %s (%d waypoints)", phase, len(poses))
        progress_every = max(1, len(poses) // 10)
        for i, pose in enumerate(poses):
            command_pose = pose
            if apply_tcp_correction:
                command_pose = self._compute_slip_corrected_pose(pose, phase)
            self.robot.servoL(command_pose, speed=self.args.speed, acc=self.args.acc, dt=self.servo_dt)
            if (i + 1) % progress_every == 0 or i + 1 == len(poses):
                _logger.info("  %s progress: %d/%d", phase, i + 1, len(poses))
            if self.servo_dt > 0:
                time.sleep(self.servo_dt)

    def _open_gripper(self) -> None:
        if self.gripper is None:
            return
        _logger.info("WSG50 OPEN width=%.2fmm speed=%.2fmm/s", WSG50_OPEN_WIDTH_MM, WSG50_GRIP_SPEED_MM_S)
        lines = self.gripper.open(WSG50_GRIP_SPEED_MM_S)
        _logger.info("WSG50 OPEN response: %s", " | ".join(lines))

    def _grip_object(self) -> None:
        if self.gripper is None:
            return
        try:
            _logger.info(
                "WSG50 GRIP force=%.2fN width=%.2fmm speed=%.2fmm/s",
                WSG50_GRIP_FORCE_N,
                WSG50_GRIP_WIDTH_MM,
                WSG50_GRIP_SPEED_MM_S,
            )
            lines = self.gripper.grip(
                force_n=WSG50_GRIP_FORCE_N,
                width_mm=WSG50_GRIP_WIDTH_MM,
                speed_mm_s=WSG50_GRIP_SPEED_MM_S,
            )
            _logger.info("WSG50 GRIP response: %s", " | ".join(lines))
        except WSG50CommandError as exc:
            _logger.warning("WSG50 GRIP failed: %s", exc)
            _logger.warning(
                "WSG50 fallback MOVE width=%.2fmm speed=%.2fmm/s",
                WSG50_CLOSE_WIDTH_MM,
                WSG50_GRIP_SPEED_MM_S,
            )
            lines = self.gripper.move(WSG50_CLOSE_WIDTH_MM, WSG50_GRIP_SPEED_MM_S)
            _logger.info("WSG50 fallback MOVE response: %s", " | ".join(lines))

    def _switch_to_grasped_tcp(self) -> None:
        if self.robot is None:
            return
        self._update_active_tcp(self.grasped_tool_tcp, reason="tool_grasped")
        self._reanchor_task_rollouts_from_current_pose()

    def execute(self) -> None:
        try:
            self.correction_provider.reset()
            self._t0 = time.monotonic()
            self._setup_rollouts()
            if not self._safety_check():
                raise RuntimeError("Safety check failed")
            if not self._check_phase_continuity():
                raise RuntimeError("Continuity checks failed")
            self._print_summary()

            if self.args.dry_run:
                _logger.info("Dry-run completed")
                return

            self._connect_robot()
            self._open_gripper()

            self._move_to_pose(self.approach_poses[0])
            self._servo_phase("approach", self.approach_poses)
            self._finish_servo_segment()

            self._grip_object()
            self._switch_to_grasped_tcp()

            self._move_to_pose(self.task1_poses[0])
            self._servo_phase("task1", self.task1_poses, apply_tcp_correction=True)
            self._finish_servo_segment()

            self._move_to_pose(self.task2_poses[0])
            self._servo_phase("task2", self.task2_poses, apply_tcp_correction=True)
            self._finish_servo_segment()

            if self.robot is not None:
                self.robot.stop_script()
            _logger.info("Execution completed successfully")

        except KeyboardInterrupt:
            _logger.warning("Interrupted by user")
            raise
        except Exception as exc:
            _logger.error("Execution failed: %s", exc, exc_info=True)
            raise
        finally:
            self._disconnect_robot()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = _parse_args()
    executor = ToolTaskExecutor(args)
    executor.execute()


if __name__ == "__main__":
    main()
