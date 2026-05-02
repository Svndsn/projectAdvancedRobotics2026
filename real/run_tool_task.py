"""Approach-grasp-task DMP executor for UR5e with tool manipulation.

This module provides a task executor that combines approach and task DMPs
with gripper control. It supports both dry-run (validation) and execution modes.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import spatialmath as sm

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from ctrl.dmp_cartesian.dmp_cartesian import DMPCartesian
from real.config import ROBOT_IP, T_TCP_GRIPPER_TCP
from real.robot import Robot
from real.wsg50_gripper import WSG50Gripper, WSG50CommandError
from real.config import (
    WSG50_IP,
    WSG50_PORT,
    WSG50_GRIP_SPEED_MM_S,
    WSG50_GRIP_FORCE_N,
    WSG50_GRIP_WIDTH_MM,
    WSG50_CLOSE_WIDTH_MM,
    WSG50_OPEN_WIDTH_MM,
)
from real.utils import se3_to_pose
from utils.math import npq2np
from utils.sm import make_tf

_logger = logging.getLogger(__name__)


def _pose_row_to_se3(row: pd.Series, prefix: str) -> sm.SE3:
    """Convert a DataFrame row to an SE3 pose."""
    position = np.array([row[f"{prefix}_{i}"] for i in range(3)], dtype=float)
    rotvec = np.array([row[f"{prefix}_{i}"] for i in range(3, 6)], dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle > 1e-12:
        R = sm.SO3.AngVec(angle, rotvec / angle).R
    else:
        R = np.eye(3)
    return sm.SE3.Rt(R, position, check=False)


def _detect_pose_prefix(df: pd.DataFrame) -> str:
    """Detect which pose column prefix is used in the CSV."""
    if all(f"actual_TCP_pose_{i}" in df.columns for i in range(6)):
        return "actual_TCP_pose"
    if all(f"target_TCP_pose_{i}" in df.columns for i in range(6)):
        return "target_TCP_pose"
    raise ValueError(
        "Could not find pose columns. Expected actual_TCP_pose_[0..5] or target_TCP_pose_[0..5]."
    )


def load_demo_poses(demo_csv: str | Path) -> list[sm.SE3]:
    """Load demo poses from a CSV file."""
    df = pd.read_csv(demo_csv)
    prefix = _detect_pose_prefix(df)
    return [_pose_row_to_se3(row, prefix) for _, row in df.iterrows()]


def rollout_dmp(
    dmp: DMPCartesian,
    tau_scale: float,
    start_pose: Optional[sm.SE3] = None,
    end_pose: Optional[sm.SE3] = None,
) -> list[sm.SE3]:
    """Roll out a trained DMP with given start and end poses.
    
    Parameters
    ----------
    dmp : DMPCartesian
        Trained DMP instance.
    tau_scale : float
        Time scaling factor (> 0).
    start_pose : sm.SE3, optional
        New start pose. If None, uses DMP's current start.
    end_pose : sm.SE3, optional
        New end pose. If None, uses DMP's current end.
    
    Returns
    -------
    list[sm.SE3]
        List of SE3 poses along the rollout trajectory.
    """
    if tau_scale <= 0:
        raise ValueError("tau_scale must be greater than 0")

    if start_pose is not None and end_pose is not None:
        dmp.set_start_goal(start_pose, end_pose)

    rollout_tau = dmp.tau * tau_scale
    n_steps = max(2, int(round(len(dmp.ts) * tau_scale)))
    rollout_ts = np.linspace(0.0, rollout_tau, n_steps)
    p, _, _, q, _, _ = dmp.rollout(rollout_ts, rollout_tau)
    return [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]


def _is_pose_valid(pose: sm.SE3) -> bool:
    """Check if a pose contains valid (non-NaN) values."""
    return not (np.isnan(pose.A).any() or np.isinf(pose.A).any())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute approach-grasp-task DMPs on UR5e with gripper."
    )
    parser.add_argument(
        "--approach-demo",
        required=True,
        help="Path to approach demo CSV file.",
    )
    parser.add_argument(
        "--task-demo",
        required=True,
        help="Path to task demo CSV file.",
    )
    parser.add_argument(
        "--robot-ip",
        default=ROBOT_IP,
        help=f"UR5e controller IP address (default: {ROBOT_IP}).",
    )
    parser.add_argument(
        "--servo-rate",
        type=float,
        default=400.0,
        help="Servo loop frequency in Hz (default: 400).",
    )
    parser.add_argument(
        "--tau-scale",
        type=float,
        default=1.0,
        help="Time scaling factor for DMPs (default: 1.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print summary without connecting to robot.",
    )
    return parser.parse_args()


class ToolTaskExecutor:
    """Executor for approach-grasp-task with tool manipulation."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the executor and load/train DMPs.
        
        Parameters
        ----------
        args : argparse.Namespace
            Parsed command-line arguments.
        """
        self.args = args
        self.servo_dt = 1.0 / args.servo_rate

        # Load demo poses from CSV files
        _logger.info("Loading demo poses from %s and %s", args.approach_demo, args.task_demo)
        approach_demo_poses = load_demo_poses(args.approach_demo)
        task_demo_poses = load_demo_poses(args.task_demo)

        _logger.info(
            "Loaded %d approach poses and %d task poses.",
            len(approach_demo_poses),
            len(task_demo_poses),
        )

        # Train DMPs from demo poses
        self.approach_dmp = DMPCartesian(roto_dilatation=True)
        self.approach_dmp.load(approach_demo_poses)

        self.task_dmp = DMPCartesian(roto_dilatation=True)
        self.task_dmp.load(task_demo_poses)

        _logger.info(
            "Trained DMPs: approach tau=%.3f s, task tau=%.3f s",
            self.approach_dmp.tau,
            self.task_dmp.tau,
        )

        # Rollout poses (will be set up after loading)
        self.approach_poses: list[sm.SE3] = []
        self.task_poses: list[sm.SE3] = []

        # Hardware references
        self.robot: Optional[Robot] = None
        self.gripper: Optional[WSG50Gripper] = None

        self._t0 = time.monotonic()

    def _setup_rollouts(self) -> None:
        """Generate rollout trajectories from trained DMPs."""
        _logger.info("Rolling out approach DMP with tau_scale=%.2f", self.args.tau_scale)
        self.approach_poses = rollout_dmp(
            self.approach_dmp,
            self.args.tau_scale,
        )

        _logger.info("Rolling out task DMP with tau_scale=%.2f", self.args.tau_scale)
        self.task_poses = rollout_dmp(
            self.task_dmp,
            self.args.tau_scale,
        )

    def _safety_check(self) -> bool:
        """Validate that all rollout poses are valid.
        
        Returns
        -------
        bool
            True if all poses are valid, False otherwise.
        """
        _logger.info("Performing safety check on %d + %d poses...", 
                     len(self.approach_poses), len(self.task_poses))
        
        all_valid = True
        for i, pose in enumerate(self.approach_poses):
            if not _is_pose_valid(pose):
                _logger.error("Invalid approach pose at index %d: %s", i, pose)
                all_valid = False

        for i, pose in enumerate(self.task_poses):
            if not _is_pose_valid(pose):
                _logger.error("Invalid task pose at index %d: %s", i, pose)
                all_valid = False

        if all_valid:
            _logger.info("✓ All poses are valid (no NaN/Inf values).")
        else:
            _logger.error("✗ Safety check failed: invalid poses found.")

        return all_valid

    def _print_summary(self) -> None:
        """Print a summary of the task configuration."""
        print("\n" + "=" * 70)
        print("TOOL TASK EXECUTION SUMMARY")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Approach demo:      {self.args.approach_demo}")
        print(f"  Task demo:          {self.args.task_demo}")
        print(f"  Servo rate:         {self.args.servo_rate:.1f} Hz")
        print(f"  DMP time scale:     {self.args.tau_scale:.2f}x")
        print(f"  Dry-run mode:       {self.args.dry_run}")

        print(f"\nApproach Phase:")
        print(f"  Demo poses:         {len(self.approach_dmp.ts)} samples")
        print(f"  Demo duration:      {self.approach_dmp.tau:.3f} s")
        print(f"  Rollout poses:      {len(self.approach_poses)} waypoints")
        print(f"  Rollout duration:   {self.approach_dmp.tau * self.args.tau_scale:.3f} s")
        if self.approach_poses:
            start_pose = self.approach_poses[0]
            end_pose = self.approach_poses[-1]
            print(f"  Start position:     [{start_pose.t[0]:.4f}, {start_pose.t[1]:.4f}, {start_pose.t[2]:.4f}]")
            print(f"  End position:       [{end_pose.t[0]:.4f}, {end_pose.t[1]:.4f}, {end_pose.t[2]:.4f}]")

        print(f"\nTask Phase:")
        print(f"  Demo poses:         {len(self.task_dmp.ts)} samples")
        print(f"  Demo duration:      {self.task_dmp.tau:.3f} s")
        print(f"  Rollout poses:      {len(self.task_poses)} waypoints")
        print(f"  Rollout duration:   {self.task_dmp.tau * self.args.tau_scale:.3f} s")
        if self.task_poses:
            start_pose = self.task_poses[0]
            end_pose = self.task_poses[-1]
            print(f"  Start position:     [{start_pose.t[0]:.4f}, {start_pose.t[1]:.4f}, {start_pose.t[2]:.4f}]")
            print(f"  End position:       [{end_pose.t[0]:.4f}, {end_pose.t[1]:.4f}, {end_pose.t[2]:.4f}]")

        print("\n" + "=" * 70 + "\n")

    def execute_dry_run(self) -> None:
        """Execute a dry-run: load, train, rollout, validate, print summary."""
        _logger.info("Starting dry-run execution...")
        self._setup_rollouts()

        if not self._safety_check():
            _logger.error("Dry-run failed safety check. Aborting.")
            return

        self._print_summary()
        _logger.info("Dry-run completed successfully.")

    def _connect_robot(self) -> None:
        """Connect to the robot and gripper."""
        _logger.info("Connecting to robot at %s...", self.args.robot_ip)
        self.robot = Robot(
            robot_ip=self.args.robot_ip,
            T_tcp_gripper_tcp=T_TCP_GRIPPER_TCP,
        )
        _logger.info("Connected to robot.")

        _logger.info("Connecting to WSG50 gripper...")
        self.gripper = WSG50Gripper(
            host=WSG50_IP,
            port=WSG50_PORT,
            default_speed_mm_s=WSG50_GRIP_SPEED_MM_S,
        )
        self.gripper.connect()
        _logger.info("Connected to WSG50 gripper at %s:%d.", WSG50_IP, WSG50_PORT)

    def _disconnect_robot(self) -> None:
        """Safely disconnect from robot and gripper."""
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

    def _move_to_pose(self, pose: sm.SE3, speed: float = 0.25, acc: float = 0.3) -> None:
        """Move to a pose using linear motion.
        
        Parameters
        ----------
        pose : sm.SE3
            Target pose.
        speed : float
            Linear speed in m/s.
        acc : float
            Linear acceleration in m/s².
        """
        if self.robot is None:
            return
        self.robot.moveL(pose, speed=speed, acc=acc)

    def _servo_to_pose(self, pose: sm.SE3) -> None:
        """Servo to a pose with the configured servo rate.
        
        Parameters
        ----------
        pose : sm.SE3
            Target pose.
        """
        if self.robot is None:
            return
        self.robot.servoL(
            pose,
            speed=0.25,
            acc=0.3,
            dt=self.servo_dt,
        )

    def _open_gripper(self) -> None:
        """Open the gripper to maximum width."""
        if self.gripper is None:
            return
        try:
            _logger.info("Opening gripper...")
            self.gripper.open(WSG50_OPEN_WIDTH_MM)
            _logger.info("Gripper opened.")
        except WSG50CommandError as exc:
            _logger.error("Failed to open gripper: %s", exc)
            raise

    def _grip_object(self) -> None:
        """Grip the object with configured force."""
        if self.gripper is None:
            return
        try:
            _logger.info("Gripping object with force %.1f N...", WSG50_GRIP_FORCE_N)
            self.gripper.grip(
                force_n=WSG50_GRIP_FORCE_N,
                width_mm=WSG50_GRIP_WIDTH_MM,
                speed_mm_s=WSG50_GRIP_SPEED_MM_S,
            )
            _logger.info("Object gripped.")
        except WSG50CommandError as exc:
            _logger.warning("Grip command failed: %s; attempting close fallback.", exc)
            try:
                self.gripper.move(
                    WSG50_CLOSE_WIDTH_MM,
                    WSG50_GRIP_SPEED_MM_S,
                )
            except WSG50CommandError as exc2:
                _logger.error("Gripper close fallback also failed: %s", exc2)
                raise

    def execute(self) -> None:
        """Execute the full approach-grasp-task sequence."""
        try:
            if self.args.dry_run:
                self.execute_dry_run()
            else:
                self._connect_robot()
                self._setup_rollouts()

                if not self._safety_check():
                    raise RuntimeError("Safety check failed.")

                self._print_summary()

                _logger.info("Opening gripper...")
                self._open_gripper()

                _logger.info("Moving to first approach pose...")
                self._move_to_pose(self.approach_poses[0])

                _logger.info("Executing approach phase (%d waypoints)...", 
                           len(self.approach_poses))
                for i, pose in enumerate(self.approach_poses):
                    self._servo_to_pose(pose)
                    if (i + 1) % max(1, len(self.approach_poses) // 10) == 0:
                        _logger.info("  Approach progress: %d/%d", i + 1, len(self.approach_poses))
                    if self.servo_dt > 0:
                        time.sleep(self.servo_dt)

                _logger.info("Approach phase complete. Gripping object...")
                self._grip_object()

                _logger.info("Retreating (reverse approach)...")
                for i, pose in enumerate(reversed(self.approach_poses)):
                    self._servo_to_pose(pose)
                    if (i + 1) % max(1, len(self.approach_poses) // 10) == 0:
                        _logger.info("  Retreat progress: %d/%d", i + 1, len(self.approach_poses))
                    if self.servo_dt > 0:
                        time.sleep(self.servo_dt)

                _logger.info("Moving to first task pose...")
                self._move_to_pose(self.task_poses[0])

                _logger.info("Executing task phase (%d waypoints)...", len(self.task_poses))
                for i, pose in enumerate(self.task_poses):
                    self._servo_to_pose(pose)
                    if (i + 1) % max(1, len(self.task_poses) // 10) == 0:
                        _logger.info("  Task progress: %d/%d", i + 1, len(self.task_poses))
                    if self.servo_dt > 0:
                        time.sleep(self.servo_dt)

                _logger.info("Task phase complete. Stopping safely...")
                if self.robot is not None:
                    self.robot.servo_stop()
                    time.sleep(0.1)
                    self.robot.stop_script()

                _logger.info("Execution completed successfully.")

        except KeyboardInterrupt:
            _logger.warning("Interrupted by user.")
            raise
        except Exception as exc:
            _logger.error("Execution failed: %s", exc, exc_info=True)
            raise
        finally:
            self._disconnect_robot()


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = _parse_args()
    executor = ToolTaskExecutor(args)
    executor.execute()


if __name__ == "__main__":
    main()
