import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spatialmath as sm
import yaml
import mujoco as mj
from dm_control import mjcf

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from ctrl.dmp_cartesian.dmp_cartesian import DMPCartesian
from real.config import (
    ROBOT_IP,
    T_TCP_GRIPPER_TCP,
    WSG50_CLOSE_WIDTH_MM,
    WSG50_GRIP_FALLBACK_TO_MOVE,
    WSG50_GRIP_FORCE_N,
    WSG50_GRIP_SPEED_MM_S,
    WSG50_GRIP_WIDTH_MM,
    WSG50_IP,
    WSG50_OPEN_WIDTH_MM,
    WSG50_PORT,
    WSG50_PREGRIP_WIDTH_MM,
    WSG50_RELEASE_DISTANCE_MM,
)
from real.slip_correction import (
    load_correction_provider,
    validate_and_clip_correction,
)
from real.utils import se3_to_pose
from utils.math import npq2np
from utils.sm import make_tf
from sims.base_sim import BaseSim
from robots.ur_robot import URRobot

_logger = logging.getLogger("sim_dmp_approach_grasp_task")


def _pose_row_to_se3(row: pd.Series, prefix: str) -> sm.SE3:
    position = np.array([row[f"{prefix}_{i}"] for i in range(3)], dtype=float)
    rotvec = np.array([row[f"{prefix}_{i}"] for i in range(3, 6)], dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle > 1e-12:
        R = sm.SO3.AngVec(angle, rotvec / angle).R
    else:
        R = np.eye(3)
    return sm.SE3.Rt(R, position, check=False)


def _detect_pose_prefix(df: pd.DataFrame) -> str:
    if all(f"actual_TCP_pose_{i}" in df.columns for i in range(6)):
        return "actual_TCP_pose"
    if all(f"target_TCP_pose_{i}" in df.columns for i in range(6)):
        return "target_TCP_pose"
    raise ValueError(
        "Could not find pose columns. Expected actual_TCP_pose_[0..5] or target_TCP_pose_[0..5]."
    )


def load_demo_poses(demo_csv: str | Path) -> list[sm.SE3]:
    df = pd.read_csv(demo_csv)
    prefix = _detect_pose_prefix(df)
    return [_pose_row_to_se3(row, prefix) for _, row in df.iterrows()]


def _parse_pose_str(s: str) -> sm.SE3:
    """Parse a CSV pose string 'x,y,z,rx,ry,rz' into an sm.SE3."""
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 6:
        raise ValueError("Pose must be 6 comma-separated floats: x,y,z,rx,ry,rz")
    pos = np.array(parts[:3], dtype=float)
    rotvec = np.array(parts[3:], dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle > 1e-12:
        R = sm.SO3.AngVec(angle, rotvec / angle).R
    else:
        R = np.eye(3)
    return sm.SE3.Rt(R, pos, check=False)


def _load_pose_from_csv(csv_path: str | Path, row_index: int = 0) -> sm.SE3:
    """Load a single pose from a CSV file by row index (0=first, -1=last)."""
    df = pd.read_csv(csv_path)
    prefix = _detect_pose_prefix(df)
    row = df.iloc[row_index]
    return _pose_row_to_se3(row, prefix)


def rollout_dmp(
    poses: list[sm.SE3],
    tau_scale: float,
    start_pose: sm.SE3 | None = None,
    end_pose: sm.SE3 | None = None,
) -> tuple[DMPCartesian, list[sm.SE3]]:
    if tau_scale <= 0:
        raise ValueError("tau_scale must be greater than 0")

    dmp = DMPCartesian(roto_dilatation=True)
    dmp.load(poses)
    # If caller supplied a new start/end pose, update the trained DMPs so
    # the learned forcing term is reused with these dynamic endpoints.
    if start_pose is not None and end_pose is not None:
        dmp.set_start_goal(start_pose, end_pose)
    rollout_tau = dmp.tau * tau_scale
    n_steps = max(2, int(round(len(dmp.ts) * tau_scale)))
    rollout_ts = np.linspace(0.0, rollout_tau, n_steps)
    p, _, _, q, _, _ = dmp.rollout(rollout_ts, rollout_tau)
    return dmp, [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]


def _default_replay_log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/demo") / f"sim_approach_grasp_task_{stamp}.csv"


class SimpleConfig:
    def __init__(self) -> None:
        self.approach_demo = "data/demo/approach.csv"
        self.task_demo = "data/demo/task.csv"
        self.replay_log_out = str(_default_replay_log_path())
        self.approach_tau_scale = 10.0
        self.task_tau_scale = 10.0
        self.speed = 0.05
        self.acc = 0.1
        self.servo_dt = 1.0 / 400
        self.close_hold_s = 0.5
        self.open_hold_s = 0.5
        self.correction_provider = ""
        self.max_translation_correction_m = 0.02
        self.max_rotation_correction_rad = 0.35
        self.start_pose = "data/demo/start_pose.csv"
        self.tool_pose = "data/demo/tool_pose.csv"
        self.goal_pose = "data/demo/goal_pose.csv"
        self.verbose = False
        self.scene_xml = "scenes/ur5e_hande.xml"


class SimDMPExecutor(BaseSim):
    def __init__(self, args: SimpleConfig) -> None:
        super().__init__()
        # Use dm_control's mjcf module to properly handle relative paths in XML
        repo_root = Path(__file__).resolve().parent.parent
        scene_path = (repo_root / args.scene_xml).resolve()
        scene = mjcf.from_path(str(scene_path))
        # Convert the mjcf model to XML and compile with mujoco
        xml_string = scene.to_xml_string()
        self._model = mj.MjModel.from_xml_string(xml_string)
        self._data = mj.MjData(self._model)
        self.robot = URRobot(
            self._model,
            self._data,
            robot_type=URRobot.Type.UR5e,
        )
        self.args = args
        self.replay_log_path = Path(args.replay_log_out).expanduser().resolve()
        self.replay_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.approach_demo = load_demo_poses(args.approach_demo)
        self.task_demo = load_demo_poses(args.task_demo)
        _logger.info(
            "Loaded demos: approach=%s poses from %s, task=%s poses from %s.",
            len(self.approach_demo),
            args.approach_demo,
            len(self.task_demo),
            args.task_demo,
        )
        self.approach_dmp = DMPCartesian(roto_dilatation=True)
        self.approach_dmp.load(self.approach_demo)
        self.task_dmp = DMPCartesian(roto_dilatation=True)
        self.task_dmp.load(self.task_demo)
        self.approach_poses: list[sm.SE3] = []
        self.task_poses: list[sm.SE3] = []
        self.task_goal_pose: sm.SE3 | None = None
        self.tool_contact_pose: sm.SE3 | None = None

        self.correction_provider = load_correction_provider(args.correction_provider)
        _logger.info(
            "Using correction provider: %s.",
            self.correction_provider.__class__.__name__,
        )
        self.rows: list[list[float | str | bool]] = []
        self._t0 = time.monotonic()

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def data(self) -> mj.MjData:
        return self._data

    def execute(self) -> None:
        self.correction_provider.reset()
        try:
            self._execute_simulation()
        except KeyboardInterrupt:
            _logger.warning("Interrupted; writing replay log.")
            raise
        finally:
            self._write_log()

    def _setup_dmp_rollouts(self) -> None:
        def resolve_pose(pose_input: str | None, default: sm.SE3) -> sm.SE3:
            if pose_input is None:
                return default
            if Path(pose_input).exists():
                return _load_pose_from_csv(pose_input, row_index=0)
            else:
                return _parse_pose_str(pose_input)

        approach_start_pose = resolve_pose(self.args.start_pose, self.approach_demo[0])
        tool_pose = resolve_pose(self.args.tool_pose, self.approach_demo[-1])
        task_goal_pose = resolve_pose(self.args.goal_pose, self.task_demo[-1])

        self.tool_contact_pose = tool_pose
        self.task_goal_pose = task_goal_pose

        self.approach_dmp.set_start_goal(approach_start_pose, tool_pose)
        self.task_dmp.set_start_goal(tool_pose, task_goal_pose)

        rollout_tau = self.approach_dmp.tau * self.args.approach_tau_scale
        n_steps = max(2, int(round(len(self.approach_dmp.ts) * self.args.approach_tau_scale)))
        rollout_ts = np.linspace(0.0, rollout_tau, n_steps)
        p, _, _, q, _, _ = self.approach_dmp.rollout(rollout_ts, rollout_tau)
        self.approach_poses = [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]

        rollout_tau = self.task_dmp.tau * self.args.task_tau_scale
        n_steps = max(2, int(round(len(self.task_dmp.ts) * self.args.task_tau_scale)))
        rollout_ts = np.linspace(0.0, rollout_tau, n_steps)
        p, _, _, q, _, _ = self.task_dmp.rollout(rollout_ts, rollout_tau)
        self.task_poses = [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]

    def _execute_simulation(self) -> None:
        self._setup_dmp_rollouts()

        _logger.info("Moving robot to first approach pose.")
        self.robot.goto_pose(self.approach_poses[0], speed=self.args.speed, acc=self.args.acc)
        self.sync()

        _logger.info("Starting approach phase.")
        self._run_phase("approach", self.approach_poses, apply_correction=False)

        _logger.info("Approach phase complete. Simulating gripper close.")
        _logger.info("Gripping is done.")
        self._hold_pose(self.approach_poses[-1], "gripper_close", self.args.close_hold_s)

        _logger.info("Deaproaching phase started.")
        self._run_phase("deapproach", list(reversed(self.approach_poses)), apply_correction=False)

        _logger.info("Starting task phase with correction enabled.")
        if self.task_goal_pose is not None:
            current_pose = self.robot.T_base_tcp
            self.task_dmp.set_start_goal(current_pose, self.task_goal_pose)
            rollout_tau = self.task_dmp.tau * self.args.task_tau_scale
            n_steps = max(2, int(round(len(self.task_dmp.ts) * self.args.task_tau_scale)))
            rollout_ts = np.linspace(0.0, rollout_tau, n_steps)
            p, _, _, q, _, _ = self.task_dmp.rollout(rollout_ts, rollout_tau)
            self.task_poses = [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]

        self._run_phase("task", self.task_poses, apply_correction=True)
        _logger.info("Task phase complete.")

        _logger.info("Simulating gripper open.")
        self._hold_pose(self.task_poses[-1], "gripper_open", self.args.open_hold_s)

    def _run_phase(
        self,
        phase: str,
        poses: list[sm.SE3],
        apply_correction: bool,
    ) -> None:
        _logger.info(
            "Phase %s: %s waypoints, correction=%s.",
            phase,
            len(poses),
            "on" if apply_correction else "off",
        )
        progress_interval = max(1, len(poses) // 10)
        for i, target_pose in enumerate(poses):
            step_idx = i + 1
            self._command_pose(target_pose, phase, apply_correction)
            if (
                step_idx == 1
                or step_idx == len(poses)
                or step_idx % progress_interval == 0
            ):
                _logger.info("Phase %s progress: %s/%s.", phase, step_idx, len(poses))
            self.sync()

    def _hold_pose(self, pose: sm.SE3, phase: str, duration_s: float) -> None:
        if duration_s <= 0:
            self._command_pose(pose, phase, phase == "task")
            self.sync()
            return
        steps = max(1, int(round(duration_s / self.dt)))
        for _ in range(steps):
            self._command_pose(pose, phase, False)
            self.sync()

    def _command_pose(
        self,
        target_pose: sm.SE3,
        phase: str,
        apply_correction: bool,
    ) -> None:
        now = time.monotonic() - self._t0
        actual_pose = self.robot.T_base_tcp
        if apply_correction:
            raw_tool_slip_tf = self.correction_provider.get_correction(
                actual_pose=actual_pose,
                target_pose=target_pose,
                phase=phase,
                t=now,
            )
            tool_slip_tf = validate_and_clip_correction(
                raw_tool_slip_tf,
                self.args.max_translation_correction_m,
                self.args.max_rotation_correction_rad,
            )
            raw_translation_norm = float(np.linalg.norm(raw_tool_slip_tf.t))
            clipped_translation_norm = float(np.linalg.norm(tool_slip_tf.t))
            raw_angle, _ = raw_tool_slip_tf.angvec()
            clipped_angle, _ = tool_slip_tf.angvec()
            if (
                abs(raw_translation_norm - clipped_translation_norm) > 1e-9
                or abs(float(raw_angle) - float(clipped_angle)) > 1e-9
            ):
                _logger.warning(
                    "Slip correction clipped: translation %.4f->%.4f m, rotation %.4f->%.4f rad.",
                    raw_translation_norm,
                    clipped_translation_norm,
                    float(raw_angle),
                    float(clipped_angle),
                )
        else:
            tool_slip_tf = sm.SE3()

        # T_base_tool_desired = target_pose
        # T_gripper_tool_slip = tool_slip_tf
        # T_base_gripper_command = T_base_tool_desired @ T_gripper_tool_slip.inv()
        command_pose = target_pose @ tool_slip_tf.inv()
        self.robot.set_tcp_target(command_pose)
        actual_pose = self.robot.T_base_tcp

        self._log_step(phase, target_pose, command_pose, actual_pose, tool_slip_tf)

    def _log_step(
        self,
        phase: str,
        target_pose: sm.SE3,
        command_pose: sm.SE3,
        actual_pose: sm.SE3,
        tool_slip_tf: sm.SE3,
    ) -> None:
        self.rows.append(
            [
                time.monotonic() - self._t0,
                phase,
                *se3_to_pose(target_pose).tolist(),
                *se3_to_pose(command_pose).tolist(),
                *se3_to_pose(actual_pose).tolist(),
                *se3_to_pose(tool_slip_tf).tolist(),
                self._gripper_state(phase),
                True,  # dry_run is always true in sim
            ]
        )

    def _gripper_state(self, phase: str) -> str:
        if phase == "gripper_close":
            return "grip"
        if phase == "gripper_open":
            return "release_open"
        if phase == "approach":
            return "open"
        return "holding"

    def _write_log(self) -> None:
        columns = [
            "time",
            "phase",
            *[f"dmp_target_TCP_pose_{i}" for i in range(6)],
            *[f"corrected_target_TCP_pose_{i}" for i in range(6)],
            *[f"actual_TCP_pose_{i}" for i in range(6)],
            *[f"correction_TCP_pose_{i}" for i in range(6)],
            "gripper_state",
            "dry_run",
        ]
        pd.DataFrame(self.rows, columns=columns).to_csv(
            self.replay_log_path,
            index=False,
        )
        _logger.info("Saved replay log: %s (%s rows)", self.replay_log_path, len(self.rows))


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    config = SimpleConfig()
    setup_logging(config.verbose)
    _logger.info("Loaded simulation DMP approach-grasp-task configuration.")
    executor = SimDMPExecutor(config)
    executor.run()


if __name__ == "__main__":
    main()
