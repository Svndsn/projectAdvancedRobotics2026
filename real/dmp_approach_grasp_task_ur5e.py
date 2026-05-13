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
from real.wsg50_gripper import WSG50CommandError, WSG50Gripper
from utils.math import npq2np
from utils.sm import make_tf


_logger = logging.getLogger("real_dmp_approach_grasp_task")


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
    return Path("data/demo") / f"real_approach_grasp_task_wsg50_{stamp}.csv"


def _apply_if_present(src: dict[str, Any], key: str, out: dict[str, Any]) -> None:
    if key in src:
        out[key] = src[key]


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config root must be a mapping/dictionary.")

    out: dict[str, Any] = {}
    for flat_key in [
        "approach_demo",
        "task_demo",
        "replay_log_out",
        "robot_ip",
        "wsg50_ip",
        "wsg50_port",
        "wsg50_open_width_mm",
        "wsg50_pregrip_width_mm",
        "wsg50_grip_width_mm",
        "wsg50_close_width_mm",
        "wsg50_grip_fallback_to_move",
        "wsg50_grip_force_n",
        "wsg50_grip_speed_mm_s",
        "wsg50_release_distance_mm",
        "approach_tau_scale",
        "task_tau_scale",
        "speed",
        "acc",
        "servo_dt",
        "close_hold_s",
        "open_hold_s",
        "correction_provider",
        "max_translation_correction_m",
        "max_rotation_correction_rad",
        "dry_run",
        "skip_gripper_home",
        "keep_gripped",
        "capture_tool_pose",
        "capture_task_goal",
        "approach_start_pose",
        "tool_pose",
        "task_goal_pose",
    ]:
        _apply_if_present(raw, flat_key, out)

    demos = raw.get("demos", {})
    if isinstance(demos, dict):
        if "approach" in demos:
            out["approach_demo"] = demos["approach"]
        if "task" in demos:
            out["task_demo"] = demos["task"]

    logging_cfg = raw.get("logging", {})
    if isinstance(logging_cfg, dict):
        _apply_if_present(logging_cfg, "replay_log_out", out)

    robot = raw.get("robot", {})
    if isinstance(robot, dict):
        if "ip" in robot:
            out["robot_ip"] = robot["ip"]

    gripper = raw.get("gripper", {})
    if isinstance(gripper, dict):
        mapping = {
            "ip": "wsg50_ip",
            "port": "wsg50_port",
            "open_width_mm": "wsg50_open_width_mm",
            "pregrip_width_mm": "wsg50_pregrip_width_mm",
            "grip_width_mm": "wsg50_grip_width_mm",
            "close_width_mm": "wsg50_close_width_mm",
            "grip_fallback_to_move": "wsg50_grip_fallback_to_move",
            "grip_force_n": "wsg50_grip_force_n",
            "grip_speed_mm_s": "wsg50_grip_speed_mm_s",
            "release_distance_mm": "wsg50_release_distance_mm",
            "skip_home": "skip_gripper_home",
            "keep_gripped": "keep_gripped",
        }
        for src_key, dst_key in mapping.items():
            if src_key in gripper:
                out[dst_key] = gripper[src_key]

    tau_scale = raw.get("tau_scale", {})
    if isinstance(tau_scale, dict):
        if "approach" in tau_scale:
            out["approach_tau_scale"] = tau_scale["approach"]
        if "task" in tau_scale:
            out["task_tau_scale"] = tau_scale["task"]

    safety = raw.get("safety", {})
    if isinstance(safety, dict):
        for key in [
            "speed",
            "acc",
            "servo_dt",
            "max_translation_correction_m",
            "max_rotation_correction_rad",
        ]:
            _apply_if_present(safety, key, out)

    timing = raw.get("timing", {})
    if isinstance(timing, dict):
        _apply_if_present(timing, "close_hold_s", out)
        _apply_if_present(timing, "open_hold_s", out)

    correction = raw.get("correction", {})
    if isinstance(correction, dict) and "provider" in correction:
        out["correction_provider"] = correction["provider"]

    return out


class RealApproachGraspTaskDMPExecutor:
    def __init__(self, args: argparse.Namespace) -> None:
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
        # Train DMPCartesian instances from demos but postpone rollouts until
        # runtime so we can set dynamic start/end poses (e.g. captured tool pose).
        self.approach_dmp = DMPCartesian(roto_dilatation=True)
        self.approach_dmp.load(self.approach_demo)
        self.task_dmp = DMPCartesian(roto_dilatation=True)
        self.task_dmp.load(self.task_demo)
        # rollout poses will be created at execution time after any dynamic
        # start/goal updates
        self.approach_poses: list[sm.SE3] = []
        self.task_poses: list[sm.SE3] = []

        self.correction_provider = load_correction_provider(args.correction_provider)
        _logger.info(
            "Using correction provider: %s.",
            self.correction_provider.__class__.__name__,
        )
        self.rows: list[list[float | str | bool]] = []
        self.robot = None
        self.gripper: WSG50Gripper | None = None
        self._t0 = time.monotonic()

    def execute(self) -> None:
        self.correction_provider.reset()
        try:
            if self.args.dry_run:
                _logger.info("Running dry-run: no UR5e or WSG50 connection will be opened.")
                self._execute_dry_run()
            else:
                self._connect_hardware()
                self._execute_hardware()
        except KeyboardInterrupt:
            _logger.warning("Interrupted; stopping hardware and writing replay log.")
            raise
        finally:
            self._stop_hardware()
            self._write_log()

    def _connect_hardware(self) -> None:
        from real.robot import Robot

        self.robot = Robot(
            robot_ip=self.args.robot_ip,
            T_tcp_gripper_tcp=T_TCP_GRIPPER_TCP,
        )
        self.gripper = WSG50Gripper(
            host=self.args.wsg50_ip,
            port=self.args.wsg50_port,
            open_width_mm=self.args.wsg50_open_width_mm,
            default_speed_mm_s=self.args.wsg50_grip_speed_mm_s,
        )
        self.gripper.connect()
        _logger.info(
            "Connected to UR5e at %s and WSG50 at %s:%s.",
            self.args.robot_ip,
            self.args.wsg50_ip,
            self.args.wsg50_port,
        )
        try:
            verbose_response = self.gripper.set_verbose(True)
            _logger.info("Enabled WSG50 verbose errors: %s", verbose_response)
        except Exception as exc:
            _logger.warning("Could not enable WSG50 verbose errors: %s", exc)

    def _setup_dmp_rollouts(self) -> None:
        """
        Set up the DMP rollouts with three independent poses:
        - approach_start: where the approach DMP begins
        - tool_pose: tool contact point (end of approach, start of task)
        - task_goal: where the task DMP ends
        """
        # Helper to load or parse a pose
        def resolve_pose(pose_input: str | None, default: sm.SE3) -> sm.SE3:
            if pose_input is None:
                return default
            # Check if it's a file path
            if Path(pose_input).exists():
                return _load_pose_from_csv(pose_input, row_index=0)
            else:
                # Parse as CSV string
                return _parse_pose_str(pose_input)

        # Determine approach start pose
        if getattr(self.args, "approach_start_pose", None):
            approach_start_pose = resolve_pose(self.args.approach_start_pose, self.approach_demo[0])
        else:
            approach_start_pose = self.approach_demo[0]

        # Determine tool pose (approach goal / task start)
        tool_pose: sm.SE3 | None = None
        if self.args.capture_tool_pose:
            if self.robot is not None:
                input(
                    "Position the tool to the desired contact pose and press Enter to capture..."
                )
                tool_pose = self.robot.T_base_tcp

        if getattr(self.args, "tool_pose", None):
            tool_pose = resolve_pose(self.args.tool_pose, self.approach_demo[-1])
        elif tool_pose is None:
            tool_pose = self.approach_demo[-1]

        # Determine task goal pose
        if getattr(self.args, "task_goal_pose", None):
            task_goal_pose = resolve_pose(self.args.task_goal_pose, self.task_demo[-1])
        elif getattr(self.args, "capture_task_goal", False):
            if self.robot is not None:
                input(
                    "Position the robot at the desired task goal and press Enter to capture..."
                )
                task_goal_pose = self.robot.T_base_tcp
            else:
                task_goal_pose = self.task_demo[-1]
        else:
            task_goal_pose = self.task_demo[-1]

        # Apply dynamic start/goals:
        # approach: starts at approach_start, ends at tool_pose
        # task: starts at tool_pose, ends at task_goal_pose
        self.approach_dmp.set_start_goal(approach_start_pose, tool_pose)
        self.task_dmp.set_start_goal(tool_pose, task_goal_pose)

        # Roll out DMPS now that start/goals are set
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

    def _execute_hardware(self) -> None:
        assert self.robot is not None
        assert self.gripper is not None

        self._acknowledge_gripper_fast_stop()
        if not self.args.skip_gripper_home:
            self._run_gripper_motion("HOME", self.gripper.home)
        self._run_gripper_motion(
            "OPEN",
            self.gripper.open,
            self.args.wsg50_grip_speed_mm_s,
        )
        # Set up DMP rollouts based on CLI options and captured poses
        self._setup_dmp_rollouts()

        _logger.info("Moving robot to first approach pose.")
        self.robot.moveL(self.approach_poses[0], speed=self.args.speed, acc=self.args.acc)
        _logger.info("Starting approach phase.")
        self._run_phase("approach", self.approach_poses, apply_correction=False)

        _logger.info("Approach phase complete. Commanding gripper close.")
        self._grip_after_approach()
        _logger.info("Holding close pose for %.3f s.", self.args.close_hold_s)
        self._hold_pose(self.approach_poses[-1], "gripper_close", self.args.close_hold_s)

        _logger.info("Deaproaching phase started.")
        self._run_phase("deapproach", reversed(self.approach_poses), apply_correction=False)

        _logger.info("Starting task phase with correction enabled.")
        self._run_phase("task", self.task_poses, apply_correction=True)
        _logger.info("Task phase complete.")

        if not self.args.keep_gripped:
            _logger.info("Releasing/opening gripper at final task pose.")
            self._run_gripper_motion(
                "RELEASE",
                self.gripper.release,
                self.args.wsg50_release_distance_mm,
                self.args.wsg50_grip_speed_mm_s,
            )
            self._run_gripper_motion(
                "OPEN",
                self.gripper.open,
                self.args.wsg50_grip_speed_mm_s,
            )
            self._hold_pose(self.task_poses[-1], "gripper_open", self.args.open_hold_s)

    def _grip_after_approach(self) -> None:
        assert self.gripper is not None
        try:
            self._run_gripper_motion(
                "GRIP",
                self.gripper.grip,
                force_n=self.args.wsg50_grip_force_n,
                width_mm=self.args.wsg50_grip_width_mm,
                speed_mm_s=(
                    self.args.wsg50_grip_speed_mm_s
                    if self.args.wsg50_grip_width_mm is not None
                    else None
                ),
                retry_once=True,
            )
        except WSG50CommandError as exc:
            if not self.args.wsg50_grip_fallback_to_move:
                raise
            _logger.warning(
                "WSG50 GRIP failed with %s; falling back to MOVE(%s, %s). "
                "Tune gripper.close_width_mm to the tool width.",
                exc,
                self.args.wsg50_close_width_mm,
                self.args.wsg50_grip_speed_mm_s,
            )
            self._run_gripper_motion(
                "fallback close MOVE",
                self.gripper.move,
                self.args.wsg50_close_width_mm,
                self.args.wsg50_grip_speed_mm_s,
                retry_once=True,
            )

    def _acknowledge_gripper_fast_stop(self) -> None:
        assert self.gripper is not None
        try:
            self.gripper.fast_stop_ack()
            _logger.info("Acknowledged WSG50 fast stop/fault state.")
        except WSG50CommandError as exc:
            _logger.warning("WSG50 FSACK returned an error: %s", exc)

    def _run_gripper_motion(
        self,
        label: str,
        fn,
        *args,
        retry_once: bool = True,
        **kwargs,
    ) -> list[str]:
        assert self.gripper is not None
        try:
            response = fn(*args, **kwargs)
            _logger.info("WSG50 %s response: %s", label, " | ".join(response))
            return response
        except WSG50CommandError as exc:
            if exc.code not in (16, 18) or not retry_once:
                raise
            _logger.warning(
                "WSG50 %s failed with %s; acknowledging fast stop/fault and retrying once.",
                label,
                exc,
            )
            self._acknowledge_gripper_fast_stop()
            response = fn(*args, **kwargs)
            _logger.info("WSG50 %s retry response: %s", label, " | ".join(response))
            return response

    def _execute_dry_run(self) -> None:
        # Set up DMP rollouts based on CLI options (no robot capture in dry-run)
        self._setup_dmp_rollouts()
        _logger.info("Dry-run approach phase.")
        self._run_phase("approach", self.approach_poses, apply_correction=False)
        _logger.info("Dry-run close hold.")
        self._hold_pose(self.approach_poses[-1], "gripper_close", self.args.close_hold_s)
        _logger.info("Dry-run task phase with correction enabled.")
        self._run_phase("task", self.task_poses, apply_correction=True)
        if not self.args.keep_gripped:
            _logger.info("Dry-run open hold.")
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
        for target_pose in poses:
            step_idx = sum(1 for row in self.rows if row[1] == phase) + 1
            self._command_pose(target_pose, phase, apply_correction)
            if (
                step_idx == 1
                or step_idx == len(poses)
                or step_idx % progress_interval == 0
            ):
                _logger.info("Phase %s progress: %s/%s.", phase, step_idx, len(poses))
            if self.args.dry_run and self.args.servo_dt > 0:
                time.sleep(min(self.args.servo_dt, 0.001))

    def _hold_pose(self, pose: sm.SE3, phase: str, duration_s: float) -> None:
        if duration_s <= 0:
            self._command_pose(pose, phase, phase == "task")
            return
        steps = max(1, int(round(duration_s / self.args.servo_dt)))
        for _ in range(steps):
            self._command_pose(pose, phase, False)
            if self.args.dry_run and self.args.servo_dt > 0:
                time.sleep(min(self.args.servo_dt, 0.001))

    def _command_pose(
        self,
        target_pose: sm.SE3,
        phase: str,
        apply_correction: bool,
    ) -> None:
        now = time.monotonic() - self._t0
        actual_pose = self._actual_pose(default=target_pose)
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
        if self.robot is not None:
            self.robot.servoL(
                command_pose,
                speed=self.args.speed,
                acc=self.args.acc,
                dt=self.args.servo_dt,
            )
            actual_pose = self._actual_pose(default=command_pose)
        else:
            actual_pose = command_pose

        self._log_step(phase, target_pose, command_pose, actual_pose, tool_slip_tf)

        if not self.args.dry_run and self.args.servo_dt > 0:
            time.sleep(self.args.servo_dt)

    def _actual_pose(self, default: sm.SE3) -> sm.SE3:
        if self.robot is None:
            return default
        return self.robot.T_base_tcp

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
                self.args.dry_run,
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

    def _stop_hardware(self) -> None:
        if self.robot is not None:
            for method_name in ("servo_stop", "stop_script"):
                try:
                    getattr(self.robot, method_name)()
                except Exception as exc:
                    _logger.warning("Failed robot %s: %s", method_name, exc)

        if self.gripper is not None:
            try:
                self.gripper.stop()
            except Exception as exc:
                _logger.warning("Failed WSG50 stop: %s", exc)
            self.gripper.disconnect()

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


def _build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        description="Real UR5e approach DMP -> WSG50 grasp -> task DMP runner."
    )
    parser.set_defaults(**defaults)

    parser.add_argument("--config", default="", help="YAML config path.")
    parser.add_argument("--approach-demo", default=defaults.get("approach_demo"))
    parser.add_argument("--task-demo", default=defaults.get("task_demo"))
    parser.add_argument(
        "--replay-log-out",
        default=defaults.get("replay_log_out", str(_default_replay_log_path())),
    )
    parser.add_argument("--robot-ip", default=defaults.get("robot_ip", ROBOT_IP))
    parser.add_argument("--wsg50-ip", default=defaults.get("wsg50_ip", WSG50_IP))
    parser.add_argument("--wsg50-port", type=int, default=defaults.get("wsg50_port", WSG50_PORT))
    parser.add_argument(
        "--wsg50-open-width-mm",
        type=float,
        default=defaults.get("wsg50_open_width_mm", WSG50_OPEN_WIDTH_MM),
    )
    parser.add_argument(
        "--wsg50-pregrip-width-mm",
        type=float,
        default=defaults.get("wsg50_pregrip_width_mm", WSG50_PREGRIP_WIDTH_MM),
    )
    parser.add_argument(
        "--wsg50-grip-width-mm",
        type=float,
        default=defaults.get("wsg50_grip_width_mm", WSG50_GRIP_WIDTH_MM),
        help="Optional nominal part width for GRIP. Omit to use GRIP(force) without part-width detection.",
    )
    parser.add_argument(
        "--wsg50-close-width-mm",
        type=float,
        default=defaults.get("wsg50_close_width_mm", WSG50_CLOSE_WIDTH_MM),
        help="Fallback MOVE width after GRIP fails.",
    )
    parser.add_argument(
        "--wsg50-grip-fallback-to-move",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("wsg50_grip_fallback_to_move", WSG50_GRIP_FALLBACK_TO_MOVE),
        help="Fallback to MOVE(close_width_mm) if GRIP fails.",
    )
    parser.add_argument(
        "--wsg50-grip-force-n",
        type=float,
        default=defaults.get("wsg50_grip_force_n", WSG50_GRIP_FORCE_N),
    )
    parser.add_argument(
        "--wsg50-grip-speed-mm-s",
        type=float,
        default=defaults.get("wsg50_grip_speed_mm_s", WSG50_GRIP_SPEED_MM_S),
    )
    parser.add_argument(
        "--wsg50-release-distance-mm",
        type=float,
        default=defaults.get("wsg50_release_distance_mm", WSG50_RELEASE_DISTANCE_MM),
    )
    parser.add_argument(
        "--approach-tau-scale",
        type=float,
        default=defaults.get("approach_tau_scale", 5.0),
    )
    parser.add_argument(
        "--task-tau-scale",
        type=float,
        default=defaults.get("task_tau_scale", 5.0),
    )
    parser.add_argument("--speed", type=float, default=defaults.get("speed", 0.05))
    parser.add_argument("--acc", type=float, default=defaults.get("acc", 0.1))
    parser.add_argument("--servo-dt", type=float, default=defaults.get("servo_dt", 1.0 / 400))
    parser.add_argument("--close-hold-s", type=float, default=defaults.get("close_hold_s", 0.5))
    parser.add_argument("--open-hold-s", type=float, default=defaults.get("open_hold_s", 0.5))
    parser.add_argument(
        "--correction-provider",
        default=defaults.get("correction_provider", ""),
        help="Optional Python provider formatted as module:ClassName.",
    )
    parser.add_argument(
        "--max-translation-correction-m",
        type=float,
        default=defaults.get("max_translation_correction_m", 0.02),
    )
    parser.add_argument(
        "--max-rotation-correction-rad",
        type=float,
        default=defaults.get("max_rotation_correction_rad", 0.35),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=defaults.get("dry_run", False),
        help="Roll out, correct, and log without connecting to hardware.",
    )
    parser.add_argument(
        "--skip-gripper-home",
        action="store_true",
        default=defaults.get("skip_gripper_home", False),
    )
    parser.add_argument(
        "--keep-gripped",
        action="store_true",
        default=defaults.get("keep_gripped", False),
    )
    parser.add_argument(
        "--capture-tool-pose",
        action="store_true",
        default=defaults.get("capture_tool_pose", False),
        help="Prompt and capture the current robot tool pose (approach goal / task start).",
    )
    parser.add_argument(
        "--capture-task-goal",
        action="store_true",
        default=defaults.get("capture_task_goal", False),
        help="Prompt and capture the current robot pose as the task goal (final task pose).",
    )
    parser.add_argument(
        "--approach-start-pose",
        type=str,
        default=defaults.get("approach_start_pose", None),
        help="Approach start pose: file path (first row) or CSV 'x,y,z,rx,ry,rz'. If omitted, demo start is used.",
    )
    parser.add_argument(
        "--tool-pose",
        type=str,
        default=defaults.get("tool_pose", None),
        help="Tool contact pose: file path (first row) or CSV 'x,y,z,rx,ry,rz'. If omitted, captured or demo contact is used.",
    )
    parser.add_argument(
        "--task-goal-pose",
        type=str,
        default=defaults.get("task_goal_pose", None),
        help="Task goal pose: file path (last row) or CSV 'x,y,z,rx,ry,rz'. If omitted, demo goal is used.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="")
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = _load_yaml_config(pre_args.config)
    parser = _build_parser(defaults)
    args = parser.parse_args(remaining)
    args.config = pre_args.config

    if not args.approach_demo or not args.task_demo:
        parser.error("Both --approach-demo and --task-demo are required.")
    if args.approach_tau_scale <= 0 or args.task_tau_scale <= 0:
        parser.error("Tau scale values must be greater than 0.")
    if args.servo_dt <= 0:
        parser.error("--servo-dt must be greater than 0.")
    return args


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = _parse_args()
    setup_logging(args.verbose)
    _logger.info("Loaded real DMP approach-grasp-task configuration.")
    executor = RealApproachGraspTaskDMPExecutor(args)
    executor.execute()


if __name__ == "__main__":
    main()
