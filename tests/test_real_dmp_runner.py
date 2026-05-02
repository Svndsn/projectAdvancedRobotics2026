import argparse

import numpy as np
import pandas as pd
import spatialmath as sm

from real.dmp_approach_grasp_task_ur5e import RealApproachGraspTaskDMPExecutor


def _write_demo(path, x0):
    rows = []
    for i in range(6):
        rows.append([x0 + 0.01 * i, 0.1, 0.2, 0.0, 0.0, 0.0])
    pd.DataFrame(
        rows,
        columns=[f"actual_TCP_pose_{i}" for i in range(6)],
    ).to_csv(path, index=False)


def test_dry_run_writes_replay_log_without_hardware(tmp_path):
    approach = tmp_path / "approach.csv"
    task = tmp_path / "task.csv"
    log_path = tmp_path / "run.csv"
    _write_demo(approach, 0.2)
    _write_demo(task, 0.3)

    args = argparse.Namespace(
        approach_demo=str(approach),
        task_demo=str(task),
        replay_log_out=str(log_path),
        approach_tau_scale=1.0,
        task_tau_scale=1.0,
        correction_provider="",
        dry_run=True,
        robot_ip="192.0.2.1",
        wsg50_ip="192.0.2.2",
        wsg50_port=1000,
        wsg50_open_width_mm=100.0,
        wsg50_pregrip_width_mm=80.0,
        wsg50_grip_width_mm=None,
        wsg50_grip_force_n=40.0,
        wsg50_grip_speed_mm_s=50.0,
        wsg50_release_distance_mm=20.0,
        speed=0.05,
        acc=0.1,
        servo_dt=0.01,
        close_hold_s=0.0,
        open_hold_s=0.0,
        max_translation_correction_m=0.02,
        max_rotation_correction_rad=0.35,
        skip_gripper_home=True,
        keep_gripped=False,
    )

    executor = RealApproachGraspTaskDMPExecutor(args)
    executor.execute()

    assert executor.robot is None
    assert executor.gripper is None
    log = pd.read_csv(log_path)
    assert {"approach", "task", "gripper_close", "gripper_open"}.issubset(
        set(log["phase"])
    )
    assert np.all(log["dry_run"])
    assert all(f"corrected_target_TCP_pose_{i}" in log.columns for i in range(6))


def test_hardware_phase_paces_each_waypoint(tmp_path, monkeypatch):
    approach = tmp_path / "approach.csv"
    task = tmp_path / "task.csv"
    log_path = tmp_path / "run.csv"
    _write_demo(approach, 0.2)
    _write_demo(task, 0.3)

    args = argparse.Namespace(
        approach_demo=str(approach),
        task_demo=str(task),
        replay_log_out=str(log_path),
        approach_tau_scale=1.0,
        task_tau_scale=1.0,
        correction_provider="",
        dry_run=False,
        robot_ip="192.0.2.1",
        wsg50_ip="192.0.2.2",
        wsg50_port=1000,
        wsg50_open_width_mm=100.0,
        wsg50_pregrip_width_mm=80.0,
        wsg50_grip_width_mm=None,
        wsg50_grip_force_n=40.0,
        wsg50_grip_speed_mm_s=50.0,
        wsg50_release_distance_mm=20.0,
        speed=0.05,
        acc=0.1,
        servo_dt=0.01,
        close_hold_s=0.0,
        open_hold_s=0.0,
        max_translation_correction_m=0.02,
        max_rotation_correction_rad=0.35,
        skip_gripper_home=True,
        keep_gripped=False,
    )

    executor = RealApproachGraspTaskDMPExecutor(args)

    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "real.dmp_approach_grasp_task_ur5e.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    class _RobotStub:
        def __init__(self) -> None:
            self.servo_calls = 0

        def servoL(self, pose, speed, acc, dt):
            self.servo_calls += 1

        @property
        def T_base_tcp(self):
            return sm.SE3()

    executor.robot = _RobotStub()
    executor.gripper = None

    executor._run_phase("approach", executor.approach_poses[:3], apply_correction=False)

    assert executor.robot.servo_calls == 3
    assert sleep_calls == [args.servo_dt] * 3
