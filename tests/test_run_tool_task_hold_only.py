from argparse import Namespace
from pathlib import Path
import sys
import time

import spatialmath as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real.run_tool_task import ToolTaskExecutor


def _make_args(**overrides: object) -> Namespace:
    base = dict(
        approach_demo=None,
        task1_demo=None,
        task2_demo=None,
        robot_ip="127.0.0.1",
        servo_rate=200.0,
        approach_tau_scale=1.0,
        task1_tau_scale=None,
        task2_tau_scale=None,
        task_tau_scale=1.0,
        tau_scale=None,
        approach_start_pose=None,
        tool_pose=None,
        task_goal_pose=None,
        correction_provider=None,
        max_translation_correction_m=0.01,
        max_rotation_correction_rad=0.1,
        max_tcp_drift_translation_m=0.03,
        max_tcp_drift_rotation_rad=0.35,
        max_phase_jump_m=0.02,
        max_phase_jump_rad=0.35,
        hold_only=True,
        hold_apply_correction=False,
        hold_phase="hold",
        speed=0.05,
        acc=0.1,
        dry_run=False,
    )
    base.update(overrides)
    return Namespace(**base)


class _FakeRobot:
    def __init__(self, pose: sm.SE3, fail_after_servo_calls: int | None = None) -> None:
        self.T_base_tcp = pose
        self.calls = 0
        self.fail_after_servo_calls = fail_after_servo_calls

    def servoL(self, pose: sm.SE3, speed: float, acc: float, dt: float) -> None:
        del pose, speed, acc, dt
        self.calls += 1
        if self.fail_after_servo_calls is not None and self.calls >= self.fail_after_servo_calls:
            raise KeyboardInterrupt()


def test_hold_only_execute_captures_live_pose_and_calls_hold(monkeypatch) -> None:
    args = _make_args(hold_apply_correction=True, hold_phase="slip_hold")
    executor = ToolTaskExecutor(args)
    live_pose = sm.SE3.Tx(0.42) @ sm.SE3.Rz(0.3)

    state: dict[str, object] = {"setup_calls": 0}

    def fake_connect_robot() -> None:
        executor.robot = _FakeRobot(live_pose)  # type: ignore[assignment]

    def fake_hold_pose(target_pose: sm.SE3, phase: str, apply_tcp_correction: bool) -> None:
        state["target_pose"] = target_pose
        state["phase"] = phase
        state["apply_tcp_correction"] = apply_tcp_correction
        raise KeyboardInterrupt()

    def fake_setup_rollouts() -> None:
        state["setup_calls"] = int(state["setup_calls"]) + 1

    monkeypatch.setattr(executor, "_connect_robot", fake_connect_robot)
    monkeypatch.setattr(executor, "_hold_pose", fake_hold_pose)
    monkeypatch.setattr(executor, "_setup_rollouts", fake_setup_rollouts)
    monkeypatch.setattr(executor, "_disconnect_robot", lambda: state.setdefault("disconnected", True))
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)

    try:
        executor.execute()
    except KeyboardInterrupt:
        pass

    assert state["setup_calls"] == 0
    assert state["phase"] == "slip_hold"
    assert state["apply_tcp_correction"] is True
    assert isinstance(state["target_pose"], sm.SE3)
    assert (state["target_pose"]).A.tolist() == live_pose.A.tolist()  # type: ignore[union-attr]
    assert state.get("disconnected") is True


def test_hold_only_execute_skips_gripper_actions(monkeypatch) -> None:
    executor = ToolTaskExecutor(_make_args())
    state = {"opened": 0, "gripped": 0}

    monkeypatch.setattr(executor, "_connect_robot", lambda: setattr(executor, "robot", _FakeRobot(sm.SE3())))
    monkeypatch.setattr(executor, "_open_gripper", lambda: state.__setitem__("opened", state["opened"] + 1))
    monkeypatch.setattr(executor, "_grip_object", lambda: state.__setitem__("gripped", state["gripped"] + 1))
    monkeypatch.setattr(executor, "_hold_pose", lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(executor, "_disconnect_robot", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)

    try:
        executor.execute()
    except KeyboardInterrupt:
        pass

    assert state["opened"] == 0
    assert state["gripped"] == 0


def test_hold_pose_uses_correction_when_enabled(monkeypatch) -> None:
    executor = ToolTaskExecutor(_make_args(hold_apply_correction=True))
    executor.robot = _FakeRobot(sm.SE3(), fail_after_servo_calls=3)  # type: ignore[assignment]
    target = sm.SE3.Tx(0.1)
    state = {"correction_calls": 0}

    def fake_compute_slip_corrected_pose(target_pose: sm.SE3, phase: str) -> sm.SE3:
        del phase
        state["correction_calls"] += 1
        return target_pose @ sm.SE3.Tx(-0.01)

    monkeypatch.setattr(executor, "_compute_slip_corrected_pose", fake_compute_slip_corrected_pose)
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)

    try:
        executor._hold_pose(target, phase="hold", apply_tcp_correction=True)
    except KeyboardInterrupt:
        pass

    assert state["correction_calls"] == 3


def test_hold_pose_skips_correction_when_disabled(monkeypatch) -> None:
    executor = ToolTaskExecutor(_make_args(hold_apply_correction=False))
    executor.robot = _FakeRobot(sm.SE3(), fail_after_servo_calls=3)  # type: ignore[assignment]
    target = sm.SE3.Tx(0.1)
    state = {"correction_calls": 0}

    def fake_compute_slip_corrected_pose(target_pose: sm.SE3, phase: str) -> sm.SE3:
        del target_pose, phase
        state["correction_calls"] += 1
        return sm.SE3()

    monkeypatch.setattr(executor, "_compute_slip_corrected_pose", fake_compute_slip_corrected_pose)
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)

    try:
        executor._hold_pose(target, phase="hold", apply_tcp_correction=False)
    except KeyboardInterrupt:
        pass

    assert state["correction_calls"] == 0

