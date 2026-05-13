from pathlib import Path
import sys

import numpy as np
import spatialmath as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real.slip_correction import compensate_tool_slip


def test_compensate_tool_slip_identity_returns_target_pose() -> None:
    target_pose = sm.SE3.Rt(sm.SO3.Rx(0.3).R, [0.1, -0.2, 0.4], check=False)

    command_pose = compensate_tool_slip(target_pose, sm.SE3())

    assert np.allclose(command_pose.A, target_pose.A)


def test_compensate_tool_slip_local_positive_z_moves_command_negative_z() -> None:
    target_pose = sm.SE3()
    slip = sm.SE3.Tz(0.05)

    command_pose = compensate_tool_slip(target_pose, slip)

    expected_pose = sm.SE3.Tz(-0.05)
    assert np.allclose(command_pose.A, expected_pose.A)


def test_compensate_tool_slip_applies_local_x_in_target_frame() -> None:
    target_pose = sm.SE3.Rz(np.pi / 2) @ sm.SE3.Tx(1.0)
    slip = sm.SE3.Tx(0.1)

    command_pose = compensate_tool_slip(target_pose, slip)

    expected_pose = target_pose @ sm.SE3.Tx(-0.1)
    world_axis_pose = sm.SE3.Tx(0.9) @ sm.SE3.Rz(np.pi / 2)

    assert np.allclose(command_pose.A, expected_pose.A)
    assert not np.allclose(command_pose.A, world_axis_pose.A)


def test_compensate_tool_slip_reconstructs_target_after_reapplying_slip() -> None:
    target_pose = sm.SE3.Rt(
        sm.SO3.RPY([0.2, -0.1, 0.4], order="zyx").R,
        [0.4, -0.3, 0.2],
        check=False,
    )
    slip = sm.SE3.Rt(
        sm.SO3.RPY([0.05, 0.02, -0.04], order="zyx").R,
        [0.03, -0.01, 0.02],
        check=False,
    )

    command_pose = compensate_tool_slip(target_pose, slip)

    assert np.allclose((command_pose @ slip).A, target_pose.A)
