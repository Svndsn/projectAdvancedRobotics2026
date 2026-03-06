import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import spatialmath as sm
import spatialmath.base as smb

from utils.sm import (
    csvread,
    ctraj,
    cubic_interpolation,
    is_ori_valid,
    is_R_valid,
    jtraj,
    load_traj,
    make_R_valid,
    make_tf,
    save_traj,
    trapezoidal_times,
)


# Test make_tf
def test_make_tf():
    # Test with default values
    T_default = make_tf()
    expected_t_default = np.array([0, 0, 0])
    expected_R_default = np.eye(3)
    assert np.allclose(
        T_default.t, expected_t_default
    ), f"Expected position {expected_t_default}, but got {T_default.t}"
    assert np.allclose(
        T_default.R, expected_R_default
    ), f"Expected rotation {expected_R_default}, but got {T_default.R}"

    # Test with custom position and quaternion (no rotation)
    pos = [1, 2, 3]
    ori = [1, 0, 0, 0]
    T_custom = make_tf(pos, ori)
    expected_t_custom = np.array(pos)
    expected_R_custom = np.eye(3, 3)
    assert np.allclose(
        T_custom.t, expected_t_custom
    ), f"Expected position {expected_t_custom}, but got {T_custom.t}"

    assert np.allclose(
        T_custom.R, expected_R_custom
    ), f"Expected rotation {expected_R_custom}, but got {T_custom.R}"

    # Test with custom position and quaternion (with rotation)
    pos = [1, 2, 3]
    ori = [0.70710678, 0.0, 0.0, 0.70710678]  # 90 degrees about z-axis
    T_rot = make_tf(pos, ori)
    expected_R_rot = smb.rpy2r(0, 0, np.pi / 2, order="zyx")

    assert np.allclose(
        T_rot.R, expected_R_rot
    ), f"Expected rotation {expected_R_rot}, but got {T_rot.R}"


def test_is_R_valid():
    # Valid rotation matrix
    R_valid = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert is_R_valid(R_valid), "Valid rotation matrix failed the check."

    # Invalid rotation matrix (determinant not 1)
    R_invalid_det = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    assert not is_R_valid(
        R_invalid_det
    ), "Matrix with invalid determinant passed the check."

    # Non-orthogonal matrix
    R_non_orthogonal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert not is_R_valid(R_non_orthogonal), "Non-orthogonal matrix passed the check."

    # Not a 3x3 matrix
    R_not_3x3 = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        is_R_valid(R_not_3x3)


def test_is_ori_valid():
    # Valid rotation matrix
    R_valid = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert is_ori_valid(R_valid), "Valid rotation matrix failed the check."

    # Valid quaternion (90 degrees about Z-axis)
    ori_quat = np.array([0, 0, 0.70710678, 0.70710678])
    assert is_ori_valid(ori_quat), "Valid quaternion failed the check."

    # Valid Euler angles (90 degrees about Z-axis)
    ori_euler = np.array([0, 0, np.pi / 2])
    assert is_ori_valid(ori_euler), "Valid Euler angles failed the check."

    # Valid Euler angles (180 degrees about Z-axis)
    ori_euler_180 = np.array([0, 0, np.pi])
    assert is_ori_valid(
        ori_euler_180
    ), "Valid Euler angles with 180 degrees failed the check."

    # Invalid rotation matrix (not orthogonal)
    R_invalid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert not is_ori_valid(R_invalid), "Invalid rotation matrix passed the check."

    # Invalid quaternion (not normalized)
    ori_invalid_quat = np.array([0, 0, 0.5, 0.5])
    assert is_ori_valid(ori_invalid_quat), "Invalid quaternion passed the check."

    # Invalid Euler angles (resulting in an invalid rotation matrix)
    ori_invalid_euler = np.array([0, 0, np.pi])
    assert is_ori_valid(ori_invalid_euler), "Invalid Euler angles failed the check."

    # Edge case: Empty input
    with pytest.raises(ValueError):
        is_ori_valid(np.array([]))

    # Edge case: Unsupported type
    with pytest.raises(ValueError):
        is_ori_valid("string")


def test_make_R_valid():
    # Valid rotation matrix
    R_valid = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(
        make_R_valid(R_valid), R_valid
    ), "Valid rotation matrix was altered."

    # Invalid rotation matrix (non-orthogonal)
    R_invalid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    R_corrected = make_R_valid(R_invalid)
    assert is_R_valid(
        R_corrected
    ), "Invalid rotation matrix was not corrected properly."


def test_csvread():
    # Create a temporary CSV file with known data
    csv_data = pd.DataFrame(
        {
            "target_TCP_pose_0": [1],
            "target_TCP_pose_1": [2],
            "target_TCP_pose_2": [3],
            "target_TCP_pose_3": [0],
            "target_TCP_pose_4": [0],
            "target_TCP_pose_5": [np.pi / 2],
        }
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
        csv_path = tmpfile.name
        csv_data.to_csv(csv_path, index=False)

    transformations = csvread(csv_path)
    assert (
        len(transformations) == 1
    ), "CSV read failed to produce expected number of transformations."

    # Validate the transformation
    T = transformations[0]
    assert np.allclose(T.t, [1, 2, 3]), "Translation component mismatch."
    expected_R = smb.rpy2r(0, 0, np.pi / 2, order="xyz")
    assert np.allclose(T.R, expected_R), "Rotation matrix mismatch."

    # Cleanup
    import os

    os.remove(csv_path)


def test_trapezoidal_times():
    # Test with specific parameters
    steps = 100
    times = trapezoidal_times(steps, accel_ratio=0.2, decel_ratio=0.2)

    assert len(times) == steps, "Number of time steps mismatch."

    # Check if the profile has the correct phases
    accel_end = int(0.2 * steps)
    decel_start = steps - int(0.2 * steps)
    assert np.all(times[:accel_end] <= 0.2), "Acceleration phase time mismatch."
    assert np.all(
        times[accel_end:decel_start] >= 0.2
    ), "Constant velocity phase time mismatch."
    assert np.all(times[decel_start:] >= 0.8), "Deceleration phase time mismatch."


def test_ctraj():
    T_start = sm.SE3.Rt(np.eye(3), [0, 0, 0])
    T_end = sm.SE3.Rt(smb.rpy2r(0, 0, np.pi), [1, 1, 1])
    num_steps = 10

    trajectory = ctraj(T_start, T_end, num_steps)
    assert len(trajectory) == num_steps, "Number of trajectory steps mismatch."

    # Check a few specific values
    T_interp = trajectory[0]
    assert np.allclose(T_interp.t, [0, 0, 0]), "Initial position mismatch."
    assert np.allclose(T_interp.R, np.eye(3)), "Initial rotation matrix mismatch."


def test_save_traj(tmpdir):
    # Create a temporary directory for saving the file
    temp_dir = tmpdir.mkdir("test_dir")
    save_path = os.path.join(temp_dir, "test_trajectory.csv")

    # Create a test trajectory with a few SE3 transformations
    traj = [
        sm.SE3(np.eye(4)),
        sm.SE3(np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])),
        sm.SE3(np.array([[0, 0, 1, 4], [0, 1, 0, 5], [-1, 0, 0, 6], [0, 0, 0, 1]])),
    ]

    # Call the save_traj function
    save_traj(traj, save_path)

    # Check if the file is created
    assert os.path.exists(save_path)

    # Load the saved file and check the contents
    with open(save_path, mode="r") as file:
        lines = file.readlines()

    print(1)

    # Expected header
    expected_header = "r11,r12,r13,r21,r22,r23,r31,r32,r33,t1,t2,t3\n"
    assert lines[0] == expected_header
    print(2)

    # Check the first saved SE3 (should be an identity matrix)
    expected_first_row = "1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0\n"
    assert lines[1] == expected_first_row

    print(3)
    # Check the second saved SE3
    expected_second_row = "0,-1,0,1,0,0,0,0,1,1,2,3\n"
    assert lines[2] == expected_second_row

    print(4)
    # Check the third saved SE3
    expected_third_row = "0,0,1,0,1,0,-1,0,0,4,5,6\n"
    assert lines[3] == expected_third_row


def test_load_traj(tmpdir):
    # Create a temporary directory for saving the file
    temp_dir = tmpdir.mkdir("test_dir")
    save_path = os.path.join(temp_dir, "test_trajectory.csv")

    # Create a CSV file with known SE3 transformations
    csv_content = """r11,r12,r13,r21,r22,r23,r31,r32,r33,t1,t2,t3
1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0
0.0,-1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,2.0,3.0
0.0,0.0,1.0,0.0,1.0,0.0,-1.0,0.0,0.0,4.0,5.0,6.0
"""
    with open(save_path, mode="w") as file:
        file.write(csv_content)

    # Load the trajectory
    loaded_traj = load_traj(save_path)
    print(loaded_traj)
    # Check the number of SE3 objects
    assert len(loaded_traj) == 3

    # Check the first SE3 (should be an identity matrix)
    assert np.allclose(loaded_traj[0].A, np.eye(4))

    # Check the second SE3
    expected_second_matrix = np.array(
        [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    )
    assert np.allclose(loaded_traj[1].A, expected_second_matrix)

    # Check the third SE3
    expected_third_matrix = np.array(
        [[0, 0, 1, 4], [0, 1, 0, 5], [-1, 0, 0, 6], [0, 0, 0, 1]]
    )
    assert np.allclose(loaded_traj[2].A, expected_third_matrix)


def test_cubic_interpolation():
    t = 0.5
    t0 = 0
    tf = 1
    q0 = np.array([0])
    qf = np.array([1])

    q_interp = cubic_interpolation(t, t0, tf, q0, qf)
    expected_q = 0.5
    assert np.allclose(q_interp, expected_q), "Cubic interpolation result mismatch."


def test_jtraj():
    q0 = np.array([0, 0])
    qf = np.array([1, 1])
    t_array = np.linspace(0, 1, 10)

    traj = jtraj(q0, qf, t_array)
    assert traj.shape == (len(t_array), len(q0)), "Trajectory shape mismatch."

    # Check some specific values
    assert np.allclose(traj[0], q0), "Initial joint configuration mismatch."
    assert np.allclose(traj[-1], qf), "Final joint configuration mismatch."
