


from typing import List, Union

import numpy as np
import pandas as pd
import spatialmath as sm
import spatialmath.base as smb
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import csv

def sm_print(T: sm.SE3) -> None:
    """
    Prints a transformation matrix in a formatted string.

    Parameters:
    - T (sm.SE3): The transformation matrix to print.

    Returns:
    - None
    """
    t = T.t
    R = T.R
    raw = f"""make_tf(pos = [{t[0]}, {t[1]}, {t[2]}], ori = [
        [{R[0, 0]}, {R[0, 1]}, {R[0, 2]}],
        [{R[1, 0]}, {R[1, 1]}, {R[1, 2]}],
        [{R[2, 0]}, {R[2, 1]}, {R[2, 2]}]
    ])
    """
    print(raw)

def make_tf(
    pos: Union[np.ndarray, list] = [0, 0, 0],
    ori: Union[np.ndarray, sm.SE3, sm.SO3] = [1, 0, 0, 0],
) -> sm.SE3:
    """
    Create an SE3 transformation matrix from the provided position and orientation.

    This function constructs a SE3 transformation matrix that combines a translation vector
    and an orientation. The orientation can be specified in various formats including rotation
    matrices, quaternions, or SE3 objects. The function handles conversion between these forma
ts
    and constructs the final SE3 transformation matrix.

    Parameters
    ----------
    pos : Union[np.ndarray, list], optional
        The translation vector as a list or ndarray with shape (3,). Defaults to [0, 0, 0].
    ori : Union[np.ndarray, sm.SE3, sm.SO3], optional
        The orientation can be a rotation matrix (3x3), quaternion (4,), or SE3 object.
        Defaults to [1, 0, 0, 0].

    Returns
    ----------
    sm.SE3
        The resulting SE3 transformation matrix combining the provided position and orientatio
n.

    Notes
    -----
    - The function handles various input formats for orientation and performs necessary conver
sions.
    - The position and orientation must be compatible with SE3 transformation.
    """

    if isinstance(ori, list):
        ori = np.array(ori)

    if isinstance(ori, sm.SO3):
        ori = ori.R

    if isinstance(pos, sm.SE3):
        pose = pos
        pos = pose.t
        ori = pose.R

    if len(ori) == 9:
        ori = np.reshape(ori, (3, 3))

    # Convert ori to SE3 if it's already a rotation matrix or a quaternion
    if isinstance(ori, np.ndarray):
        if ori.shape == (3, 3):  # Assuming ori is a rotation matrix
            ori = ori
        elif ori.shape == (4,):  # Assuming ori is a quaternion
            ori = sm.UnitQuaternion(s=ori[0], v=ori[1:]).R
        elif ori.shape == (3,):  # Assuming ori is rpy
            ori = sm.SE3.Eul(ori, unit="rad").R

    T_R = smb.r2t(ori) if is_R_valid(ori) else smb.r2t(make_R_valid(ori))
    R = sm.SE3(T_R, check=False).R

    # Combine translation and orientation
    T = sm.SE3.Rt(R=R, t=pos, check=False)

    return T


def is_R_valid(R: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if the given matrix is a valid 3x3 rotation matrix.

    This function verifies that the provided matrix is a valid rotation matrix by checking
    its orthogonality and ensuring that its determinant is close to 1. The function uses a
    tolerance level to account for numerical inaccuracies.

    Parameters
    ----------
    R : np.ndarray
        The matrix to be checked.
    tol : float, optional
        Tolerance for numerical comparison. Defaults to 1e-8.

    Returns
    ----------
    bool
        True if the matrix is a valid rotation matrix, False otherwise.

    Raises
    ------
    ValueError
        If the input matrix is not 3x3.

    Notes
    -----
    - The function performs orthogonality check and determinant check.
    """
    # Check if R is a 3x3 matrix
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        raise ValueError(f"Input is not a 3x3 matrix. R is \n{R}")

    # Check if R is orthogonal
    is_orthogonal = np.allclose(np.dot(R.T, R), np.eye(3), atol=tol)

    # Check if the determinant is 1
    det = np.linalg.det(R)

    return is_orthogonal and np.isclose(det, 1.0, atol=tol)




def is_ori_valid(ori: Union[np.ndarray, sm.SE3] = [1, 0, 0, 0]) -> bool:
    """
    Check if the input orientation representation is valid.

    This function verifies if the provided orientation is valid, which can be in the form of
    a rotation matrix, quaternion, or Euler angles. It checks the validity of the rotation
    matrix derived from these representations.

    Parameters
    ----------
    ori : Union[np.ndarray, sm.SE3], optional
        The orientation representation to be checked. Could be a rotation matrix (3x3),
        quaternion (4,), or Euler angles (3,). Defaults to [1, 0, 0, 0].

    Returns
    ----------
    bool
        True if the orientation representation is valid, False otherwise.

    Raises
    ------
    ValueError
        If the input orientation is not of a recognized format or invalid dimensions.

    Notes
    -----
    - The function performs conversion to rotation matrix if necessary and validates it.
    """
    if isinstance(ori, np.ndarray):
        if ori.shape == (3, 3):  # Assuming ori is a rotation matrix
            R = ori
        elif ori.shape == (4,):  # Assuming ori is a quaternion
            R = sm.UnitQuaternion(s=ori[0], v=ori[1:]).R
        elif ori.shape == (3,):  # Assuming ori is Euler angles
            R = sm.SE3.Eul(ori, unit="rad").R
        else:
            raise ValueError(f"Invalid array shape for orientation: {ori.shape}")
    elif isinstance(ori, sm.SE3):
        R = ori.R
    else:
        raise ValueError("Unsupported type for orientation")

    return is_R_valid(R)



def make_R_valid(R: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Make the input matrix a valid 3x3 rotation matrix.

    This function corrects the input matrix to ensure it is a valid rotation matrix. It
    uses Gram-Schmidt orthogonalization and adjusts the determinant to be positive.

    Parameters
    ----------
    R : np.ndarray
        The matrix to be corrected.
    tol : float, optional
        Tolerance for numerical comparison. Defaults to 1e-6.

    Returns
    ----------
    np.ndarray
        A valid 3x3 rotation matrix derived from the input matrix.

    Raises
    ------
    ValueError
        If the input matrix cannot be made a valid rotation matrix.

    Notes
    -----
    - The function performs orthogonalization and adjusts the determinant if necessary.
    """

    if is_R_valid(R):
        return R

    if not isinstance(R, np.ndarray):
        R = np.array(R)

    # Check if R is a 3x3 matrix
    if R.shape != (3, 3):
        raise ValueError("Input is not a 3x3 matrix")

    # Step 1: Gram-Schmidt Orthogonalization
    Q, _ = np.linalg.qr(R)

    # Step 2: Ensure determinant is 1
    det = np.linalg.det(Q)
    if np.isclose(det, 0.0, atol=tol):
        raise ValueError("Invalid rotation matrix (determinant is zero)")

    # Step 3: Ensure determinant is positive
    if det < 0:
        Q[:, 2] *= -1

    return Q

def se3_to_pose(se3_matrix: sm.SE3) -> np.ndarray:
    """
    Convert an SE(3) matrix into a pose of the form [position, orientation],
    where the orientation is a rotation vector.

    Parameters
    ----------
    se3_matrix : SE3
        The spatialmath SE(3) matrix.

    Returns
    -------
    np.ndarray
        A 6-element array where the first 3 elements are the position (x, y, z)
        and the last 3 elements are the rotation vector (rx, ry, rz).
    """
    if not isinstance(se3_matrix, sm.SE3):
        raise ValueError("Input must be a spatialmath SE3 object.")

    # Extract the position from the SE(3) matrix
    position = se3_matrix.t  # Returns a (3,) NumPy array [x, y, z]

    # Extract the rotation matrix from the SE(3) matrix
    rotation_matrix = se3_matrix.R  # Returns a (3, 3) NumPy array

    # Convert the rotation matrix to a rotation vector
    rotation = R.from_matrix(rotation_matrix)
    rotation_vector = rotation.as_rotvec()  # Returns a (3,) NumPy array [rx, ry, rz]

    # Combine position and rotation vector into a single pose
    pose = np.hstack((position, rotation_vector))
    return pose
