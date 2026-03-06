from typing import Tuple, Union

import jaxlie as jaxl
import mujoco.mjx as mjx
from jax import numpy as jp
from scipy.spatial.transform import Rotation

from utils.mj import ObjType, name2id


def get_pose(
    model: mjx.Model, data: mjx.Data, identifier: Union[str, int], obj_type: ObjType
) -> jaxl.SE3:
    pose_mapping = {
        # ObjType.ACTUATOR: data.actuator,
        # ObjType.BODY: data.body,
        # ObjType.JOINT: data.joint,
        ObjType.GEOM: (data.geom_xpos, data.geom_xmat),
        ObjType.SITE: (data.site_xpos, data.site_xmat),
        # ObjType.CAMERA: data.cam,
        # ObjType.LIGHT: data.light,
    }
    # Check if type has a pose; raise an error if it does not
    if obj_type not in pose_mapping:
        raise ValueError(f"obj_type {obj_type.name} cannot provide a pose in mjx...")

    if isinstance(identifier, str):
        identifier: int = name2id(model, identifier, obj_type)

    T = make_tf(
        pos=pose_mapping[obj_type][0][identifier],
        ori=pose_mapping[obj_type][1][identifier],
    )

    return T


def make_tf(pos: jp.ndarray, ori: jp.ndarray) -> jaxl.SE3:
    """
    Creates a jaxlie SE3 transformation from position and orientation.

    Orientation (ori) can be:
    - A rotation matrix (1x9 or 3x3)
    - A quaternion (1x4, [x, y, z, w])
    - Euler angles (1x3, [roll, pitch, yaw])
    """
    if ori.shape == (9,) or ori.shape == (3, 3):
        R = ori.reshape(3, 3)
    elif ori.shape == (4,):
        R = Rotation.from_quat(ori).as_matrix()
    elif ori.shape == (3,):
        R = Rotation.from_euler("xyz", ori).as_matrix()
    else:
        raise ValueError(
            "Invalid orientation shape. Expected (9,), (3,3), (4,), or (3,)."
        )

    T = jp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(pos)

    return jaxl.SE3.from_matrix(T)


def get_contact_states(model: mjx.Model, data: mjx.Data):
    raise NotImplementedError("Have not come around to implementing this yet :P...")


def pose_error(T_current: jaxl.SE3, T_goal: jaxl.SE3) -> Tuple[float, float]:
    """
    Computes the pose error between two SE3 transformations i.e. (displacement, angle (from axis-angle) ).

    The error is defined as the sum of:
    - The Euclidean distance between translations.
    - The magnitude of the axis-angle representation of the rotational difference.
    """
    # Compute relative transformation
    T_rel = T_current.inverse() @ T_goal

    # Translation error (Euclidean distance)
    trans_error = jp.linalg.norm(T_rel.translation())

    # Rotation error (axis-angle magnitude)
    rot_error = jp.linalg.norm(T_rel.rotation().log())

    return jp.array([trans_error, rot_error])
