"""Read-only robot state interface using RTDEReceiveInterface.

This module provides a minimal, read-only wrapper around RTDEReceiveInterface
that does not require RTDEControlInterface or Remote Control mode. It only
reads the current robot state and is suitable for monitoring and recording.
"""

from typing import Optional

import numpy as np
from rtde_receive import RTDEReceiveInterface
from transform3d import Transform as T
import spatialmath as sm

from .config import UR5E_BASE_POS, UR5E_BASE_QUAT, ROBOT_IP
from .utils import make_tf


class RobotState:
    """Read-only wrapper for RTDEReceiveInterface providing robot state access.
    
    Only requires RTDEReceiveInterface and does not require Remote Control mode.
    Suitable for monitoring and recording robot trajectories without needing
    control capabilities.
    """

    def __init__(
        self,
        robot_ip: str = ROBOT_IP,
        T_tcp_gripper_tcp: Optional[sm.SE3] = None,
    ):
        """Initialize the robot state interface.
        
        Parameters
        ----------
        robot_ip : str
            IP address of the UR controller (default: ROBOT_IP from config).
        T_tcp_gripper_tcp : sm.SE3, optional
            Transform from UR flange TCP to active gripper/tool TCP.
            If None, assumes no gripper (identity transform).
        """
        self.recv = RTDEReceiveInterface(robot_ip)
        self.T_tcp_gripper_tcp = (
            T_tcp_gripper_tcp if T_tcp_gripper_tcp is not None else sm.SE3()
        )
        self.has_gripper = not np.allclose(self.T_tcp_gripper_tcp.A, sm.SE3().A)

    @property
    def T_base_tcp(self) -> sm.SE3:
        """Get transform from robot base to TCP (accounting for gripper offset).
        
        Returns
        -------
        sm.SE3
            Transform from robot base to the active TCP (gripper or flange).
        """
        # Get the UR flange TCP pose from the robot
        tcp_pose = self.recv.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        Ti = T.from_xyz_rotvec(tcp_pose)
        
        # Convert to SE3 and apply gripper TCP offset
        T_base_flange = make_tf(pos=Ti.p, ori=Ti.R)
        return T_base_flange @ self.T_tcp_gripper_tcp

    def get_q(self) -> np.ndarray:
        """Get current joint angles.
        
        Returns
        -------
        np.ndarray
            Joint angles [q0, q1, q2, q3, q4, q5] in radians.
        """
        return self.recv.getActualQ()

    @property
    def T_w_base(self) -> sm.SE3:
        """Get transform from world to robot base.
        
        Returns
        -------
        sm.SE3
            Transform from world to robot base.
        """
        return make_tf(pos=UR5E_BASE_POS, ori=UR5E_BASE_QUAT) @ sm.SE3.Rz(np.pi)

    @property
    def T_w_tcp(self) -> sm.SE3:
        """Get transform from world to TCP.
        
        Returns
        -------
        sm.SE3
            Transform from world to the active TCP (gripper or flange).
        """
        return self.T_w_base @ self.T_base_tcp
