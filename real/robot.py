import json
import time
from contextlib import contextmanager
from typing import Optional, Union

import numpy as np
from rtde_control import RTDEControlInterface
from rtde_io import RTDEIOInterface
from rtde_receive import RTDEReceiveInterface
from transform3d import Transform as T
import spatialmath as sm
import spatialmath.base as smb

from scripts.robotiq_gripper import RobotiqGripper

from .config import T_FOLLOW_INIT, UR5E_BASE_POS, UR5E_BASE_QUAT, ROBOTIQ_PORT, ROBOT_IP
from .utils import make_tf, se3_to_pose

_DEFAULT_J_SPEED = 0.3
# _DEFAULT_J_SPEED = 1.0
_DEFAULT_J_ACC = 0.3
# _DEFAULT_J_ACC = 1.0
_DEFAULT_L_SPEED = 0.25
_DEFAULT_L_ACC = 0.3
# _DEFAULT_L_ACC = 1.0



class Robot:
    def __init__(self, robot_ip: str = ROBOT_IP):
        self.ctrl = RTDEControlInterface(robot_ip)
        self.recv = RTDEReceiveInterface(robot_ip)
        self.io = RTDEIOInterface(robot_ip)
        self.hande = RobotiqGripper()
        self.hande.connect(robot_ip, ROBOTIQ_PORT)
        self.hande.activate()
        self.has_gripper = self.hande.is_active()

        self.T_tcp_gripper_tcp = sm.SE3.Tz(0.20) if self.has_gripper else sm.SE3()
        # self.T_tcp_gripper_tcp = sm.SE3.Tz(0.18) if self.has_gripper else sm.SE3()

        self.ctrl.setTcp([0] * 6)


    def moveL(self, pose: sm.SE3, speed: int = _DEFAULT_L_SPEED, acc: int =_DEFAULT_L_ACC):
        pose = pose @ self.T_tcp_gripper_tcp.inv()
        T = se3_to_pose(pose)
        assert self.ctrl.moveL(T, speed, acc)

    def moveJ(self, q: Union[list, np.ndarray], speed: int = _DEFAULT_J_SPEED, acc: int = _DEFAULT_J_ACC):
        assert self.ctrl.moveJ(q, speed, acc)

    def moveJ_IK(self, pose: sm.SE3, speed: int = _DEFAULT_J_SPEED, acc: int = _DEFAULT_J_ACC):
        pose = pose @ self.T_tcp_gripper_tcp.inv()
        T = se3_to_pose(pose)
        assert self.ctrl.moveJ_IK(T, speed, acc)

    def servoJ(self, q: np.ndarray, speed: int = _DEFAULT_J_SPEED, acc: int = _DEFAULT_J_ACC) -> None:
        assert self.ctrl.servoJ(q, speed, acc)

    def servoL(self, pose: sm.SE3, speed: int = _DEFAULT_L_SPEED, acc: int = _DEFAULT_L_SPEED) -> None:
        pose = pose @ self.T_tcp_gripper_tcp.inv()
        dt = 1.0/400  # 2ms
        # dt = 1.0/500  # 2ms
        lookahead_time = 0.1
        gain = 300
        T = se3_to_pose(pose)
        assert self.ctrl.servoL(T.tolist(), speed, acc, dt, lookahead_time, gain)
        # assert self.ctrl.servoL(T, speed, acc)

    @property
    def T_base_tcp(self):
        Ti =  T.from_xyz_rotvec(self.recv.getActualTCPPose())
        return make_tf(pos = Ti.p, ori = Ti.R) @ self.T_tcp_gripper_tcp

    def get_q(self):
        return self.recv.getActualQ()

    def zero_ft(self):
        time.sleep(0.2)  # a sleep is needed to reset properly
        self.ctrl.zeroFtSensor()

    @property
    def T_w_base(self) -> sm.SE3:
        return make_tf(pos=UR5E_BASE_POS, ori=UR5E_BASE_QUAT) @ sm.SE3.Rz(np.pi)

    @property
    def T_w_tcp(self) -> sm.SE3:
        return self.T_w_base @ self.T_base_tcp

    @contextmanager
    def teachmode_ctx(self):
        try:
            self.start_teachmode()
            yield
        finally:
            self.end_techmode()

