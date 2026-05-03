import time
from contextlib import contextmanager
from typing import Optional, Union

import numpy as np
from rtde_control import RTDEControlInterface
from rtde_io import RTDEIOInterface
from rtde_receive import RTDEReceiveInterface
from transform3d import Transform as T
import spatialmath as sm

#from scripts.robotiq_gripper import RobotiqGripper

from .config import T_TCP_GRIPPER_TCP, UR5E_BASE_POS, UR5E_BASE_QUAT, ROBOT_IP
from .utils import make_tf, se3_to_pose

_DEFAULT_J_SPEED = 0.3
# _DEFAULT_J_SPEED = 1.0
_DEFAULT_J_ACC = 0.3
# _DEFAULT_J_ACC = 1.0
_DEFAULT_L_SPEED = 0.25
_DEFAULT_L_ACC = 0.3
# _DEFAULT_L_ACC = 1.0



class Robot:
    def __init__(
        self,
        robot_ip: str = ROBOT_IP,
        T_tcp_gripper_tcp: Optional[sm.SE3] = T_TCP_GRIPPER_TCP,
    ):
        self.ctrl = RTDEControlInterface(robot_ip)
        self.recv = RTDEReceiveInterface(robot_ip)
        self.io = RTDEIOInterface(robot_ip)
        self.T_tcp_gripper_tcp = T_tcp_gripper_tcp if T_tcp_gripper_tcp is not None else sm.SE3()
        self.has_gripper = not np.allclose(self.T_tcp_gripper_tcp.A, sm.SE3().A)

        # self.T_tcp_gripper_tcp = sm.SE3.Tz(0.18) if self.has_gripper else sm.SE3()

        self.ctrl.setTcp([0] * 6)
        self._ensure_control_ready()

    def _ensure_control_ready(self) -> None:
        """Raise a clear error when RTDE control is not ready for motion."""
        is_connected = getattr(self.ctrl, "isConnected", None)
        if callable(is_connected) and not is_connected():
            raise RuntimeError(
                f"Unable to connect to UR RTDE control at {self.ctrl}. "
                "Check the robot IP and network connectivity."
            )

        is_program_running = getattr(self.ctrl, "isProgramRunning", None)
        if callable(is_program_running) and not is_program_running():
            raise RuntimeError(
                "UR RTDE control script is not running. "
                "Start the robot program in Remote Control mode before executing motions."
            )

    def _is_program_running(self) -> Optional[bool]:
        is_program_running = getattr(self.ctrl, "isProgramRunning", None)
        if callable(is_program_running):
            return bool(is_program_running())
        return None

    @staticmethod
    def _raise_motion_failure(name: str) -> None:
        raise RuntimeError(
            f"{name} failed. The UR controller rejected the command; "
            "verify Remote Control mode and that the RTDE control script is running."
        )


    def moveL(self, pose: sm.SE3, speed: int = _DEFAULT_L_SPEED, acc: int =_DEFAULT_L_ACC):
        self._ensure_control_ready()
        pose = pose @ self.T_tcp_gripper_tcp.inv()
        T = se3_to_pose(pose)
        if not self.ctrl.moveL(T, speed, acc):
            self._raise_motion_failure("moveL")

    def moveJ(self, q: Union[list, np.ndarray], speed: int = _DEFAULT_J_SPEED, acc: int = _DEFAULT_J_ACC):
        self._ensure_control_ready()
        if not self.ctrl.moveJ(q, speed, acc):
            self._raise_motion_failure("moveJ")

    def moveJ_IK(self, pose: sm.SE3, speed: int = _DEFAULT_J_SPEED, acc: int = _DEFAULT_J_ACC):
        self._ensure_control_ready()
        pose = pose @ self.T_tcp_gripper_tcp.inv()
        T = se3_to_pose(pose)
        if not self.ctrl.moveJ_IK(T, speed, acc):
            self._raise_motion_failure("moveJ_IK")

    def servoJ(self, q: np.ndarray, speed: int = _DEFAULT_J_SPEED, acc: int = _DEFAULT_J_ACC) -> None:
        self._ensure_control_ready()
        if not self.ctrl.servoJ(q, speed, acc):
            self._raise_motion_failure("servoJ")

    def servoL(
        self,
        pose: sm.SE3,
        speed: int = _DEFAULT_L_SPEED,
        acc: int = _DEFAULT_L_SPEED,
        dt: float = 1.0 / 400,
        lookahead_time: float = 0.1,
        gain: int = 300,
    ) -> None:
        self._ensure_control_ready()
        pose = pose @ self.T_tcp_gripper_tcp.inv()
        T = se3_to_pose(pose)
        if not self.ctrl.servoL(T.tolist(), speed, acc, dt, lookahead_time, gain):
            self._raise_motion_failure("servoL")
        # assert self.ctrl.servoL(T, speed, acc)

    def servo_stop(self) -> None:
        if self._is_program_running() is False:
            return
        self.ctrl.servoStop()

    def stop_script(self) -> None:
        if self._is_program_running() is False:
            return
        self.ctrl.stopScript()

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
