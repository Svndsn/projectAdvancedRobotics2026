from collections import deque

import mujoco as mj
import mujoco.viewer
import spatialmath as sm

from robots import BaseRobot
from utils.mj import ObjType, RobotInfo, get_pose


class Twof85(BaseRobot):
    def __init__(self, model: mj.MjModel, data: mj.MjData, args=None) -> None:
        self._args = args
        self._data = data
        self._model = model

        self._info = RobotInfo(self._model, self.name)

        self._info.print()

        self._task_queue = deque()

        self.tcp_name = [sn for sn in self._info.site_names if "tcp" in sn][0]

    def step(self) -> None:
        if self._task_queue:
            ctrl_target = self._task_queue.popleft()
            self.set_ctrl(ctrl_target)

    @property
    def info(self) -> RobotInfo:
        return self._info

    @property
    def data(self) -> mj.MjData:
        """
        Get the MuJoCo data object.

        This property returns the MuJoCo data object associated with the robot. The `mj.MjData`
        object contains the dynamic state of the simulation, including positions, velocities, forces,
        and other simulation-specific data for the robot.

        Returns
        -------
        mj.MjData
            The MuJoCo data object containing the dynamic state of the simulation.
        """
        return self._data

    @property
    def model(self) -> mj.MjModel:
        """
        Get the MuJoCo model object.

        This property returns the MuJoCo model object associated with the robot. The `mj.MjModel`
        object represents the static model of the simulation, including the robot's physical
        structure, joint configurations, and other model-specific parameters.

        Returns
        -------
        mj.MjModel
            The MuJoCo model object representing the robot's static configuration.
        """
        return self._model

    @property
    def name(self) -> str:
        return "2f85"

    def get_ee_pose(self) -> sm.SE3:
        """
        Get the end-effector pose.

        This method retrieves the pose of the robot's end-effector, specifically the TCP (Tool Center
        Point). The pose is returned as an instance of `sm.SE3`, representing the position and
        orientation of the end-effector in 3D space.

        Returns
        -------
        sm.SE3
            The pose of the robot's end-effector (TCP) in 3D space.
        """
        return get_pose(self.model, self.data, self.tcp_name, ObjType.SITE)
