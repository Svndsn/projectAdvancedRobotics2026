from collections import deque
from typing import List, Union

import mujoco as mj
import numpy as np
import spatialmath as sm

from robots import BaseRobot
from sims.base_sim import SimSync, sleep
from utils.mj import ObjType, RobotInfo, get_pose, set_pose
from utils.sm import ctraj


class Mocap(BaseRobot):
    """
    A controller class for handling motion capture (Mocap) data and robot transformations.
    """

    def __init__(self, model: mj.MjModel, data: mj.MjData, name: str = "mocap") -> None:
        """
        Initialize the Mocap controller.

        Parameters
        ----------
        args : Namespace
            Arguments for the controller.
        robot : Robot
            Robot instance with model and data attributes.
        """
        self._data = data
        self._model = model
        self._name = name

        self._info = RobotInfo(self._model, self._name)

        self._task_queue = deque()

        self.T_target = self.T_world_base

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def info(self) -> RobotInfo:
        return self._info

    def step(self) -> None:
        set_pose(self._model, self._data, self._name, ObjType.BODY, self.T_target)

    @property
    def T_world_base(self) -> sm.SE3:
        """
        Get the transformation matrix from the world frame to the mocap system.

        Returns
        -------
        sm.SE3
            Transformation matrix from world to mocap system.
        """
        return get_pose(self._model, self._data, self._name, ObjType.BODY)

    def move_l(
        self,
        T: sm.SE3,
        ss: SimSync,
        velocity: Union[list, np.ndarray] = 0.25,
        acceleration: Union[list, np.ndarray] = 1.2,
    ) -> bool:
        """
        Move to a given position in task-space (or cartesian space).
        The robot guides the TCP at a defined velocity along a straight path to the end point defined by T.

        Args:
            T (sm.SE3): The desired end-effector pose in the base frame.
            velocity (Union[list, np.ndarray]): tool velocity [m/s]
            acceleration (Union[list, np.ndarray]): tool acceleration [m/s^2]

        Returns:
            success (bool): True if the move succeeds and False otherwise.
        """

        success = True

        T0 = self.T_world_base
        T1 = T

        if T0 == T1:
            success = False
            return success

        delta = T0.delta(T1)
        duration = np.linalg.norm(delta) / velocity

        trajectory_samples = int(duration * int(1 / self._model.opt.timestep))
        t_array = np.linspace(0.0, duration, num=trajectory_samples)
        c_traj = ctraj(T0, T1, t_array)

        # Add task poses to the task queue
        for task_pose in c_traj:
            set_pose(self.model, self.data, self.name, ObjType.BODY, task_pose)
            sleep(self.model.opt.timestep, ss, self.model)
        return success

    def move_traj(self, T: Union[sm.SE3, List], ss: SimSync) -> None:
        """
        Move the robot along a given trajectory.

        This function enqueues a series of task poses for the robot to follow sequentially.

        Args
        ----------
                T (Union[sm.SE3, List]): List of desired end-effector poses in the base frame.
        """
        # add task poses to the robot task queue
        for task_pose in T:
            set_pose(self.model, self.data, self.name, ObjType.BODY, task_pose)
            sleep(self.model.opt.timestep, ss, self.model)