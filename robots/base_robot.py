from abc import ABC, abstractmethod
from typing import List, Union

import mujoco as mj
import numpy as np

from utils.mj import (
    ObjType,
    RobotInfo,
    get_joint_ddq,
    get_joint_dq,
    get_joint_q,
    get_pose,
)


class BaseRobot(ABC):
    """
    Base class for robot simulation in MuJoCo.

    This class provides a framework for simulating robots in MuJoCo environments. It defines
    key properties and methods that should be implemented in child classes, including access
    to the robot's model, data, and control mechanisms.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the robot.

        This property returns the name of the robot as a string. The name is typically a unique identifier
        used to distinguish between different robots in the simulation environment.

        Returns
        -------
        str
            The name of the robot as a string.
        """
        raise NotImplementedError("property 'name' must be implemented in robot.")

    @property
    @abstractmethod
    def data(self) -> mj.MjData:
        """
        Access the current simulation data.

        This property provides access to an instance of the `MjData` class, which contains the dynamic
        simulation state. This includes quantities such as joint positions, velocities,
        actuator forces, and sensory information. The `MjData` object is updated at each simulation step
        and can be used to inspect the real-time state of the robot during the simulation.

        Returns
        -------
        mj.MjData
            An object representing the current dynamic state of the simulation.
        """
        raise NotImplementedError("property 'data' must be implemented in robot.")

    @property
    @abstractmethod
    def model(self) -> mj.MjModel:
        """
        Access the model of the MuJoCo simulation.

        This property returns an instance of the `MjModel` class, which describes the physical and
        mechanical properties of the simulation. The `MjModel` object contains static information about the
        robot such as its kinematic tree, inertial properties, joint and actuator definitions, and geometry
        configurations. It is used to define the robot's structure and behavior within the simulation.

        Returns
        -------
        mj.MjModel
            An object representing the static model of the robot and overall MuJoCo simulation.
        """
        raise NotImplementedError("property 'model' must be implemented in robot.")

    @property
    @abstractmethod
    def info(self) -> RobotInfo:
        """
        Get detailed information about the robot.

        This property returns an instance of the `RobotInfo` class, which provides comprehensive
        details about the robot's structure and components. This includes information on the robot's
        bodies, joints, actuators, and geometries, among other attributes. The `RobotInfo` instance
        can be used to access various properties such as the number of joints, actuator limits, joint
        limits, and more.

        Returns
        -------
        RobotInfo
            An object containing detailed information about the robot's configuration and components.
        """
        raise NotImplementedError(
            "property 'info' of type RobotInfo must be implemented in robot."
        )

    @abstractmethod
    def step(self) -> None:
        """
        Perform a step in the controller.

        This method calls the `step()` method of the controller object and
        before doing so it checks if there are any tasks to be performed in
        the robot task queue
        """
        raise NotImplementedError("method 'step' must be implemented in robot.")

    def set_ctrl(self, x: Union[list, np.ndarray]) -> None:
        """
        This function sends the control signal to the simulated robot.

        Args
        ----------
                x (Union[list, np.ndarray]): control signal
        """
        assert len(x) == self.info.n_actuators
        for i, xi in enumerate(x):
            self.data.actuator(self.info.actuator_ids[i]).ctrl = xi

    @property
    def ctrl(self) -> List[float]:
        """
        The control signal sent to the robot's actuator(s).
        """
        return np.array(
            [self.data.actuator(aid).ctrl for aid in self.info._actuator_ids]
        )

    @property
    def Jp(self) -> np.ndarray:
        """
        Get the position Jacobian in base frame.

        Returns
        ----------
                Position Jacobian as a numpy array.
        """
        return self.J[:3, :]

    @property
    def Jo(self) -> np.ndarray:
        """
        Get the orientation Jacobian in base frame.

        Returns
        ----------
                Orientation Jacobian as a numpy array.
        """
        # Jacobian.
        return self.J[3:, :]

    @property
    def J(self) -> np.ndarray:
        """
        Get the full Jacobian in base frame.

        Returns
        ----------
                Full Jacobian as a numpy array.
        """
        sys_J = np.zeros((6, self.model.nv))

        mj.mj_jacSite(
            self.model,
            self.data,
            sys_J[:3],
            sys_J[3:],
            self.info.site_ids[0],
            # name2id(self.model, f"{self.name}/{self.info.site_names[0]}", ObjType.SITE),
        )

        # get only the dofs for this robot
        sys_J = sys_J[:, self.info._dof_indxs].reshape(6, -1)

        # convert from world frame to base frame
        sys_J[3:, :] = (
            get_pose(self.model, self.data, f"{self.name}/base", ObjType.BODY).R
            @ sys_J[3:, :]
        )
        sys_J[:3, :] = (
            get_pose(self.model, self.data, f"{self.name}/base", ObjType.BODY).R
            @ sys_J[:3, :]
        )
        return sys_J

    @property
    def c(self) -> np.ndarray:
        """
        bias force: Coriolis, centrifugal, gravitational
        """
        return self.data.qfrc_bias[np.ravel(self.info._dof_indxs)]

    @property
    def Mq(self) -> np.ndarray:
        """
        Getter property for the inertia matrix M(q) in joint space.

        Returns
        ----------
        - numpy.ndarray: Symmetric inertia matrix in joint space.
        """
        sys_Mq_inv = np.zeros((self.model.nv, self.model.nv))

        mj.mj_solveM(self.model, self.data, sys_Mq_inv, np.eye(self.model.nv))

        dof_indices = np.ravel(self.info._dof_indxs)  # Flatten to 1D if not already
        Mq_inv = sys_Mq_inv[np.ix_(dof_indices, dof_indices)]

        if abs(np.linalg.det(Mq_inv)) >= 1e-2:
            self._Mq = np.linalg.inv(Mq_inv)
        else:
            self._Mq = np.linalg.pinv(Mq_inv, rcond=1e-2)
        return self._Mq

    @property
    def Mx(self) -> np.ndarray:
        """
        Getter property for the inertia matrix M(q) in task space.

        Returns
        ----------
        - numpy.ndarray: Symmetric inertia matrix in task space.
        """
        Mx_inv = self.J @ np.linalg.inv(self.Mq) @ self.J.T

        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            self._Mx = np.linalg.inv(Mx_inv)
        else:
            self._Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        return self._Mx

    @property
    def q(self) -> np.ndarray:
        """
        Get the joint positions.

        Returns
        ----------
                Joint positions as a numpy array.
        """
        q = np.array(
            [get_joint_q(self.data, self.model, jn) for jn in self.info._joint_ids]
        ).flatten()
        return q

    @property
    def dq(self) -> np.ndarray:
        """
        Get the joint velocities.

        Returns
        ----------
                Joint velocities as a numpy array.
        """
        dq = np.array(
            [get_joint_dq(self.data, self.model, jn) for jn in self.info._joint_ids]
        ).flatten()
        return dq

    @property
    def ddq(self) -> np.ndarray:
        """
        Get the joint accelerations.

        Returns
        ----------
                Joint accelerations as a numpy array.
        """
        ddq = np.array(
            [get_joint_ddq(self.data, self.model, jn) for jn in self.info._joint_ids]
        ).flatten()
        return ddq
