import enum
import math
import time as t
from collections import deque
from typing import List, Optional, Tuple, Union

import mujoco as mj
import mujoco.viewer
import numpy as np
import spatialmath as sm
import spatialmath.base as smb
from mujoco import minimize

# from ctrl.dmp_position import DMPPosition
from ctrl.base_ctrl import BaseController
from robots.base_robot import BaseRobot
from sims.base_sim import SimSync, sleep
from utils.mj import (
    ObjType,
    RobotInfo,
    get_pose,
    name2id,
    set_joint_q,
)
from utils.sm import ctraj, jtraj, make_tf


class URRobot(BaseRobot):
    class Type(enum.Enum):
        UR3e = "ur3e"
        UR5e = "ur5e"
        UR10e = "ur10e"

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        robot_type: Type = Type.UR5e,
    ) -> None:
        self._data = data
        self._model = model
        self._robot_type = robot_type.value
        self.dt = self._model.opt.timestep

        self.home_qpos = [2.8, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        self._task_queue = deque()

        self._info = RobotInfo(self._model, self.name)

        self.tcp_id = name2id(self._model, f"{self.name}/attachment_site", ObjType.SITE)
        self._controller = None

    def step(self):
        """
        Perform a step in the controller.

        This method calls the `step()` method of the controller object and
        before doing so it checks if there are any tasks to be performed in
        the robot task queue (URRobot._task_queue)
        """
        if self._task_queue:
            task_pose = self._task_queue.popleft()
            self.set_ee_pose(T=task_pose)

        x = self.controller.step()
        self.set_ctrl(x)

    @property
    def controller(self) -> Optional[BaseController]:
        return self._controller

    @controller.setter
    def controller(self, new_controller) -> None:
        self._controller = new_controller

    @property
    def info(self) -> RobotInfo:
        return self._info

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def name(self) -> str:
        return self._robot_type

    @property
    def type(self) -> str:
        """
        Get the type of the UR robot.

        Returns
        ----------
                Type of the UR robot.
        """
        return self._robot_type

    @property
    def actuator_values(self) -> List[float]:
        """
        Get the values of the actuators.

        Returns
        ----------
                List of actuator values.
        """
        return [self.data.actuator(an).ctrl[0] for an in self.info.actuator_names]

    @property
    def w(self) -> np.ndarray:
        """
        Get the sensor wrench (force,torque) data from the force and toque sensor.

        Returns
        ----------
                np.ndarray: The sensor data as a NumPy array.
        """
        return np.append(
            self.data.sensor(f"{self.name}/force").data,
            self.data.sensor(f"{self.name}/torque").data,
        )

    def fk(self, q: Union[list, np.ndarray]) -> sm.SE3:
        """
        Compute the forward kinematics of the UR robot given joint positions.

        Args
        ----------
                q (Union[list, np.ndarray]): Joint positions.

        Returns
        ----------
                sm.SE3: The pose of the end-effector in the base frame.

        Raises:
                ValueError: If the length of `q` does not match the number of actuators.
        """
        if len(q) != self.info.n_actuators:
            raise ValueError(
                f"Length of q should be {self.info.n_actuators}, q had length {len(q)}"
            )

        # save the original state before performing forward kinematics
        q0 = self.q
        [
            set_joint_q(self._data, self._model, jn, q[i])
            for i, jn in enumerate(self.info.joint_names)
        ]

        # compute forward kinematics
        mj.mj_kinematics(self._model, self._data)

        # build tcp pose in base frame
        T_world_tcp = get_pose(self._model, self._data, self.tcp_id, ObjType.SITE)

        T_base_tcp = (
            get_pose(self.model, self.data, f"{self.name}/base", ObjType.BODY).inv()
            @ T_world_tcp
        )

        # return to pre- forward kinematics state
        [
            set_joint_q(self._data, self._model, jn, q0[i])
            for i, jn in enumerate(self.info.joint_names)
        ]

        return T_base_tcp

    def ik(
        self,
        T: sm.SE3,
        iter: int = 10,
        iter_interpolation: int = 3,
        epsilon: float = 1e-6,
        radius: float = 0.4,
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute the inverse kinematics for the given target pose.

        This function calculates the joint angles required to achieve the specified
        end-effector pose using inverse kinematics. It uses an analytic Jacobian
        and residual function to iteratively solve for the joint angles that minimize
        the difference between the current end-effector pose and the target pose.

        Args:
        ----------
        T : sm.SE3
            The target pose for the end-effector, represented as a transformation matrix.
        iter : int
            Maximum number of iterations for the solver.
        epsilon : float
            Convergence threshold for the distance measure.

        Returns:
        ----------
        Tuple[np.ndarray, bool]
            The joint angles that achieve the target pose and a boolean indicating if the pose was reachable.
        """

        def ik_jac(
            x: np.ndarray,
            res: np.ndarray,
            pos: Optional[np.ndarray] = None,
            quat: Optional[np.ndarray] = None,
            radius: float = 0.04,
            reg: float = 1e-3,
        ) -> np.ndarray:
            """Analytic Jacobian of inverse kinematics residual

            Args:
                x: joint angles.
                pos: target position for the end effector.
                quat: target orientation for the end effector.
                radius: scaling of the 3D cross.

            Returns:
                The Jacobian of the Inverse Kinematics task.
            """
            # least_squares() passes the value of the residual at x which is sometimes
            # useful, but we don't need it here.
            del res

            T = make_tf(pos=pos, ori=quat)

            # Call mj_kinematics and mj_comPos (required for Jacobians).
            mujoco.mj_kinematics(self._model, self._data)
            mujoco.mj_comPos(self._model, self._data)

            # Get Deffector, the 3x3 mju_subquat Jacobian
            effector_quat = np.empty(4)

            mujoco.mju_mat2Quat(effector_quat, self.get_ee_pose().R.flatten())
            target_quat = smb.r2q(T.R)
            # target_quat =  data.body("target").xquat
            Deffector = np.empty((3, 3))
            mujoco.mjd_subQuat(target_quat, effector_quat, None, Deffector)

            # Rotate into target frame, multiply by subQuat Jacobian, scale by radius.
            target_mat = T.R
            mat = radius * Deffector.T @ target_mat.T
            Jo = mat @ self.Jo

            # Regularization Jacobian.
            Jr = reg * np.eye(self.info.n_joints)
            return np.vstack((self.Jp, Jo, Jr))

        def ik_res(
            x: np.ndarray,
            pos: Optional[np.ndarray] = None,
            quat: Optional[np.ndarray] = None,
            radius: float = 0.04,
            reg: float = 1e-3,
            reg_target: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            """Residual for inverse kinematics.

            Args:
                x: joint angles.
                pos: target position for the end effector.
                quat: target orientation for the end effector.
                radius: scaling of the 3D cross.

            Returns:
                The residual of the Inverse Kinematics task.
            """

            T = make_tf(pos=pos, ori=quat)

            # Set qpos, compute forward kinematics.
            res = []

            [
                set_joint_q(self._data, self.info.joint_names[i], x[i])
                for i in range(len(x))
            ]
            mujoco.mj_kinematics(self._model, self._data)

            # Position residual.
            # res_pos = T.t - self.get_ee_pose().t
            res_pos = self.get_ee_pose().t - T.t

            # Effector quat, use mju_mat2quat.
            effector_quat = np.empty(4)
            mujoco.mju_mat2Quat(effector_quat, self.get_ee_pose().R.flatten())
            # mujoco.mju_mat2Quat(effector_quat, data.site("effector").xmat)

            # Target quat, exploit the fact that the site is aligned with the body.
            target_quat = smb.r2q(T.R)
            # target_quat = data.body("target").xquat

            # Orientation residual: quaternion difference.
            res_quat = np.empty(3)
            mujoco.mju_subQuat(res_quat, target_quat, effector_quat)
            res_quat *= radius

            # Regularization residual.
            reg_target = q0_init if reg_target is None else reg_target
            res_reg = reg * (x.flatten() - reg_target.flatten())

            res_i = np.hstack((res_pos, res_quat, res_reg)).T

            res.append(np.atleast_2d(res_i).T)
            return np.hstack(res)

        def distance_measure(pos_error: np.ndarray, ori_error: np.ndarray) -> float:
            """Compute the distance measure for convergence."""
            return np.linalg.norm(pos_error) + np.linalg.norm(ori_error)

        # Define IK problem
        # https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/least_squares.ipynb#scrollTo=mfhCxqQiio5l

        pos = T.t
        quat = smb.r2q(T.R)
        bounds = self.info.joint_limits
        q0_init = self.q
        q0 = self.q
        converged = False
        q_error = np.finfo(np.float64).max

        _T = ctraj(self.get_ee_pose(), T, iter_interpolation)

        t0 = t.perf_counter()

        for Ti in _T:
            pos = Ti.t
            quat = smb.r2q(T.R)
            for _ in range(iter):
                ik_target = lambda x: ik_res(x, pos=pos, quat=quat, reg_target=q0)
                jac_target = lambda x, r: ik_jac(x, r, pos=pos, quat=quat)

                q, _ = minimize.least_squares(
                    q0,
                    ik_target,
                    bounds,
                    jacobian=jac_target,
                    verbose=0,
                )

                [
                    set_joint_q(self._data, self.info.joint_names[i], qi)
                    for i, qi in enumerate(q)
                ]

                mj.mj_kinematics(self._model, self._data)

                pos_error = self.get_ee_pose().t - pos
                effector_quat = np.empty(4)
                mujoco.mju_mat2Quat(effector_quat, self.get_ee_pose().R.flatten())
                res_quat = np.empty(3)
                mujoco.mju_subQuat(res_quat, quat, effector_quat)
                res_quat *= radius
                distance = distance_measure(pos_error, res_quat)

                q_error = q - q0

                # eval steps and epsilon
                if distance < epsilon:
                    converged = True
                    break
                q0 = q

        # reset to current q
        [
            set_joint_q(self._data, self.info.joint_names[i], q0_init[i])
            for i in range(len(q0_init))
        ]

        time = t.perf_counter() - t0

        t_error = self.fk(q).t - pos

        return q, converged, q_error, t_error, {"time": time}

    def get_ee_pose(self) -> sm.SE3:
        """
        Get the end-effector pose for the UR robot.

        Returns
        ----------
                T (sm.SE3): The end-effector pose in the base frame.
        """
        return self.fk(self.q)

    def set_ee_pose(self, T: sm.SE3):
        """
        Set the desired end-effector pose for the UR robot.

        Args
        ----------
                T (sm.SE3): The desired end-effector pose in the base frame.
        """
        self.controller.T_target = T

    def move_l(
        self,
        T: sm.SE3,
        ss: SimSync,
        velocity: Union[list, np.ndarray] = 0.25,
        acceleration: Union[list, np.ndarray] = 1.2,
    ):
        """
        Move to a given position in task-space (or cartesian space)

        The robot guides the TCP at a defined velocity along a straight path to the end point defined by T.

        Args:
                T (sm.SE3): The desired end-effector pose in the base frame.
                velocity (Union[list, np.ndarray]): tool velocity [m/s]
                acceleration (Union[list, np.ndarray]): tool acceleration [m/s^2]

        Returns:
                success (bool): True if the move succeeds and False otherwise.
        """

        if self.controller is None:
            raise Exception("""
                            You attempted to send a command to the robot without a controller, set a controller 
                            ur5e.controller = DiffIk(ur5e)
                            or
                            ur5e.controller = OpSpace(ur5e)
                            """)

        success = True

        # get the tcp pose in base frame
        T_base_tcp: sm.SE3 = self.fk(self.q)
        T0 = T_base_tcp
        T1 = T

        # check if current and target is identical within some tolerance and abort if so.
        if T0 == T1:
            success = False
            return success

        # calculate trajectory duration based on provided velocity and difference of SE(3) matrices as differential motion.
        delta = T0.delta(T1)
        duration = np.linalg.norm(delta) / velocity

        with np.printoptions(precision=3, suppress=True):
            print(
                "> performing move_l:\n  from:\n"
                + str(T0)
                + "\n  to:\n"
                + str(T1)
                + "\n  duration:\t"
                + str("{:.3f}".format(duration))
                + " s"
            )

        trajectory_samples = int(duration * int(1 / self.dt))
        t_array = np.linspace(0.0, duration, num=trajectory_samples)

        c_traj = ctraj(T0, T1, t_array)

        # Add task poses to the task queue
        for task_pose in c_traj:
            self.controller.T_target = task_pose
            x = self.controller.step()
            self.set_ctrl(x)
            sleep(self.model.opt.timestep, ss, self.model)
        return success

    def move_j(
        self,
        q: Union[list, np.ndarray],
        ss: SimSync,
        velocity: Union[list, np.ndarray] = 1.05,
        acceleration: Union[list, np.ndarray] = 1.4,
    ):
        """
        Move to a given joint position in joint-space.

        The robot moves the joints to achieve the fastest path to the end point. The fastest
        path is generally not the shortest path and is thus not a straight line. As the
        motions of the robot axes are rotational, curved paths can be executed faster than
        straight paths. The exact path of the motion cannot be predicted.

        Args:
                q (Union[list, np.ndarray]): q specifies joint positions of the robot axes [radians].
                velocity (Union[list, np.ndarray]): joint velocity [rad/s]
                acceleration (Union[list, np.ndarray]): joint acceleration [rad/s^2]

        Returns:
                success (bool): True if the move succeeds and False otherwise.
        """

        if self.controller is None:
            raise Exception("""
                            You attempted to send a command to the robot without a controller, set a controller 
                            ur5e.controller = DiffIk(ur5e)
                            or
                            ur5e.controller = OpSpace(ur5e)
                            """)

        success = True

        current_q = self.q
        target_q = np.array(q)

        # check if current and target is identical within some tolerance and abort if so.
        if np.allclose(current_q, target_q):
            success = False
            return success

        # calculate trajectory duration based on provided velocity and leading axis movement.
        max_dist = 0.0
        for i in range(0, 6):
            max_dist = max(max_dist, math.fabs(current_q[i] - target_q[i]))
        duration = max_dist / velocity
        with np.printoptions(precision=3, suppress=True):
            print(
                "> performing move_j:\n\tfrom\t\t:\t"
                + str(current_q)
                + "\n\tto\t\t:\t"
                + str(target_q)
                + "\n\tduration\t:\t"
                + str("{:.3f}".format(duration))
                + " s"
            )
        trajectory_samples = int(duration * int(1 / self.dt))
        t_array = np.linspace(0.0, duration, num=trajectory_samples)
        j_traj = jtraj(current_q, target_q, t_array).q

        for joint_q in j_traj:
            task_pose = self.fk(joint_q)
            self.controller.T_target = task_pose
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
            self.controller.T_target = task_pose
            x = self.controller.step()
            self.set_ctrl(x)
            sleep(self.model.opt.timestep, ss, self.model)
