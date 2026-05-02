from typing import List, Optional, Tuple

import numpy as np
import spatialmath as sm
import spatialmath.base as smb

from ctrl.dmp_cartesian.canonical_system import CanonicalSystem
from ctrl.dmp_position import DMPPosition
from ctrl.dmp_quaternion import DMPQuaternion
import quaternion


class DMPCartesian:
    def __init__(
        self,
        n_bfs: int = 100,
        alpha: float = 100,
        beta: Optional[float] = None,
        cs_alpha: Optional[float] = -np.log(0.0001),
        cs: Optional[CanonicalSystem] = None,
        roto_dilatation: bool = False,
    ) -> None:
        """
        Initialize the DMPCartesian object.

        Parameters:
        n_bfs (int): Number of basis functions.
        alpha (float): Alpha parameter for DMP.
        beta (Optional[float]): Beta parameter for DMP. Defaults to alpha / 4 if not provided.
        cs_alpha (Optional[float]): Alpha parameter for the canonical system.
        cs (Optional[CanonicalSystem]): Custom canonical system.
        roto_dilatation (bool): Flag for roto-dilatation.
        """

        self.n_bfs = n_bfs
        self.cs = (
            cs
            if cs is not None
            else CanonicalSystem(alpha=cs_alpha if cs_alpha is not None else alpha / 2)
        )

        # Centres of the Gaussian basis functions
        self.position_dmp = DMPPosition(
            n_bfs, alpha, beta, cs=self.cs, roto_dilatation=roto_dilatation
        )
        self.quaternion_dmp = DMPQuaternion(
            n_bfs, alpha, beta, cs=self.cs, roto_dilatation=roto_dilatation
        )

        self.dt = None
        self.tau = None
        self.ts = None
        self.trained = False

        self.reset()

    def step(
        self,
        x: float,
        dt: float,
        tau: float,
        force_disturbance: np.ndarray = np.array([0, 0, 0]),
        torque_disturbance: np.ndarray = np.array([0, 0, 0]),
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.quaternion, np.ndarray, np.ndarray
    ]:
        """
        Perform a single DMP step.

        Parameters:
         - x (float): Phase variable.
         - dt (float): Time step.
         - tau (float): Temporal scaling factor.
         - force_disturbance (np.ndarray): External force disturbance.
         - torque_disturbance (np.ndarray): External torque disturbance.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.quaternion, np.ndarray, np.ndarray]:
            Position, velocity, and acceleration of position DMP.
            Orientation, angular velocity, and angular acceleration of quaternion DMP.
        """
        p, dp, ddp = self.position_dmp.step(x, dt, tau, force_disturbance)
        q, omega, d_omega = self.quaternion_dmp.step(x, dt, tau, torque_disturbance)
        return p, dp, ddp, q, omega, d_omega

    def rollout(
        self, ts: np.ndarray, tau: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a DMP rollout over time.

        Parameters:
         - ts (np.ndarray): Time steps.
         - tau (float): Temporal scaling factor.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Position, velocity, and acceleration of position DMP.
            Orientation, angular velocity, and angular acceleration of quaternion DMP.
        """
        self.reset()

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts)  # Differential time vector

        n_steps = len(ts)
        p = np.empty((n_steps, 3))
        dp = np.empty((n_steps, 3))
        ddp = np.empty((n_steps, 3))

        q = np.empty((n_steps,), dtype=np.quaternion)
        omega = np.empty((n_steps, 3))
        d_omega = np.empty((n_steps, 3))

        for i in range(n_steps):
            p[i], dp[i], ddp[i] = self.position_dmp.step(x[i], dt[i], tau[i])
            q[i], omega[i], d_omega[i] = self.quaternion_dmp.step(x[i], dt[i], tau[i])

        return p, dp, ddp, q, omega, d_omega

    def reset(self) -> None:
        """
        Reset the DMP to its initial state.
        """
        self.position_dmp.reset()
        self.quaternion_dmp.reset()
        self.cs.reset()

    def train(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        ts: np.ndarray,
        tau: float,
        T: List[sm.SE3] = None,
    ) -> None:
        """
        Train the DMP with given positions and quaternions.

        Parameters:
         - positions (np.ndarray): Training positions.
         - quaternions (np.ndarray): Training orientations as quaternions.
         - ts (np.ndarray): Time steps.
         - tau (float): Temporal scaling factor.
        """

        if T is not None:
            positions = np.array([Ti.t for Ti in T])
            quaternions = np.array([smb.r2q(Ti.R) for Ti in T])

        self.position_dmp.train(positions, ts, tau)
        self.quaternion_dmp.train(quaternions, ts, tau)
        self.trained = True

    def load(self, Traj: sm.SE3, dt: float = 0.05) -> None:
        """
        Load and train the DMP with a trajectory.

        Parameters:
         - Traj (sm.SE3): Trajectory as an SE3 object.
         - dt (float): Time step size.
        """
        positions = np.array([pose.t for pose in Traj])
        quaternions = np.array([smb.r2q(pose.R) for pose in Traj])

        for i in range(1, len(quaternions)):
            quaternions[i] = (
                -quaternions[i]
                if np.dot(quaternions[i][1:], quaternions[i - 1][1:]) < 0
                else quaternions[i]
            )

        self.dt = dt
        self.tau = (len(positions)) * self.dt
        self.ts = np.arange(0, self.tau, self.dt)
        self.ts = self.ts[: len(positions)]
        print(positions.shape, quaternions.shape, self.ts.shape)
        self.train(positions, quaternions, self.ts, self.tau)
        self.reset()

    def load_with_time(self, Traj: sm.SE3, ts: np.ndarray) -> None:
        """
        Load and train the DMP with a trajectory and explicit timestamp vector.

        This method trains the DMP from SE3 poses using the provided timestamp vector
        instead of assuming a fixed dt. Useful when training from trajectories with
        non-uniform sampling or known timestamps.

        Parameters:
         - Traj (sm.SE3): Trajectory as an SE3 object.
         - ts (np.ndarray): Timestamp vector for trajectory points.
        """
        positions = np.array([pose.t for pose in Traj])
        quaternions = np.array([smb.r2q(pose.R) for pose in Traj])

        for i in range(1, len(quaternions)):
            quaternions[i] = (
                -quaternions[i]
                if np.dot(quaternions[i][1:], quaternions[i - 1][1:]) < 0
                else quaternions[i]
            )

        self.ts = ts
        self.tau = ts[-1] if len(ts) > 0 else 1.0
        self.dt = np.mean(np.gradient(ts)) if len(ts) > 1 else None
        print(positions.shape, quaternions.shape, self.ts.shape)
        self.train(positions, quaternions, self.ts, self.tau)
        self.reset()

    def is_trained(self) -> bool:
        return self.trained
        # if self.dt is None and self.tau is None and self.ts is None:
        #     return False
        # else:
        #     return True

    def set_start_goal(self, start: sm.SE3, goal: sm.SE3) -> None:
        """
        Set a new start and goal pose for the trained DMPs.

        This updates both the position and orientation DMPs so the learned
        forcing terms can be reused with different start/goal SE3 poses.
        """
        # positions
        self.position_dmp.p0 = np.array(start.t)
        self.position_dmp.gp = np.array(goal.t)

        # orientations: convert rotation matrices to quaternion scalar-first
        q0_arr = smb.r2q(start.R)
        go_arr = smb.r2q(goal.R)
        # quaternion.quaternion(w, x, y, z)
        self.quaternion_dmp.q0 = quaternion.quaternion(q0_arr[0], q0_arr[1], q0_arr[2], q0_arr[3])
        self.quaternion_dmp.go = quaternion.quaternion(go_arr[0], go_arr[1], go_arr[2], go_arr[3])
