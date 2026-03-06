from __future__ import division, print_function

from typing import Optional, Tuple, Union

import numpy as np
import quaternion  # add-on numpy quaternion type (https://github.com/moble/quaternion)

from ctrl.dmp_cartesian.canonical_system import CanonicalSystem
from utils.math import calculate_rotation_between_vectors, npq2np


class DMPQuaternion:
    """
    Dynamic Movement Primitives (DMP) for orientation in Cartesian space using quaternions.

    Based on:
    [1] A. Ude, B. Nemec, T. Petric, and J. Morimoto, "Orientation in Cartesian
    space dynamic movement primitives", in 2014 IEEE International Conference on
    Robotics and Automation (ICRA), 2014, no. 3, pp 2997-3004.
    """

    def __init__(
        self,
        n_bfs: int = 10,
        alpha: float = 48.0,
        beta: Optional[float] = None,
        cs_alpha: Optional[float] = None,
        cs: Optional[CanonicalSystem] = None,
        roto_dilatation: bool = False,
    ):
        """
        Initialize the DMPQuaternion.

        Parameters
        ----------
        n_bfs : int
            Number of basis functions.
        alpha : float
            Filter constant.
        beta : Optional[float]
            Filter constant. If None, set to alpha / 4.
        cs_alpha : Optional[float]
            Alpha value for the canonical system. If None, set to alpha / 2.
        cs : Optional[CanonicalSystem]
            Canonical system instance. If None, create a new one with cs_alpha.
        roto_dilatation : bool
            Flag to enable roto-dilatation.
        """
        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta if beta is not None else self.alpha / 4
        self.cs = (
            cs
            if cs is not None
            else CanonicalSystem(
                alpha=cs_alpha if cs_alpha is not None else self.alpha / 2
            )
        )

        # Centres of the Gaussian basis functions
        self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c) ** 2

        # Scaling factor
        self.Do = np.identity(3)

        # Initially weights are zero (no forcing term)
        self.w = np.zeros((3, self.n_bfs))

        # Initial and goal orientations
        self._q0 = quaternion.one
        self._go = quaternion.one

        self._q0_train = quaternion.one
        self._go_train = quaternion.one

        self._R_fx = np.identity(3)

        # Reset
        self.q = self._q0.copy()
        self.omega = np.zeros(3)
        self.d_omega = np.zeros(3)
        self.train_quats = None
        self.train_omega = None
        self.train_d_omega = None

        self._roto_dilatation = roto_dilatation

    def step(
        self,
        x: float,
        dt: float,
        tau: float,
        torque_disturbance: np.ndarray = np.array([0, 0, 0]),
    ) -> Tuple[quaternion.quaternion, np.ndarray, np.ndarray]:
        """
        Perform a single DMP step.

        Parameters
        ----------
        x : float
            Phase variable.
        dt : float
            Time step.
        tau : float
            Temporal scaling factor.
        torque_disturbance : np.ndarray, optional
            External torque disturbance, by default np.array([0, 0, 0]).

        Returns
        -------
        Tuple[quaternion.quaternion, np.ndarray, np.ndarray]
            Current quaternion, angular velocity, and angular acceleration.
        """

        def fo(xj):
            psi = np.exp(-self.h * (xj - self.c) ** 2)
            return self.Do.dot(self.w.dot(psi) / psi.sum() * xj)

        # DMP system acceleration
        self.d_omega = (
            self.alpha
            * (
                self.beta * 2 * np.log(self._go * self.q.conjugate()).vec
                - tau * self.omega
            )
            + self._R_fx @ fo(x)
            + torque_disturbance
        ) / tau**2

        # Integrate rotational acceleration
        self.omega += self.d_omega * dt

        # Integrate rotational velocity (to obtain quaternion)
        self.q = np.exp(dt / 2 * np.quaternion(0, *self.omega)) * self.q

        return npq2np(self.q), self.omega, self.d_omega

    def rollout(
        self, ts: np.ndarray, tau: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a DMP rollout over the given time steps.

        Parameters
        ----------
        ts : np.ndarray
            Array of time steps.
        tau : Union[float, np.ndarray]
            Temporal scaling factor.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Arrays of quaternions, angular velocities, and angular accelerations.
        """
        self.reset()

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts)  # Differential time vector

        n_steps = len(ts)
        q = np.empty((n_steps,), dtype=np.quaternion)
        omega = np.empty((n_steps, 3))
        d_omega = np.empty((n_steps, 3))

        for i in range(n_steps):
            q[i], omega[i], d_omega[i] = self.step(x[i], dt[i], tau[i])

        return q, omega, d_omega

    def reset(self) -> None:
        """
        Reset the DMP state.
        """
        self.q = self._q0.copy()
        self.omega = np.zeros(3)
        self.d_omega = np.zeros(3)

    def train(self, quaternions: np.ndarray, ts: np.ndarray, tau: float) -> None:
        """
        Train the DMP with given quaternion trajectory.

        Parameters
        ----------
        quaternions : np.ndarray
            Array of quaternions.
        ts : np.ndarray
            Array of time steps.
        tau : float
            Temporal scaling factor.

        Raises
        ------
        RuntimeError
            If the length of quaternions does not match the length of time steps.
        """
        # View input as numpy quaternion type
        if quaternions.dtype == np.quaternion:
            quats = quaternions
        else:
            quats = quaternion.as_quat_array(quaternions)

        # Sanity-check input
        if len(quats) != len(ts):
            raise RuntimeError("len(quats) != len(ts)")

        # Initial and goal orientations
        self._q0 = quats[0]
        self._go = quats[-1]

        self._q0_train = quats[0]
        self._go_train = quats[-1]

        # Differential time vector
        dt = np.gradient(ts)[:, np.newaxis]

        # Scaling factor
        Do_inv = np.identity(3)

        # Compute finite difference velocity between orientations
        omega = 2 * np.log(np.roll(quats, -1) * quats.conjugate())  # In unit time
        omega[-1] = omega[-2]  # Last element is no good
        omega = quaternion.as_float_array(omega)[:, 1:] / dt  # Scale by dt

        # Compute desired angular accelerations
        d_omega = np.gradient(omega, axis=0) / dt

        # Integrate canonical system at time points
        x = self.cs.rollout(ts, tau)

        # Set up system of equations to solve for weights
        def features(xj):
            psi = np.exp(-self.h * (xj - self.c) ** 2)
            return xj * psi / psi.sum()

        def forcing(j):
            return Do_inv.dot(
                tau**2 * d_omega[j]
                - self.alpha
                * (
                    self.beta * (2 * np.log(self._go * quats[j].conjugate())).vec
                    - tau * omega[j]
                )
            )

        A = np.stack([features(xj) for xj in x])
        f = np.stack([forcing(j) for j in range(len(ts))])

        # Least squares solution for Aw = f (for each column of f)
        self.w = np.linalg.lstsq(A, f, rcond=None)[0].T

        # Cache variables for later inspection
        self.train_quats = quats
        self.train_omega = omega
        self.train_d_omega = d_omega

    def set_trained(
        self,
        w: np.ndarray,
        c: np.ndarray,
        h: np.ndarray,
        q0: quaternion.quaternion,
        go: quaternion.quaternion,
    ) -> None:
        """
        Set trained parameters for the DMP.

        Parameters
        ----------
        w : np.ndarray
            Weight matrix.
        c : np.ndarray
            Centers of the Gaussian basis functions.
        h : np.ndarray
            Variances of the Gaussian basis functions.
        q0 : quaternion.quaternion
            Initial quaternion.
        go : quaternion.quaternion
            Goal quaternion.
        """
        self.w = w
        self.c = c
        self.h = h
        self._q0 = q0
        self._go = go

        # Scaling factor
        self.Do = np.diag((2 * np.log(self._go * self._q0.conjugate())).vec)

    def _update_goal_change_parameters(self) -> None:
        """
        Update parameters when the goal quaternion is changed.
        """
        self._sg = (
            np.log(self._go_train * self._q0_train.conjugate()).norm()
            / np.log(self._go * self._q0.conjugate()).norm()
        )

        v_new = (2 * np.log(self._go * self._q0.conjugate())).vec
        v_train = (2 * np.log(self._go_train * self._q0_train.conjugate())).vec
        self._R_fx = calculate_rotation_between_vectors(v_train, v_new)

    @property
    def go(self) -> quaternion.quaternion:
        """
        Get the goal quaternion.

        Returns
        -------
        quaternion.quaternion
            Goal quaternion.
        """
        return self._go

    @go.setter
    def go(self, value: quaternion.quaternion) -> None:
        """
        Set the goal quaternion and update parameters if roto-dilatation is enabled.

        Parameters
        ----------
        value : quaternion.quaternion
            Goal quaternion.
        """
        self._go = value
        if self._roto_dilatation:
            self._update_goal_change_parameters()

    @property
    def q0(self) -> quaternion.quaternion:
        """
        Get the initial quaternion.

        Returns
        -------
        quaternion.quaternion
            Initial quaternion.
        """
        return self._q0

    @q0.setter
    def q0(self, value: quaternion.quaternion) -> None:
        """
        Set the initial quaternion and update parameters if roto-dilatation is enabled.

        Parameters
        ----------
        value : quaternion.quaternion
            Initial quaternion.
        """
        self._q0 = value
        if self._roto_dilatation:
            self._update_goal_change_parameters()
