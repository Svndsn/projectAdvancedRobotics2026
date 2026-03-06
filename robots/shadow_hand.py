import enum
from collections import deque
from pathlib import Path
from typing import List, Tuple, Union

import mujoco as mj
import numpy as np
from dm_control import mjcf

from robots import BaseRobot
from sims import BaseSim
from sims.base_sim import SimSync, sleep
from utils.mj import ObjType, RobotInfo, get_names, get_pose, id2name, name2id
from utils.sm import jtraj


class ShadowHand(BaseRobot):
    class Type(enum.Enum):
        RIGHT = "right"
        LEFT = "left"

    class ShadowFinger(BaseRobot):
        class FingerType(enum.Enum):
            FF = "FF"
            MF = "MF"
            RF = "RF"
            LF = "LF"
            TH = "TH"

        def __init__(self, sh: "ShadowHand", type: FingerType):
            self._data = sh.data
            self._model = sh.model
            self._type = type
            self._name = sh.name + f"_{type}"
            self._info = RobotInfo(self.model, self._type.value)

            # custom naming
            self._info.body_names = [
                bna for bna in get_names(self._model, ObjType.BODY) if "ff" in bna
            ]
            self._info.body_ids = [
                name2id(self._model, bna, ObjType.BODY) for bna in self._info.body_names
            ]

            self._info.geom_ids = [
                geom_id
                for geom_id in range(self._model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) in self.info.body_ids
            ]
            self._info.geom_names = [
                id2name(self._model, gid, ObjType.GEOM) for gid in self._info.geom_ids
            ]

            self._info.site_names = [
                sna
                for sna in get_names(self._model, ObjType.SITE)
                if self._type.value.lower() in sna
            ]
            self._info.site_ids = [
                name2id(self._model, sna, ObjType.SITE) for sna in self._info.site_names
            ]

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
                get_pose(
                    self.model,
                    self.data,
                    f"{self.sh.name}/rh_{self._type.lower()}knuckle",
                    ObjType.BODY,
                ).R
                @ sys_J[3:, :]
            )
            sys_J[:3, :] = (
                get_pose(
                    self.model,
                    self.data,
                    f"{self.sh.name}/rh_{self._type.lower()}knuckle",
                    ObjType.BODY,
                ).R
                @ sys_J[:3, :]
            )
            return sys_J

        @property
        def data(self) -> mj.MjData:
            return self._data

        @property
        def model(self) -> mj.MjModel:
            return self._model

        @property
        def info(self) -> RobotInfo:
            return self._info

        @property
        def name(self) -> str:
            return self._name

        @property
        def step(self) -> None:
            return

        def set_ctrl(
            self, x: Union[List, np.ndarray], ss: SimSync, time: float = 5.0
        ) -> None:
            assert len(x) == self.info.n_actuators

            if isinstance(x, list):
                x = np.array(x)

            duration = time
            trajectory_samples = int(duration * int(1 / self.model.opt.timestep))
            t_array = np.linspace(0.0, duration, num=trajectory_samples)
            a_limits = self.info.actuator_limits.T

            Q = jtraj(self.ctrl, x, t_array, interp_type="linear")

            x_total = x[0] + x[1]
            is_curling = np.sum(x[:2]) > np.sum(self.ctrl[:2])
            in_overflow = x_total >= a_limits[1][1]

            half_duration = duration / 2
            half_samples = trajectory_samples // 2
            t_array_1 = np.linspace(0.0, half_duration, num=half_samples)

            if is_curling:
                if not in_overflow:
                    x[1], x[0] = x_total, 0
                    Q[:, 0] = np.zeros_like(t_array)
                    Q[:, 1] = jtraj(
                        self.ctrl[1], x_total, t_array, interp_type="linear"
                    ).flatten()
                else:  # is overflow
                    # Fill out first phase
                    Q[: len(t_array_1), 1] = jtraj(
                        self.ctrl[1],
                        a_limits[1][1],
                        t_array_1,
                        interp_type="linear",
                    ).flatten()
                    Q[: len(t_array_1), 0] = np.zeros_like(t_array_1)

                    # Fill out second phase
                    Q[len(t_array_1) :, 1] = (
                        np.ones(len(t_array) - len(t_array_1)) * a_limits[1][1]
                    )
                    Q[len(t_array_1) :, 0] = jtraj(
                        self.ctrl[0],
                        x_total - a_limits[1][1],
                        t_array[len(t_array_1) :],
                        interp_type="linear",
                    ).flatten()

            else:  # is extending
                in_overflow = x_total > a_limits[1][1]
                is_overflow = np.sum(self.ctrl[:2]) > a_limits[1][1]

                if in_overflow and is_overflow:
                    # When we extend from a fully curled position
                    Q[:, 1] = np.ones_like(t_array) * self.ctrl[1]
                    Q[:, 0] = jtraj(
                        self.ctrl[0],
                        np.sum(self.ctrl[:2]) - x_total,
                        t_array,
                        interp_type="linear",
                    ).flatten()

                if not in_overflow and is_overflow:
                    # Fill out first phase
                    Q[: len(t_array_1), 0] = jtraj(
                        self.ctrl[0],
                        a_limits[0][0],
                        t_array_1,
                        interp_type="linear",
                    ).flatten()
                    Q[: len(t_array_1), 1] = np.ones_like(t_array_1) * self.ctrl[1]

                    # Fill out second phase
                    Q[len(t_array_1) :, 0] = (
                        np.ones(len(t_array) - len(t_array_1)) * a_limits[0][0]
                    )

                    Q[len(t_array_1) :, 1] = jtraj(
                        self.ctrl[1],
                        x_total,
                        t_array[len(t_array_1) :],
                        interp_type="linear",
                    ).flatten()

            def _set_ctrl(q):
                for i, qi in enumerate(q):
                    self.data.actuator(self.info.actuator_ids[i]).ctrl = qi

            # execute Q
            for q in Q:
                _set_ctrl(q)
                sleep(self.model.opt.timestep, ss, self.model)

    def __init__(
        self, model: mj.MjModel, data: mj.MjData, type: Type = Type.RIGHT, args=None
    ) -> None:
        self._args = args
        self._data = data
        self._model = model
        self._name = f"{type.value}_shadow_hand"

        self._info = RobotInfo(self._model, self.name)

        self.ff = self.ShadowFinger(self, self.ShadowFinger.FingerType.FF)
        self.mf = self.ShadowFinger(self, self.ShadowFinger.FingerType.MF)
        self.rf = self.ShadowFinger(self, self.ShadowFinger.FingerType.RF)
        self.lf = self.ShadowFinger(self, self.ShadowFinger.FingerType.LF)
        self.th = self.ShadowFinger(self, self.ShadowFinger.FingerType.TH)

        self._task_queue = deque()

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def name(self) -> str:
        return self._name

    @property
    def info(self) -> RobotInfo:
        return self._info

    def step(self) -> None:
        if self._task_queue:
            ctrl_target = self._task_queue.popleft()
            self.set_ctrl(ctrl_target)


if __name__ == "__main__":

    class MjSim(BaseSim):
        def __init__(self):
            super().__init__()

            self._model, self._data = self.init()

            sh = ShadowHand(self._model, self._data)

            sh.info.print()

        def init(self) -> Tuple[mj.MjModel, mj.MjData]:
            # root
            _HERE = Path(__file__).parent.parent
            # scene path
            _XML_SCENE = Path(_HERE / "scenes/empty.xml")
            scene = mjcf.from_path(_XML_SCENE)

            # shadow hand path
            _XML_SHADOW_HAND = Path(_HERE / "assets/shadow_hand/shadow_rh.xml")
            shadow_hand = mjcf.from_path(_XML_SHADOW_HAND)

            scene.attach(shadow_hand)

            m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
            d = mj.MjData(m)

            # step once to compute the poses of objects
            mj.mj_step(m, d)

            return m, d

        @property
        def data(self):
            return self._data

        @property
        def model(self):
            return self._model

    sim = MjSim()

    sim.run()
