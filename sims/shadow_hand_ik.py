from enum import Enum
from pathlib import Path

import glfw
import mujoco as mj
import numpy as np
import spatialmath as sm
from dm_control import mjcf

from robots import Mocap, ShadowHand
from robots.base_robot import BaseRobot
from sims import BaseSim
from sims.base_sim import SimSync
from utils.mj import (
    ObjType,
    RobotInfo,
    get_names,
    get_pose,
    id2name,
    load_keyframe,
    name2id,
)


class ShadowFinger(BaseRobot):
    class FingerType(Enum):
        FF = "FF"
        MF = "MF"
        RF = "RF"
        LF = "LF"
        TH = "TH"

    def __init__(self, sh: ShadowHand, type: FingerType):
        self.sh = sh
        self._data = sh.data
        self._model = sh.model
        self._type = type
        self._name = sh.name + f"_{type}"
        self._info = RobotInfo(self.model, self.data, self._type)

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
            if self._type.lower() in sna
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


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()
        self._model, self._data = self.init()

        self.sh = ShadowHand(self.model, self.data)

        self.mocap = Mocap(self.model, self.data)

        self.tasks = []
        # self.tasks = [self.script, self.log]

        ff = ShadowFinger(self.sh, ShadowFinger.FingerType.FF)
        mf = ShadowFinger(self.sh, ShadowFinger.FingerType.MF)
        rf = ShadowFinger(self.sh, ShadowFinger.FingerType.RF)
        lf = ShadowFinger(self.sh, ShadowFinger.FingerType.LF)
        th = ShadowFinger(self.sh, ShadowFinger.FingerType.TH)

    def init(self):
        # root
        _HERE = Path(__file__).parent.parent

        # create keyframe file
        self.keyframe_path = Path(_HERE, "keyframes", Path(__file__).stem + ".xml")
        # Ensure the parent directory exists
        self.keyframe_path.parent.mkdir(parents=True, exist_ok=True)

        # Create an empty file if it doesn't exist
        if not self.keyframe_path.exists():
            self.keyframe_path.touch()
            self.keyframe_path.write_text(
                r"""
                <mujoco>

                </mujoco>
                """
            )

        # find keyframes file and load in
        keyframe = mjcf.from_path(self.keyframe_path)

        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        # add keyframes
        scene.attach(keyframe)

        # shadow hand path
        _XML_SHADOW_HAND = Path(_HERE / "assets/shadow_hand/right_hand.xml")
        shadow_hand = mjcf.from_path(_XML_SHADOW_HAND)

        # add sites to finger tips
        for body in shadow_hand.find_all("body"):
            if "distal" in body.name:
                body.add("site", name=f"{body.name}_tcp", pos="0 0 0.026")
                # since the site is placed as the root of the body, I shoft it 2.6 cm to be at the tip
                # source: https://shadow-robot-company-dexterous-hand.readthedocs-hosted.com/en/latest/user_guide/md_finger.html

        mocap_body = scene.worldbody.add(
            "body",
            name="mocap",
            pos="0 0 1",
            mocap="true",
        )
        mocap_geom = mocap_body.add(
            "geom", type="box", size="0.01 0.01 0.01", contype="0", conaffinity="0"
        )
        mocap_site = mocap_body.add("site")
        mocap_site.attach(shadow_hand)

        # spawn cable
        _XML_CABLE = Path(_HERE / "assets/props/cable.xml")
        cable = mjcf.from_path(_XML_CABLE)
        cable_root = cable.find("body", "root")
        cable_root.pos = (1, 1, 0.1)
        scene.attach(cable)

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        return m, d

    def script(self, ss: SimSync):
        print("Move to (0,0,1) in meters...")
        self.mocap.move_l(sm.SE3.Tz(1), ss)
        print("Move to (1,0,0) in meters...")
        self.mocap.move_l(sm.SE3.Tx(1), ss)

        # 6 actuator [5]

        # once moved, make sure that the target of the mocap is where it currently is
        self.mocap.T_target = self.mocap.T_world_base

    def log(self, ss: SimSync):
        for i in range(100):
            print(f"I logged {i}")
            ss.step()

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    def keyboard_callback(self, key: int):
        if key is glfw.KEY_SPACE:
            print("You pressed space...")
            print(self._model.nkey)
            # save_mj_keyframe(self.data, "s0", self.keyframe_path)
            load_keyframe(
                self._model,
                self._data,
                "unnamed_model/s0",
                file_path=self.keyframe_path,
            )

    def control_loop(self):
        self.mocap.step()


if __name__ == "__main__":
    sim = MjSim()
    sim.run()
