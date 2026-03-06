from pathlib import Path

import glfw
import mujoco as mj
from dm_control import mjcf

from robots.base_robot import BaseRobot
from sims import BaseSim
from sims.base_sim import SimSync
from utils.mj import (
    RobotInfo,
    get_contact_states,
)


class HandE(BaseRobot):
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        super().__init__()
        self._model = model
        self._data = data
        self._name = "hande"
        self._info = RobotInfo(self._model, self._name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def info(self) -> RobotInfo:
        return self._info

    @property
    def step(self) -> None:
        return


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()

        self._model, self._data = self.init()

        self.hande = HandE(self._model, self._data)

        self.tasks = [self.spin]

    def init(self):
        # root
        _HERE = Path(__file__).parent.parent
        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        # shadow hand path
        _XML_HANDE = Path(_HERE / "assets/robotiq_hande/hande.xml")
        hande = mjcf.from_path(_XML_HANDE)

        cube = scene.worldbody.add("body", name="prop", pos="0 0 0.13")
        cube.add("geom", name="prop", type="box", size="0.01 0.01 0.01")

        scene.attach(hande)

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        return m, d

    def spin(self, ss: SimSync):
        while True:
            css = get_contact_states(self._data, self._model, "prop")
            print(css)
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


if __name__ == "__main__":
    sim = MjSim()
    sim.run()
