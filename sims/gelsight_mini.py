from pathlib import Path

import glfw
import mujoco as mj
from dm_control import mjcf

from sensors.gelsight_mini.gelsight_mini import GelSightMini
from sims import BaseSim
from sims.base_sim import SimSync


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()

        self._model, self._data = self.init()

        # self.cam = Camera(self._model, self._data, "gelsight_mini/gelsight_mini")
        self.gs = GelSightMini(self._model, self._data)

        self.tasks = [self.spin]

    def init(self):
        # root
        _HERE = Path(__file__).parent.parent
        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        _XML_GS_MINI = Path(_HERE / "assets/gelsight_mini/gelsight_mini.xml")
        gs1 = mjcf.from_path(_XML_GS_MINI)

        scene.attach(gs1)

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        return m, d

    def spin(self, ss: SimSync):
        while True:
            ss.step()

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    def keyboard_callback(self, key: int):
        if key is glfw.KEY_SPACE:
            self.gs.save("demo.png")
            print("You pressed space...")


if __name__ == "__main__":
    sim = MjSim()
    sim.run()
