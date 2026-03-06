from pathlib import Path

import glfw
import mujoco as mj
import spatialmath as sm
from dm_control import mjcf

from robots import Mocap, ShadowHand
from sims import BaseSim
from sims.base_sim import SimSync


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()
        self._model, self._data = self.init()

        self.sh = ShadowHand(self.model, self.data)

        self.mocap = Mocap(self.model, self.data)

        # self.tasks = []
        self.tasks = [self.script]

        self.begin = False

    def init(self):
        # root
        _HERE = Path(__file__).parent.parent

        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        # get global solref
        scene.option.o_solref = [0.00000001, 4]

        # prop
        cube = scene.worldbody.add("body", name="cube", pos="0.3 0 1.05")
        cube.add("geom", name="cube", type="box", size="0.02 0.02 0.02")
        cube.add("joint", name="cube", type="free")

        # shadow hand path
        _XML_SHADOW_HAND = Path(_HERE / "assets/shadow_hand/shadow_hand.xml")

        shadow_hand = mjcf.from_path(_XML_SHADOW_HAND)

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

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        return m, d

    def script(self, ss: SimSync):
        while not self.begin:
            ss.step()
        print("Move to (0,0,1) in meters...")
        self.mocap.move_l(sm.SE3.Ty(0.6) @ self.mocap.T_world_base, ss)

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
            self.begin = True

    def control_loop(self):
        self.mocap.step()


if __name__ == "__main__":
    sim = MjSim()
    sim.run()
