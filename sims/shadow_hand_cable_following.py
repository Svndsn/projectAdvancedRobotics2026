from pathlib import Path

import glfw
import mujoco as mj
import spatialmath as sm
from dm_control import mjcf

from robots import Mocap, ShadowHand
from sims import BaseSim
from sims.base_sim import SimSync
from utils.mj import load_keyframe


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()
        self._model, self._data = self.init()

        # load_keyframe(
        #     self._model, self._data, "unnamed_model/s2", file_path=self.keyframe_path
        # )

        self.sh = ShadowHand(self.model, self.data)

        self.mocap = Mocap(self.model, self.data)

        # self.tasks = []
        self.tasks = [self.spin, self.script]

        self.begin = False

        # self.i = 1

    def init(self):
        # Root directory
        _HERE = Path(__file__).parent.parent

        # Define keyframe file path
        self.keyframe_path = Path(_HERE, "keyframes", Path(__file__).stem + ".xml")

        # Ensure the parent directory exists
        self.keyframe_path.parent.mkdir(parents=True, exist_ok=True)

        # Create an empty keyframe file with necessary tags if it doesn't exist
        if not self.keyframe_path.exists() or self.keyframe_path.stat().st_size == 0:
            with open(self.keyframe_path, "w", encoding="utf-8") as file:
                file.write("<mujoco>\n    <keyframe />\n</mujoco>")

        # Load the keyframe file
        keyframe = mjcf.from_path(str(self.keyframe_path))

        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        scene.attach(keyframe)

        # get global solref
        scene.option.o_solref = [0.00000001, 4]

        # shadow hand path
        _XML_SHADOW_HAND = Path(_HERE / "assets/shadow_hand/shadow_rh.xml")
        # _XML_SHADOW_HAND = Path(_HERE / "assets/shadow_hand/right_hand.xml")
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

        # spawn cable
        _XML_CABLE = Path(_HERE / "assets/props/cable.xml")
        cable_1 = mjcf.from_path(_XML_CABLE)
        cable_1_body = cable_1.worldbody.find("body", "cable")
        cable_1_body.pos = [0.3, 0, 0.1]
        cable_1.model = "cable_1"

        cable_2 = mjcf.from_path(_XML_CABLE)
        cable_2.model = "cable_2"
        cable_2_body = cable_2.worldbody.find("body", "cable")
        cable_2_body.pos = [-0.3, 0.0, 0.1]

        cable_mocap = scene.worldbody.add(
            "body", pos="0.3 0.915 0.1", name="cable_mocap", mocap="true"
        )
        cable_mocap.add(
            "geom",
            name="cable_mocap",
            type="box",
            size="0.01 0.01 0.01",
            rgba="0 0 0 1",
        )

        scene.equality.add(
            "weld",
            body1="cable_mocap",
            body2="cable_1/cable:B_last",
            solref="0.0000001 1",
        )
        scene.attach(cable_1)
        scene.attach(cable_2)

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        return m, d

    def script(self, ss: SimSync):
        while not self.begin:
            ss.step()
        for i in range(6):
            load_keyframe(
                self.model, self.data, f"unnamed_model/s{i}", self.keyframe_path
            )
            self.mocap.move_l(
                sm.SE3.Ty(0.4) @ self.mocap.T_world_base, ss, velocity=0.3
            )

        # once moved, make sure that the target of the mocap is where it currently is
        self.mocap.T_target = self.mocap.T_world_base

    def log(self, ss: SimSync):
        for i in range(100):
            print(f"I logged {i}")
            ss.step()

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
            print("You pressed space...")
            # save_keyframe(self.model, self.data, "s2", self.keyframe_path)
            # self.i += 1
            self.begin = True
        if key is glfw.KEY_M:
            load_keyframe(self.model, self.data, "s0", self.keyframe_path)

    def control_loop(self):
        self.mocap.step()


if __name__ == "__main__":
    sim = MjSim()
    sim.run(show_left_ui=False, show_right_ui=False)
