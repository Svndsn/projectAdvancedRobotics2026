from pathlib import Path

import glfw
import mujoco as mj
import spatialmath.base as smb
from dm_control import mjcf

from sims import BaseSim
from sims.base_sim import SimSync
from utils.mj import get_contact_states
from utils.sm import make_tf


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()

        self._model, self._data = self.init()

        self.tasks = [self.spin, self.log]

    def init(self):
        # root
        _HERE = Path(__file__).parent.parent
        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        # look at the scene
        getattr(scene.visual, "global").azimuth = -178.64700780572414
        getattr(scene.visual, "global").elevation = -0.017346053772759384
        scene.statistic.center = [0.035, -0.04, 0.1]
        scene.statistic.extent = scene.statistic.extent * 0.5

        # Load finger asset
        _XML_FINGER = Path(_HERE / "assets/shadow_hand/finger.xml")
        finger = mjcf.from_path(_XML_FINGER)

        def tip_cfg(id: int):
            cfg = {
                "0": [-0.009, 0.001, -0.006, -0.513, -0.459, 0.483, 0.540],
                "1": [-0.006, -0.007, -0.008, 0.253, 0.670, -0.653, -0.247],
                "2": [-0.007, -0.006, -0.013, 0.225, 0.703, -0.642, -0.209],
                "3": [-0.009, 0.001, -0.012, -0.531, -0.451, 0.464, 0.546],
                "4": [-0.009, 0.001, 0.000, -0.435, -0.415, 0.553, 0.577],
                "5": [0.006, -0.005, -0.001, 0.146, -0.545, 0.775, -0.284],
                "6": [0.009, 0.001, 0.000, -0.436, 0.414, -0.551, 0.578],
                "7": [0.007, -0.006, -0.007, -0.258, 0.671, -0.649, 0.250],
                "8": [0.000, -0.008, -0.011, -0.001, 0.737, -0.676, 0.002],
                "9": [-0.006, -0.005, -0.001, -0.146, -0.546, 0.775, 0.285],
                "10": [0.009, 0.001, -0.011, -0.528, 0.456, -0.469, 0.542],
                "11": [0.004, -0.000, 0.006, -0.105, 0.214, -0.749, 0.618],
                "12": [0.000, -0.005, 0.003, 0.009, -0.439, 0.898, -0.014],
                "13": [-0.004, -0.000, 0.006, 0.230, -0.062, -0.749, 0.618],
                "14": [0.007, -0.006, -0.014, -0.231, 0.699, -0.642, 0.216],
                "15": [0.009, 0.001, -0.005, -0.506, 0.467, -0.492, 0.533],
                "16": [0.000, -0.008, -0.004, 0.004, -0.638, 0.770, -0.010],
            }
            pose = cfg[str(id)]
            return make_tf(pos=pose[:3], ori=pose[3:])

        N_SENSORS = 17

        # add sites to finger tips
        for body in finger.find_all("body"):
            if "distal" in body.name:
                body.add("site", name=f"{body.name}_tip", pos="0 0 0.026")
                for i in range(N_SENSORS):
                    Ti = make_tf(pos=[0, 0, 0.026]) @ tip_cfg(i)
                    # since the site is placed as the root of the body, I shift it 2.6 cm to be at the tip
                    # source: https://shadow-robot-company-dexterous-hand.readthedocs-hosted.com/en/latest/user_guide/md_finger.html
                    si = body.add(
                        "site",
                        name=f"{body.name}_{i}",
                        pos=" ".join(str(x) for x in Ti.t),
                        quat=" ".join(str(x) for x in smb.r2q(Ti.R)),
                        size="0.001",
                        rgba="0 0 0 0.1",
                    )
                    sens = scene.sensor.add(
                        "plugin",
                        name=f"touch_{body.name}_{i}",
                        plugin="mujoco.sensor.touch_grid",
                        objtype="site",
                        objname=f"{finger.model}/{body.name}_{i}",
                    )
                    sens.add("config", key="size", value="1 1")
                    sens.add("config", key="fov", value="10 10")
                    sens.add("config", key="gamma", value="0")
                    sens.add("config", key="nchannel", value="3")

        scene.attach(finger)

        # prop
        prop = scene.worldbody.add(
            "body", name="prop", pos=" 0.015 -0.04 0.1", euler="0 1.57 0"
        )
        prop.add("geom", name="prop", type="cylinder", size="0.01 0.02")
        prop.add(
            "geom",
            name="prop_dir",
            type="cylinder",
            size="0.001 0.01",
            pos="0 0 0.02",
            euler="0 1.57 0",
            rgba="1 0 0 1",
            mass="0",
        )
        prop.add("joint", name="revolute")

        # m = mj.MjModel.from_xml_string(XML)
        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        return m, d

    def spin(self, ss: SimSync):
        while True:
            ss.step()

    def log(self, ss: SimSync):
        while True:
            is_in_contact, css = get_contact_states(
                self.data,
                self.model,
                "finger/finger_tip",
            )
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
