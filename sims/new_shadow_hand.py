from pathlib import Path

import glfw
import mujoco as mj
import spatialmath as sm
import spatialmath.base as smb
from dm_control import mjcf

from robots import Mocap, ShadowHand
from sims import BaseSim
from sims.base_sim import SimSync
from utils.sm import make_tf


def tip_cfg(id: int, thumb: bool = False):
    if not thumb:
        finger_cfg = {
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
        pose = finger_cfg[str(id)]
    else:
        thumb_cfg = {
            "0": [-0.010, 0.000, -0.004, -0.462, -0.476, 0.537, 0.522],
            "1": [-0.007, -0.007, -0.005, -0.246, -0.642, 0.678, 0.262],
            "2": [-0.007, -0.007, -0.012, 0.223, 0.671, -0.671, -0.223],
            "3": [-0.010, -0.000, -0.011, -0.481, -0.482, 0.519, 0.518],
            "4": [-0.009, 0.001, 0.002, -0.439, -0.425, 0.551, 0.568],
            "5": [0.006, -0.006, 0.001, 0.149, -0.551, 0.772, -0.281],
            "6": [0.009, 0.001, 0.002, -0.441, 0.430, -0.550, 0.564],
            "7": [0.007, -0.007, -0.006, 0.250, -0.640, 0.676, -0.266],
            "8": [0.000, -0.009, -0.009, -0.002, 0.722, -0.691, 0.005],
            "9": [-0.006, -0.006, 0.001, -0.145, -0.549, 0.774, 0.279],
            "10": [0.010, -0.000, -0.010, -0.482, 0.482, -0.518, 0.517],
            "11": [0.004, 0.001, 0.008, -0.180, 0.160, -0.674, 0.698],
            "12": [0.000, -0.005, 0.005, 0.011, -0.443, 0.896, -0.017],
            "13": [-0.005, 0.001, 0.008, 0.184, -0.216, -0.666, 0.690],
            "14": [0.007, -0.007, -0.012, -0.218, 0.675, -0.670, 0.217],
            "15": [0.010, 0.000, -0.005, -0.465, 0.473, -0.533, 0.525],
            "16": [0.000, -0.009, -0.003, 0.003, -0.622, 0.783, -0.007],
        }
        pose = thumb_cfg[str(id)]

    return make_tf(pos=pose[:3], ori=pose[3:])


N_SENSORS = 17


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()
        self._model, self._data = self.init()

        self.sh = ShadowHand(self.model, self.data)

        self.mocap = Mocap(self.model, self.data)

        self.tasks = []
        # self.tasks = [self.spin]

        self.begin = False

        self.N_SENSORS = N_SENSORS

    def init(self):
        # root
        _HERE = Path(__file__).parent.parent

        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        # get global solref
        scene.option.o_solref = [0.00000001, 4]

        # prop
        cube = scene.worldbody.add("body", name="cube", pos="0.35 0 1.1")
        # cube = scene.worldbody.add("body", name="cube", pos="0.3 0 1.1")
        cube.add("geom", name="cube", type="box", size="0.02 0.02 0.02")

        # shadow hand path
        # _XML_SHADOW_HAND = Path(_HERE / "assets/shadow_hand/mjx_shadow_hand.xml")
        _XML_SHADOW_HAND = Path(_HERE / "assets/shadow_hand/shadow_rh.xml")
        # _XML_SHADOW_HAND = Path(_HERE / "assets/shadow_hand/rh.xml")
        # _XML_SHADOW_HAND = Path(_HERE / "assets/shadow_hand/right_hand.xml")
        shadow_hand = mjcf.from_path(_XML_SHADOW_HAND)

        # add sites to finger tips
        for body in shadow_hand.find_all("body"):
            if "distal" in body.name:
                THUMB = True if "th" in body.name else False
                body.add("site", name=f"{body.name}_tip", pos="0 0 0.026")
                for i in range(N_SENSORS):
                    Ti = make_tf(pos=[0, 0, 0.026]) @ tip_cfg(i, thumb=THUMB)
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
                        objname=f"{shadow_hand.model}/{body.name}_{i}",
                    )
                    sens.add("config", key="size", value="1 1")
                    sens.add("config", key="fov", value="10 10")
                    sens.add("config", key="gamma", value="0")
                    sens.add("config", key="nchannel", value="3")

        # mjcf.export_with_assets(
        #     shadow_hand,
        #     "/home/vims/git/gitlab/mj_sim/assets/shadow_hand",
        #     out_file_name="test.xml",
        # )

        # quit()

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
