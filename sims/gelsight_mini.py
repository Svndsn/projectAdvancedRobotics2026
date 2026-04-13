import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glfw
import mujoco as mj
import numpy as np
from dm_control import mjcf

from sensors.gelsight_mini.gelsight_mini import GelSightMini
from sims import BaseSim
from sims.base_sim import SimSync


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()

        self._model, self._data = self.init()
        self._probe_pressed = False

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
        probe = scene.worldbody.add("body", name="press_probe")
        probe.add("freejoint", name="press_probe_joint")
        probe.add("geom", name="press_probe", type="sphere", size="0.006", rgba="0.85 0.2 0.2 1")
        press_sensor = scene.sensor.add(
            "plugin",
            name="gelsight_touch",
            plugin="mujoco.sensor.touch_grid",
            objtype="site",
            objname=f"{gs1.model}/gelsight_mini_press_site",
        )
        scene.sensor.add(
            "force",
            name="gelsight_force",
            site=f"{gs1.model}/gelsight_mini_press_site",
        )
        press_sensor.add("config", key="size", value="8 6")
        press_sensor.add("config", key="fov", value="18 14")
        press_sensor.add("config", key="gamma", value="0")
        press_sensor.add("config", key="nchannel", value="1")

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        press_site_pos = d.site("gelsight_mini/gelsight_mini_press_site").xpos.copy()
        d.joint("press_probe_joint").qpos[:] = np.r_[press_site_pos + np.array([0.0, 0.0, 0.02]), [1.0, 0.0, 0.0, 0.0]]
        mj.mj_forward(m, d)

        return m, d

    def set_probe_pressed(self, pressed: bool) -> None:
        press_site_pos = self.data.site("gelsight_mini/gelsight_mini_press_site").xpos.copy()
        z_offset = 0.006 if pressed else 0.02
        self.data.joint("press_probe_joint").qpos[:] = np.r_[press_site_pos + np.array([0.0, 0.0, z_offset]), [1.0, 0.0, 0.0, 0.0]]
        mj.mj_forward(self.model, self.data)
        self._probe_pressed = pressed

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
            self.set_probe_pressed(not self._probe_pressed)
            self.gs.save("demo.png")
            press = self.gs.record_press_measurement()
            print(
                f"probe_pressed={self._probe_pressed}  "
                f"pressed={self.gs.is_pressed}  "
                f"total_press={press['total_press']:.4f}  "
                f"normal_force={press['normal_force']:.4f}  "
                f"peak_press={press['peak_press']:.4f}  "
                f"active_taxels={press['active_taxels']}"
            )


if __name__ == "__main__":
    sim = MjSim()
    sim.run()
