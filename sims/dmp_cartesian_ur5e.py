import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Tuple

import glfw
import mujoco as mj
import numpy as np
import pandas as pd
import spatialmath as sm
from dm_control import mjcf

from ctrl.dmp_cartesian.dmp_cartesian import DMPCartesian
from ctrl.opspace.opspace import OpSpace
from robots.ur_robot import URRobot
from scripts.plot_dmp_demo import generate_plots
from sims import BaseSim
from sims.base_sim import SimSync, sleep
from utils.math import npq2np
from utils.mj import ObjType, get_names
from utils.sm import make_tf


def _rotvec_from_pose(T: sm.SE3) -> list[float]:
    theta, v = T.angvec()
    rotvec = theta * v
    return [rotvec[0], rotvec[1], rotvec[2]]


def _default_replay_log_path(demo_path: str, tau_scale: float) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(demo_path).stem
    scale_tag = str(tau_scale).replace(".", "p")
    return Path("data/demo") / f"{stem}_dmp_replay_tau{scale_tag}_{stamp}.csv"


class MjSim(BaseSim):
    def __init__(
        self,
        demo_path: str,
        replay_log_path: str,
        tau_scale: float,
    ) -> None:
        super().__init__()

        self._model, self._data = self.init()

        # lock to set state
        self._lock = Lock()

        # UR5e robot interface
        self.ur5e = URRobot(self._model, self._data, robot_type=URRobot.Type.UR5e)

        # give the robot the opspace controller
        self.ur5e.controller = OpSpace(self.ur5e)

        # load in demonstration
        df = pd.read_csv(demo_path)
        self.demo_path = Path(demo_path).expanduser().resolve()
        self.replay_log_path = Path(replay_log_path).expanduser().resolve()
        self.replay_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.tau_scale = tau_scale
        self.pose_matrices = self.df_2_se3(df, "actual_TCP_pose")

        self.dmp = DMPCartesian()

        self.dmp.load(self.pose_matrices)

        rollout_tau = self.dmp.tau * self.tau_scale
        n_steps = max(2, int(round(len(self.dmp.ts) * self.tau_scale)))
        rollout_ts = np.linspace(0.0, rollout_tau, n_steps)
        p, v, a, q, w, dw = self.dmp.rollout(rollout_ts, rollout_tau)

        self.dmp_poses = [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]
        self.replay_rows: list[list[float]] = []

        self.tasks = [self.spin]

    def df_2_se3(self, df: pd.DataFrame, pose_columns_prefix: str) -> List[sm.SE3]:
        """
        Converts pose columns in a DataFrame to a list of spatialmath SE3 matrices.

        Args:
            df: DataFrame containing pose data.
            pose_columns_prefix: The prefix of the pose columns (e.g., 'actual_TCP_pose' or 'target_TCP_pose').

        Returns:
            A list of SE3 objects representing the poses.
        """
        se3_matrices = []
        for _, row in df.iterrows():
            # Extract position (first 3 values)
            position = row[
                [
                    f"{pose_columns_prefix}_0",
                    f"{pose_columns_prefix}_1",
                    f"{pose_columns_prefix}_2",
                ]
            ].values
            # Extract axis-angle orientation (last 3 values)
            axis_angle = row[
                [
                    f"{pose_columns_prefix}_3",
                    f"{pose_columns_prefix}_4",
                    f"{pose_columns_prefix}_5",
                ]
            ].values
            # Convert axis-angle to an SE3 rotation matrix
            angle = np.linalg.norm(axis_angle)
            if angle > 1e-12:
                orientation = sm.SO3.AngVec(angle, axis_angle / angle)
            else:
                orientation = sm.SO3()
            # Create the SE3 matrix
            se3_matrices.append(sm.SE3.Rt(orientation.R, position))
        return se3_matrices

    def init(self) -> Tuple[mj.MjModel, mj.MjData]:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        # get ur5e
        _XML_UR5E = Path(_HERE / "assets/universal_robots_ur5e/ur5e.xml")
        ur5e = mjcf.from_path(_XML_UR5E)
        ur5e_base = ur5e.worldbody.find("body", "base")
        ur5e_base.pos = [
            0.22331901,
            0.37537452,
            0.08791326,
        ]  # from hand to table calibration
        ur5e_base.quat = [
            -0.19858483999999996,
            -0.00311175,
            0.0012299899999999998,
            0.98007799,
        ]  # from hand to table calibration

        # Get the actuator root element
        actuator_root = ur5e.actuator

        names_and_joints = []
        # save all the actuator names, joints they link to and classes
        for general_actuator in actuator_root.all_children():
            names_and_joints.append(
                (
                    general_actuator.name,
                    general_actuator.joint.name,
                    general_actuator.dclass.dclass
                    if not isinstance(general_actuator.dclass, list)
                    else general_actuator.dclass[0].dclass,
                )
            )

        # remove all the general actuators
        for general_actuator in actuator_root.all_children():
            actuator_root.remove(general_actuator)

        # List of actuators with their new kp and kv values
        actuator_config: dict = {
            "shoulder_pan": {"ctrlrange": "-150 150"},
            "shoulder_lift": {"ctrlrange": "-150 150"},
            "elbow": {"ctrlrange": "-150 150"},
            "wrist_1": {"ctrlrange": "-28 28"},
            "wrist_2": {"ctrlrange": "-28 28"},
            "wrist_3": {"ctrlrange": "-28 28"},
        }

        # create position actuators instead
        for name, joint, dclass in names_and_joints:
            # Create a new position actuator with kp and kv
            actuator_root.add(
                "motor",
                name=name,
                joint=joint,
                dclass=dclass,
                ctrlrange=actuator_config[name]["ctrlrange"],
            )

        scene.attach(ur5e)

        # spawn flange tool
        flange_tool = mjcf.from_path(_HERE / "assets/props/flange_tool.xml")
        ur5e_attach_site = ur5e.find("site", "attachment_site")
        ur5e_attach_site.attach(flange_tool)

        # spawn the table
        table = mjcf.from_path(_HERE / "assets/props/flexcell_top.xml")
        scene.attach(table)

        # spawn the mounting plate
        mounting_plate = mjcf.from_path(_HERE / "assets/props/mounting_plate.xml")
        scene.attach(mounting_plate)

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        # get the init pose of the ur5e
        ur_joint_names = [jn for jn in get_names(m, ObjType.JOINT) if "ur5e" in jn]
        j0 = [2.8, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        for i, jn in enumerate(ur_joint_names):
            d.joint(jn).qpos = j0[i]

        return m, d

    def log(self, ss: SimSync):
        while True:
            print(
                "UR5e force/torque measurement: ",
                np.array_str(self.ur5e.w, precision=2, suppress_small=True),
            )
            ss.step()

    def spin(self, ss: SimSync):
        self.ur5e.move_l(self.dmp_poses[0], ss)

        for target_pose in self.dmp_poses:
            self.ur5e.controller.T_target = target_pose
            x = self.ur5e.controller.step()
            self.ur5e.set_ctrl(x)
            sleep(self.model.opt.timestep, ss, self.model)

            actual_pose = self.ur5e.get_ee_pose()
            self.replay_rows.append(
                [
                    target_pose.t[0],
                    target_pose.t[1],
                    target_pose.t[2],
                    *_rotvec_from_pose(target_pose),
                    actual_pose.t[0],
                    actual_pose.t[1],
                    actual_pose.t[2],
                    *_rotvec_from_pose(actual_pose),
                ]
            )

        pd.DataFrame(
            self.replay_rows,
            columns=[f"dmp_target_TCP_pose_{i}" for i in range(6)]
            + [f"actual_TCP_pose_{i}" for i in range(6)],
        ).to_csv(self.replay_log_path, index=False)
        print(f"Saved replay log: {self.replay_log_path}")

        # to keep the robot in place after having performed the dmp motion
        self.ur5e.controller.T_target = self.dmp_poses[-1]
        while True:
            tau = self.ur5e.controller.step()
            self.ur5e.set_ctrl(tau)
            ss.step()

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    def keyboard_callback(self, key: int) -> None:
        if key is glfw.KEY_SPACE:
            print("Hello, MuJoCo engineer!")
            self.ur5e.move_l(self.ur5e.get_ee_pose() @ sm.SE3.Tz(0.2))

        if key is glfw.KEY_PERIOD:
            print("UR5e force/torque measurement: ", self.ur5e.w)


_logger = logging.getLogger("mj_sim")


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo simulation of manipulators and controllers."
    )
    parser.add_argument(
        "--demo",
        default="data/demo/demonstration.csv",
        help="Path to recorded demonstration CSV.",
    )
    parser.add_argument(
        "--replay-log-out",
        default="",
        help="Optional output CSV path for logged DMP target vs executed replay.",
    )
    parser.add_argument(
        "--tau-scale",
        type=float,
        default=1.0,
        help="Temporal scaling for the DMP rollout. Values > 1 slow it down.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )

    args, _ = parser.parse_known_args()

    if args.tau_scale <= 0:
        parser.error("--tau-scale must be greater than 0.")

    # set the default log level to INFO
    _logger.setLevel(logging.INFO)
    setup_logging(args.loglevel)

    _logger.info(" > Loaded configuration:")
    for key, value in vars(args).items():
        _logger.info(f"\t{key:30}{value}")

    replay_log_path = (
        args.replay_log_out
        if args.replay_log_out
        else str(_default_replay_log_path(args.demo, args.tau_scale))
    )
    sim = MjSim(args.demo, replay_log_path, args.tau_scale)
    sim.run()
    generate_plots(args.demo, replay_log=replay_log_path)


if __name__ == "__main__":
    main()
