import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

import mujoco as mj
import numpy as np
import pandas as pd
import quaternion
import spatialmath as sm
import spatialmath.base as smb
from dm_control import mjcf
import yaml

# Allow running this file directly (e.g. `uv run sims/...py`) without
# requiring manual PYTHONPATH changes.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ctrl.dmp_cartesian.dmp_cartesian import DMPCartesian
from ctrl.opspace.opspace import OpSpace
from robots.ur_robot import URRobot
from sims import BaseSim
from sims.base_sim import SimSync
from utils.math import npq2np
from utils.mj import ObjType, get_contact_states, get_names, get_pose, name2id
from utils.sm import make_tf


def _rotvec_from_pose(T: sm.SE3) -> List[float]:
    theta, v = T.angvec()
    rotvec = theta * v
    return [float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]


def _pose_row_to_se3(row: pd.Series, prefix: str) -> sm.SE3:
    p = np.array(
        [
            row[f"{prefix}_0"],
            row[f"{prefix}_1"],
            row[f"{prefix}_2"],
        ],
        dtype=float,
    )
    rotvec = np.array(
        [
            row[f"{prefix}_3"],
            row[f"{prefix}_4"],
            row[f"{prefix}_5"],
        ],
        dtype=float,
    )
    angle = np.linalg.norm(rotvec)
    if angle > 1e-12:
        R = sm.SO3.AngVec(angle, rotvec / angle).R
    else:
        R = np.eye(3)
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = p
    return sm.SE3(A, check=False)


def _detect_pose_prefix(df: pd.DataFrame) -> str:
    if all(f"actual_TCP_pose_{i}" in df.columns for i in range(6)):
        return "actual_TCP_pose"
    if all(f"target_TCP_pose_{i}" in df.columns for i in range(6)):
        return "target_TCP_pose"
    raise ValueError(
        "Could not find pose columns. Expected actual_TCP_pose_[0..5] or target_TCP_pose_[0..5]."
    )


def _load_demo_poses(demo_csv: str) -> List[sm.SE3]:
    df = pd.read_csv(demo_csv)
    prefix = _detect_pose_prefix(df)
    return [_pose_row_to_se3(row, prefix) for _, row in df.iterrows()]


def _default_replay_log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/demo") / f"approach_grasp_task_{stamp}.csv"


def _apply_if_present(src: Dict[str, Any], key: str, out: Dict[str, Any]) -> None:
    if key in src:
        out[key] = src[key]


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("YAML config root must be a mapping/dictionary.")

    out: Dict[str, Any] = {}

    # Flat keys matching argparse destinations are accepted directly.
    for flat_key in [
        "approach_demo",
        "task_demo",
        "replay_log_out",
        "object_name",
        "object_pos",
        "object_rotvec",
        "object_size",
        "object_mass",
        "approach_start_offset",
        "approach_goal_offset",
        "task_start_offset",
        "task_goal_offset",
        "approach_tau_scale",
        "task_tau_scale",
        "gripper_open",
        "gripper_close",
        "close_hold_s",
        "grasp_timeout",
        "open_hold_s",
        "hold_final_s",
    ]:
        _apply_if_present(raw, flat_key, out)

    # Nested sections are also supported.
    demos = raw.get("demos", {})
    if isinstance(demos, dict):
        if "approach" in demos:
            out["approach_demo"] = demos["approach"]
        if "task" in demos:
            out["task_demo"] = demos["task"]

    logging_cfg = raw.get("logging", {})
    if isinstance(logging_cfg, dict) and "replay_log_out" in logging_cfg:
        out["replay_log_out"] = logging_cfg["replay_log_out"]

    object_cfg = raw.get("object", {})
    if isinstance(object_cfg, dict):
        for key in ["name", "pos", "rotvec", "size", "mass"]:
            if key in object_cfg:
                out[f"object_{key}"] = object_cfg[key]

    offsets = raw.get("offsets", {})
    if isinstance(offsets, dict):
        approach_offsets = offsets.get("approach", {})
        task_offsets = offsets.get("task", {})
        if isinstance(approach_offsets, dict):
            if "start" in approach_offsets:
                out["approach_start_offset"] = approach_offsets["start"]
            if "goal" in approach_offsets:
                out["approach_goal_offset"] = approach_offsets["goal"]
        if isinstance(task_offsets, dict):
            if "start" in task_offsets:
                out["task_start_offset"] = task_offsets["start"]
            if "goal" in task_offsets:
                out["task_goal_offset"] = task_offsets["goal"]

    tau_scale = raw.get("tau_scale", {})
    if isinstance(tau_scale, dict):
        if "approach" in tau_scale:
            out["approach_tau_scale"] = tau_scale["approach"]
        if "task" in tau_scale:
            out["task_tau_scale"] = tau_scale["task"]

    gripper = raw.get("gripper", {})
    if isinstance(gripper, dict):
        if "open" in gripper:
            out["gripper_open"] = gripper["open"]
        if "close" in gripper:
            out["gripper_close"] = gripper["close"]

    timing = raw.get("timing", {})
    if isinstance(timing, dict):
        for key in ["close_hold_s", "grasp_timeout", "open_hold_s", "hold_final_s"]:
            _apply_if_present(timing, key, out)

    return out


@dataclass
class PhaseSpec:
    name: str
    dmp: DMPCartesian
    start_offset: np.ndarray
    goal_offset: np.ndarray
    tau_scale: float
    ref_start: sm.SE3
    ref_goal: sm.SE3


class RobotiqHandE:
    def __init__(self, model: mj.MjModel, data: mj.MjData, open_pos: float, close_pos: float):
        self.model = model
        self.data = data
        self.open_pos = open_pos
        self.close_pos = close_pos

        actuator_names = get_names(self.model, ObjType.ACTUATOR)
        candidates = [
            name
            for name in actuator_names
            if "hande_fingers_actuator" in name
        ]
        if not candidates:
            raise ValueError(
                "Could not find HandE actuator. Available actuators: "
                f"{actuator_names}"
            )
        self.actuator_name = candidates[0]
        self.actuator_id = name2id(self.model, self.actuator_name, ObjType.ACTUATOR)

        geom_names = get_names(self.model, ObjType.GEOM)
        self.finger_geom_names = [
            g
            for g in geom_names
            if ("hande_left_finger" in g or "hande_right_finger" in g)
        ]
        if not self.finger_geom_names:
            self.finger_geom_names = [
                g for g in geom_names if ("left_finger" in g or "right_finger" in g)
            ]

    def command(self, target: float) -> None:
        clipped = float(
            np.clip(
                target,
                min(self.open_pos, self.close_pos),
                max(self.open_pos, self.close_pos),
            )
        )
        self.data.actuator(self.actuator_id).ctrl = clipped

    def open(self) -> None:
        self.command(self.open_pos)

    def close(self) -> None:
        self.command(self.close_pos)


class MjSim(BaseSim):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self.args = args
        self._model, self._data = self.init(args)

        self._lock = Lock()

        self.ur5e = URRobot(self._model, self._data, robot_type=URRobot.Type.UR5e)
        self._configure_ur5e_arm_interface()
        self.ur5e.controller = OpSpace(self.ur5e)

        self.gripper = RobotiqHandE(
            self._model,
            self._data,
            open_pos=args.gripper_open,
            close_pos=args.gripper_close,
        )

        # Keep default stiff control from existing DMP replay script.
        self.ur5e.controller.Kpos = 0.95
        self.ur5e.controller.Kori = 0.95

        approach_demo = _load_demo_poses(args.approach_demo)
        task_demo = _load_demo_poses(args.task_demo)

        self.approach_dmp = DMPCartesian(roto_dilatation=True)
        self.approach_dmp.load(approach_demo)

        self.task_dmp = DMPCartesian(roto_dilatation=True)
        self.task_dmp.load(task_demo)

        self.object_body_name = args.object_name
        self.object_geom_name = f"{args.object_name}_geom"

        self.phase_specs = [
            PhaseSpec(
                name="approach",
                dmp=self.approach_dmp,
                start_offset=np.array(args.approach_start_offset, dtype=float),
                goal_offset=np.array(args.approach_goal_offset, dtype=float),
                tau_scale=args.approach_tau_scale,
                ref_start=approach_demo[0],
                ref_goal=approach_demo[-1],
            ),
            PhaseSpec(
                name="task",
                dmp=self.task_dmp,
                start_offset=np.array(args.task_start_offset, dtype=float),
                goal_offset=np.array(args.task_goal_offset, dtype=float),
                tau_scale=args.task_tau_scale,
                ref_start=task_demo[0],
                ref_goal=task_demo[-1],
            ),
        ]

        self.replay_log_path = Path(args.replay_log_out).expanduser().resolve()
        self.replay_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.rows: list[list[float | str | bool]] = []
        self.current_gripper_cmd = float(args.gripper_open)
        self.grasped = False

        self.tasks = [self.spin]

    def _configure_ur5e_arm_interface(self) -> None:
        arm_joint_suffixes = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        arm_actuator_suffixes = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]

        def _ordered_match(names: List[str], suffixes: List[str]) -> List[str]:
            selected = []
            for suffix in suffixes:
                candidates = [name for name in names if name.endswith(suffix)]
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected exactly one match for suffix '{suffix}', found {candidates}"
                    )
                selected.append(candidates[0])
            return selected

        info = self.ur5e.info
        arm_joint_names = _ordered_match(info.joint_names, arm_joint_suffixes)
        arm_actuator_names = _ordered_match(info.actuator_names, arm_actuator_suffixes)
        site_candidates = [sn for sn in info.site_names if sn.endswith("attachment_site")]
        if len(site_candidates) != 1:
            raise ValueError(
                "Expected exactly one UR5e attachment site, found "
                f"{site_candidates}"
            )

        info.joint_names = arm_joint_names
        info.joint_ids = [name2id(self.model, jn, ObjType.JOINT) for jn in arm_joint_names]
        info.joint_indxs = [int(self.model.joint(jn).qposadr[0]) for jn in arm_joint_names]
        info.dof_indxs = [int(self.model.joint(jn).dofadr[0]) for jn in arm_joint_names]

        info.actuator_names = arm_actuator_names
        info.actuator_ids = [
            name2id(self.model, an, ObjType.ACTUATOR) for an in arm_actuator_names
        ]

        info.site_names = site_candidates
        info.site_ids = [name2id(self.model, site_candidates[0], ObjType.SITE)]

    def init(self, args: argparse.Namespace) -> Tuple[mj.MjModel, mj.MjData]:
        here = Path(__file__).parent.parent
        scene = mjcf.from_path(here / "scenes/empty.xml")

        ur5e = mjcf.from_path(here / "assets/universal_robots_ur5e/ur5e.xml")
        ur5e_base = ur5e.worldbody.find("body", "base")
        ur5e_base.pos = [0.22331901, 0.37537452, 0.08791326]
        ur5e_base.quat = [-0.19858483999999996, -0.00311175, 0.0012299899999999998, 0.98007799]

        actuator_root = ur5e.actuator
        names_and_joints = []
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

        for general_actuator in actuator_root.all_children():
            actuator_root.remove(general_actuator)

        actuator_config: dict[str, dict[str, str]] = {
            "shoulder_pan": {"ctrlrange": "-150 150"},
            "shoulder_lift": {"ctrlrange": "-150 150"},
            "elbow": {"ctrlrange": "-150 150"},
            "wrist_1": {"ctrlrange": "-28 28"},
            "wrist_2": {"ctrlrange": "-28 28"},
            "wrist_3": {"ctrlrange": "-28 28"},
        }

        for name, joint, dclass in names_and_joints:
            actuator_root.add(
                "motor",
                name=name,
                joint=joint,
                dclass=dclass,
                ctrlrange=actuator_config[name]["ctrlrange"],
            )

        scene.attach(ur5e)

        hande = mjcf.from_path(here / "assets/robotiq_hande/hande.xml")
        ur5e_attach_site = ur5e.find("site", "attachment_site")
        ur5e_attach_site.attach(hande)

        table = mjcf.from_path(here / "assets/props/flexcell_top.xml")
        scene.attach(table)

        mounting_plate = mjcf.from_path(here / "assets/props/mounting_plate.xml")
        scene.attach(mounting_plate)

        # Add a simple tool/object body. Its placement is runtime-configurable via CLI.
        object_rotvec = np.asarray(args.object_rotvec, dtype=float)
        object_angle = np.linalg.norm(object_rotvec)
        if object_angle > 1e-12:
            object_quat = smb.r2q(
                sm.SO3.AngVec(object_angle, object_rotvec / object_angle).R
            )
        else:
            object_quat = [1.0, 0.0, 0.0, 0.0]

        object_body = scene.worldbody.add(
            "body",
            name=args.object_name,
            pos=list(np.asarray(args.object_pos, dtype=float)),
            quat=list(object_quat),
        )
        object_body.add(
            "geom",
            name=f"{args.object_name}_geom",
            type="box",
            size=list(np.asarray(args.object_size, dtype=float)),
            rgba=[0.85, 0.35, 0.2, 1.0],
            mass=float(args.object_mass),
        )

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        mj.mj_step(m, d)

        arm_joint_suffixes = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        ur_joint_names = [
            jn
            for jn in get_names(m, ObjType.JOINT)
            if ("ur5e/" in jn and any(jn.endswith(s) for s in arm_joint_suffixes))
        ]

        if len(ur_joint_names) != 6:
            raise ValueError(
                "Expected 6 UR arm joints, found "
                f"{len(ur_joint_names)}: {ur_joint_names}"
            )

        j0 = [2.8, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        for joint_name, q0 in zip(ur_joint_names, j0):
            d.joint(joint_name).qpos = q0

        return m, d

    def _pose_from_offsets(self, T_obj: sm.SE3, offset: np.ndarray, ref_pose: sm.SE3) -> sm.SE3:
        p = T_obj.t + T_obj.R @ offset
        A = np.eye(4)
        A[:3, :3] = ref_pose.R
        A[:3, 3] = p
        return sm.SE3(A, check=False)

    def _retarget_dmp(self, spec: PhaseSpec, T_obj: sm.SE3) -> List[sm.SE3]:
        if spec.tau_scale <= 0:
            raise ValueError(f"{spec.name} tau scale must be > 0")

        start_pose = self._pose_from_offsets(T_obj, spec.start_offset, spec.ref_start)
        goal_pose = self._pose_from_offsets(T_obj, spec.goal_offset, spec.ref_goal)

        spec.dmp.position_dmp.p0 = start_pose.t
        spec.dmp.position_dmp.gp = goal_pose.t

        spec.dmp.quaternion_dmp.q0 = quaternion.quaternion(*smb.r2q(start_pose.R))
        spec.dmp.quaternion_dmp.go = quaternion.quaternion(*smb.r2q(goal_pose.R))
        spec.dmp.reset()

        tau = spec.dmp.tau * spec.tau_scale
        n_steps = max(2, int(round(len(spec.dmp.ts) * spec.tau_scale)))
        ts = np.linspace(0.0, tau, n_steps)

        p, _, _, q, _, _ = spec.dmp.rollout(ts, tau)
        return [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]

    def _log_step(self, phase: str, target_pose: sm.SE3, actual_pose: sm.SE3) -> None:
        self.rows.append(
            [
                float(self.data.time),
                phase,
                target_pose.t[0],
                target_pose.t[1],
                target_pose.t[2],
                *_rotvec_from_pose(target_pose),
                actual_pose.t[0],
                actual_pose.t[1],
                actual_pose.t[2],
                *_rotvec_from_pose(actual_pose),
                self.current_gripper_cmd,
                self.grasped,
            ]
        )

    def _control_step(self, ss: SimSync, target_pose: sm.SE3, phase: str) -> None:
        self.ur5e.controller.T_target = target_pose
        tau = self.ur5e.controller.step()
        self.ur5e.set_ctrl(tau)
        self.gripper.command(self.current_gripper_cmd)
        ss.step()

        actual_pose = self.ur5e.get_ee_pose()
        self._log_step(phase, target_pose, actual_pose)

    def _run_phase(self, ss: SimSync, phase_name: str, poses: List[sm.SE3]) -> None:
        for target_pose in poses:
            self._control_step(ss, target_pose, phase_name)

    def _is_object_in_gripper_contact(self) -> bool:
        in_contact, _ = get_contact_states(
            self.data,
            self.model,
            geom_names1=self.gripper.finger_geom_names,
            geom_names2=self.object_geom_name,
        )
        return in_contact

    def _hold_pose(self, ss: SimSync, pose: sm.SE3, phase: str, duration: float) -> None:
        n_steps = int(round(duration / self.model.opt.timestep))
        for _ in range(max(1, n_steps)):
            self._control_step(ss, pose, phase)

    def _wait_for_grasp(self, ss: SimSync, hold_pose: sm.SE3) -> bool:
        t0 = self.data.time
        while self.data.time - t0 < self.args.grasp_timeout:
            self.grasped = self._is_object_in_gripper_contact()
            self._control_step(ss, hold_pose, "wait_grasp")
            if self.grasped:
                return True
        return False

    def spin(self, ss: SimSync) -> None:
        T_obj = get_pose(self.model, self.data, self.object_body_name, ObjType.BODY)

        approach_poses = self._retarget_dmp(self.phase_specs[0], T_obj)
        task_poses = self._retarget_dmp(self.phase_specs[1], T_obj)

        self.current_gripper_cmd = self.args.gripper_open
        self.gripper.open()

        self.ur5e.move_l(approach_poses[0], ss)
        self._run_phase(ss, "approach", approach_poses)

        self.current_gripper_cmd = self.args.gripper_close
        self.gripper.close()
        self._hold_pose(ss, approach_poses[-1], "gripper_close", self.args.close_hold_s)

        self.grasped = self._wait_for_grasp(ss, approach_poses[-1])
        if not self.grasped:
            print("Grasp timeout reached, continuing to task phase.")

        self._run_phase(ss, "task", task_poses)

        self.current_gripper_cmd = self.args.gripper_open
        self.gripper.open()
        self._hold_pose(ss, task_poses[-1], "gripper_open", self.args.open_hold_s)

        if self.args.hold_final_s > 0:
            self._hold_pose(ss, task_poses[-1], "hold_final", self.args.hold_final_s)

        pd.DataFrame(
            self.rows,
            columns=[
                "sim_time",
                "phase",
                *[f"target_TCP_pose_{i}" for i in range(6)],
                *[f"actual_TCP_pose_{i}" for i in range(6)],
                "gripper_ctrl",
                "grasped",
            ],
        ).to_csv(self.replay_log_path, index=False)

        print(f"Saved orchestrated replay log: {self.replay_log_path}")

        # Keep the robot stable at the last pose.
        while True:
            self._control_step(ss, task_poses[-1], "hold_end")

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model


def setup_logging(loglevel: int) -> None:
    logformat = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=loglevel if loglevel is not None else logging.INFO,
        stream=sys.stdout,
        format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default="",
        help="Path to YAML config file. CLI flags override config values.",
    )

    pre_args, _ = pre_parser.parse_known_args()
    config_defaults: Dict[str, Any] = {}
    if pre_args.config:
        config_defaults = _load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Multi-phase UR5e simulation: approach DMP -> grasp -> task DMP.",
    )

    parser.add_argument("--approach-demo", default=None, help="Path to approach demo CSV.")
    parser.add_argument("--task-demo", default=None, help="Path to task demo CSV.")

    parser.add_argument(
        "--replay-log-out",
        default="",
        help="Output CSV path for phase replay logging.",
    )

    parser.add_argument(
        "--object-name",
        default="pickup_object",
        help="Name of the dynamic object body in the scene.",
    )
    parser.add_argument(
        "--object-pos",
        nargs=3,
        type=float,
        default=[0.62, -0.18, 0.93],
        metavar=("X", "Y", "Z"),
        help="Object body position in world frame.",
    )
    parser.add_argument(
        "--object-rotvec",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("RX", "RY", "RZ"),
        help="Object body orientation as axis-angle rotation vector.",
    )
    parser.add_argument(
        "--object-size",
        nargs=3,
        type=float,
        default=[0.015, 0.015, 0.03],
        metavar=("SX", "SY", "SZ"),
        help="Object half-sizes for generated box geom.",
    )
    parser.add_argument(
        "--object-mass",
        type=float,
        default=0.2,
        help="Mass of generated object body.",
    )

    parser.add_argument(
        "--approach-start-offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.20],
        metavar=("X", "Y", "Z"),
        help="Approach start offset from object frame, in meters.",
    )
    parser.add_argument(
        "--approach-goal-offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.10],
        metavar=("X", "Y", "Z"),
        help="Approach goal offset from object frame, in meters.",
    )
    parser.add_argument(
        "--task-start-offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.10],
        metavar=("X", "Y", "Z"),
        help="Task start offset from object frame, in meters.",
    )
    parser.add_argument(
        "--task-goal-offset",
        nargs=3,
        type=float,
        default=[0.16, 0.0, 0.12],
        metavar=("X", "Y", "Z"),
        help="Task goal offset from object frame, in meters.",
    )

    parser.add_argument(
        "--approach-tau-scale",
        type=float,
        default=1.2,
        help="Temporal scale for approach DMP rollout.",
    )
    parser.add_argument(
        "--task-tau-scale",
        type=float,
        default=1.0,
        help="Temporal scale for task DMP rollout.",
    )

    parser.add_argument(
        "--gripper-open",
        type=float,
        default=0.0,
        help="Robotiq HandE open command (0..255).",
    )
    parser.add_argument(
        "--gripper-close",
        type=float,
        default=255.0,
        help="Robotiq HandE close command (0..255).",
    )
    parser.add_argument(
        "--close-hold-s",
        type=float,
        default=0.5,
        help="Time to hold close command before grasp check.",
    )
    parser.add_argument(
        "--grasp-timeout",
        type=float,
        default=1.5,
        help="Timeout while waiting for grasp contact before task phase.",
    )
    parser.add_argument(
        "--open-hold-s",
        type=float,
        default=0.5,
        help="Time to hold open command at the end.",
    )
    parser.add_argument(
        "--hold-final-s",
        type=float,
        default=2.0,
        help="Optional duration to keep final pose before infinite hold.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        action="store_const",
        const=logging.INFO,
        help="Set log level to INFO.",
    )

    if config_defaults:
        parser.set_defaults(**config_defaults)

    args, _ = parser.parse_known_args()

    if not args.approach_demo or not args.task_demo:
        parser.error("Both --approach-demo and --task-demo must be provided via CLI or --config.")

    if args.approach_tau_scale <= 0 or args.task_tau_scale <= 0:
        parser.error("Tau scale values must be greater than 0.")

    return args


def main() -> None:
    args = _parse_args()

    if not args.replay_log_out:
        args.replay_log_out = str(_default_replay_log_path())

    setup_logging(args.loglevel)

    sim = MjSim(args)
    sim.run()


if __name__ == "__main__":
    main()
