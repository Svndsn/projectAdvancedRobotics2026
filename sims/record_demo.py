import argparse
import time
from datetime import datetime
from pathlib import Path

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
import pandas as pd
import spatialmath as sm
import spatialmath.base as smb
from dm_control import mjcf

from ctrl.opspace.opspace import OpSpace
from robots.ur_robot import URRobot
from utils.mj import ObjType, get_names, get_pose, name2id


def _default_out_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/demo") / f"sim_{stamp}.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a MuJoCo UR5e TCP trajectory and save to data/demo."
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output CSV path (default: data/demo/sim_YYYYMMDD_HHMMSS.csv).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Sampling rate in Hz (default: 50).",
    )
    parser.add_argument(
        "--teach-alpha",
        type=float,
        default=0.08,
        help=(
            "How quickly the compliant hold target follows the current pose. "
            "Higher values make the robot easier to reposition (default: 0.08)."
        ),
    )
    return parser.parse_args()


def _build_ur5e_scene() -> tuple[mj.MjModel, mj.MjData]:
    repo_root = Path(__file__).parent.parent
    scene = mjcf.from_path(repo_root / "scenes/empty.xml")

    ur5e = mjcf.from_path(repo_root / "assets/universal_robots_ur5e/ur5e.xml")
    ur5e_base = ur5e.worldbody.find("body", "base")
    ur5e_base.pos = [
        0.22331901,
        0.37537452,
        0.08791326,
    ]
    ur5e_base.quat = [
        -0.19858483999999996,
        -0.00311175,
        0.0012299899999999998,
        0.98007799,
    ]

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

    actuator_config: dict = {
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

    flange_tool = mjcf.from_path(repo_root / "assets/props/flange_tool.xml")
    ur5e_attach_site = ur5e.find("site", "attachment_site")
    ur5e_attach_site.attach(flange_tool)

    table = mjcf.from_path(repo_root / "assets/props/flexcell_top.xml")
    scene.attach(table)

    mounting_plate = mjcf.from_path(repo_root / "assets/props/mounting_plate.xml")
    scene.attach(mounting_plate)

    drag_target = scene.worldbody.add(
        "body",
        name="drag_target",
        mocap="true",
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    drag_target.add(
        "geom",
        name="drag_target_vis",
        type="sphere",
        size=[0.025],
        rgba=[0.95, 0.35, 0.1, 0.55],
        contype="0",
        conaffinity="0",
    )
    drag_target.add(
        "site",
        name="drag_target_site",
        type="sphere",
        size=[0.01],
        rgba=[1.0, 0.9, 0.9, 1.0],
    )

    model = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
    data = mj.MjData(model)

    mj.mj_step(model, data)

    ur_joint_names = [jn for jn in get_names(model, ObjType.JOINT) if "ur5e" in jn]
    j0 = [2.8, -1.5708, 1.5708, -1.5708, -1.5708, 0]
    for i, jn in enumerate(ur_joint_names):
        data.joint(jn).qpos = j0[i]

    return model, data


def main() -> None:
    args = _parse_args()
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else _default_out_path().resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, data = _build_ur5e_scene()
    ur5e = URRobot(model, data, robot_type=URRobot.Type.UR5e)
    ur5e.controller = OpSpace(ur5e)

    # Use a soft task-space impedance so the arm supports itself in gravity,
    # but still yields when manually repositioned in the viewer.
    ur5e.controller.impedance_pos = np.asarray([250.0, 250.0, 250.0])
    ur5e.controller.impedance_ori = np.asarray([80.0, 80.0, 80.0])
    ur5e.controller.Kp_null = np.asarray([10.0, 10.0, 8.0, 6.0, 4.0, 2.0])
    ur5e.controller.Kpos = 1.0
    ur5e.controller.Kori = 1.0
    ur5e.controller.damping_pos = (
        ur5e.controller.damping_ratio * 2 * np.sqrt(ur5e.controller.impedance_pos)
    )
    ur5e.controller.damping_ori = (
        ur5e.controller.damping_ratio * 2 * np.sqrt(ur5e.controller.impedance_ori)
    )
    ur5e.controller.Kp = np.concatenate(
        [ur5e.controller.impedance_pos, ur5e.controller.impedance_ori],
        axis=0,
    )
    ur5e.controller.Kd = np.concatenate(
        [ur5e.controller.damping_pos, ur5e.controller.damping_ori],
        axis=0,
    )
    ur5e.controller.Kd_null = (
        ur5e.controller.damping_ratio * 2 * np.sqrt(ur5e.controller.Kp_null)
    )

    is_recording = False
    should_exit = False
    should_save = False
    last_sample_time = -np.inf
    rows = []

    target_body_id = name2id(model, "drag_target", ObjType.BODY)
    target_mocap_id = model.body_mocapid[target_body_id]

    def _key_callback(key: int) -> None:
        nonlocal is_recording, should_exit, should_save
        if key == glfw.KEY_SPACE:
            is_recording = not is_recording
            state = "ON" if is_recording else "OFF"
            print(f"Recording {state}")
        if key == glfw.KEY_ENTER:
            should_save = True
            should_exit = True
        if key == glfw.KEY_ESCAPE:
            should_exit = True

    # Initialize the draggable mocap target at the current TCP pose.
    T_world_tcp = get_pose(model, data, ur5e.tcp_id, ObjType.SITE)
    data.mocap_pos[target_mocap_id] = T_world_tcp.t
    data.mocap_quat[target_mocap_id] = smb.r2q(T_world_tcp.R)
    ur5e.controller.T_target = ur5e.get_ee_pose()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        key_callback=_key_callback,
    ) as viewer:
        viewer.cam.azimuth = model.vis.global_.azimuth
        viewer.cam.elevation = model.vis.global_.elevation
        viewer.cam.lookat = model.stat.center
        viewer.cam.distance = model.stat.extent

        print(
            "SPACE: toggle record | ENTER: save & exit | ESC: exit without saving"
        )
        print(
            "Drag the orange target sphere with the mouse and the arm will follow it."
        )
        while viewer.is_running():
            step_start = time.time()
            T_world_target = get_pose(model, data, "drag_target", ObjType.BODY)
            ur5e.controller.T_target = (
                get_pose(model, data, f"{ur5e.name}/base", ObjType.BODY).inv()
                @ T_world_target
            )

            # Apply task-space hold torques every step.
            tau = ur5e.controller.step()
            ur5e.set_ctrl(tau)
            mj.mj_step(model, data)

            if is_recording and data.time - last_sample_time >= 1.0 / args.rate:
                T_base_tcp: sm.SE3 = ur5e.get_ee_pose()
                theta, v = smb.r2angvec(T_base_tcp.R)
                rotvec = theta * v
                row = [
                    T_base_tcp.t[0],
                    T_base_tcp.t[1],
                    T_base_tcp.t[2],
                    rotvec[0],
                    rotvec[1],
                    rotvec[2],
                ]
                rows.append(row)
                last_sample_time = data.time

            viewer.sync()

            if should_exit:
                break

            sleep_for = model.opt.timestep - (time.time() - step_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    if should_save and rows:
        columns = [f"actual_TCP_pose_{i}" for i in range(6)]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    else:
        print("No data saved.")


if __name__ == "__main__":
    main()
