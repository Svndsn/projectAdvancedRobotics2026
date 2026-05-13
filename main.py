from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer
from dm_control import mjcf


def _resolve_path(path_like: str) -> Path:
    """Resolve a potentially relative path against the current working directory."""
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _build_model(
    scene_path: Path,
    add_gelsight_mini: bool = True,
    add_unbrako: bool = False,
    add_hande: bool = True,
) -> mj.MjModel:
    """Build a MuJoCo model from a scene and optional attached assets."""
    scene = mjcf.from_path(scene_path)

    repo_root = Path(__file__).resolve().parent
    assets_root = repo_root / "assets"

    table_geom = scene.find("geom", "sigmund_table")
    base_body = scene.find("body", "base")
    if table_geom is not None and base_body is not None:
        base_body.pos = [0.0, 0.0, float(table_geom.pos[2])]

    attachment_site = scene.find("site", "attachment_site")

    if add_hande and attachment_site is not None:
        hande = mjcf.from_path(assets_root / "robotiq_hande" / "hande.xml")
        attachment_site.attach(hande)

    if add_gelsight_mini and attachment_site is not None:
        gelsight = mjcf.from_path(assets_root / "gelsight_mini" / "gelsight_mini.xml")
        if add_hande:
            # Attach to one of the fingers instead of the TCP
            left_finger = hande.find("body", "hande_left_finger")
            # Create a mounting site on the finger
            mount_site = left_finger.add("site", name="gelsight_mount", pos=[-0.020, 0.0, 0.040], quat=[-0.5, 0.5, -0.5, 0.5])
            mount_site.attach(gelsight)
        else:
            attachment_site.attach(gelsight)

    if add_unbrako:
        unbrako = mjcf.from_path(assets_root / "unbrako" / "unbrako.xml")
        
        # To make it graspable, add a free joint and collision mass if they don't exist
        unbrako_body = unbrako.find("body", "unbrako")
        if unbrako_body is not None:
            if not unbrako_body.find_all("freejoint"):
                unbrako_body.add("freejoint", name="unbrako_free")
            if not unbrako_body.find_all("inertial"):
                unbrako_body.add("inertial", mass=0.1, pos=[0, 0, 0], diaginertia=[1e-4, 1e-4, 1e-4])
        
        # Attach to the worldbody (the table) instead of the robot's TCP
        if table_geom is not None:
            spawn_pos = [float(table_geom.pos[0]), float(table_geom.pos[1]) + 0.3, float(table_geom.pos[2]) + 0.05]
        else:
            spawn_pos = [0.4, -0.3, 0.8]
            
        spawn_site = scene.worldbody.add("site", name="unbrako_spawn", pos=spawn_pos, size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 0])
        spawn_site.attach(unbrako)

    return mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())


def _run_viewer(model: mj.MjModel) -> None:
    """Run a simple passive MuJoCo viewer loop."""
    data = mj.MjData(model)

    # Step once so that poses and cameras are initialized before rendering.
    mj.mj_step(model, data)

    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        viewer.cam.azimuth = model.vis.global_.azimuth
        viewer.cam.elevation = model.vis.global_.elevation
        viewer.cam.lookat = model.stat.center
        viewer.cam.distance = model.stat.extent

        while viewer.is_running():
            step_start = time.time()
            mj.mj_step(model, data)
            viewer.sync()

            # Basic wall-clock synchronization.
            sleep_for = model.opt.timestep - (time.time() - step_start)
            if sleep_for > 0:
                time.sleep(sleep_for)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a MuJoCo scene and optionally attach extra assets."
    )
    parser.add_argument(
        "scene",
        help="Path to the MuJoCo scene XML file (relative to cwd or absolute).",
    )
    parser.add_argument(
        "--with-gelsight-mini",
        action="store_true",
        default=True,
        help="Attach assets/gelsight_mini/gelsight_mini.xml to the scene.",
    )
    parser.add_argument(
        "--with-unbrako",
        action="store_true",
        help="Attach assets/unbrako/unbrako.xml to the scene.",
    )
    parser.add_argument(
        "--no-hande",
        dest="with_hande",
        action="store_false",
        default=True,
        help="Skip attaching the Robotiq HandE gripper.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scene_path = _resolve_path(args.scene)

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    model = _build_model(
        scene_path,
        add_gelsight_mini=args.with_gelsight_mini,
        add_unbrako=args.with_unbrako,
        add_hande=args.with_hande,
    )
    _run_viewer(model)


if __name__ == "__main__":
    main()