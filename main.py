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
    scene_path: Path, add_gelsight_mini: bool = False, add_unbrako: bool = False
) -> mj.MjModel:
    """Build a MuJoCo model from a scene and optional attached assets."""
    scene = mjcf.from_path(scene_path)

    repo_root = Path(__file__).resolve().parent

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
        help="Attach assets/gelsight_mini/gelsight_mini.xml to the scene.",
    )
    parser.add_argument(
        "--with-unbrako",
        action="store_true",
        help="Attach assets/unbrako/unbrako.xml to the scene.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scene_path = _resolve_path(args.scene)

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    model = _build_model(
        scene_path
    )
    _run_viewer(model)


if __name__ == "__main__":
    main()
