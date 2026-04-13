import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialmath as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ctrl.dmp_cartesian.dmp_cartesian import DMPCartesian
from utils.math import npq2np
from utils.sm import make_tf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot recorded Cartesian demonstrations against a DMP fit."
    )
    parser.add_argument(
        "--demo",
        required=True,
        help="Path to a recorded demonstration CSV.",
    )
    parser.add_argument(
        "--outdir",
        default="data/plots",
        help="Directory to save output PNGs.",
    )
    parser.add_argument(
        "--replay-log",
        default="",
        help="Optional replay log CSV from sims.dmp_cartesian_ur5e.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open the figures interactively after saving.",
    )
    return parser.parse_args()


def _df_to_se3(df: pd.DataFrame, pose_columns_prefix: str) -> list[sm.SE3]:
    poses = []
    for _, row in df.iterrows():
        position = row[[f"{pose_columns_prefix}_{i}" for i in range(3)]].to_numpy()
        axis_angle = row[[f"{pose_columns_prefix}_{i}" for i in range(3, 6)]].to_numpy()
        angle = np.linalg.norm(axis_angle)
        if angle > 1e-12:
            orientation = sm.SO3.AngVec(angle, axis_angle / angle)
        else:
            orientation = sm.SO3()
        poses.append(sm.SE3.Rt(orientation.R, position))
    return poses


def _poses_to_rotvecs(poses: list[sm.SE3]) -> np.ndarray:
    rotvecs = np.zeros((len(poses), 3))
    for i, pose in enumerate(poses):
        rotvecs[i] = pose.angvec()[0] * pose.angvec()[1]
    return rotvecs


def _relative_orientation_error_deg(reference: list[sm.SE3], compared: list[sm.SE3]) -> np.ndarray:
    errs = np.zeros(len(reference))
    for i, (T_ref, T_cmp) in enumerate(zip(reference, compared)):
        dR = T_ref.R.T @ T_cmp.R
        angle = np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
        errs[i] = np.degrees(angle)
    return errs


def _save_position_plot(
    t: np.ndarray,
    actual_pos: np.ndarray,
    dmp_pos: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = ["x [m]", "y [m]", "z [m]"]
    colors = ["#b03a2e", "#2874a6", "#239b56"]

    for i, ax in enumerate(axes):
        ax.plot(t, actual_pos[:, i], color=colors[i], linewidth=2, label="actual")
        ax.plot(
            t, dmp_pos[:, i], color="black", linestyle="--", linewidth=1.5, label="dmp"
        )
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)

    axes[0].legend(loc="best", ncol=4)
    axes[-1].set_xlabel("sample")
    fig.suptitle("Cartesian Position: Recorded vs DMP")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_orientation_plot(
    t: np.ndarray,
    actual_rotvec: np.ndarray,
    dmp_rotvec: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = ["rx [rad]", "ry [rad]", "rz [rad]"]
    colors = ["#8e44ad", "#16a085", "#c0392b"]

    for i, ax in enumerate(axes):
        ax.plot(t, actual_rotvec[:, i], color=colors[i], linewidth=2, label="actual")
        ax.plot(
            t, dmp_rotvec[:, i], color="black", linestyle="--", linewidth=1.5, label="dmp"
        )
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)

    axes[0].legend(loc="best", ncol=4)
    axes[-1].set_xlabel("sample")
    fig.suptitle("Orientation Rotvec: Recorded vs DMP")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_path_plot(
    actual_pos: np.ndarray,
    dmp_pos: np.ndarray,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], linewidth=2, label="actual")
    ax.plot(
        dmp_pos[:, 0], dmp_pos[:, 1], dmp_pos[:, 2], linestyle="--", linewidth=1.5, label="dmp"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D Path Comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_error_plot(
    t: np.ndarray,
    pos_err: np.ndarray,
    ori_err_deg: np.ndarray,
    out_path: Path,
    title: str = "DMP Fit Error vs Recorded Pose",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, pos_err * 1000.0, color="#c0392b", linewidth=2)
    axes[0].set_ylabel("pos err [mm]")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, ori_err_deg, color="#2c3e50", linewidth=2)
    axes[1].set_ylabel("ori err [deg]")
    axes[1].set_xlabel("sample")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_replay_comparison_plot(
    t: np.ndarray,
    dmp_target_pos: np.ndarray,
    replay_actual_pos: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = ["x [m]", "y [m]", "z [m]"]
    for i, ax in enumerate(axes):
        ax.plot(t, dmp_target_pos[:, i], linewidth=2, label="dmp_target")
        ax.plot(
            t,
            replay_actual_pos[:, i],
            linestyle="--",
            linewidth=1.5,
            label="executed_actual",
        )
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("sample")
    fig.suptitle("Replay Execution: DMP Target vs Executed Actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_plots(
    demo_path: str | Path,
    outdir: str | Path = "data/plots",
    replay_log: str | Path = "",
) -> None:
    demo_path = Path(demo_path).expanduser().resolve()
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(demo_path)
    actual_poses = _df_to_se3(df, "actual_TCP_pose")
    dmp = DMPCartesian()
    dmp.load(actual_poses)
    p, _, _, q, _, _ = dmp.rollout(dmp.ts, dmp.tau)
    dmp_poses = [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]

    actual_pos = np.array([pose.t for pose in actual_poses])
    actual_rotvec = _poses_to_rotvecs(actual_poses)
    dmp_pos = np.array([pose.t for pose in dmp_poses])
    dmp_rotvec = _poses_to_rotvecs(dmp_poses)
    pos_err = np.linalg.norm(dmp_pos - actual_pos, axis=1)
    ori_err_deg = _relative_orientation_error_deg(actual_poses, dmp_poses)
    t = np.arange(len(actual_poses))

    stem = demo_path.stem
    _save_position_plot(
        t,
        actual_pos,
        dmp_pos,
        outdir / f"{stem}_actual_position.png",
    )
    _save_orientation_plot(
        t,
        actual_rotvec,
        dmp_rotvec,
        outdir / f"{stem}_actual_orientation.png",
    )
    _save_path_plot(
        actual_pos,
        dmp_pos,
        outdir / f"{stem}_actual_path3d.png",
    )
    _save_error_plot(
        t,
        pos_err,
        ori_err_deg,
        outdir / f"{stem}_actual_dmp_error.png",
    )

    print(f"Demo file: {demo_path}")
    print("Pose source: actual_TCP_pose")
    print(f"Saved plots to: {outdir}")
    print(
        "DMP fit summary: "
        f"mean_pos_err={pos_err.mean() * 1000.0:.2f} mm, "
        f"max_pos_err={pos_err.max() * 1000.0:.2f} mm, "
        f"mean_ori_err={ori_err_deg.mean():.3f} deg, "
        f"max_ori_err={ori_err_deg.max():.3f} deg"
    )

    if replay_log:
        replay_log_path = Path(replay_log).expanduser().resolve()
        if not replay_log_path.exists():
            print(f"Replay log not found, skipping replay plots: {replay_log_path}")
            return
        replay_df = pd.read_csv(replay_log_path)
        dmp_target_poses = _df_to_se3(replay_df, "dmp_target_TCP_pose")
        replay_actual_poses = _df_to_se3(replay_df, "actual_TCP_pose")
        replay_target_pos = np.array([pose.t for pose in dmp_target_poses])
        replay_actual_pos = np.array([pose.t for pose in replay_actual_poses])
        replay_pos_err = np.linalg.norm(replay_actual_pos - replay_target_pos, axis=1)
        replay_ori_err = _relative_orientation_error_deg(
            dmp_target_poses, replay_actual_poses
        )
        replay_t = np.arange(len(replay_actual_poses))
        replay_stem = replay_log_path.stem

        _save_replay_comparison_plot(
            replay_t,
            replay_target_pos,
            replay_actual_pos,
            outdir / f"{replay_stem}_position_compare.png",
        )
        _save_error_plot(
            replay_t,
            replay_pos_err,
            replay_ori_err,
            outdir / f"{replay_stem}_execution_error.png",
            title="Replay Execution Error: DMP Target vs Executed Actual",
        )
        print(f"Replay log: {replay_log_path}")
        print(
            "Replay execution summary: "
            f"mean_pos_err={replay_pos_err.mean() * 1000.0:.2f} mm, "
            f"max_pos_err={replay_pos_err.max() * 1000.0:.2f} mm, "
            f"mean_ori_err={replay_ori_err.mean():.3f} deg, "
            f"max_ori_err={replay_ori_err.max():.3f} deg"
        )


def main() -> None:
    args = _parse_args()
    generate_plots(args.demo, args.outdir, args.replay_log)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
