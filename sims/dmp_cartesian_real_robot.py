import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add project root to path FIRST so modules can be found
_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import spatialmath as sm
import spatialmath.base as smb

from ctrl.dmp_cartesian.dmp_cartesian import DMPCartesian
from real.robot import Robot
from utils.math import npq2np
from utils.sm import make_tf


def _rotvec_from_pose(T: sm.SE3) -> list[float]:
    theta, v = T.angvec()
    rotvec = theta * v
    return [rotvec[0], rotvec[1], rotvec[2]]


def _default_replay_log_path(demo_path: str, tau_scale: float) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(demo_path).stem
    scale_tag = str(tau_scale).replace(".", "p")
    return Path("data/demo") / f"{stem}_dmp_real_replay_tau{scale_tag}_{stamp}.csv"


class RealRobotDMPExecutor:
    def __init__(
        self,
        demo_path: str,
        replay_log_path: str,
        tau_scale: float,
        robot_ip: str = "192.168.1.20",
        speed: float = 0.25,
        acc: float = 0.3,
    ) -> None:
        # Load demonstration
        df = pd.read_csv(demo_path)
        self.demo_path = Path(demo_path).expanduser().resolve()
        self.replay_log_path = Path(replay_log_path).expanduser().resolve()
        self.replay_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.tau_scale = tau_scale
        
        # Convert demo to SE3 poses
        self.pose_matrices = self.df_2_se3(df, "actual_TCP_pose")
        
        # Create and train DMP
        self.dmp = DMPCartesian()
        self.dmp.load(self.pose_matrices)
        
        # Rollout DMP with scaled time
        rollout_tau = self.dmp.tau * self.tau_scale
        n_steps = max(2, int(round(len(self.dmp.ts) * self.tau_scale)))
        rollout_ts = np.linspace(0.0, rollout_tau, n_steps)
        p, v, a, q, w, dw = self.dmp.rollout(rollout_ts, rollout_tau)
        
        self.dmp_poses = [make_tf(p[i], npq2np(q[i])) for i in range(len(p))]
        self.replay_rows: list[list[float]] = []
        
        # Connect to real robot
        self.robot = Robot(robot_ip=robot_ip)
        self.speed = speed
        self.acc = acc
        
        _logger.info(f"Connected to robot at {robot_ip}")
        _logger.info(f"Loaded {len(self.pose_matrices)} poses from demonstration")
        _logger.info(f"Generated {len(self.dmp_poses)} poses from DMP rollout")

    def df_2_se3(self, df: pd.DataFrame, pose_columns_prefix: str) -> List[sm.SE3]:
        """
        Converts pose columns in a DataFrame to a list of spatialmath SE3 matrices.

        Args:
            df: DataFrame containing pose data.
            pose_columns_prefix: The prefix of the pose columns (e.g., 'actual_TCP_pose').

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
            T_pose = sm.SE3.Rt(orientation.R, position)
            
            se3_matrices.append(T_pose)
        return se3_matrices

    def execute(self) -> None:
        """Execute the DMP trajectory on the real robot."""
        try:
            _logger.info("Starting DMP execution on real robot...")
            
            # Move to first pose
            _logger.info("Moving to initial pose...")
            self.robot.moveL(self.dmp_poses[0], speed=self.speed, acc=self.acc)
            _logger.info("Reached initial pose")
            
            # Get time step between poses
            dt = self.dmp.ts[1] - self.dmp.ts[0] if len(self.dmp.ts) > 1 else 0.01
            dt_scaled = dt * self.tau_scale
            
            _logger.info(f"Starting trajectory execution with {len(self.dmp_poses)} waypoints")
            
            # Execute trajectory
            for i, target_pose in enumerate(self.dmp_poses):
                # Use servoL for smooth continuous control
                self.robot.servoL(target_pose, speed=self.speed, acc=self.acc)
                
                # Get actual pose from robot
                actual_pose = self.robot.T_w_tcp
                
                # Log poses
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
                
                if (i + 1) % max(1, len(self.dmp_poses) // 10) == 0:
                    _logger.info(f"Progress: {i + 1}/{len(self.dmp_poses)} waypoints")
            
            # Save log
            pd.DataFrame(
                self.replay_rows,
                columns=[f"dmp_target_TCP_pose_{i}" for i in range(6)]
                + [f"actual_TCP_pose_{i}" for i in range(6)],
            ).to_csv(self.replay_log_path, index=False)
            _logger.info(f"Saved replay log: {self.replay_log_path}")
            
            _logger.info("DMP execution completed successfully")
            
        except Exception as e:
            _logger.error(f"Error during DMP execution: {e}")
            raise


_logger = logging.getLogger("dmp_real_robot")


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
        description="Execute DMP trajectory on real UR5e robot."
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
        "--robot-ip",
        default="192.168.1.20",
        help="IP address of the UR5e robot.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.25,
        help="Linear speed for robot motion (m/s).",
    )
    parser.add_argument(
        "--acc",
        type=float,
        default=0.3,
        help="Linear acceleration for robot motion (m/s^2).",
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
    
    executor = RealRobotDMPExecutor(
        args.demo,
        replay_log_path,
        args.tau_scale,
        robot_ip=args.robot_ip,
        speed=args.speed,
        acc=args.acc,
    )
    executor.execute()


if __name__ == "__main__":
    main()
