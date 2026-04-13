import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from rtde_receive import RTDEReceiveInterface


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a UR TCP trajectory and save to data/demo."
    )
    parser.add_argument("--robot-ip", required=True, help="UR controller IP address.")
    parser.add_argument(
        "--out",
        default="",
        help="Output CSV path (default: data/demo/real_YYYYMMDD_HHMMSS.csv).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Sampling rate in Hz (default: 50).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording duration in seconds (0 means manual stop).",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use enter-to-start and enter-to-stop (recommended).",
    )
    return parser.parse_args()


def _default_out_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/demo") / f"real_{stamp}.csv"


def main() -> None:
    args = _parse_args()
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else _default_out_path().resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recv = RTDEReceiveInterface(args.robot_ip)

    dt = 1.0 / args.rate
    rows = []

    if args.manual or args.duration <= 0:
        input("Press Enter to start recording...")

        stop_event = threading.Event()

        def _wait_for_stop() -> None:
            input("Press Enter to stop recording...")
            stop_event.set()

        threading.Thread(target=_wait_for_stop, daemon=True).start()
        t0 = time.time()
        while not stop_event.is_set():
            pose = recv.getActualTCPPose()  # [x, y, z, rx, ry, rz]
            rows.append(pose)
            time.sleep(dt)
        elapsed = time.time() - t0
        print(f"Stopped. Recorded {len(rows)} samples in {elapsed:.2f}s.")
    else:
        n_samples = max(1, int(args.duration * args.rate))
        print(
            f"Recording {n_samples} samples at {args.rate:.1f} Hz to {out_path}.\n"
            "Move the robot in freedrive/teach mode now."
        )
        t0 = time.time()
        for _ in range(n_samples):
            pose = recv.getActualTCPPose()  # [x, y, z, rx, ry, rz]
            rows.append(pose)
            time.sleep(dt)
        elapsed = time.time() - t0
        print(f"Done. Recorded {len(rows)} samples in {elapsed:.2f}s.")

    columns = [f"actual_TCP_pose_{i}" for i in range(6)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
