import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import spatialmath as sm

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from live_gelsight_pca import (
    FRAME_SIZE,
    GelSightPCATracker,
    SlipCalibration,
    TrackingResult,
    draw_axis_projection,
    draw_component_pca,
    draw_mask_overlay,
    draw_simple_prediction_window,
    draw_text,
)
from real.config import (
    GELSIGHT_CAMERA_INDEX,
    GELSIGHT_FRAME_SIZE,
    GELSIGHT_SLIP_HOLD_TIMEOUT_S,
    GELSIGHT_SLIP_ONLY_WHEN_MOVED,
    GELSIGHT_SLIP_ROTATION_AXIS,
    GELSIGHT_SLIP_ROTATION_COMPONENT,
    GELSIGHT_SLIP_ROTATION_DEADBAND_DEG,
    GELSIGHT_SLIP_ROTATION_RAD_PER_DEG,
    GELSIGHT_SLIP_ROTATION_SIGN,
    GELSIGHT_SLIP_SMOOTHING_ALPHA,
    GELSIGHT_SLIP_STALE_TIMEOUT_S,
    GELSIGHT_SLIP_STATE_PATH,
    GELSIGHT_SLIP_X_DEADBAND_PX,
    GELSIGHT_SLIP_X_M_PER_PX,
    GELSIGHT_SLIP_X_SIGN,
    GELSIGHT_SLIP_Y_DEADBAND_PX,
    GELSIGHT_SLIP_Y_M_PER_PX,
    GELSIGHT_SLIP_Y_SIGN,
)


@dataclass
class GelSightSidecarConfig:
    camera_index: int = GELSIGHT_CAMERA_INDEX
    frame_size: tuple[int, int] = GELSIGHT_FRAME_SIZE
    state_path: Path = GELSIGHT_SLIP_STATE_PATH
    show_preview: bool = True

    def make_calibration(self) -> SlipCalibration:
        return SlipCalibration(
            x_m_per_px=GELSIGHT_SLIP_X_M_PER_PX,
            y_m_per_px=GELSIGHT_SLIP_Y_M_PER_PX,
            rotation_rad_per_deg=GELSIGHT_SLIP_ROTATION_RAD_PER_DEG,
            x_sign=GELSIGHT_SLIP_X_SIGN,
            y_sign=GELSIGHT_SLIP_Y_SIGN,
            rotation_sign=GELSIGHT_SLIP_ROTATION_SIGN,
            x_deadband_px=GELSIGHT_SLIP_X_DEADBAND_PX,
            y_deadband_px=GELSIGHT_SLIP_Y_DEADBAND_PX,
            rotation_deadband_deg=GELSIGHT_SLIP_ROTATION_DEADBAND_DEG,
            rotation_component=GELSIGHT_SLIP_ROTATION_COMPONENT,
        )


def _identity_payload(timestamp: float, sequence: int, error: str | None = None) -> dict[str, Any]:
    return {
        "valid": False,
        "sequence": int(sequence),
        "timestamp": float(timestamp),
        "short_dx_px": 0.0,
        "short_dy_px": 0.0,
        "long_dx_px": 0.0,
        "long_dy_px": 0.0,
        "intersection_dx_px": 0.0,
        "intersection_dy_px": 0.0,
        "short_angle_diff_deg": 0.0,
        "long_angle_diff_deg": 0.0,
        "moved": False,
        "calibrated_x_m": 0.0,
        "calibrated_y_m": 0.0,
        "calibrated_rotation_rad": 0.0,
        "error": error,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_latest_sample(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


class GelSightPCASidecar:
    def __init__(self, config: GelSightSidecarConfig | None = None) -> None:
        self.config = config or GelSightSidecarConfig()
        self.tracker = GelSightPCATracker(
            frame_size=self.config.frame_size,
            calibration=self.config.make_calibration(),
        )
        self.sequence = 0

    def _publish_result(
        self,
        result: TrackingResult | None,
        timestamp: float,
        error: str | None = None,
    ) -> None:
        if result is not None and result.slip_sample is not None:
            payload = result.slip_sample.to_dict()
        else:
            payload = _identity_payload(timestamp, self.sequence, error=error)
        atomic_write_json(self.config.state_path, payload)

    def _render_preview(self, frame: np.ndarray, result: TrackingResult | None, error: str | None) -> None:
        vis = frame.copy()
        if self.tracker.ref_gray is None:
            draw_text(
                vis,
                "Press n to set neutral GelSight reference",
                (10, 30),
                (0, 255, 255),
                0.55,
                2,
            )
        elif error is not None:
            draw_text(vis, error, (10, 30), (0, 0, 255), 0.6, 2)
        elif result is not None:
            vis = draw_mask_overlay(vis, result.features.mask)
            colors = [(0, 0, 255), (255, 0, 0)]
            for i, component in enumerate(result.features.components):
                vis = draw_component_pca(vis, component, colors[i])
                draw_text(
                    vis,
                    f"C{i+1}: center=({component.center[0]:.1f}, {component.center[1]:.1f}) angle={component.angle_deg:.1f}",
                    (10, 25 + i * 20),
                    colors[i],
                )
            vis = draw_axis_projection(vis, result.projection, result.long_component)
            draw_text(
                vis,
                f"Intersection: ({result.projection.closest_point[0]:.1f}, {result.projection.closest_point[1]:.1f})",
                (10, 75),
                (255, 255, 0),
            )
            if result.slip_sample is not None:
                sample = result.slip_sample
                draw_text(
                    vis,
                    (
                        f"x={sample.calibrated_x_m:.4f}m y={sample.calibrated_y_m:.4f}m "
                        f"rot={sample.calibrated_rotation_rad:.4f}rad moved={sample.moved}"
                    ),
                    (10, 100),
                    (0, 255, 255),
                )
            else:
                draw_text(
                    vis,
                    "Press r to set movement reference",
                    (10, 100),
                    (0, 255, 255),
                )
            prediction_view = draw_simple_prediction_window(
                result.short_component,
                result.long_component,
                result.projection.closest_point,
            )
            cv2.imshow("GelSight Sidecar Prediction", prediction_view)
        cv2.imshow("GelSight Slip Sidecar", vis)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open GelSight camera at index {self.config.camera_index}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    now = time.monotonic()
                    self._publish_result(None, now, error="Failed to grab frame")
                    continue

                frame = self.tracker._prepare_frame(frame)
                now = time.monotonic()
                result = None
                error = None

                if self.tracker.ref_gray is not None:
                    try:
                        result = self.tracker.process_frame(frame, self.sequence, now)
                    except ValueError as exc:
                        error = str(exc)

                self._publish_result(result, now, error=error)

                if self.config.show_preview:
                    self._render_preview(frame, result, error)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("n"):
                        self.tracker.set_neutral_reference(frame)
                    elif key == ord("r"):
                        if self.tracker.ref_gray is not None:
                            try:
                                self.tracker.set_movement_reference(frame)
                            except ValueError:
                                pass
                    elif key == ord("q") or key == 27:
                        break

                self.sequence += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()


class GelSightPCASlipCorrectionProvider:
    def __init__(self) -> None:
        self.state_path = Path(GELSIGHT_SLIP_STATE_PATH)
        self.stale_timeout_s = float(GELSIGHT_SLIP_STALE_TIMEOUT_S)
        self.hold_timeout_s = float(GELSIGHT_SLIP_HOLD_TIMEOUT_S)
        self.smoothing_alpha = float(GELSIGHT_SLIP_SMOOTHING_ALPHA)
        self.only_when_moved = bool(GELSIGHT_SLIP_ONLY_WHEN_MOVED)
        self.rotation_axis = GELSIGHT_SLIP_ROTATION_AXIS
        self.reset()

    def reset(self) -> None:
        self._reset_timestamp = time.monotonic()
        self._filtered_translation = np.zeros(3, dtype=float)
        self._filtered_rotation_rad = 0.0
        self._last_valid_sample_timestamp: float | None = None

    def _decay(self) -> sm.SE3:
        alpha = min(max(self.smoothing_alpha, 0.0), 1.0)
        self._filtered_translation *= max(0.0, 1.0 - alpha)
        self._filtered_rotation_rad *= max(0.0, 1.0 - alpha)
        return self._build_pose()

    def _build_pose(self) -> sm.SE3:
        if self.rotation_axis == "x":
            R = sm.SO3.Rx(self._filtered_rotation_rad).R
        elif self.rotation_axis == "y":
            R = sm.SO3.Ry(self._filtered_rotation_rad).R
        else:
            R = sm.SO3.Rz(self._filtered_rotation_rad).R
        return sm.SE3.Rt(R, self._filtered_translation, check=False)

    def _sample_is_fresh(self, sample: dict[str, Any], now: float) -> bool:
        timestamp = float(sample.get("timestamp", -np.inf))
        return (now - timestamp) <= self.stale_timeout_s and timestamp > self._reset_timestamp

    def _should_hold_previous_value(self, now: float) -> bool:
        return (
            self._last_valid_sample_timestamp is not None
            and (now - self._last_valid_sample_timestamp) <= self.hold_timeout_s
        )

    def get_correction(
        self,
        actual_pose: sm.SE3,
        target_pose: sm.SE3,
        phase: str,
        t: float,
    ) -> sm.SE3:
        del actual_pose, target_pose, phase, t
        now = time.monotonic()
        sample = load_latest_sample(self.state_path)
        if sample is None:
            return self._decay()

        try:
            valid = bool(sample.get("valid", False))
            moved = bool(sample.get("moved", False))
            x_m = float(sample.get("calibrated_x_m", 0.0))
            y_m = float(sample.get("calibrated_y_m", 0.0))
            rotation_rad = float(sample.get("calibrated_rotation_rad", 0.0))
        except (TypeError, ValueError):
            return self._decay()

        if not valid or not self._sample_is_fresh(sample, now):
            if self._should_hold_previous_value(now):
                return self._build_pose()
            return self._decay()

        if self.only_when_moved and not moved:
            return self._decay()

        self._last_valid_sample_timestamp = float(sample["timestamp"])
        alpha = min(max(self.smoothing_alpha, 0.0), 1.0)
        measurement = np.array([x_m, y_m, 0.0], dtype=float)
        self._filtered_translation = alpha * measurement + (1.0 - alpha) * self._filtered_translation
        self._filtered_rotation_rad = alpha * rotation_rad + (1.0 - alpha) * self._filtered_rotation_rad
        return self._build_pose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GelSight PCA slip sidecar.")
    parser.add_argument("--camera-index", type=int, default=GELSIGHT_CAMERA_INDEX)
    parser.add_argument("--state-path", type=Path, default=GELSIGHT_SLIP_STATE_PATH)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without OpenCV preview windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = GelSightSidecarConfig(
        camera_index=args.camera_index,
        frame_size=GELSIGHT_FRAME_SIZE or FRAME_SIZE,
        state_path=args.state_path,
        show_preview=not args.no_preview,
    )
    GelSightPCASidecar(config).run()


if __name__ == "__main__":
    main()
