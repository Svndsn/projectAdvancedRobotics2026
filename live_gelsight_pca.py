import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np


CAMERA_INDEX = 2
FRAME_SIZE = (640, 640)

DIFF_THRESHOLD = 10
MIN_AREA_PIXELS = 10

CENTER_THRESHOLD_PX = 10.0
ANGLE_THRESHOLD_DEG = 10.0
AREA_THRESHOLD_RATIO = 0.20

AXIS_DRAW_LENGTH_PX = 350

PIXELS_PER_MM = 5.0
KEY_LENGTH_MM = 130.0
TIP_DIRECTION_SIGN = -1


@dataclass
class ComponentFeatures:
    center: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    angle_deg: float
    area_pixels: int
    mask: np.ndarray


@dataclass
class ImageFeatures:
    components: list[ComponentFeatures]
    mask: np.ndarray


@dataclass
class ComponentMovement:
    dx_px: float
    dy_px: float
    distance_px: float
    angle_diff_deg: float
    area_diff_pixels: int
    area_diff_ratio: float
    moved: bool


@dataclass
class AxisProjection:
    axis_point: np.ndarray
    axis_direction: np.ndarray
    closest_point: np.ndarray
    distance_px: float
    signed_distance_px: float


@dataclass
class SlipCalibration:
    x_m_per_px: float = 1e-4
    y_m_per_px: float = 1e-4
    rotation_rad_per_deg: float = np.pi / 180.0
    x_sign: float = 1.0
    y_sign: float = 1.0
    rotation_sign: float = 1.0
    x_deadband_px: float = 0.0
    y_deadband_px: float = 0.0
    rotation_deadband_deg: float = 0.0
    rotation_component: str = "long"


@dataclass
class SlipEstimate:
    valid: bool
    sequence: int
    timestamp: float
    short_dx_px: float
    short_dy_px: float
    long_dx_px: float
    long_dy_px: float
    intersection_dx_px: float
    intersection_dy_px: float
    short_angle_diff_deg: float
    long_angle_diff_deg: float
    moved: bool
    calibrated_x_m: float
    calibrated_y_m: float
    calibrated_rotation_rad: float
    error: str | None = None

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        return asdict(self)


@dataclass
class TrackingResult:
    features: ImageFeatures
    short_component: ComponentFeatures
    long_component: ComponentFeatures
    projection: AxisProjection
    slip_sample: SlipEstimate | None


def normalize_pca_angle_deg(angle_deg: float) -> float:
    angle_deg = angle_deg % 180.0
    if angle_deg < 0:
        angle_deg += 180.0
    return angle_deg


def angle_difference_deg(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_a - angle_b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def signed_pca_angle_difference_deg(angle_deg: float, ref_angle_deg: float) -> float:
    return float((angle_deg - ref_angle_deg + 90.0) % 180.0 - 90.0)


def compute_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    centered = points - mean

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return mean, eigenvalues, eigenvectors


def extract_two_component_masks(mask: np.ndarray) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        raise ValueError("No object detected")

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_indices = np.argsort(component_areas)[::-1]

    component_masks = []
    for idx in sorted_indices:
        label = idx + 1
        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_AREA_PIXELS:
            continue

        component_mask = np.zeros_like(mask)
        component_mask[labels == label] = 255
        component_masks.append(component_mask)

        if len(component_masks) == 2:
            break

    if len(component_masks) < 2:
        raise ValueError("Less than two components detected")

    return component_masks


def extract_component_features(component_mask: np.ndarray) -> ComponentFeatures:
    ys, xs = np.where(component_mask > 0)
    points = np.column_stack((xs, ys)).astype(np.float64)

    if len(points) < 2:
        raise ValueError("Not enough pixels for PCA")

    center, eigenvalues, eigenvectors = compute_pca(points)
    principal_axis = eigenvectors[:, 0]

    angle_rad = np.arctan2(-principal_axis[1], principal_axis[0])
    angle_deg = normalize_pca_angle_deg(np.degrees(angle_rad))
    area_pixels = int(cv2.countNonZero(component_mask))

    return ComponentFeatures(
        center=center,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        angle_deg=angle_deg,
        area_pixels=area_pixels,
        mask=component_mask,
    )


def extract_features_from_frame(frame: np.ndarray, ref_gray: np.ndarray) -> ImageFeatures:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, ref_gray)

    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    component_masks = extract_two_component_masks(mask)
    components = [extract_component_features(component_mask) for component_mask in component_masks]

    combined_mask = np.zeros_like(mask)
    for component in components:
        combined_mask[component.mask > 0] = 255

    return ImageFeatures(
        components=components,
        mask=combined_mask,
    )


def detect_component_movement(
    ref_component: ComponentFeatures,
    current_component: ComponentFeatures,
) -> ComponentMovement:
    dx_px = float(current_component.center[0] - ref_component.center[0])
    dy_px = float(current_component.center[1] - ref_component.center[1])
    distance_px = float(np.hypot(dx_px, dy_px))
    angle_diff_deg = float(
        angle_difference_deg(
            current_component.angle_deg,
            ref_component.angle_deg,
        )
    )
    area_diff_pixels = int(current_component.area_pixels - ref_component.area_pixels)
    area_diff_ratio = abs(area_diff_pixels) / max(ref_component.area_pixels, 1)

    moved = (
        distance_px > CENTER_THRESHOLD_PX
        or angle_diff_deg > ANGLE_THRESHOLD_DEG
        or area_diff_ratio > AREA_THRESHOLD_RATIO
    )

    return ComponentMovement(
        dx_px=dx_px,
        dy_px=dy_px,
        distance_px=distance_px,
        angle_diff_deg=angle_diff_deg,
        area_diff_pixels=area_diff_pixels,
        area_diff_ratio=area_diff_ratio,
        moved=moved,
    )


def get_short_and_long_components(
    features: ImageFeatures,
) -> tuple[ComponentFeatures, ComponentFeatures]:
    c1, c2 = features.components

    vertical_score_1 = abs(c1.angle_deg - 90.0)
    vertical_score_2 = abs(c2.angle_deg - 90.0)

    if vertical_score_1 < vertical_score_2:
        short_component = c1
        long_component = c2
    else:
        short_component = c2
        long_component = c1

    return short_component, long_component


def compute_axis_projection(
    short_component: ComponentFeatures,
    long_component: ComponentFeatures,
) -> AxisProjection:
    p = short_component.center
    r = short_component.eigenvectors[:, 0]
    r = r / np.linalg.norm(r)

    q = long_component.center
    s = long_component.eigenvectors[:, 0]
    s = s / np.linalg.norm(s)

    cross_rs = r[0] * s[1] - r[1] * s[0]
    if abs(cross_rs) < 1e-6:
        raise ValueError("Major PCA axes are parallel, no intersection found")

    q_minus_p = q - p
    t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / cross_rs
    intersection_point = p + t * r

    return AxisProjection(
        axis_point=p,
        axis_direction=r,
        closest_point=intersection_point,
        distance_px=0.0,
        signed_distance_px=0.0,
    )


def _apply_deadband(value: float, deadband: float) -> float:
    if abs(value) <= deadband:
        return 0.0
    return value


def calibrate_slip_estimate(
    intersection_dx_px: float,
    intersection_dy_px: float,
    rotation_diff_deg: float,
    calibration: SlipCalibration,
) -> tuple[float, float, float]:
    dx_px = _apply_deadband(intersection_dx_px, calibration.x_deadband_px)
    dy_px = _apply_deadband(intersection_dy_px, calibration.y_deadband_px)
    dtheta_deg = _apply_deadband(rotation_diff_deg, calibration.rotation_deadband_deg)

    x_m = calibration.x_sign * calibration.x_m_per_px * dx_px
    y_m = calibration.y_sign * calibration.y_m_per_px * dy_px
    rotation_rad = (
        calibration.rotation_sign * calibration.rotation_rad_per_deg * dtheta_deg
    )
    return float(x_m), float(y_m), float(rotation_rad)


def compute_slip_estimate(
    ref_features: ImageFeatures,
    current_features: ImageFeatures,
    sequence: int,
    timestamp: float,
    calibration: SlipCalibration | None = None,
) -> SlipEstimate:
    calibration = calibration or SlipCalibration()

    ref_short, ref_long = get_short_and_long_components(ref_features)
    current_short, current_long = get_short_and_long_components(current_features)

    ref_projection = compute_axis_projection(ref_short, ref_long)
    current_projection = compute_axis_projection(current_short, current_long)

    short_movement = detect_component_movement(ref_short, current_short)
    long_movement = detect_component_movement(ref_long, current_long)

    intersection_dx_px = float(
        current_projection.closest_point[0] - ref_projection.closest_point[0]
    )
    intersection_dy_px = float(
        current_projection.closest_point[1] - ref_projection.closest_point[1]
    )
    intersection_distance_px = float(np.hypot(intersection_dx_px, intersection_dy_px))

    short_angle_diff_deg = signed_pca_angle_difference_deg(
        current_short.angle_deg,
        ref_short.angle_deg,
    )
    long_angle_diff_deg = signed_pca_angle_difference_deg(
        current_long.angle_deg,
        ref_long.angle_deg,
    )
    rotation_source_deg = (
        long_angle_diff_deg
        if calibration.rotation_component == "long"
        else short_angle_diff_deg
    )
    calibrated_x_m, calibrated_y_m, calibrated_rotation_rad = calibrate_slip_estimate(
        intersection_dx_px=intersection_dx_px,
        intersection_dy_px=intersection_dy_px,
        rotation_diff_deg=rotation_source_deg,
        calibration=calibration,
    )

    moved = (
        short_movement.moved
        or long_movement.moved
        or intersection_distance_px > CENTER_THRESHOLD_PX
    )

    return SlipEstimate(
        valid=True,
        sequence=sequence,
        timestamp=float(timestamp),
        short_dx_px=short_movement.dx_px,
        short_dy_px=short_movement.dy_px,
        long_dx_px=long_movement.dx_px,
        long_dy_px=long_movement.dy_px,
        intersection_dx_px=intersection_dx_px,
        intersection_dy_px=intersection_dy_px,
        short_angle_diff_deg=float(short_angle_diff_deg),
        long_angle_diff_deg=float(long_angle_diff_deg),
        moved=bool(moved),
        calibrated_x_m=calibrated_x_m,
        calibrated_y_m=calibrated_y_m,
        calibrated_rotation_rad=calibrated_rotation_rad,
    )


class GelSightPCATracker:
    def __init__(
        self,
        frame_size: tuple[int, int] = FRAME_SIZE,
        calibration: SlipCalibration | None = None,
    ) -> None:
        self.frame_size = frame_size
        self.calibration = calibration or SlipCalibration()
        self.ref_gray: np.ndarray | None = None
        self.ref_features: ImageFeatures | None = None

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if tuple(frame.shape[1::-1]) != self.frame_size:
            return cv2.resize(frame, self.frame_size)
        return frame

    def set_neutral_reference(self, frame: np.ndarray) -> np.ndarray:
        frame = self._prepare_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.ref_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        self.ref_features = None
        return frame

    def set_movement_reference(self, frame: np.ndarray) -> ImageFeatures:
        if self.ref_gray is None:
            raise ValueError("Set neutral reference first")
        frame = self._prepare_frame(frame)
        self.ref_features = extract_features_from_frame(frame, self.ref_gray)
        return self.ref_features

    def process_frame(
        self,
        frame: np.ndarray,
        sequence: int = 0,
        timestamp: float | None = None,
    ) -> TrackingResult:
        if self.ref_gray is None:
            raise ValueError("Neutral reference has not been set")

        frame = self._prepare_frame(frame)
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        current_features = extract_features_from_frame(frame, self.ref_gray)
        short_component, long_component = get_short_and_long_components(current_features)
        projection = compute_axis_projection(short_component, long_component)

        slip_sample = None
        if self.ref_features is not None:
            slip_sample = compute_slip_estimate(
                ref_features=self.ref_features,
                current_features=current_features,
                sequence=sequence,
                timestamp=timestamp,
                calibration=self.calibration,
            )

        return TrackingResult(
            features=current_features,
            short_component=short_component,
            long_component=long_component,
            projection=projection,
            slip_sample=slip_sample,
        )


def draw_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


def draw_component_pca(
    frame: np.ndarray,
    component: ComponentFeatures,
    color: tuple[int, int, int],
) -> np.ndarray:
    vis = frame.copy()

    cx, cy = component.center
    center = (int(round(cx)), int(round(cy)))

    scale = 3.0
    major_len = int(scale * np.sqrt(component.eigenvalues[0]))
    minor_len = int(scale * np.sqrt(component.eigenvalues[1]))

    major_vec = component.eigenvectors[:, 0]
    minor_vec = component.eigenvectors[:, 1]

    major_p1 = (
        int(round(cx - major_vec[0] * major_len)),
        int(round(cy - major_vec[1] * major_len)),
    )
    major_p2 = (
        int(round(cx + major_vec[0] * major_len)),
        int(round(cy + major_vec[1] * major_len)),
    )

    minor_p1 = (
        int(round(cx - minor_vec[0] * minor_len)),
        int(round(cy - minor_vec[1] * minor_len)),
    )
    minor_p2 = (
        int(round(cx + minor_vec[0] * minor_len)),
        int(round(cy + minor_vec[1] * minor_len)),
    )

    cv2.line(vis, major_p1, major_p2, color, 2)
    cv2.line(vis, minor_p1, minor_p2, (0, 255, 255), 2)
    cv2.circle(vis, center, 5, color, -1)

    return vis


def draw_axis_projection(
    frame: np.ndarray,
    projection: AxisProjection,
    long_component: ComponentFeatures,
) -> np.ndarray:
    vis = frame.copy()

    axis_point = projection.axis_point
    axis_dir = projection.axis_direction
    closest_point = projection.closest_point
    long_center = long_component.center

    p_axis_1 = axis_point - AXIS_DRAW_LENGTH_PX * axis_dir
    p_axis_2 = axis_point + AXIS_DRAW_LENGTH_PX * axis_dir

    p_axis_1 = tuple(np.round(p_axis_1).astype(int))
    p_axis_2 = tuple(np.round(p_axis_2).astype(int))

    p_long = tuple(np.round(long_center).astype(int))
    p_closest = tuple(np.round(closest_point).astype(int))

    cv2.line(vis, p_axis_1, p_axis_2, (255, 255, 0), 2)
    cv2.line(vis, p_long, p_closest, (255, 0, 255), 2)

    cv2.circle(vis, p_closest, 6, (255, 255, 0), -1)
    cv2.circle(vis, p_long, 6, (255, 0, 255), -1)

    cv2.putText(
        vis,
        "projected point",
        (p_closest[0] + 8, p_closest[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 0),
        1,
    )

    return vis


def draw_text(
    vis: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    cv2.putText(
        vis,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


def draw_simple_prediction_window(
    short_component: ComponentFeatures,
    long_component: ComponentFeatures,
    estimated_point: np.ndarray,
) -> np.ndarray:
    canvas = np.zeros((640, 1250, 3), dtype=np.uint8)

    short_pt = short_component.center
    long_pt = long_component.center
    est_pt = estimated_point

    short_center = tuple(np.round(short_pt).astype(int))
    long_center = tuple(np.round(long_pt).astype(int))
    estimated = tuple(np.round(est_pt).astype(int))

    long_axis = long_component.eigenvectors[:, 0]
    long_axis = long_axis / np.linalg.norm(long_axis)

    length_px = KEY_LENGTH_MM * PIXELS_PER_MM
    estimated_tip_point = est_pt + TIP_DIRECTION_SIGN * length_px * long_axis
    estimated_tip = tuple(np.round(estimated_tip_point).astype(int))

    pts = np.array([short_pt, long_pt, est_pt])
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)

    margin = 30
    top_left = (int(min_xy[0] - margin), int(min_xy[1] - margin))
    bottom_right = (int(max_xy[0] + margin), int(max_xy[1] + margin))

    cv2.rectangle(canvas, top_left, bottom_right, (200, 200, 200), 2)
    cv2.putText(
        canvas,
        "visible region",
        (top_left[0], top_left[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    cv2.circle(canvas, short_center, 10, (0, 0, 255), -1)
    cv2.circle(canvas, long_center, 10, (255, 0, 0), -1)
    cv2.circle(canvas, estimated, 10, (0, 255, 255), -1)
    cv2.circle(canvas, estimated_tip, 10, (0, 255, 0), -1)

    cv2.line(canvas, short_center, estimated, (0, 0, 255), 2)
    cv2.line(canvas, long_center, estimated, (255, 0, 0), 2)
    cv2.line(canvas, estimated, estimated_tip, (0, 255, 0), 2)

    cv2.putText(
        canvas,
        "short",
        (short_center[0] + 10, short_center[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        canvas,
        "long",
        (long_center[0] + 10, long_center[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        canvas,
        "est",
        (estimated[0] + 10, estimated[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        "tip",
        (estimated_tip[0] + 10, estimated_tip[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    return canvas


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open camera")
        return

    tracker = GelSightPCATracker(frame_size=FRAME_SIZE)
    sequence = 0

    print("Controls:")
    print("  n = set neutral GelSight reference")
    print("  r = set two-component movement reference")
    print("  q / ESC = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = tracker._prepare_frame(frame)
        vis = frame.copy()

        if tracker.ref_gray is None:
            draw_text(
                vis,
                "Press n to set neutral GelSight reference",
                (10, 30),
                (0, 255, 255),
                0.55,
                2,
            )
        else:
            try:
                result = tracker.process_frame(
                    frame,
                    sequence=sequence,
                    timestamp=time.monotonic(),
                )
                sequence += 1

                vis = draw_mask_overlay(frame, result.features.mask)

                colors = [(0, 0, 255), (255, 0, 0)]
                for i, component in enumerate(result.features.components):
                    vis = draw_component_pca(vis, component, colors[i])
                    draw_text(
                        vis,
                        f"C{i+1}: center=({component.center[0]:.1f}, {component.center[1]:.1f}) angle={component.angle_deg:.1f} area={component.area_pixels}",
                        (10, 25 + i * 20),
                        colors[i],
                    )

                vis = draw_axis_projection(vis, result.projection, result.long_component)
                prediction_view = draw_simple_prediction_window(
                    result.short_component,
                    result.long_component,
                    result.projection.closest_point,
                )
                cv2.imshow("Simple Prediction", prediction_view)

                draw_text(
                    vis,
                    f"Intersection: ({result.projection.closest_point[0]:.1f}, {result.projection.closest_point[1]:.1f})",
                    (10, 70),
                    (255, 255, 0),
                )

                if result.slip_sample is not None:
                    sample = result.slip_sample
                    draw_text(
                        vis,
                        (
                            f"Slip dx={sample.intersection_dx_px:.1f}px "
                            f"dy={sample.intersection_dy_px:.1f}px "
                            f"rot={sample.long_angle_diff_deg:.1f}deg"
                        ),
                        (10, 100),
                        (0, 255, 255),
                    )
                    draw_text(
                        vis,
                        (
                            f"Calib x={sample.calibrated_x_m:.4f}m "
                            f"y={sample.calibrated_y_m:.4f}m "
                            f"pitch={sample.calibrated_rotation_rad:.4f}rad moved={sample.moved}"
                        ),
                        (10, 125),
                        (0, 255, 255),
                    )
                else:
                    draw_text(
                        vis,
                        "Press r to set two-component reference",
                        (10, 100),
                        (0, 255, 255),
                    )

            except ValueError as exc:
                draw_text(
                    vis,
                    str(exc),
                    (10, 30),
                    (0, 0, 255),
                    0.6,
                    2,
                )

        cv2.imshow("GelSight Two Component Tracking", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            tracker.set_neutral_reference(frame)
            print("Neutral GelSight reference set")
        elif key == ord("r"):
            if tracker.ref_gray is None:
                print("Set neutral reference first with n")
            else:
                try:
                    tracker.set_movement_reference(frame)
                    print("Two-component movement reference set")
                except ValueError as exc:
                    print(f"Could not set movement reference: {exc}")
        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
