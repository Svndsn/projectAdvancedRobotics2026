import cv2
import numpy as np
from dataclasses import dataclass


CAMERA_INDEX = 2
FRAME_SIZE = (640, 480)

DIFF_THRESHOLD = 10
MIN_AREA_PIXELS = 20

# Measure this: distance from your grip/contact center to the end of the long shaft
LONG_SHAFT_FROM_GRIP_MM = 130.0

# Calibrate this for your camera/image setup
PIXELS_PER_MM = 5

CENTER_THRESHOLD_PX = 5.0
ANGLE_THRESHOLD_DEG = 5.0
ENDPOINT_THRESHOLD_MM = 2.0
AREA_THRESHOLD_RATIO = 0.20

# PCA has no direction, only an axis.
# Use 1 or -1 depending on which side the long shaft points.
DIRECTION_SIGN = 1


@dataclass
class ImageFeatures:
    center: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    angle_deg: float
    area_pixels: int
    mask: np.ndarray
    virtual_endpoint: np.ndarray


@dataclass
class MovementResult:
    center_dx_px: float
    center_dy_px: float
    center_distance_px: float

    endpoint_dx_px: float
    endpoint_dy_px: float
    endpoint_distance_px: float

    endpoint_dx_mm: float
    endpoint_dy_mm: float
    endpoint_distance_mm: float

    angle_diff_deg: float
    area_diff_pixels: int
    area_diff_ratio: float

    moved: bool


def normalize_pca_angle_deg(angle_deg):
    angle_deg = angle_deg % 180.0
    if angle_deg < 0:
        angle_deg += 180.0
    return angle_deg


def angle_difference_deg(angle_a, angle_b):
    diff = abs(angle_a - angle_b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def extract_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        raise ValueError("No object detected")

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[largest_label, cv2.CC_STAT_AREA]

    if area < MIN_AREA_PIXELS:
        raise ValueError("Object too small")

    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255

    return largest_mask


def compute_pca(points):
    mean = points.mean(axis=0)
    centered = points - mean

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return mean, eigenvalues, eigenvectors


def estimate_virtual_endpoint(center, angle_deg):
    angle_rad = np.deg2rad(angle_deg)

    direction = np.array([
        np.cos(angle_rad),
        -np.sin(angle_rad)
    ], dtype=np.float64)

    length_px = LONG_SHAFT_FROM_GRIP_MM * PIXELS_PER_MM

    endpoint = center + DIRECTION_SIGN * length_px * direction

    return endpoint


def extract_features_from_frame(frame, ref_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    diff = cv2.absdiff(gray, ref_gray)

    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = extract_largest_component(mask)

    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys)).astype(np.float64)

    if len(points) < 2:
        raise ValueError("Not enough pixels for PCA")

    center, eigenvalues, eigenvectors = compute_pca(points)

    principal_axis = eigenvectors[:, 0]

    angle_rad = np.arctan2(-principal_axis[1], principal_axis[0])
    angle_deg = normalize_pca_angle_deg(np.degrees(angle_rad))

    area_pixels = int(cv2.countNonZero(mask))

    virtual_endpoint = estimate_virtual_endpoint(center, angle_deg)

    return ImageFeatures(
        center=center,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        angle_deg=angle_deg,
        area_pixels=area_pixels,
        mask=mask,
        virtual_endpoint=virtual_endpoint,
    )


def detect_movement(ref, current):
    center_dx_px = float(current.center[0] - ref.center[0])
    center_dy_px = float(current.center[1] - ref.center[1])
    center_distance_px = float(np.hypot(center_dx_px, center_dy_px))

    endpoint_dx_px = float(current.virtual_endpoint[0] - ref.virtual_endpoint[0])
    endpoint_dy_px = float(current.virtual_endpoint[1] - ref.virtual_endpoint[1])
    endpoint_distance_px = float(np.hypot(endpoint_dx_px, endpoint_dy_px))

    endpoint_dx_mm = endpoint_dx_px / PIXELS_PER_MM
    endpoint_dy_mm = endpoint_dy_px / PIXELS_PER_MM
    endpoint_distance_mm = endpoint_distance_px / PIXELS_PER_MM

    angle_diff_deg = float(angle_difference_deg(current.angle_deg, ref.angle_deg))

    area_diff_pixels = int(current.area_pixels - ref.area_pixels)
    area_diff_ratio = abs(area_diff_pixels) / max(ref.area_pixels, 1)

    moved = (
        endpoint_distance_mm > ENDPOINT_THRESHOLD_MM
        or angle_diff_deg > ANGLE_THRESHOLD_DEG
        or area_diff_ratio > AREA_THRESHOLD_RATIO
    )

    return MovementResult(
        center_dx_px=center_dx_px,
        center_dy_px=center_dy_px,
        center_distance_px=center_distance_px,

        endpoint_dx_px=endpoint_dx_px,
        endpoint_dy_px=endpoint_dy_px,
        endpoint_distance_px=endpoint_distance_px,

        endpoint_dx_mm=endpoint_dx_mm,
        endpoint_dy_mm=endpoint_dy_mm,
        endpoint_distance_mm=endpoint_distance_mm,

        angle_diff_deg=angle_diff_deg,
        area_diff_pixels=area_diff_pixels,
        area_diff_ratio=area_diff_ratio,

        moved=moved,
    )


def draw_mask_overlay(frame, mask):
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


def draw_features(frame, features):
    vis = frame.copy()

    cx, cy = features.center
    center = (int(round(cx)), int(round(cy)))

    scale = 3.0
    major_len = int(scale * np.sqrt(features.eigenvalues[0]))
    minor_len = int(scale * np.sqrt(features.eigenvalues[1]))

    major_vec = features.eigenvectors[:, 0]
    minor_vec = features.eigenvectors[:, 1]

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

    endpoint = tuple(np.round(features.virtual_endpoint).astype(int))

    cv2.line(vis, major_p1, major_p2, (0, 0, 255), 2)
    cv2.line(vis, minor_p1, minor_p2, (0, 255, 255), 2)

    cv2.circle(vis, center, 5, (0, 255, 0), -1)

    cv2.line(vis, center, endpoint, (255, 0, 255), 2)
    cv2.circle(vis, endpoint, 8, (255, 0, 255), -1)

    cv2.putText(
        vis,
        "virtual shaft end",
        (endpoint[0] + 10, endpoint[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2,
    )

    return vis


def draw_text(vis, text, pos, color=(255, 255, 255), scale=0.5, thickness=1):
    cv2.putText(
        vis,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Could not open camera")
        return

    ref_gray = None
    ref_features = None

    print("Controls:")
    print("  n = set neutral GelSight reference")
    print("  r = set current grip/key reference")
    print("  q / ESC = quit")
    print()
    print("Important settings:")
    print(f"  LONG_SHAFT_FROM_GRIP_MM = {LONG_SHAFT_FROM_GRIP_MM}")
    print(f"  PIXELS_PER_MM = {PIXELS_PER_MM}")
    print(f"  DIRECTION_SIGN = {DIRECTION_SIGN}")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        vis = frame.copy()

        if ref_gray is None:
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
                current_features = extract_features_from_frame(frame, ref_gray)

                vis = draw_mask_overlay(frame, current_features.mask)
                vis = draw_features(vis, current_features)

                draw_text(
                    vis,
                    f"Center px: {current_features.center[0]:.1f}, {current_features.center[1]:.1f}",
                    (10, 25),
                )

                draw_text(
                    vis,
                    f"PCA angle: {current_features.angle_deg:.2f} deg",
                    (10, 45),
                )

                draw_text(
                    vis,
                    f"Virtual end px: {current_features.virtual_endpoint[0]:.1f}, {current_features.virtual_endpoint[1]:.1f}",
                    (10, 65),
                )

                draw_text(
                    vis,
                    f"Area: {current_features.area_pixels}",
                    (10, 85),
                )

                if ref_features is not None:
                    result = detect_movement(ref_features, current_features)

                    color = (0, 0, 255) if result.moved else (0, 255, 0)

                    draw_text(
                        vis,
                        f"Moved: {result.moved}",
                        (10, 115),
                        color,
                        0.6,
                        2,
                    )

                    draw_text(
                        vis,
                        f"Endpoint px: dx={result.endpoint_dx_px:.1f}, dy={result.endpoint_dy_px:.1f}, d={result.endpoint_distance_px:.1f}",
                        (10, 140),
                        color,
                    )

                    draw_text(
                        vis,
                        f"Endpoint mm: dx={result.endpoint_dx_mm:.2f}, dy={result.endpoint_dy_mm:.2f}, d={result.endpoint_distance_mm:.2f}",
                        (10, 160),
                        color,
                    )

                    draw_text(
                        vis,
                        f"Angle diff: {result.angle_diff_deg:.2f} deg",
                        (10, 180),
                        color,
                    )

                    draw_text(
                        vis,
                        f"Center px: dx={result.center_dx_px:.1f}, dy={result.center_dy_px:.1f}, d={result.center_distance_px:.1f}",
                        (10, 200),
                        color,
                    )

                else:
                    draw_text(
                        vis,
                        "Press r to set movement reference",
                        (10, 115),
                        (0, 255, 255),
                    )

            except ValueError:
                draw_text(
                    vis,
                    "No contact/object detected",
                    (10, 30),
                    (0, 0, 255),
                    0.6,
                    2,
                )

        cv2.imshow("GelSight Virtual Umbraco Endpoint Tracking", vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.GaussianBlur(gray, (5, 5), 0)
            ref_features = None
            print("Neutral GelSight reference set")

        elif key == ord("r"):
            if ref_gray is None:
                print("Set neutral reference first with n")
            else:
                try:
                    ref_features = extract_features_from_frame(frame, ref_gray)

                    print("Movement reference set")
                    print(f"Reference center px: {ref_features.center}")
                    print(f"Reference PCA angle: {ref_features.angle_deg:.2f}")
                    print(f"Reference virtual endpoint px: {ref_features.virtual_endpoint}")
                    print(f"Reference area px: {ref_features.area_pixels}")

                except ValueError:
                    print("Could not set movement reference")

        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()