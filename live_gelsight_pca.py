import cv2
import numpy as np
from dataclasses import dataclass


CAMERA_INDEX = 4
FRAME_SIZE = (640, 640)

DIFF_THRESHOLD = 10
MIN_AREA_PIXELS = 20

CENTER_THRESHOLD_PX = 5.0
ANGLE_THRESHOLD_DEG = 5.0
AREA_THRESHOLD_RATIO = 0.20


@dataclass
class ImageFeatures:
    center: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    angle_deg: float
    area_pixels: int
    mask: np.ndarray


@dataclass
class MovementResult:
    dx: float
    dy: float
    distance: float
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
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

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

    mean, eigenvalues, eigenvectors = compute_pca(points)

    principal_axis = eigenvectors[:, 0]
    angle_rad = np.arctan2(-principal_axis[1], principal_axis[0])
    angle_deg = normalize_pca_angle_deg(np.degrees(angle_rad))

    area_pixels = int(cv2.countNonZero(mask))

    return ImageFeatures(
        center=mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        angle_deg=angle_deg,
        area_pixels=area_pixels,
        mask=mask,
    )


def detect_movement(ref, current):
    dx = float(current.center[0] - ref.center[0])
    dy = float(current.center[1] - ref.center[1])
    distance = float(np.hypot(dx, dy))

    angle_diff_deg = float(angle_difference_deg(current.angle_deg, ref.angle_deg))

    area_diff_pixels = int(current.area_pixels - ref.area_pixels)
    area_diff_ratio = abs(area_diff_pixels) / max(ref.area_pixels, 1)

    moved = (
        distance > CENTER_THRESHOLD_PX
        or angle_diff_deg > ANGLE_THRESHOLD_DEG
        or area_diff_ratio > AREA_THRESHOLD_RATIO
    )

    return MovementResult(
        dx=dx,
        dy=dy,
        distance=distance,
        angle_diff_deg=angle_diff_deg,
        area_diff_pixels=area_diff_pixels,
        area_diff_ratio=area_diff_ratio,
        moved=moved,
    )


def draw_pca(frame, features):
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

    cv2.line(vis, major_p1, major_p2, (0, 0, 255), 2)
    cv2.line(vis, minor_p1, minor_p2, (0, 255, 255), 2)
    cv2.circle(vis, center, 4, (0, 255, 0), -1)

    return vis


def draw_mask_overlay(frame, mask):
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Could not open camera")
        return

    ref_gray = None
    ref_features = None

    print("Controls:")
    print("  n = set neutral membrane reference")
    print("  r = set current object as movement reference")
    print("  q / ESC = quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        vis = frame.copy()

        if ref_gray is None:
            cv2.putText(
                vis,
                "Press n to set neutral reference",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        else:
            try:
                current_features = extract_features_from_frame(frame, ref_gray)

                vis = draw_mask_overlay(frame, current_features.mask)
                vis = draw_pca(vis, current_features)

                cv2.putText(
                    vis,
                    f"Center: {current_features.center[0]:.1f}, {current_features.center[1]:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.putText(
                    vis,
                    f"Angle: {current_features.angle_deg:.1f} deg",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.putText(
                    vis,
                    f"Area: {current_features.area_pixels}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                if ref_features is not None:
                    result = detect_movement(ref_features, current_features)

                    color = (0, 0, 255) if result.moved else (0, 255, 0)

                    cv2.putText(
                        vis,
                        f"Moved: {result.moved}",
                        (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    cv2.putText(
                        vis,
                        f"dx={result.dx:.1f}, dy={result.dy:.1f}, d={result.distance:.1f}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

                    cv2.putText(
                        vis,
                        f"angle diff={result.angle_diff_deg:.1f}",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )
                else:
                    cv2.putText(
                        vis,
                        "Press r to set object reference",
                        (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

            except ValueError:
                cv2.putText(
                    vis,
                    "No contact/object detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("GelSight Live PCA", vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.GaussianBlur(gray, (5, 5), 0)
            ref_features = None
            print("Neutral membrane reference set")

        elif key == ord("r"):
            if ref_gray is None:
                print("Set neutral reference first with n")
            else:
                try:
                    ref_features = extract_features_from_frame(frame, ref_gray)
                    print("Object movement reference set")
                    print(f"Reference center: {ref_features.center}")
                    print(f"Reference angle: {ref_features.angle_deg:.2f}")
                    print(f"Reference area: {ref_features.area_pixels}")
                except ValueError:
                    print("Could not set object reference")

        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()