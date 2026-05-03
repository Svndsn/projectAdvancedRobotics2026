import cv2
import numpy as np
from dataclasses import dataclass


CAMERA_INDEX = 2
FRAME_SIZE = (640, 640)

DIFF_THRESHOLD = 10
MIN_AREA_PIXELS = 15

CENTER_THRESHOLD_PX = 5.0
ANGLE_THRESHOLD_DEG = 5.0
AREA_THRESHOLD_RATIO = 0.20

AXIS_DRAW_LENGTH_PX = 350


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


def compute_pca(points):
    mean = points.mean(axis=0)
    centered = points - mean

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return mean, eigenvalues, eigenvectors


def extract_two_component_masks(mask):
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


def extract_component_features(component_mask):
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


def extract_features_from_frame(frame, ref_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    diff = cv2.absdiff(gray, ref_gray)

    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    component_masks = extract_two_component_masks(mask)

    components = []

    for component_mask in component_masks:
        component_features = extract_component_features(component_mask)
        components.append(component_features)

    combined_mask = np.zeros_like(mask)
    for component in components:
        combined_mask[component.mask > 0] = 255

    return ImageFeatures(
        components=components,
        mask=combined_mask,
    )


def detect_component_movement(ref_component, current_component):
    dx_px = float(current_component.center[0] - ref_component.center[0])
    dy_px = float(current_component.center[1] - ref_component.center[1])
    distance_px = float(np.hypot(dx_px, dy_px))

    angle_diff_deg = float(
        angle_difference_deg(
            current_component.angle_deg,
            ref_component.angle_deg,
        )
    )

    area_diff_pixels = int(
        current_component.area_pixels - ref_component.area_pixels
    )
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


def get_short_and_long_components(features):
    """
    Assumption from your image:
    - short shaft contact is more vertical
    - long shaft contact is more horizontal

    We use PCA angle:
    - near 90 deg = short/vertical component
    - near 0 or 180 deg = long/horizontal component
    """
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


def compute_axis_projection(short_component, long_component):
    """
    Finds the intersection point between the two major PCA axes.

    Axis 1 = short component major PCA axis
    Axis 2 = long component major PCA axis
    """
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
    
def draw_mask_overlay(frame, mask):
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


def draw_component_pca(frame, component, color):
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


def draw_axis_projection(frame, projection, long_component):
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


# --- everything above stays EXACTLY the same ---


def draw_simple_prediction_window(short_component, long_component, estimated_point):
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)

    short_center = tuple(np.round(short_component.center).astype(int))
    long_center = tuple(np.round(long_component.center).astype(int))
    estimated = tuple(np.round(estimated_point).astype(int))

    # draw points
    cv2.circle(canvas, short_center, 10, (0, 0, 255), -1)   # red
    cv2.circle(canvas, long_center, 10, (255, 0, 0), -1)    # blue
    cv2.circle(canvas, estimated, 10, (0, 255, 255), -1)    # yellow

    # draw simple connections
    cv2.line(canvas, short_center, estimated, (0, 0, 255), 2)
    cv2.line(canvas, long_center, estimated, (255, 0, 0), 2)

    # labels
    cv2.putText(canvas, "short", (short_center[0] + 10, short_center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(canvas, "long", (long_center[0] + 10, long_center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(canvas, "est", (estimated[0] + 10, estimated[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return canvas


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Could not open camera")
        return

    ref_gray = None
    ref_features = None

    print("Controls:")
    print("  n = set neutral GelSight reference")
    print("  r = set two-component movement reference")
    print("  q / ESC = quit")

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

                colors = [
                    (0, 0, 255),
                    (255, 0, 0),
                ]

                for i, component in enumerate(current_features.components):
                    vis = draw_component_pca(vis, component, colors[i])

                    draw_text(
                        vis,
                        f"C{i+1}: center=({component.center[0]:.1f}, {component.center[1]:.1f}) angle={component.angle_deg:.1f} area={component.area_pixels}",
                        (10, 25 + i * 20),
                        colors[i],
                    )

                short_component, long_component = get_short_and_long_components(
                    current_features
                )

                projection = compute_axis_projection(
                    short_component,
                    long_component,
                )

                vis = draw_axis_projection(
                    vis,
                    projection,
                    long_component,
                )

                # ✅ NEW (simple visualization window)
                prediction_view = draw_simple_prediction_window(
                    short_component,
                    long_component,
                    projection.closest_point,
                )

                cv2.imshow("Simple Prediction", prediction_view)

                draw_text(
                    vis,
                    f"Intersection: ({projection.closest_point[0]:.1f}, {projection.closest_point[1]:.1f})",
                    (10, 70),
                    (255, 255, 0),
                )

                if ref_features is not None:
                    for i in range(2):
                        result = detect_component_movement(
                            ref_features.components[i],
                            current_features.components[i],
                        )

                        color = (0, 0, 255) if result.moved else (0, 255, 0)

                        draw_text(
                            vis,
                            f"C{i+1} moved={result.moved} dx={result.dx_px:.1f} dy={result.dy_px:.1f} d={result.distance_px:.1f} angle={result.angle_diff_deg:.1f}",
                            (10, 100 + i * 25),
                            color,
                        )

                else:
                    draw_text(
                        vis,
                        "Press r to set two-component reference",
                        (10, 100),
                        (0, 255, 255),
                    )

            except ValueError as e:
                draw_text(
                    vis,
                    str(e),
                    (10, 30),
                    (0, 0, 255),
                    0.6,
                    2,
                )

        cv2.imshow("GelSight Two Component Tracking", vis)

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

                    print("Two-component movement reference set")

                except ValueError as e:
                    print(f"Could not set movement reference: {e}")

        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()