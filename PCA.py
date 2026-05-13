import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE_PATH = "data/img/gelsight_mini/gelsight_mini/demo.png_rgb.png"
REF_IMAGE_PATH = "screenshot_20260501_143024.png"
X_MOVEMENT_IMAGE_PATH = "testing_pictures/x_movement.png"
Y_MOVEMENT_IMAGE_PATH = "testing_pictures/y_movement.png"


@dataclass
class ImageFeatures:
    center: np.ndarray          # [cx, cy]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    angle_rad: float
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find dominant PCA axis of the foreground object in an image."
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE_PATH,
        help="Path to input image.",
    )
    parser.add_argument(
        "--ref_image",
        default=REF_IMAGE_PATH,
        help="Path to reference image.",
    )
    parser.add_argument(
        "--x_movement_image",
        default=X_MOVEMENT_IMAGE_PATH,
        help="Path to image to compare against reference.",
    )
    parser.add_argument(
        "--y_movement_image",
        default=Y_MOVEMENT_IMAGE_PATH,
        help="Path to image to compare against reference.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Threshold in [0, 255]. Used if --use-otsu is not enabled.",
    )
    parser.add_argument(
        "--use-otsu",
        action="store_true",
        help="Use Otsu thresholding instead of a fixed threshold.",
    )
    parser.add_argument(
        "--output",
        default="pca_principal_component.png",
        help="Path to save annotated output image.",
    )
    return parser.parse_args()


def load_grayscale_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def create_foreground_mask(img: np.ndarray, threshold: int, use_otsu: bool) -> tuple[np.ndarray, str]:
    if use_otsu:
        ret, mask = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        mode = f"otsu + binary_inv (threshold={ret:.1f})"
    else:
        if not 0 <= threshold <= 255:
            raise ValueError("--threshold must be between 0 and 255.")
        _, mask = cv2.threshold(
            img, threshold, 255, cv2.THRESH_BINARY_INV
        )
        mode = f"fixed + binary_inv (threshold={threshold})"

    return mask, mode


def extract_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        raise ValueError("No foreground object found in mask.")

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255
    return largest_mask


def compute_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    centered = points - mean

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return mean, eigenvalues, eigenvectors


def draw_axes(
    img: np.ndarray,
    mean: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cx, cy = mean
    center = (int(round(cx)), int(round(cy)))

    scale = 3.0
    major_len = int(scale * np.sqrt(eigenvalues[0]))
    minor_len = int(scale * np.sqrt(eigenvalues[1]))

    major_vec = eigenvectors[:, 0]
    minor_vec = eigenvectors[:, 1]

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


def normalize_pca_angle_deg(angle_deg: float) -> float:
    """
    Normalize angle to [0, 180), because a PCA axis has 180 degree symmetry.
    """
    angle_deg = angle_deg % 180.0
    if angle_deg < 0:
        angle_deg += 180.0
    return angle_deg


def angle_difference_deg(angle_a: float, angle_b: float) -> float:
    """
    Smallest difference between two PCA axis angles in degrees.
    Result is in [0, 90].
    """
    diff = abs(angle_a - angle_b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def extract_features(image_path: str, threshold: int, use_otsu: bool, output_path: str) -> ImageFeatures:
    img = load_grayscale_image(image_path)
    mask, threshold_mode = create_foreground_mask(img, threshold, use_otsu)
    mask = extract_largest_component(mask)

    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys)).astype(np.float64)

    if len(points) < 2:
        raise ValueError("Not enough foreground pixels for PCA.")

    mean, eigenvalues, eigenvectors = compute_pca(points)
    principal_axis = eigenvectors[:, 0]

    angle_rad = np.arctan2(-principal_axis[1], principal_axis[0])
    angle_deg = normalize_pca_angle_deg(np.degrees(angle_rad))

    area_pixels = int(cv2.countNonZero(mask))

    vis = draw_axes(img, mean, eigenvalues, eigenvectors)

    output_path = Path(output_path)
    if not cv2.imwrite(str(output_path), vis):
        raise IOError(f"Failed to write output image to: {output_path}")

    print(f"Processing: {image_path}")
    print(f"Threshold mode: {threshold_mode}")
    print(f"Center: ({mean[0]:.2f}, {mean[1]:.2f})")
    print(f"Angle (deg): {angle_deg:.2f}")
    print(f"Area (pixels): {area_pixels}")
    print(f"Saved: {output_path}")

    return ImageFeatures(
        center=mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        angle_rad=angle_rad,
        angle_deg=angle_deg,
        area_pixels=area_pixels,
        mask=mask,
    )


def detect_movement(
    ref: ImageFeatures,
    current: ImageFeatures,
    center_threshold_px: float = 5.0,
    angle_threshold_deg: float = 5.0,
    area_threshold_ratio: float = 0.10,
) -> MovementResult:
    dx = float(current.center[0] - ref.center[0])
    dy = float(current.center[1] - ref.center[1])
    distance = float(np.hypot(dx, dy))

    angle_diff_deg = float(angle_difference_deg(current.angle_deg, ref.angle_deg))

    area_diff_pixels = int(current.area_pixels - ref.area_pixels)
    area_diff_ratio = abs(area_diff_pixels) / max(ref.area_pixels, 1)

    moved = (
        distance > center_threshold_px
        or angle_diff_deg > angle_threshold_deg
        or area_diff_ratio > area_threshold_ratio
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


def main() -> None:
    args = parse_args()

    ref_output = f"pca_{Path(args.ref_image).stem}.png"
    movement_output = f"pca_{Path(args.y_movement_image).stem}.png"

    ref_features = extract_features(
        image_path=args.ref_image,
        threshold=args.threshold,
        use_otsu=args.use_otsu,
        output_path=ref_output,
    )

    movement_features = extract_features(
        image_path=args.y_movement_image,
        threshold=args.threshold,
        use_otsu=args.use_otsu,
        output_path=movement_output,
    )

    result = detect_movement(
        ref=ref_features,
        current=movement_features,
        center_threshold_px=5.0,
        angle_threshold_deg=5.0,
        area_threshold_ratio=0.10,
    )

    print("\nComparison result")
    print(f"dx: {result.dx:.2f} px")
    print(f"dy: {result.dy:.2f} px")
    print(f"distance: {result.distance:.2f} px")
    print(f"angle difference: {result.angle_diff_deg:.2f} deg")
    print(f"area difference: {result.area_diff_pixels} px")
    print(f"area difference ratio: {result.area_diff_ratio:.3f}")
    print(f"Moved: {result.moved}")


if __name__ == "__main__":
    main()