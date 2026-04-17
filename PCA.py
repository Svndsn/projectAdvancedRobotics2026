import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE_PATH = "data/img/gelsight_mini/gelsight_mini/demo.png_rgb.png"
REF_IMAGE_PATH = "data/img/gelsight_mini/gelsight_mini/testing_pictures/ref_frame.png"
X_MOVEMENT_IMAGE_PATH = "data/img/gelsight_mini/gelsight_mini/testing_pictures/x_movement.png"

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
        help="Path to input image.",
    )
    parser.add_argument(
        "--movement_image",
        default=X_MOVEMENT_IMAGE_PATH,
        help="Path to input image.",
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
    """
    Create a binary mask where the dark object becomes foreground (255).
    """
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
    """
    Keep only the largest connected foreground component.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        raise ValueError("No foreground object found in mask.")

    # Ignore background at index 0
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255
    return largest_mask


def compute_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        mean: shape (2,)
        eigenvalues: shape (2,) sorted descending
        eigenvectors: shape (2, 2), columns are eigenvectors sorted descending
    """
    mean = points.mean(axis=0)
    centered = points - mean

    cov = np.cov(centered, rowvar=False)

    # eigh is better for symmetric matrices like covariance matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
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

    # Scale axis lengths for visualization
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

    # Major axis = red, minor axis = yellow, center = green
    cv2.line(vis, major_p1, major_p2, (0, 0, 255), 2)
    cv2.line(vis, minor_p1, minor_p2, (0, 255, 255), 2)
    cv2.circle(vis, center, 4, (0, 255, 0), -1)

    return vis


def pca_image(image_path: str, threshold: int, use_otsu: bool, output_path: str):
    
    
    

    img = load_grayscale_image(image_path)
    mask, threshold_mode = create_foreground_mask(img, threshold, use_otsu)
    mask = extract_largest_component(mask)
    

    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys)).astype(np.float64)

    if len(points) < 2:
        raise ValueError("Not enough foreground pixels for PCA.")

    mean, eigenvalues, eigenvectors = compute_pca(points)
    principal_axis = eigenvectors[:, 0]
    angle_rad = np.arctan2(max(abs(principal_axis)), min(abs(principal_axis)))
    angle_deg = np.degrees(angle_rad)

    vis = draw_axes(img, mean, eigenvalues, eigenvectors)

    output_path = Path(output_path)
    if not cv2.imwrite(str(output_path), vis):
        raise IOError(f"Failed to write output image to: {output_path}")

    print(f"Threshold mode: {threshold_mode}")
    print(f"Center: ({mean[0]:.2f}, {mean[1]:.2f})")
    print(f"Principal axis: [{principal_axis[0]:.6f}, {principal_axis[1]:.6f}]")
    print(f"Angle (rad): {angle_rad:.6f}")
    print(f"Angle (deg): {angle_deg:.2f}")
    print(f"Saved: {output_path}")


def main() -> None:

    args = parse_args()

    image_paths = [
        args.ref_image,
        #args.image,
        args.movement_image,
    ]

    for image_path in image_paths:
        output_path = f"pca_{Path(image_path).stem}.png"
        print(f"\nProcessing: {image_path}")
        pca_image(
            image_path=image_path,
            threshold=args.threshold,
            use_otsu=args.use_otsu,
            output_path=output_path,
        )



if __name__ == "__main__":
    main()