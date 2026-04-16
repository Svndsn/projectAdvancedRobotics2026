import argparse

import cv2
import numpy as np


DEFAULT_IMAGE_PATH = "/home/simon/git/projectAdvancedRobotics2026/data/img/gelsight_mini/gelsight_mini/demo.png_rgb.png"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compute dominant PCA axis angle from a binary mask.")
	parser.add_argument(
		"--image",
		default=DEFAULT_IMAGE_PATH,
		help="Path to grayscale image.",
	)
	parser.add_argument(
		"--threshold",
		type=int,
		default=127,
		help="Binary threshold in [0, 255]. If result is empty, Otsu fallback is attempted.",
	)
	parser.add_argument(
		"--output",
		default="pca_principal_component.png",
		help="Path to save the annotated image with the principal component axis.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not 0 <= args.threshold <= 255:
		raise ValueError("--threshold must be between 0 and 255.")

	img = cv2.imread(args.image, 0)
	if img is None:
		raise FileNotFoundError(
			f"Could not load image at '{args.image}'. Check that the file path exists and is readable."
		)

	_, thresh = cv2.threshold(img, args.threshold, 255, cv2.THRESH_BINARY)
	threshold_mode = f"fixed ({args.threshold})"

	ys, xs = np.where(thresh > 0)
	points = np.column_stack((xs, ys))

	if points.shape[0] < 2:
		# Otsu is a robust fallback when a fixed threshold produces an empty mask.
		_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		threshold_mode = "otsu fallback"
		ys, xs = np.where(thresh > 0)
		points = np.column_stack((xs, ys))

	if points.shape[0] < 2:
		raise ValueError(
			"Not enough foreground pixels for PCA. Try adjusting --threshold or use an image with a visible object."
		)

	mean = np.mean(points, axis=0)
	centered = points - mean

	cov = np.cov(centered, rowvar=False)
	eigvals, eigvecs = np.linalg.eig(cov)

	principal_axis = eigvecs[:, np.argmax(eigvals)]
	angle = np.arctan2(principal_axis[1], principal_axis[0])

	# Draw the principal axis through the mean point and save for inspection.
	vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	h, w = img.shape
	line_length = int(0.4 * max(h, w))

	cx, cy = mean
	x1 = int(round(cx - principal_axis[0] * line_length))
	y1 = int(round(cy - principal_axis[1] * line_length))
	x2 = int(round(cx + principal_axis[0] * line_length))
	y2 = int(round(cy + principal_axis[1] * line_length))

	cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
	cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 255, 0), -1)

	if not cv2.imwrite(args.output, vis):
		raise IOError(f"Failed to write output image to '{args.output}'.")

	print("Threshold mode:", threshold_mode)
	print("Angle (radians):", angle)
	print("Annotated image:", args.output)


if __name__ == "__main__":
	main()