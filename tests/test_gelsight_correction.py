from pathlib import Path
import sys
import time

import numpy as np
import spatialmath as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_gelsight_pca import (
    ComponentFeatures,
    ImageFeatures,
    SlipCalibration,
    compute_slip_estimate,
)
from real.gelsight_correction import (
    GelSightPCASlipCorrectionProvider,
    atomic_write_json,
)


def _make_component(center: tuple[float, float], angle_deg: float) -> ComponentFeatures:
    angle_rad = np.deg2rad(angle_deg)
    major_axis = np.array([np.cos(angle_rad), -np.sin(angle_rad)], dtype=float)
    minor_axis = np.array([major_axis[1], -major_axis[0]], dtype=float)
    eigenvectors = np.column_stack((major_axis, minor_axis))
    return ComponentFeatures(
        center=np.array(center, dtype=float),
        eigenvalues=np.array([9.0, 4.0], dtype=float),
        eigenvectors=eigenvectors,
        angle_deg=float(angle_deg),
        area_pixels=100,
        mask=np.zeros((8, 8), dtype=np.uint8),
    )


def _make_features(
    short_center: tuple[float, float],
    long_center: tuple[float, float],
    short_angle_deg: float = 90.0,
    long_angle_deg: float = 0.0,
) -> ImageFeatures:
    return ImageFeatures(
        components=[
            _make_component(short_center, short_angle_deg),
            _make_component(long_center, long_angle_deg),
        ],
        mask=np.zeros((8, 8), dtype=np.uint8),
    )


def test_compute_slip_estimate_maps_intersection_translation_to_xy() -> None:
    ref_features = _make_features(short_center=(10.0, 20.0), long_center=(40.0, 30.0))
    current_features = _make_features(short_center=(14.0, 18.0), long_center=(44.0, 28.0))
    calibration = SlipCalibration(
        x_m_per_px=0.01,
        y_m_per_px=0.02,
        rotation_rad_per_deg=0.1,
        x_sign=1.0,
        y_sign=-1.0,
        rotation_sign=1.0,
        rotation_component="long",
    )

    sample = compute_slip_estimate(
        ref_features=ref_features,
        current_features=current_features,
        sequence=7,
        timestamp=12.3,
        calibration=calibration,
    )

    assert sample.valid is True
    assert sample.sequence == 7
    assert sample.intersection_dx_px == 4.0
    assert sample.intersection_dy_px == -2.0
    assert np.isclose(sample.calibrated_x_m, 0.04)
    assert np.isclose(sample.calibrated_y_m, 0.04)
    assert np.isclose(sample.calibrated_rotation_rad, 0.0)


def test_compute_slip_estimate_maps_long_axis_rotation() -> None:
    ref_features = _make_features(short_center=(10.0, 20.0), long_center=(40.0, 30.0))
    current_features = _make_features(
        short_center=(10.0, 20.0),
        long_center=(40.0, 30.0),
        long_angle_deg=15.0,
    )
    calibration = SlipCalibration(
        x_m_per_px=0.01,
        y_m_per_px=0.01,
        rotation_rad_per_deg=0.1,
        x_sign=1.0,
        y_sign=1.0,
        rotation_sign=-1.0,
        rotation_component="long",
    )

    sample = compute_slip_estimate(
        ref_features=ref_features,
        current_features=current_features,
        sequence=1,
        timestamp=1.0,
        calibration=calibration,
    )

    assert np.isclose(sample.long_angle_diff_deg, 15.0)
    assert np.isclose(sample.calibrated_rotation_rad, -1.5)


def test_provider_returns_xy_and_rotation_from_fresh_sample(tmp_path: Path) -> None:
    state_path = tmp_path / "gelsight.json"
    provider = GelSightPCASlipCorrectionProvider()
    provider.state_path = state_path
    provider.smoothing_alpha = 1.0
    provider.only_when_moved = False
    provider.stale_timeout_s = 1.0
    provider.reset()

    atomic_write_json(
        state_path,
        {
            "valid": True,
            "sequence": 1,
            "timestamp": time.monotonic(),
            "short_dx_px": 0.0,
            "short_dy_px": 0.0,
            "long_dx_px": 0.0,
            "long_dy_px": 0.0,
            "intersection_dx_px": 0.0,
            "intersection_dy_px": 0.0,
            "short_angle_diff_deg": 0.0,
            "long_angle_diff_deg": 0.0,
            "moved": True,
            "calibrated_x_m": 0.012,
            "calibrated_y_m": -0.004,
            "calibrated_rotation_rad": 0.3,
            "error": None,
        },
    )

    correction = provider.get_correction(sm.SE3(), sm.SE3(), "task1", 0.0)
    angle, axis = correction.angvec()

    assert np.allclose(correction.t, [0.012, -0.004, 0.0])
    assert np.isclose(float(angle), 0.3)
    assert np.allclose(axis, [0.0, 0.0, 1.0])


def test_provider_ignores_stale_and_malformed_samples(tmp_path: Path) -> None:
    state_path = tmp_path / "gelsight.json"
    provider = GelSightPCASlipCorrectionProvider()
    provider.state_path = state_path
    provider.smoothing_alpha = 1.0
    provider.only_when_moved = False
    provider.stale_timeout_s = 0.01
    provider.hold_timeout_s = 0.0
    provider.reset()

    atomic_write_json(
        state_path,
        {
            "valid": True,
            "sequence": 1,
            "timestamp": time.monotonic() - 1.0,
            "short_dx_px": 0.0,
            "short_dy_px": 0.0,
            "long_dx_px": 0.0,
            "long_dy_px": 0.0,
            "intersection_dx_px": 0.0,
            "intersection_dy_px": 0.0,
            "short_angle_diff_deg": 0.0,
            "long_angle_diff_deg": 0.0,
            "moved": True,
            "calibrated_x_m": 0.012,
            "calibrated_y_m": -0.004,
            "calibrated_rotation_rad": 0.3,
            "error": None,
        },
    )

    stale_correction = provider.get_correction(sm.SE3(), sm.SE3(), "task1", 0.0)
    assert np.allclose(stale_correction.A, sm.SE3().A)

    state_path.write_text("{", encoding="utf-8")
    malformed_correction = provider.get_correction(sm.SE3(), sm.SE3(), "task1", 0.0)
    assert np.allclose(malformed_correction.A, sm.SE3().A)


def test_provider_requires_fresh_sample_after_reset(tmp_path: Path) -> None:
    state_path = tmp_path / "gelsight.json"
    provider = GelSightPCASlipCorrectionProvider()
    provider.state_path = state_path
    provider.smoothing_alpha = 1.0
    provider.only_when_moved = False
    provider.stale_timeout_s = 1.0

    old_timestamp = time.monotonic()
    atomic_write_json(
        state_path,
        {
            "valid": True,
            "sequence": 1,
            "timestamp": old_timestamp,
            "short_dx_px": 0.0,
            "short_dy_px": 0.0,
            "long_dx_px": 0.0,
            "long_dy_px": 0.0,
            "intersection_dx_px": 0.0,
            "intersection_dy_px": 0.0,
            "short_angle_diff_deg": 0.0,
            "long_angle_diff_deg": 0.0,
            "moved": True,
            "calibrated_x_m": 0.01,
            "calibrated_y_m": 0.0,
            "calibrated_rotation_rad": 0.1,
            "error": None,
        },
    )
    provider.reset()

    correction_before_fresh_sample = provider.get_correction(sm.SE3(), sm.SE3(), "task1", 0.0)
    assert np.allclose(correction_before_fresh_sample.A, sm.SE3().A)

    atomic_write_json(
        state_path,
        {
            "valid": True,
            "sequence": 2,
            "timestamp": time.monotonic(),
            "short_dx_px": 0.0,
            "short_dy_px": 0.0,
            "long_dx_px": 0.0,
            "long_dy_px": 0.0,
            "intersection_dx_px": 0.0,
            "intersection_dy_px": 0.0,
            "short_angle_diff_deg": 0.0,
            "long_angle_diff_deg": 0.0,
            "moved": True,
            "calibrated_x_m": 0.01,
            "calibrated_y_m": 0.0,
            "calibrated_rotation_rad": 0.1,
            "error": None,
        },
    )

    correction_after_fresh_sample = provider.get_correction(sm.SE3(), sm.SE3(), "task1", 0.0)
    assert np.allclose(correction_after_fresh_sample.t, [0.01, 0.0, 0.0])
