import sys

import numpy as np
import pytest
import spatialmath as sm

from real.slip_correction import (
    NoSlipCorrection,
    load_correction_provider,
    validate_and_clip_correction,
)


def test_no_slip_correction_returns_identity():
    provider = NoSlipCorrection()
    correction = provider.get_correction(sm.SE3(), sm.SE3.Tx(1.0), "task", 0.0)

    assert np.allclose(correction.A, sm.SE3().A)


def test_dynamic_provider_loading(tmp_path):
    module_path = tmp_path / "fake_provider.py"
    module_path.write_text(
        "import spatialmath as sm\n"
        "class Provider:\n"
        "    def __init__(self):\n"
        "        self.reset_called = False\n"
        "    def reset(self):\n"
        "        self.reset_called = True\n"
        "    def get_correction(self, actual_pose, target_pose, phase, t):\n"
        "        return sm.SE3.Tx(0.01)\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        provider = load_correction_provider("fake_provider:Provider")
        provider.reset()
        correction = provider.get_correction(sm.SE3(), sm.SE3(), "task", 0.0)
    finally:
        sys.path.remove(str(tmp_path))

    assert provider.reset_called
    assert np.allclose(correction.t, [0.01, 0.0, 0.0])


def test_validate_and_clip_correction_clips_translation_and_rotation():
    correction = sm.SE3.Tx(0.1) @ sm.SE3.Rz(1.0)

    clipped = validate_and_clip_correction(
        correction,
        max_translation_m=0.02,
        max_rotation_rad=0.25,
    )

    assert np.isclose(np.linalg.norm(clipped.t), 0.02)
    angle, _ = clipped.angvec()
    assert np.isclose(angle, 0.25)


def test_validate_and_clip_correction_rejects_non_se3():
    with pytest.raises(TypeError):
        validate_and_clip_correction(np.eye(4), 0.02, 0.2)
