import importlib
from typing import Protocol

import numpy as np
import spatialmath as sm


class SlipCorrectionProvider(Protocol):
    def reset(self) -> None:
        ...

    def get_correction(
        self,
        actual_pose: sm.SE3,
        target_pose: sm.SE3,
        phase: str,
        t: float,
    ) -> sm.SE3:
        ...


class NoSlipCorrection:
    def reset(self) -> None:
        pass

    def get_correction(
        self,
        actual_pose: sm.SE3,
        target_pose: sm.SE3,
        phase: str,
        t: float,
    ) -> sm.SE3:
        return sm.SE3()


def compensate_tool_slip(
    target_tool_pose: sm.SE3,
    T_gripper_tool_slip: sm.SE3,
) -> sm.SE3:
    return target_tool_pose @ T_gripper_tool_slip.inv()


def load_correction_provider(spec: str | None) -> SlipCorrectionProvider:
    if not spec:
        return NoSlipCorrection()

    if ":" not in spec:
        raise ValueError(
            "Correction provider must be formatted as 'module:ClassName'."
        )

    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    provider_cls = getattr(module, class_name)
    provider = provider_cls()

    if not hasattr(provider, "get_correction"):
        raise TypeError(f"{spec} does not define get_correction().")
    if not hasattr(provider, "reset"):
        provider.reset = lambda: None

    return provider


def validate_and_clip_correction(
    correction: sm.SE3,
    max_translation_m: float,
    max_rotation_rad: float,
) -> sm.SE3:
    if not isinstance(correction, sm.SE3):
        raise TypeError("Slip correction provider must return spatialmath.SE3.")

    A = correction.A
    if not np.all(np.isfinite(A)):
        raise ValueError("Slip correction contains non-finite values.")

    translation = np.array(correction.t, dtype=float)
    translation_norm = float(np.linalg.norm(translation))
    if max_translation_m >= 0 and translation_norm > max_translation_m > 0:
        translation = translation * (max_translation_m / translation_norm)
    elif max_translation_m == 0:
        translation = np.zeros(3)

    angle, axis = correction.angvec()
    angle = float(angle)
    axis = np.array(axis, dtype=float)
    if not np.all(np.isfinite(axis)):
        raise ValueError("Slip correction rotation axis contains non-finite values.")

    if max_rotation_rad >= 0 and abs(angle) > max_rotation_rad > 0:
        angle = np.sign(angle) * max_rotation_rad
    elif max_rotation_rad == 0:
        angle = 0.0

    if abs(angle) > 1e-12 and np.linalg.norm(axis) > 1e-12:
        R = sm.SO3.AngVec(angle, axis / np.linalg.norm(axis)).R
    else:
        R = np.eye(3)
    return sm.SE3.Rt(R, translation, check=False)
