"""IO helpers to convert between UR rotvec CSV rows and spatialmath.SE3.

Functions
- rotvec_to_R, R_to_rotvec: conversion helpers
- row_to_se3: convert numeric pose row to :class:`spatialmath.SE3`
- load_pose_csv(path, prefix): read timestamps and poses from CSV
- save_pose_csv(path, timestamps, poses, prefix): write CSV

This module accepts CSVs where a pose is stored as 6 values
([x, y, z, rx, ry, rz]) where rx,ry,rz is the UR rotation vector
(axis * angle). The loader will try to find columns that include the
given `prefix` and component names. Basic validation is performed and
an informative ValueError is raised when required columns are missing.
"""
from __future__ import annotations

import csv
import math
from typing import List, Tuple

import numpy as np
from spatialmath import SE3


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotvec_to_R(rvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector (axis * angle) to a 3x3 rotation matrix.

    Parameters
    - rvec: array-like shape (3,)
    """
    r = np.asarray(rvec, dtype=float)
    theta = np.linalg.norm(r)
    if theta <= 1e-12:
        return np.eye(3)
    axis = r / theta
    K = _skew(axis)
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a rotation vector (axis * angle).

    Handles small-angle cases robustly.
    """
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    cos_theta = max(min((tr - 1.0) / 2.0, 1.0), -1.0)
    theta = math.acos(cos_theta)
    if abs(theta) <= 1e-12:
        return np.zeros(3)
    denom = 2.0 * math.sin(theta)
    rx = (R[2, 1] - R[1, 2]) / denom
    ry = (R[0, 2] - R[2, 0]) / denom
    rz = (R[1, 0] - R[0, 1]) / denom
    return np.array([rx, ry, rz]) * theta


def _row_to_se3_components(x: float, y: float, z: float, rx: float, ry: float, rz: float) -> SE3:
    R = rotvec_to_R(np.array([rx, ry, rz], dtype=float))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z], dtype=float)
    return SE3(T)


def _find_column(headers: List[str], prefix: str, component: str) -> str | None:
    p = prefix.lower()
    c = component.lower()
    # First try headers that contain both prefix and the component
    for h in headers:
        low = h.lower()
        if p in low and c in low:
            return h
    # Fallback: accept any header that contains the component (e.g. 'x')
    for h in headers:
        low = h.lower()
        if c == 'timestamp' and ('time' in low or 'timestamp' in low):
            return h
        if c in ('x', 'y', 'z') and (low.endswith('_' + c) or low == c):
            return h
        if c in ('rx', 'ry', 'rz') and (('_' + c) in low or low.endswith(c)):
            return h
    return None


def load_pose_csv(path: str, prefix: str = "actual_TCP_pose") -> Tuple[np.ndarray, List[SE3]]:
    """Load timestamps and poses from CSV where poses are UR rotvec rows.

    Returns (timestamps, poses) where timestamps is a 1D numpy array of
    floats and poses is a list of :class:`spatialmath.SE3` instances.

    The loader searches for columns that include the given `prefix` and
    the components `x,y,z,rx,ry,rz`. It also tries to find a timestamp
    column (containing "time" or "timestamp") and falls back to the
    first column if none is found.
    """
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        if not headers:
            raise ValueError(f"CSV {path} has no header row or is empty")

        cx = _find_column(headers, prefix, 'x')
        cy = _find_column(headers, prefix, 'y')
        cz = _find_column(headers, prefix, 'z')
        crx = _find_column(headers, prefix, 'rx')
        cry = _find_column(headers, prefix, 'ry')
        crz = _find_column(headers, prefix, 'rz')
        ctime = _find_column(headers, prefix, 'timestamp') or headers[0]

        missing = [name for name, col in (('x', cx), ('y', cy), ('z', cz), ('rx', crx), ('ry', cry), ('rz', crz)) if col is None]
        if missing:
            raise ValueError(
                f"Missing required pose columns for prefix '{prefix}': {missing}. Available headers: {headers}"
            )

        timestamps: List[float] = []
        poses: List[SE3] = []
        for row in reader:
            try:
                t = float(row.get(ctime, '').strip() or 0.0)
                x = float(row[cx])
                y = float(row[cy])
                z = float(row[cz])
                rx = float(row[crx])
                ry = float(row[cry])
                rz = float(row[crz])
            except Exception as e:  # pragma: no cover - input parsing error
                raise ValueError(f"Failed parsing row {reader.line_num}: {e}")
            timestamps.append(t)
            poses.append(_row_to_se3_components(x, y, z, rx, ry, rz))

    return np.asarray(timestamps, dtype=float), poses


def save_pose_csv(path: str, timestamps: List[float], poses: List[SE3], prefix: str = "actual_TCP_pose") -> None:
    """Save timestamps and poses to CSV using UR rotvec representation.

    The output CSV will contain columns:
    `{prefix}_timestamp,{prefix}_x,{prefix}_y,{prefix}_z,{prefix}_rx,{prefix}_ry,{prefix}_rz`
    """
    if len(timestamps) != len(poses):
        raise ValueError("timestamps and poses must have the same length")

    header = [f"{prefix}_timestamp", f"{prefix}_x", f"{prefix}_y", f"{prefix}_z", f"{prefix}_rx", f"{prefix}_ry", f"{prefix}_rz"]
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for t, p in zip(timestamps, poses):
            # p can be SE3; extract R and translation
            T = p.A if hasattr(p, 'A') else np.asarray(p)
            R = T[:3, :3]
            trans = T[:3, 3]
            rvec = R_to_rotvec(R)
            row = [f"{float(t):.9g}", f"{float(trans[0]):.9g}", f"{float(trans[1]):.9g}", f"{float(trans[2]):.9g}", f"{float(rvec[0]):.9g}", f"{float(rvec[1]):.9g}", f"{float(rvec[2]):.9g}"]
            writer.writerow(row)


__all__ = [
    'rotvec_to_R',
    'R_to_rotvec',
    'load_pose_csv',
    'save_pose_csv',
]
