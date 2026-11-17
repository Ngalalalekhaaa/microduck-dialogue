"""Provisional kick envelope expressed at the encoder-derived foot site.

The sensor-policy sweep foot_reach_calibration_v4 found both feet successful
at x=90 mm, right-foot misses at 120 mm, and early contacts at 60 mm.
This interior foot-site envelope still needs closed-loop validation; it is
not a real-robot calibration or a guarantee over the continuous region.
"""

import numpy as np

from .rgb_tof_geometry import check_rotation

MIN_FORWARD_M = 0.07
MAX_FORWARD_M = 0.105


def foot_point(point, pose):
    """Transform a measured body-frame ball centre using a serialized foot pose."""
    position = np.asarray(pose["position_body_m"], dtype=float)
    rotation = np.asarray(pose["rotation_body_from_foot"], dtype=float)
    point = np.asarray(point, dtype=float)
    if position.shape != (3,) or point.shape != (3,) or not np.isfinite([position, point]).all():
        raise ValueError("Use finite three-dimensional ball and foot positions.")
    check_rotation(rotation)
    return rotation.T @ (point - position)


def reachable_from_foot(point, foot, pose):
    if foot not in ("left", "right"):
        return False
    relative = foot_point(point, pose)
    x, y, z = relative
    outward = y if foot == "left" else -y
    return bool(MIN_FORWARD_M <= x <= MAX_FORWARD_M and -0.045 <= outward <= 0.025 and 0.015 <= z <= 0.07)


def choose_reachable_foot(point, poses):
    candidates = [foot for foot in ("left", "right") if reachable_from_foot(point, foot, poses[foot])]
    return min(candidates, key=lambda foot: abs(foot_point(point, poses[foot])[1]), default=None)
