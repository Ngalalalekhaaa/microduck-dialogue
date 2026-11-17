"""Bounded ToF-to-RGB handoff while returning the head to a kickable posture.

This is explicitly a short memory of a measured, stationary ball, not a fresh
ToF detection. Current complete RGB silhouette and the known radius must
independently agree with that memory. Never use it to acquire or approach.
"""

import numpy as np
from scipy import ndimage

from .rgb_tof_fusion import _hypotheses, _masks
from .rgb_tof_geometry import check_rotation, pixel_unit_rays, project_camera


def _cone_outline(config, direction, alpha):
    tangent = np.cross(direction, [0, 1, 0])
    tangent /= np.linalg.norm(tangent)
    other = np.cross(direction, tangent)
    phase = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rays = np.cos(alpha) * direction + np.sin(alpha) * (
        np.cos(phase)[:, None] * tangent + np.sin(phase)[:, None] * other
    )
    return project_camera(config, rays)


def _in_frame(config, direction, alpha):
    outline = _cone_outline(config, direction, alpha)
    return bool(
        np.isfinite(outline).all()
        and (outline > [1, 1]).all()
        and (outline < [config.width - 2, config.height - 2]).all()
    )


def _partial_arc(rgb, config, pose, locked):
    """Fit observed upper boundary rays; the ToF lock only bounds the search.

    The boundary is computed before selecting a region, so no crop boundary is
    fitted as a ball edge. This handles a red ball joined to a red foot in the
    mask, but refuses image-clipped balls and short or straight edges.
    """
    centre = pose[1].T @ (np.asarray(locked["center_body_m"]) - pose[0])
    distance = np.linalg.norm(centre)
    if centre[2] <= 0 or distance <= config.ball_radius_m:
        return None
    direction = centre / distance
    alpha = np.arcsin(config.ball_radius_m / distance)
    if not _in_frame(config, direction, alpha):
        return None
    uv = project_camera(config, centre)
    mask = _masks(rgb).get(locked["color"])
    if mask is None:
        return None
    boundary = mask & ~ndimage.binary_erosion(mask)
    rows, columns = np.where(boundary)
    rays = pixel_unit_rays(config, np.column_stack((columns, rows)))
    angles = np.arccos(np.clip(rays @ direction, -1, 1))
    keep = (
        (np.abs(angles - alpha) < max(0.035, 0.25 * alpha))
        & (rows <= uv[1] + 0.15 * config.focal_px * np.tan(alpha))
        & (rows > 0)
        & (rows < config.height - 1)
        & (columns > 0)
        & (columns < config.width - 1)
    )
    rays = rays[keep]
    if len(rays) < 16:
        return None
    threshold = 0.9 / config.focal_px
    best = np.zeros(len(rays), dtype=bool)
    rng = np.random.default_rng(23)
    for _ in range(120):
        sample = rays[rng.choice(len(rays), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        radius = np.arccos(np.clip(sample @ normal, -1, 1)).mean()
        errors = np.abs(np.arccos(np.clip(rays @ normal, -1, 1)) - radius)
        inliers = errors < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < max(16, 0.7 * len(rays)):
        return None
    accepted = rays[best]
    _, _, vectors = np.linalg.svd(accepted - accepted.mean(axis=0), full_matrices=False)
    normal = vectors[-1] * (1 if vectors[-1, 2] > 0 else -1)
    angles = np.arccos(np.clip(accepted @ normal, -1, 1))
    raw_alpha = float(np.median(angles))
    residual = float(np.quantile(np.abs(angles - raw_alpha), 0.9))
    tangent = vectors[0]
    phase = np.sort(np.arctan2(accepted @ np.cross(normal, tangent), accepted @ tangent))
    coverage = float(2 * np.pi - np.diff(np.r_[phase, phase[0] + 2 * np.pi]).max())
    alpha = raw_alpha + 0.45 / config.focal_px
    if residual > threshold or coverage < np.deg2rad(130) or not 0 < alpha < np.pi / 2:
        return None
    if not _in_frame(config, normal, alpha):
        return None
    centre = pose[0] + pose[1] @ (normal * config.ball_radius_m / np.sin(alpha))
    return centre, dict(
        method="partial_arc",
        arc_pixels=int(best.sum()),
        arc_coverage_deg=float(np.rad2deg(coverage)),
        arc_residual_rad=residual,
        edge_source="uncropped_rgb_mask_boundary",
    )


def check_handoff(rgb, *, config, pose, locked, lock_time, observation, foot, foot_pose=None):
    rgb = np.asarray(rgb)
    age = observation["time_s"] - lock_time
    result = dict(accepted=False, source="recent_tof_lock_plus_fresh_known_radius_rgb", range_lock_age_s=age)
    sensor_age = observation["sensor_age_s"]
    stationary = observation["stationary"]
    if (
        not np.isfinite(age)
        or not 0 <= age <= 1.4
        or not np.isfinite(sensor_age)
        or not 0 <= sensor_age <= 0.12
        or not isinstance(stationary, (bool, np.bool_))
        or not stationary
    ):
        return {**result, "reason": "lock_expired_or_not_settled"}
    if foot not in ("left", "right"):
        return {**result, "reason": "invalid_foot"}
    position, rotation = np.asarray(pose[0], dtype=float), np.asarray(pose[1], dtype=float)
    try:
        check_rotation(rotation)
    except ValueError:
        return {**result, "reason": "invalid_camera_pose"}
    locked_point = np.asarray(locked["center_body_m"], dtype=float)
    if (
        position.shape != (3,)
        or not np.isfinite(position).all()
        or locked_point.shape != (3,)
        or not np.isfinite(locked_point).all()
    ):
        return {**result, "reason": "invalid_camera_pose_or_lock"}
    if (
        locked.get("sensor_source") != "rgb_silhouette_and_coarse_radial_tof_only"
        or not isinstance(locked.get("tof_zone_count"), int)
        or locked["tof_zone_count"] < 1
    ):
        return {**result, "reason": "missing_tof_lock_evidence"}
    if rgb.dtype != np.uint8 or rgb.shape != (config.height, config.width, 3):
        return {**result, "reason": "invalid_rgb"}
    pose = position, rotation
    candidates = []

    def add(point, evidence):
        difference = float(np.linalg.norm(point - locked_point))
        x, y, z = point
        lateral = y if foot == "left" else -y
        accepted = difference <= 0.025 and 0.075 <= x <= 0.105 and 0.018 <= lateral <= 0.052 and -0.11 <= z <= -0.04
        if foot_pose is not None:
            from .rgb_tof_reach import foot_point, reachable_from_foot

            try:
                accepted = difference <= 0.025 and reachable_from_foot(point, foot, foot_pose)
                evidence["center_foot_m"] = foot_point(point, foot_pose).tolist()
            except (KeyError, TypeError, ValueError):
                accepted = False
        candidates.append(dict(accepted=bool(accepted), center_body_m=point.tolist(), change_m=difference, **evidence))

    for item, direction, distance, _ in _hypotheses(rgb, config, []):
        if item["color"] != locked["color"]:
            continue
        point = pose[0] + pose[1] @ (direction * distance)
        add(point, dict(method="complete_silhouette"))
    if not any(item["accepted"] for item in candidates):
        arc = _partial_arc(rgb, config, pose, locked)
        if arc is not None:
            add(*arc)
    if not candidates:
        return {**result, "reason": "rgb_ball_geometry_missing"}
    best = min(candidates, key=lambda item: (not item["accepted"], item["change_m"]))
    return {**result, **best, "reason": "verified" if best["accepted"] else "ball_moved_or_outside_foot_region"}
