"""Recover a near ball's RGB arc only when fresh coarse ToF corroborates it.

A recent sensor detection bounds the image search. Its coordinates are never
returned as a new measurement: current RGB boundary geometry and current ToF
surface ranges are fitted anew. This is separate from the RGB-only kick handoff.
"""

import numpy as np
from scipy import ndimage

from .rgb_tof_fusion import _fit, _masks, _prediction
from .rgb_tof_geometry import check_rotation, pixel_unit_rays, project_camera, project_zones, zone_unit_rays
from .rgb_tof_handoff import _cone_outline, _in_frame

SOURCE = "rgb_silhouette_and_coarse_radial_tof_only"


def _arc(rgb, config, pose, locked):
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
    # Extract edges before selecting the search region; never fit a crop edge.
    rows, columns = np.where(mask & ~ndimage.binary_erosion(mask))
    rays = pixel_unit_rays(config, np.column_stack((columns, rows)))
    angles = np.arccos(np.clip(rays @ direction, -1, 1))
    uncertainty = max(0.035, 0.25 * alpha, np.arcsin(min(0.99, 0.025 / distance)))
    keep = (
        (np.abs(angles - alpha) < uncertainty)
        & (rows <= uv[1] + 0.5 * config.focal_px * np.tan(alpha))
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
    direction = vectors[-1] * (1 if vectors[-1, 2] > 0 else -1)
    angles = np.arccos(np.clip(accepted @ direction, -1, 1))
    raw_alpha = float(np.median(angles))
    residual = float(np.quantile(np.abs(angles - raw_alpha), 0.9))
    phase = np.sort(np.arctan2(accepted @ np.cross(direction, vectors[0]), accepted @ vectors[0]))
    coverage = float(2 * np.pi - np.diff(np.r_[phase, phase[0] + 2 * np.pi]).max())
    alpha = raw_alpha + 0.45 / config.focal_px
    if residual > threshold or coverage < np.deg2rad(130) or not 0 < alpha < np.pi / 2:
        return None
    if not _in_frame(config, direction, alpha):
        return None
    evidence = dict(
        rgb_geometry="partial_arc",
        arc_pixels=int(best.sum()),
        arc_coverage_deg=float(np.rad2deg(coverage)),
        arc_residual_rad=residual,
        edge_source="uncropped_rgb_mask_boundary",
        roi_position_uncertainty_m=0.025,
        roi_angular_uncertainty_rad=float(uncertainty),
    )
    return direction, float(config.ball_radius_m / np.sin(alpha)), evidence


def _range_fit(direction, distance, zones, config):
    candidates = []
    for zone in zones:
        rays = zone_unit_rays(config, zone["row"], zone["column"], samples=5)
        predicted = _prediction(distance, direction, rays, config)
        if predicted is None:
            continue
        expected, fraction, spread = predicted
        tolerance = max(0.015, distance * 0.12, 4 * config.range_sigma_m) + 0.2 * spread
        if fraction >= 0.05 and abs(expected - zone["range_m"]) <= tolerance:
            candidates.append(dict(rays=rays, observed=zone["range_m"], zone=zone))
    if len(candidates) < 2:
        return None
    fitted = _fit(direction, distance, candidates, config)
    evidence = []
    for candidate in candidates:
        predicted = _prediction(fitted, direction, candidate["rays"], config)
        if predicted is None:
            continue
        expected, fraction, spread = predicted
        residual = abs(expected - candidate["observed"])
        if residual > max(0.012, 4 * config.range_sigma_m, 0.45 * spread):
            continue
        zone = candidate["zone"]
        evidence.append(
            dict(
                row=zone["row"],
                column=zone["column"],
                observed_surface_range_m=candidate["observed"],
                predicted_surface_range_m=expected,
                absolute_residual_m=residual,
                angular_support_fraction=fraction,
            )
        )
    return (fitted, evidence) if len(evidence) >= 2 else None


def recover_ball(rgb, ranges_m, *, config, pose, valid, locked, lock_age_s):
    """Return a new detection plus association diagnostics, or None plus refusal.

    The caller must invalidate stale current ranges and advance a track timestamp
    only for a newly captured sensor frame. A prior <=0.8 s old and in front of
    the body within 0.2 m is required; every recovery needs >=2 current ToF zones.
    """
    diagnostic = dict(
        recovered=False,
        source=SOURCE,
        rgb_geometry="partial_arc",
        prior_age_s=lock_age_s,
        prior_role="search_region_only",
        supporting_zones=[],
        range_convention="euclidean_radial_surface_range_from_tof_origin",
        zone_response_model="5x5_subrays_nearest_12mm_cluster_with_at_least_2_returns",
    )

    def refuse(reason):
        return None, {**diagnostic, "reason": reason}

    if not np.isfinite(lock_age_s) or not 0 <= lock_age_s <= 0.8 + 1e-9:
        return refuse("prior_expired_or_invalid_age")
    count = locked.get("tof_zone_count") if isinstance(locked, dict) else None
    if (
        not isinstance(locked, dict)
        or locked.get("sensor_source") != SOURCE
        or not isinstance(count, (int, np.integer))
        or isinstance(count, (bool, np.bool_))
        or count < 1
    ):
        return refuse("missing_sensor_prior")
    prior = np.asarray(locked.get("center_body_m"), dtype=float)
    if prior.shape != (3,) or not np.isfinite(prior).all() or not 0 < prior[0] < 0.2:
        return refuse("prior_not_near_ball")
    diagnostic["color"] = locked.get("color")
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (config.height, config.width, 3):
        return refuse("invalid_rgb")
    position, rotation = np.asarray(pose[0], dtype=float), np.asarray(pose[1], dtype=float)
    try:
        check_rotation(rotation)
    except ValueError:
        return refuse("invalid_camera_pose")
    if position.shape != (3,) or not np.isfinite(position).all():
        return refuse("invalid_camera_pose")
    zones = project_zones(ranges_m, config=config, valid=valid)
    if not zones:
        return refuse("no_current_tof")
    arc = _arc(rgb, config, (position, rotation), locked)
    if arc is None:
        return refuse("insufficient_rgb_arc")
    direction, distance, evidence = arc
    diagnostic.update(evidence, rgb_distance_hypothesis_m=distance)
    fitted = _range_fit(direction, distance, zones, config)
    if fitted is None:
        return refuse("no_consistent_current_tof")
    distance, supporting = fitted
    alpha = np.arcsin(config.ball_radius_m / distance)
    if not _in_frame(config, direction, alpha):
        return refuse("fused_ball_clipped")
    camera = distance * direction
    body = position + rotation @ camera
    pixels = project_camera(config, camera)
    outline = _cone_outline(config, direction, alpha)
    residual = float(np.mean([zone["absolute_residual_m"] for zone in supporting]))
    confidence = float(min(1, evidence["arc_pixels"] / 40) * np.exp(-residual / 0.015))
    detection = dict(
        color=locked["color"],
        center_body_m=body.tolist(),
        center_camera_m=camera.tolist(),
        pixel_uv=pixels.tolist(),
        pixels=evidence["arc_pixels"],
        confidence=confidence,
        camera="front_rgb_tof",
        fit_residual_m=residual,
        tof_zone_count=len(supporting),
        sensor_source=SOURCE,
        rgb_geometry="partial_arc",
    )
    diagnostic.update(
        recovered=True,
        reason="fresh_rgb_arc_and_tof_verified",
        pixels=evidence["arc_pixels"],
        box_uv=[*outline.min(axis=0).tolist(), *outline.max(axis=0).tolist()],
        pixel_uv=pixels.tolist(),
        box_source="fitted_sphere_cone_not_connected_component",
        supporting_zones=supporting,
        fused_center_range_camera_m=distance,
        angular_radius_rad=float(alpha),
        prior_change_m=float(np.linalg.norm(body - prior)),
        current_tof_required=True,
    )
    return detection, diagnostic
