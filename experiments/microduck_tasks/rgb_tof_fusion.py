"""RGB colour/silhouette and coarse radial ToF fusion without world state.

The known-radius sphere is a scene prior. ToF support is mandatory: a visual
hypothesis alone never becomes a detection. A zone's return is never treated
as dense image depth, nor assigned to its centre ray without uncertainty.
"""

import numpy as np
from scipy import ndimage, optimize

from .rgb_tof_geometry import check_rotation, pixel_unit_rays, project_zones, zone_unit_rays


def _masks(rgb):
    r, g, b = np.moveaxis(rgb.astype(float), -1, 0)
    return {
        "red": (r > 55) & (r > 2.2 * g) & (r > 1.8 * b),
        "yellow": (r > 65) & (g > 65) & (r > 2 * b) & (g > 2 * b) & (r > 0.72 * g) & (r < 1.25 * g),
        "blue": (b > 55) & (b > 1.65 * r) & (b > 1.3 * g),
    }


def _hypotheses(rgb, config, rejected):
    for color, mask in _masks(rgb).items():
        labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
        for index in range(1, count + 1):
            region = labels == index
            rows, columns = np.where(region)
            if len(rows) < 16:
                continue
            box = [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]
            entry = {"color": color, "box_uv": box, "pixels": len(rows)}
            width, height = box[2] - box[0] + 1, box[3] - box[1] + 1
            fill = len(rows) / (width * height)
            if box[0] == 0 or box[1] == 0 or box[2] == config.width - 1 or box[3] == config.height - 1:
                rejected.append({**entry, "reason": "clipped_rgb_silhouette"})
                continue
            if not 0.55 <= fill <= 0.93 or not 0.45 < width / height < 2.2:
                rejected.append({**entry, "reason": "non_spherical_rgb_silhouette"})
                continue
            boundary = region & ~ndimage.binary_erosion(region)
            v, u = np.where(boundary)
            rays = pixel_unit_rays(config, np.column_stack((u, v)))
            _, _, vectors = np.linalg.svd(rays - rays.mean(axis=0), full_matrices=False)
            direction = vectors[-1]
            if direction[2] < 0:
                direction = -direction
            angles = np.arccos(np.clip(rays @ direction, -1, 1))
            # Boundary pixel centres are slightly inside the silhouette.
            alpha = float(np.median(angles) + 0.45 / config.focal_px)
            if not 0 < alpha < np.pi / 2 or np.std(angles) > 0.2 * alpha:
                rejected.append({**entry, "reason": "irregular_rgb_silhouette"})
                continue
            distance = config.ball_radius_m / np.sin(alpha)
            yield entry, direction, float(distance), float(alpha)


def _surface_ranges(centres_tof, rays, radius):
    """Ray/sphere surfaces predicted only from the visual sphere hypothesis."""
    b = np.einsum("...j,kj->...k", centres_tof, rays)
    discriminant = b * b - (np.sum(centres_tof**2, axis=-1, keepdims=True) - radius * radius)
    ranges = b - np.sqrt(np.maximum(0, discriminant))
    return np.where((discriminant >= 0) & (ranges > 0), ranges, np.nan)


def _prediction(distance, direction, rays, config):
    centre = (distance * direction - config.translation) @ config.rotation
    ranges = _surface_ranges(centre, rays, config.ball_radius_m)
    hits = np.sort(ranges[np.isfinite(ranges) & (ranges >= 0.04) & (ranges <= 2.0)])
    # The calibrated simulator reports the nearest cluster supported by two
    # subrays within 12 mm, rather than the median of every visible surface.
    # Only the visual sphere is hypothesized here; background is not known.
    for first in hits:
        cluster = hits[(hits >= first) & (hits <= first + 0.012)]
        if len(cluster) >= 2:
            return float(np.median(cluster)), float(len(cluster) / len(rays)), float(np.ptp(cluster))
    return None


def _fit(direction, distance, zones, config):
    noise = max(0.005, 3 * config.range_sigma_m)
    prior_sigma = max(0.008, 0.08 * distance)

    def objective(candidate):
        total = 0.25 * ((candidate - distance) / prior_sigma) ** 2
        for zone in zones:
            predicted = _prediction(candidate, direction, zone["rays"], config)
            if predicted is None:
                total += 25
                continue
            expected, fraction, spread = predicted
            uncertainty = max(noise, 0.3 * spread)
            error = (expected - zone["observed"]) / uncertainty
            total += max(0.1, fraction) * (np.sqrt(1 + error * error) - 1)
        return total

    grid = np.linspace(max(config.ball_radius_m * 1.05, distance * 0.78), distance * 1.25, 41)
    best = int(np.argmin([objective(value) for value in grid]))
    lower, upper = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    result = optimize.minimize_scalar(objective, bounds=(lower, upper), method="bounded", options={"xatol": 0.00005})
    return float(result.x)


def fuse_balls(rgb, ranges_m, *, config, camera_position_body, camera_rotation_body, valid=None):
    """Return (body-frame detections, auditable projection/association details).

    ``camera_rotation_body`` maps OpenCV camera axes into body axes. Inputs are
    only RGB, one radial surface range per zone, calibration and encoder-based
    camera pose. This module does not accept depth images, segmentation IDs,
    world poses, ball positions or simulation handles.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (config.height, config.width, 3):
        raise ValueError("Provide uint8 RGB with the configured camera dimensions.")
    position = np.asarray(camera_position_body, dtype=float)
    rotation = np.asarray(camera_rotation_body, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Provide the finite camera origin in body axes.")
    check_rotation(rotation)
    zones = project_zones(ranges_m, config=config, valid=valid)
    diagnostics = {
        "source": "rgb_silhouette_and_coarse_radial_tof_only",
        "projections": zones,
        "rejected_rgb_candidates": [],
        "associations": [],
        "ball_radius_prior_m": config.ball_radius_m,
        "range_convention": "euclidean_radial_surface_range_from_tof_origin",
        "zone_response_model": "5x5_subrays_nearest_12mm_cluster_with_at_least_2_returns",
    }
    detections = []
    for entry, direction, distance, alpha in _hypotheses(rgb, config, diagnostics["rejected_rgb_candidates"]):
        candidates = []
        for zone in zones:
            rays = zone_unit_rays(config, zone["row"], zone["column"], samples=5)
            predicted = _prediction(distance, direction, rays, config)
            if predicted is None:
                continue
            expected, fraction, spread = predicted
            tolerance = max(0.015, distance * 0.12, 4 * config.range_sigma_m) + 0.2 * spread
            if fraction >= 0.05 and abs(expected - zone["range_m"]) <= tolerance:
                candidates.append({"rays": rays, "observed": zone["range_m"], "zone": zone})
        if not candidates:
            diagnostics["rejected_rgb_candidates"].append(
                {
                    **entry,
                    "reason": "no_consistent_tof_support",
                    "rgb_distance_hypothesis_m": distance,
                }
            )
            continue
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
                {
                    "row": zone["row"],
                    "column": zone["column"],
                    "observed_surface_range_m": candidate["observed"],
                    "predicted_surface_range_m": expected,
                    "absolute_residual_m": residual,
                    "angular_support_fraction": fraction,
                }
            )
        if not evidence:
            diagnostics["rejected_rgb_candidates"].append({**entry, "reason": "tof_surface_fit_inconsistent"})
            continue
        centre_camera = fitted * direction
        centre_body = rotation @ centre_camera + position
        residual = float(np.mean([item["absolute_residual_m"] for item in evidence]))
        confidence = float(min(1, entry["pixels"] / 40) * np.exp(-residual / 0.015))
        uv = config.focal_px * direction[:2] / direction[2] + [(config.width - 1) / 2, (config.height - 1) / 2]
        detection = {
            "color": entry["color"],
            "center_body_m": centre_body.tolist(),
            "pixel_uv": uv.tolist(),
            "pixels": entry["pixels"],
            "confidence": confidence,
            "camera": "front_rgb_tof",
            "fit_residual_m": residual,
            "center_camera_m": centre_camera.tolist(),
            "tof_zone_count": len(evidence),
            "sensor_source": diagnostics["source"],
        }
        detections.append(detection)
        diagnostics["associations"].append(
            {
                **entry,
                "pixel_uv": uv.tolist(),
                "rgb_distance_hypothesis_m": distance,
                "fused_center_range_camera_m": fitted,
                "angular_radius_rad": alpha,
                "supporting_zones": evidence,
            }
        )
    return sorted(detections, key=lambda item: (item["color"], item["pixel_uv"][0])), diagnostics
