"""Coloured ball measurements from RGB-D pixels and camera calibration only."""

import numpy as np


def _components(mask):
    """Yield eight-connected regions, preserving separate balls of one colour."""
    height, width = mask.shape
    pending = set(np.flatnonzero(mask).tolist())
    while pending:
        seed = pending.pop()
        region, frontier = [seed], [seed]
        while frontier:
            row, column = divmod(frontier.pop(), width)
            for v in range(max(0, row - 1), min(height, row + 2)):
                for u in range(max(0, column - 1), min(width, column + 2)):
                    neighbour = v * width + u
                    if neighbour in pending:
                        pending.remove(neighbour)
                        region.append(neighbour)
                        frontier.append(neighbour)
        if len(region) >= 10:
            yield np.divmod(np.asarray(region, dtype=np.intp), width)


def _sphere(points, radius):
    """Fit a known-radius sphere, refusing planes and insufficient curvature.

    An unconstrained algebraic fit checks observed curvature before robust
    geometric refinement with the radius prior. The prior must not manufacture
    a plausible sphere from a coloured flat surface.
    """
    median = np.median(points, axis=0)
    keep = np.linalg.norm(points - median, axis=1) < 2.5 * radius
    points = points[keep]
    if len(points) < 10:
        return None
    origin = np.mean(points, axis=0)
    local = points - origin
    design = np.column_stack((2 * local, np.ones(len(local))))
    solution, _, rank, _ = np.linalg.lstsq(design, np.sum(local**2, axis=1), rcond=None)
    if rank != 4:
        return None
    centre = solution[:3] + origin
    radius_squared = solution[3] + np.dot(solution[:3], solution[:3])
    if not np.isfinite(radius_squared) or not (0.6 * radius) ** 2 < radius_squared < (1.5 * radius) ** 2:
        return None
    cutoff = max(0.0005, 0.04 * radius)
    for _ in range(12):
        delta = centre - points
        distances = np.linalg.norm(delta, axis=1)
        if np.any(distances < 1e-9):
            return None
        residuals = distances - radius
        weights = np.sqrt(np.minimum(1, cutoff / np.maximum(np.abs(residuals), 1e-12)))
        jacobian = delta / distances[:, None]
        update, _, rank, _ = np.linalg.lstsq(jacobian * weights[:, None], -residuals * weights, rcond=None)
        if rank != 3 or not np.all(np.isfinite(update)):
            return None
        centre += update
        if np.linalg.norm(update) < 1e-8:
            break
    distances = np.linalg.norm(points - centre, axis=1)
    residuals = np.abs(distances - radius)
    inliers = residuals <= 2.5 * cutoff
    if inliers.sum() < 10 or np.mean(inliers) < 0.75 or not np.all(np.isfinite(centre)):
        return None
    view_distance = np.linalg.norm(centre)
    if view_distance <= radius or centre[2] >= -radius:
        return None
    direction = centre / view_distance
    observed_cap = points[inliers] - np.mean(points[inliers], axis=0)
    tangent = observed_cap @ (np.eye(3) - np.outer(direction, direction))
    spread = np.linalg.svd(tangent, compute_uv=False) / np.sqrt(inliers.sum())
    # A thin coloured edge can fit a sphere without constraining its centre.
    # Metric spread keeps distant small images valid while rejecting close specks.
    if spread[1] < 0.12 * radius:
        return None
    # A visible sphere cap lies nearer to the camera than the sphere centre.
    cap_depth = np.dot(centre - np.mean(points[inliers], axis=0), centre / view_distance)
    if cap_depth < 0.15 * radius:
        return None
    residual = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
    if residual > 0.06 * radius:
        return None
    return centre, residual, int(inliers.sum()), float(np.mean(inliers))


def detect_balls(
    rgb,
    depth,
    *,
    fovy_deg,
    camera_position_body,
    camera_rotation_body,
    camera_name="front",
    radius=0.035,
) -> list[dict]:
    """Find red, yellow and blue ball centres in robot body axes, in metres.

    RGB is uint8 HxWx3; registered depth is positive metric optical-axis depth.
    MuJoCo camera axes are x right, y up, -z forward. The rotation maps these
    axes into body axes. Pixel centres use ((width-1)/2, (height-1)/2), and the
    vertical field of view assumes square pixels. Invalid calibration raises
    ValueError; absent or geometrically implausible detections yield no entry.

    ``confidence`` is a geometric quality score, not a calibrated probability.
    ``pixels`` counts fitted inliers. The known radius is a scene assumption;
    differently sized real objects need different calibration or a new detector.
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float64)
    position = np.asarray(camera_position_body, dtype=np.float64)
    rotation = np.asarray(camera_rotation_body, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Provide an RGB uint8 image with shape (height, width, 3).")
    height, width = rgb.shape[:2]
    if not height or not width or depth.shape != (height, width):
        raise ValueError("Provide nonempty RGB and depth images with matching dimensions.")
    if not np.isscalar(fovy_deg) or not np.isfinite(fovy_deg) or not 0 < fovy_deg < 180:
        raise ValueError("Provide a finite vertical field of view between 0 and 180 degrees.")
    if not np.isscalar(radius) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Provide a finite positive ball radius in metres.")
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("Provide a finite camera position with shape (3,).")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Provide a finite camera-to-body rotation with shape (3, 3).")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-6
    ):
        raise ValueError("Provide an orthonormal, right-handed camera-to-body rotation.")
    if not isinstance(camera_name, str) or not camera_name:
        raise ValueError("Provide a nonempty camera name.")
    colour = rgb.astype(np.float64)
    red, green, blue = colour[..., 0], colour[..., 1], colour[..., 2]
    masks = {
        "red": (red > 55) & (red > 2.2 * green) & (red > 1.8 * blue),
        "yellow": (red > 65)
        & (green > 65)
        & (red > 2 * blue)
        & (green > 2 * blue)
        & (red > 0.72 * green)
        & (red < 1.25 * green),
        "blue": (blue > 55) & (blue > 1.65 * red) & (blue > 1.3 * green),
    }
    focal = height / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Provide a field of view yielding a finite positive focal length.")
    detections = []
    for name, mask in masks.items():
        for rows, columns in _components(mask):
            z = depth[rows, columns]
            valid = np.isfinite(z) & (z > 0)
            rows, columns, z = rows[valid], columns[valid], z[valid]
            with np.errstate(over="ignore", invalid="ignore"):
                points = np.column_stack(
                    ((columns - (width - 1) / 2) * z / focal, -(rows - (height - 1) / 2) * z / focal, -z)
                )
            finite = np.all(np.isfinite(points), axis=1)
            if finite.sum() < 10:
                continue
            rows, columns, points = rows[finite], columns[finite], points[finite]
            fitted = _sphere(points, float(radius))
            if fitted is None:
                continue
            centre, residual, count, fraction = fitted
            with np.errstate(over="ignore", invalid="ignore"):
                body = rotation @ centre + position
            if not np.all(np.isfinite(body)):
                continue
            confidence = fraction * np.exp(-residual / (0.04 * radius)) * min(1.0, count / 40)
            detections.append(
                {
                    "color": name,
                    "center_body_m": body.tolist(),
                    "pixel_uv": [float(np.median(columns)), float(np.median(rows))],
                    "pixels": count,
                    "confidence": float(confidence),
                    "fit_residual_m": residual,
                    "camera": camera_name,
                }
            )
    return sorted(detections, key=lambda item: (item["color"], item["pixel_uv"][0]))
