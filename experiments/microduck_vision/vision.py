"""Locate one red marker using only an RGB-D frame and calibrated camera geometry."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetObservation:
    """Metric target position in body axes; ``distance_m`` is horizontal distance."""

    found: bool
    forward_m: float | None
    left_m: float | None
    distance_m: float | None
    pixel_uv: tuple[float, float] | None
    pixels: int
    reason: str


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the largest eight-connected colour region, ignoring isolated noise."""
    height, width = mask.shape
    pending = set(np.flatnonzero(mask).tolist())
    largest: list[int] = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        component = [seed]
        frontier = [seed]
        while frontier:
            row, column = divmod(frontier.pop(), width)
            for v in range(max(0, row - 1), min(height, row + 2)):
                for u in range(max(0, column - 1), min(width, column + 2)):
                    neighbour = v * width + u
                    if neighbour in pending:
                        pending.remove(neighbour)
                        component.append(neighbour)
                        frontier.append(neighbour)
        if len(component) > len(largest):
            largest = component
    return np.divmod(np.asarray(largest, dtype=np.intp), width)


def detect_target(
    rgb: np.ndarray,
    depth: np.ndarray,
    fovy_deg: float,
    camera_position_body: np.ndarray,
    camera_rotation_body: np.ndarray,
    min_pixels: int = 6,
) -> TargetObservation:
    """Measure a red marker, without reading simulator object positions.

    ``depth`` is MuJoCo Renderer metric optical-axis depth, not raw OpenGL depth
    or radial range. RGB and depth must be registered, top-row-first images from
    the same camera. The camera has square pixels and vertical field of view
    ``fovy_deg``; cameras with explicit custom intrinsics need a different API.

    ``camera_rotation_body`` maps camera coordinates into body coordinates.
    MuJoCo camera axes are x right, y up, and -z forward. Body axes are x forward,
    y left, z up. Pixel coordinates refer to pixel centres, so the principal
    point is ((width - 1) / 2, (height - 1) / 2). Medians of the reconstructed
    points reduce the influence of occasional depth outliers.

    Invalid calibration or image dimensions raise ValueError. Missing markers
    and unusable target depth return ``found=False`` with no fabricated position.
    ``pixels`` counts valid-depth pixels in the largest red connected component.
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float64)
    position = np.asarray(camera_position_body, dtype=np.float64)
    rotation = np.asarray(camera_rotation_body, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Provide an RGB uint8 image with shape (height, width, 3).")
    height, width = rgb.shape[:2]
    if height == 0 or width == 0 or depth.shape != (height, width):
        raise ValueError("Provide nonempty RGB and depth images with matching dimensions.")
    if not np.isscalar(fovy_deg) or not np.isfinite(fovy_deg) or not 0 < fovy_deg < 180:
        raise ValueError("Provide a finite vertical field of view strictly between 0 and 180 degrees.")
    if isinstance(min_pixels, bool) or not isinstance(min_pixels, (int, np.integer)) or min_pixels < 1:
        raise ValueError("Set min_pixels to a positive integer.")
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("Provide a finite camera position with shape (3,).")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Provide a finite camera-to-body rotation with shape (3, 3).")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-6
    ):
        raise ValueError("Provide an orthonormal, right-handed camera-to-body rotation.")

    colour = rgb.astype(np.float32)
    red = (colour[..., 0] > 100) & (colour[..., 0] > 1.5 * colour[..., 1])
    red &= colour[..., 0] > 1.5 * colour[..., 2]
    rows, columns = _largest_component(red)
    if len(rows) < min_pixels:
        return TargetObservation(False, None, None, None, None, 0, "insufficient_target_pixels")
    z = depth[rows, columns]
    valid = np.isfinite(z) & (z > 0)
    rows, columns, z = rows[valid], columns[valid], z[valid]
    pixels = len(z)
    if pixels < min_pixels:
        return TargetObservation(False, None, None, None, None, pixels, "invalid_target_depth")

    focal_length = height / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
    if not np.isfinite(focal_length) or focal_length <= 0:
        raise ValueError("Provide a field of view that yields a finite positive focal length.")
    cx, cy = (width - 1) / 2, (height - 1) / 2
    with np.errstate(over="ignore", invalid="ignore"):
        points_camera = np.column_stack(((columns - cx) * z / focal_length, -(rows - cy) * z / focal_length, -z))
        points_body = points_camera @ rotation.T + position
    finite = np.all(np.isfinite(points_body), axis=1)
    if np.count_nonzero(finite) < min_pixels:
        return TargetObservation(False, None, None, None, None, int(finite.sum()), "invalid_target_depth")
    centre = np.median(points_body[finite], axis=0)
    forward, left = float(centre[0]), float(centre[1])
    distance = float(np.hypot(forward, left))
    if not np.isfinite(distance):
        return TargetObservation(False, None, None, None, None, 0, "invalid_target_depth")
    uv = (float(np.median(columns[finite])), float(np.median(rows[finite])))
    return TargetObservation(True, forward, left, distance, uv, int(finite.sum()), "ok")
