"""Calibrated RGB/ToF projection, with explicit radial surface ranges.

All optical coordinates are OpenCV axes: right, down, forward. The default
baseline is an illustrative simulation assumption, not a hardware calibration.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBToFConfig:
    width: int = 320
    height: int = 240
    fovy_deg: float = 60.0
    tof_rows: int = 8
    tof_columns: int = 8
    tof_fov_deg: float = 45.0
    tof_translation_camera_m: tuple = (-0.02, 0.0, 0.0)
    tof_rotation_camera: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    ball_radius_m: float = 0.035
    range_sigma_m: float = 0.002

    def __post_init__(self):
        for value in (self.width, self.height, self.tof_rows, self.tof_columns):
            if not isinstance(value, int) or value < 1:
                raise ValueError("Image and ToF grid sizes must be positive integers.")
        for angle in (self.fovy_deg, self.tof_fov_deg):
            if not np.isfinite(angle) or not 0 < angle < 180:
                raise ValueError("Camera and ToF fields of view must be finite and between 0 and 180 degrees.")
        if not np.isfinite(self.ball_radius_m) or self.ball_radius_m <= 0:
            raise ValueError("Provide a finite positive ball radius.")
        if not np.isfinite(self.range_sigma_m) or self.range_sigma_m < 0:
            raise ValueError("Provide a finite nonnegative radial range uncertainty.")
        translation = np.asarray(self.tof_translation_camera_m, dtype=float)
        rotation = np.asarray(self.tof_rotation_camera, dtype=float)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("Provide the finite ToF origin in camera axes.")
        if rotation.size != 9:
            raise ValueError("Provide nine values for the ToF-to-camera rotation.")
        check_rotation(rotation.reshape(3, 3))

    @property
    def focal_px(self):
        return self.height / (2 * np.tan(np.deg2rad(self.fovy_deg) / 2))

    @property
    def rotation(self):
        return np.asarray(self.tof_rotation_camera, dtype=float).reshape(3, 3)

    @property
    def translation(self):
        return np.asarray(self.tof_translation_camera_m, dtype=float)


def check_rotation(rotation):
    rotation = np.asarray(rotation, dtype=float)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise ValueError("Provide an orthonormal, right-handed rotation.")


def pixel_unit_rays(config, pixels_uv):
    pixels = np.asarray(pixels_uv, dtype=float)
    rays = np.concatenate(
        (
            (pixels - [(config.width - 1) / 2, (config.height - 1) / 2]) / config.focal_px,
            np.ones((*pixels.shape[:-1], 1)),
        ),
        axis=-1,
    )
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def project_camera(config, points):
    """Project camera points; points behind the camera have NaN coordinates."""
    points = np.asarray(points, dtype=float)
    depth = points[..., 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = config.focal_px * points[..., :2] / depth
    pixels += [(config.width - 1) / 2, (config.height - 1) / 2]
    return np.where(depth > 0, pixels, np.nan)


def zone_unit_rays(config, row, column, samples=3):
    """Uniform tangent-plane samples inside a zone, at subcell centres.

    The square field of view and zone response are simulation approximations, not
    the manufacturer's measured optical response. Returned rays are unit vectors.
    """
    if not 0 <= row < config.tof_rows or not 0 <= column < config.tof_columns:
        raise ValueError("Select a zone inside the configured ToF grid.")
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("Use a positive integer number of samples per zone axis.")
    half = np.tan(np.deg2rad(config.tof_fov_deg) / 2)
    offsets = (np.arange(samples) + 0.5) / samples
    x = (2 * (column + offsets) / config.tof_columns - 1) * half
    y = (2 * (row + offsets) / config.tof_rows - 1) * half
    xx, yy = np.meshgrid(x, y)
    rays = np.column_stack((xx.ravel(), yy.ravel(), np.ones(samples * samples)))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def zone_corners(config, row, column):
    half = np.tan(np.deg2rad(config.tof_fov_deg) / 2)
    corners = np.asarray([[column, row], [column + 1, row], [column + 1, row + 1], [column, row + 1]])
    xy = (2 * corners / [config.tof_columns, config.tof_rows] - 1) * half
    rays = np.column_stack((xy, np.ones(4)))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def project_zones(ranges_m, *, config, valid=None):
    """Project each scalar range's angular footprint as a radial shell.

    Boxes describe a coarse measurement area, not per-pixel depth or the actual
    return location. The baseline makes these footprints depend on range.
    """
    ranges = np.asarray(ranges_m, dtype=float)
    shape = (config.tof_rows, config.tof_columns)
    if ranges.shape != shape:
        raise ValueError(f"Provide exactly one radial range for each ToF zone: shape {shape}.")
    validity = np.ones(shape, dtype=bool) if valid is None else np.array(valid, dtype=bool, copy=True)
    if validity.shape != shape:
        raise ValueError("ToF validity must have the same shape as the coarse range grid.")
    validity &= np.isfinite(ranges) & (ranges > 0)
    projections = []
    for row, column in zip(*np.where(validity), strict=True):
        distance = float(ranges[row, column])
        rays = zone_corners(config, row, column)
        points = distance * rays @ config.rotation.T + config.translation
        pixels = project_camera(config, points)
        if not np.isfinite(pixels).all():
            continue
        projections.append(
            {
                "row": int(row),
                "column": int(column),
                "range_m": distance,
                "polygon_uv": pixels.tolist(),
                "box_uv": [*pixels.min(axis=0).tolist(), *pixels.max(axis=0).tolist()],
                "footprint_kind": "coarse_angular_zone_at_radial_range",
            }
        )
    return projections
