"""Independent synthetic rays check geometry and coarse sensor evidence."""

import inspect
from dataclasses import replace

import numpy as np
import pytest

from experiments.microduck_tasks.rgb_tof_fusion import _prediction, fuse_balls
from experiments.microduck_tasks.rgb_tof_geometry import (
    RGBToFConfig,
    pixel_unit_rays,
    project_camera,
    project_zones,
    zone_unit_rays,
)


def _intersections(rays, centre, radius):
    dot = rays @ centre
    determinant = dot**2 - (np.dot(centre, centre) - radius**2)
    distance = dot - np.sqrt(np.maximum(determinant, 0))
    return np.where((determinant >= 0) & (distance > 0), distance, np.nan)


def _sensor_inputs(config, centre, *, background=1.5, calibrated_response=False):
    """Generate RGB and a separate single-range-per-zone sensor fixture."""
    centre = np.asarray(centre, dtype=float)
    rows, columns = np.indices((config.height, config.width))
    rays = pixel_unit_rays(config, np.stack((columns, rows), axis=-1))
    distances = _intersections(rays, centre, config.ball_radius_m)
    rgb = np.full((config.height, config.width, 3), 100, dtype=np.uint8)
    rgb[np.isfinite(distances)] = [230, 12, 12]
    ranges = np.full((config.tof_rows, config.tof_columns), background)
    centre_tof = config.rotation.T @ (centre - config.translation)
    for row in range(config.tof_rows):
        for column in range(config.tof_columns):
            # The independent robustness fixture deliberately changes both the
            # quadrature and reducer; the calibrated fixture tests its contract.
            count = 5 if calibrated_response else 11
            samples = zone_unit_rays(config, row, column, samples=count)
            distances = _intersections(samples, centre_tof, config.ball_radius_m)
            hits = distances[np.isfinite(distances)]
            if calibrated_response:
                returns = np.sort(np.where(np.isfinite(distances), distances, background))
                separation = returns[None, :] - returns[:, None]
                windows = (separation >= 0) & (separation <= 0.012)
                supported = np.flatnonzero(windows.sum(axis=1) >= 2)
                if len(supported):
                    ranges[row, column] = np.median(returns[windows[supported[0]]])
            elif len(hits) >= 5:
                ranges[row, column] = np.median(hits)
    return rgb, ranges


def _fuse(rgb, ranges, config, **kwargs):
    return fuse_balls(
        rgb, ranges, config=config, camera_position_body=[0, 0, 0], camera_rotation_body=np.eye(3), **kwargs
    )


def test_left_baseline_projection_depends_on_range():
    config = RGBToFConfig()
    near = project_camera(config, config.translation + [0, 0, 0.1])
    far = project_camera(config, config.translation + [0, 0, 1.0])
    centre = (config.width - 1) / 2
    assert near[0] < far[0] < centre
    assert centre - near[0] == pytest.approx(10 * (centre - far[0]))
    near_boxes = project_zones(np.full((8, 8), 0.1), config=config)
    far_boxes = project_zones(np.full((8, 8), 1.0), config=config)
    assert near_boxes[27]["polygon_uv"][0][0] < far_boxes[27]["polygon_uv"][0][0]


def test_rotated_baseline_and_radial_convention():
    theta = np.deg2rad(15)
    rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    config = RGBToFConfig(tof_rotation_camera=tuple(rotation.ravel()), tof_translation_camera_m=(-0.03, 0.01, 0))
    ray = zone_unit_rays(config, 6, 5, samples=1)[0]
    assert np.linalg.norm(ray) == pytest.approx(1)
    point = rotation @ (0.4 * ray) + config.translation
    projected = project_camera(config, point)
    expected = config.focal_px * point[:2] / point[2] + [159.5, 119.5]
    np.testing.assert_allclose(projected, expected)
    assert 0.4 * ray[2] < 0.4  # A radial scalar is not an optical-axis depth.


@pytest.mark.parametrize("centre", [[0, 0, 0.4], [-0.045, 0.02, 0.3], [0.02, 0.015, 0.14]])
def test_rgb_and_coarse_tof_recover_ball_centre_not_surface(centre):
    config = RGBToFConfig(fovy_deg=100)
    rgb, ranges = _sensor_inputs(config, centre)
    detections, diagnostics = _fuse(rgb, ranges, config)
    assert len(detections) == 1, diagnostics
    found = detections[0]
    np.testing.assert_allclose(found["center_body_m"], centre, atol=0.008)
    assert found["tof_zone_count"] > 0
    association = diagnostics["associations"][0]
    supported = association["supporting_zones"]
    assert min(zone["observed_surface_range_m"] for zone in supported) < np.linalg.norm(centre) - 0.015
    assert association["fused_center_range_camera_m"] > min(zone["observed_surface_range_m"] for zone in supported)


@pytest.mark.parametrize("angles,indices", [([0, 1, 18, 19, 20], [0, 1]), ([0, 18, 19, 20], [1, 2, 3])])
def test_prediction_uses_nearest_supported_cluster_and_skips_isolated_return(angles, indices):
    config = RGBToFConfig(tof_translation_camera_m=(0, 0, 0))
    radians = np.deg2rad(angles)
    rays = np.column_stack((np.sin(radians), np.zeros(len(angles)), np.cos(radians)))
    expected = _intersections(rays, np.array([0, 0, 0.1]), 0.035)
    prediction = _prediction(0.1, np.array([0, 0, 1]), rays, config)
    assert prediction[0] == pytest.approx(np.median(expected[indices]))
    assert prediction[1] == pytest.approx(len(indices) / len(angles))
    assert abs(prediction[0] - np.median(expected)) > 0.001


@pytest.mark.parametrize("centre", [[0, 0, 0.4], [-0.045, 0.02, 0.3], [0.02, 0.015, 0.14]])
def test_calibrated_five_by_five_cluster_response_recovers_centres(centre):
    config = RGBToFConfig(fovy_deg=100)
    rgb, ranges = _sensor_inputs(config, centre, calibrated_response=True)
    detections, diagnostics = _fuse(rgb, ranges, config)
    assert len(detections) == 1, diagnostics
    np.testing.assert_allclose(detections[0]["center_body_m"], centre, atol=0.008)
    assert diagnostics["zone_response_model"] == "5x5_subrays_nearest_12mm_cluster_with_at_least_2_returns"


def test_background_ranges_cannot_support_visible_ball():
    config = RGBToFConfig()
    rgb, _ = _sensor_inputs(config, [0, 0, 0.3])
    detections, diagnostics = _fuse(rgb, np.full((8, 8), 1.5), config)
    assert detections == []
    assert diagnostics["rejected_rgb_candidates"][0]["reason"] == "no_consistent_tof_support"


def test_no_tof_no_detection_and_input_validity_is_not_modified():
    config = RGBToFConfig()
    rgb, _ = _sensor_inputs(config, [0, 0, 0.3])
    valid = np.ones((8, 8), dtype=bool)
    detections, diagnostics = _fuse(rgb, np.full((8, 8), np.nan), config, valid=valid)
    assert detections == []
    assert diagnostics["projections"] == []
    assert valid.all()


def test_rotated_sensor_camera_and_body_pose_are_used():
    angle = np.deg2rad(-8)
    rotation_tof = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    config = RGBToFConfig(tof_rotation_camera=tuple(rotation_tof.ravel()))
    centre = np.array([-0.025, 0, 0.25])
    rgb, ranges = _sensor_inputs(config, centre)
    body_rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    body_position = np.array([0.03, 0.01, 0.14])
    detections, _ = fuse_balls(
        rgb, ranges, config=config, camera_position_body=body_position, camera_rotation_body=body_rotation
    )
    np.testing.assert_allclose(detections[0]["center_body_m"], body_rotation @ centre + body_position, atol=0.008)


def test_no_dense_depth_or_truth_input_is_accepted():
    config = RGBToFConfig()
    rgb, ranges = _sensor_inputs(config, [0, 0, 0.3])
    with pytest.raises(ValueError, match="one radial range"):
        _fuse(rgb, np.full((240, 320), 0.3), config)
    with pytest.raises(TypeError):
        _fuse(rgb, ranges, config, ball_world_position=[0, 0, 0.3])
    assert set(inspect.signature(fuse_balls).parameters) == {
        "rgb",
        "ranges_m",
        "config",
        "camera_position_body",
        "camera_rotation_body",
        "valid",
    }


def test_invalid_calibration_is_rejected():
    with pytest.raises(ValueError, match="rotation"):
        RGBToFConfig(tof_rotation_camera=(2, 0, 0, 0, 1, 0, 0, 0, 1))
    with pytest.raises(ValueError, match="positive ball radius"):
        replace(RGBToFConfig(), ball_radius_m=np.nan)
