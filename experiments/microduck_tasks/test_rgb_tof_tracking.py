"""A recent track can guide search but cannot replace fresh RGB and ToF."""

import inspect

import numpy as np
import pytest

from experiments.microduck_tasks.rgb_tof_fusion import fuse_balls
from experiments.microduck_tasks.rgb_tof_geometry import project_camera
from experiments.microduck_tasks.rgb_tof_tracking import recover_ball
from experiments.microduck_tasks.test_rgb_tof_fusion import _sensor_inputs
from experiments.microduck_tasks.test_rgb_tof_handoff import _case, _joined_red_foot


def _input():
    _, handoff = _case((0.09, 0.035, -0.077))
    config, pose, locked = handoff["config"], handoff["pose"], handoff["locked"]
    camera = pose[1].T @ (np.asarray(locked["center_body_m"]) - pose[0])
    rgb, ranges = _sensor_inputs(config, camera, calibrated_response=True)
    arguments = dict(config=config, pose=pose, valid=np.ones((8, 8), dtype=bool), locked=locked, lock_age_s=0.4)
    return _joined_red_foot(rgb), ranges, arguments


def test_joined_ball_is_recovered_only_with_current_range_support():
    rgb, ranges, args = _input()
    original, _ = fuse_balls(
        rgb,
        ranges,
        config=args["config"],
        valid=args["valid"],
        camera_position_body=args["pose"][0],
        camera_rotation_body=args["pose"][1],
    )
    assert original == []
    found, diagnostic = recover_ball(rgb, ranges, **args)
    assert found is not None, diagnostic
    np.testing.assert_allclose(found["center_body_m"], args["locked"]["center_body_m"], atol=0.008)
    assert found["tof_zone_count"] >= 2
    assert found["rgb_geometry"] == diagnostic["rgb_geometry"] == "partial_arc"
    assert diagnostic["current_tof_required"]
    assert diagnostic["arc_coverage_deg"] >= 130
    assert diagnostic["edge_source"] == "uncropped_rgb_mask_boundary"
    assert diagnostic["box_source"] == "fitted_sphere_cone_not_connected_component"
    assert diagnostic["box_uv"][3] < 230  # The joined red foot extends to row 239.


def test_prior_displacement_does_not_become_the_output_measurement():
    rgb, ranges, args = _input()
    actual = np.array(args["locked"]["center_body_m"])
    args["locked"]["center_body_m"] = (actual + [0.01, -0.008, 0.005]).tolist()
    found, diagnostic = recover_ball(rgb, ranges, **args)
    assert found is not None, diagnostic
    np.testing.assert_allclose(found["center_body_m"], actual, atol=0.008)
    assert np.linalg.norm(np.asarray(found["center_body_m"]) - args["locked"]["center_body_m"]) > 0.006
    assert diagnostic["prior_role"] == "search_region_only"


@pytest.mark.parametrize("failure", ["invalid_all", "nan_all", "shift_300mm", "background", "single_zone"])
def test_current_tof_failures_never_return_rgb_or_old_lock_coordinates(failure):
    rgb, ranges, args = _input()
    if failure == "invalid_all":
        args["valid"][:] = False
    elif failure == "nan_all":
        ranges[:] = np.nan
    elif failure == "shift_300mm":
        ranges += 0.3
    elif failure == "background":
        ranges[:] = 1.5
    else:
        found, diagnostic = recover_ball(rgb, ranges, **args)
        assert found is not None
        zone = diagnostic["supporting_zones"][0]
        args["valid"][:] = False
        args["valid"][zone["row"], zone["column"]] = True
    found, diagnostic = recover_ball(rgb, ranges, **args)
    assert found is None
    assert not diagnostic["recovered"]


@pytest.mark.parametrize("age", [-0.01, 0.80001, np.nan, np.inf])
def test_expired_or_invalid_prior_is_rejected(age):
    rgb, ranges, args = _input()
    args["lock_age_s"] = age
    assert recover_ball(rgb, ranges, **args)[0] is None


@pytest.mark.parametrize("position", [[0.2, 0.03, -0.08], [-0.03, 0.03, -0.08], [np.nan, 0.03, -0.08]])
def test_distant_behind_or_nonfinite_prior_is_rejected(position):
    rgb, ranges, args = _input()
    args["locked"]["center_body_m"] = position
    assert recover_ball(rgb, ranges, **args)[0] is None


@pytest.mark.parametrize("value", [None, np.nan, True, 0])
def test_prior_requires_a_real_sensor_support_count(value):
    rgb, ranges, args = _input()
    args["locked"]["tof_zone_count"] = value
    assert recover_ball(rgb, ranges, **args)[0] is None


@pytest.mark.parametrize("kind", ["missing", "wrong_color", "block", "short_arc"])
def test_current_rgb_evidence_is_required(kind):
    rgb, ranges, args = _input()
    if kind == "wrong_color":
        rgb[rgb[..., 0] == 230] = [12, 35, 230]
    elif kind in ("missing", "block"):
        rgb[:] = 100
        if kind == "block":
            rgb[65:, 100:220] = [230, 12, 12]
    else:
        point = np.asarray(args["locked"]["center_body_m"])
        uv = project_camera(args["config"], args["pose"][1].T @ (point - args["pose"][0]))
        rgb[int(uv[1]) - 5 :, :] = 100
        rgb[:, : int(uv[0]) + 2] = 100
    assert recover_ball(rgb, ranges, **args)[0] is None


def test_rgb_moved_outside_prior_search_cannot_reuse_old_ranges():
    _, ranges, args = _input()
    point = np.asarray(args["locked"]["center_body_m"]) + [0, -0.12, 0]
    camera = args["pose"][1].T @ (point - args["pose"][0])
    rgb, _ = _sensor_inputs(args["config"], camera, calibrated_response=True)
    assert recover_ball(rgb, ranges, **args)[0] is None


def test_interface_accepts_no_dense_depth_world_state_or_segmentation():
    assert set(inspect.signature(recover_ball).parameters) == {
        "rgb",
        "ranges_m",
        "config",
        "pose",
        "valid",
        "locked",
        "lock_age_s",
    }
    rgb, _, args = _input()
    with pytest.raises(ValueError, match="one radial range"):
        recover_ball(rgb, np.ones((240, 320)), **args)
