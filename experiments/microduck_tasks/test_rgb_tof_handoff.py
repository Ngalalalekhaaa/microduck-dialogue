"""Fresh RGB geometry must independently support the bounded ToF memory."""

import numpy as np
import pytest

from experiments.microduck_tasks.rgb_tof_geometry import RGBToFConfig, project_camera
from experiments.microduck_tasks.rgb_tof_handoff import check_handoff
from experiments.microduck_tasks.test_rgb_tof_fusion import _sensor_inputs


def _case(point=(0.09, 0.035, -0.077), *, camera_position=(0.14, 0, 0.16)):
    config = RGBToFConfig(fovy_deg=100)
    rotation = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    pose = np.asarray(camera_position, dtype=float), rotation
    centre = rotation.T @ (np.asarray(point) - pose[0])
    rgb, _ = _sensor_inputs(config, centre)
    arguments = dict(
        config=config,
        pose=pose,
        locked=dict(
            color="red",
            center_body_m=list(point),
            tof_zone_count=4,
            sensor_source="rgb_silhouette_and_coarse_radial_tof_only",
        ),
        lock_time=10.0,
        observation=dict(time_s=10.6, sensor_age_s=0.0, stationary=True),
        foot="left",
    )
    return rgb, arguments


def _joined_red_foot(rgb):
    rgb = rgb.copy()
    rows, columns = np.where(rgb[..., 0] == 230)
    u, v = int(np.median(columns)), int(np.median(rows))
    rgb[v + 3 :, u - 5 : u + 6] = [230, 12, 12]
    return rgb


def test_complete_rgb_handoff_accepts_valid_recent_stationary_lock():
    rgb, arguments = _case()
    result = check_handoff(rgb, **arguments)
    assert result["accepted"]
    assert result["method"] == "complete_silhouette"
    np.testing.assert_allclose(result["center_body_m"], arguments["locked"]["center_body_m"], atol=0.006)
    assert result["source"] == "recent_tof_lock_plus_fresh_known_radius_rgb"


def test_partial_arc_recovers_ball_connected_to_red_foot_without_crop_edges():
    rgb, arguments = _case()
    result = check_handoff(_joined_red_foot(rgb), **arguments)
    assert result["accepted"], result
    assert result["method"] == "partial_arc"
    assert result["edge_source"] == "uncropped_rgb_mask_boundary"
    assert result["arc_coverage_deg"] >= 130
    assert result["arc_pixels"] >= 16
    np.testing.assert_allclose(result["center_body_m"], arguments["locked"]["center_body_m"], atol=0.008)


@pytest.mark.parametrize("age", [-0.01, 1.4001, np.nan, np.inf])
def test_expired_or_invalid_lock_time_is_rejected(age):
    rgb, arguments = _case()
    arguments["observation"]["time_s"] = arguments["lock_time"] + age
    assert not check_handoff(rgb, **arguments)["accepted"]


@pytest.mark.parametrize("age", [-0.01, 0.1201, np.nan, np.inf])
def test_stale_future_or_nonfinite_rgb_timestamp_is_rejected(age):
    rgb, arguments = _case()
    arguments["observation"]["sensor_age_s"] = age
    assert not check_handoff(rgb, **arguments)["accepted"]


@pytest.mark.parametrize("stationary", [False, np.nan, 1, "yes", None])
def test_unsettled_or_nonboolean_stationary_is_rejected(stationary):
    rgb, arguments = _case()
    arguments["observation"]["stationary"] = stationary
    assert not check_handoff(rgb, **arguments)["accepted"]


@pytest.mark.parametrize("foot", ["right", "invalid", None])
def test_wrong_or_unknown_foot_is_rejected(foot):
    rgb, arguments = _case()
    arguments["foot"] = foot
    assert not check_handoff(rgb, **arguments)["accepted"]


@pytest.mark.parametrize("kind", ["missing", "wrong_color", "large_block", "thin_strip"])
def test_missing_ball_or_nonspherical_colour_region_is_rejected(kind):
    rgb, arguments = _case()
    if kind == "wrong_color":
        rgb[rgb[..., 0] == 230] = [12, 35, 230]
    else:
        rgb[:] = 100
        if kind == "large_block":
            rgb[60:, 100:220] = [230, 12, 12]
        elif kind == "thin_strip":
            rgb[125:128, 110:190] = [230, 12, 12]
    assert not check_handoff(rgb, **arguments)["accepted"]


@pytest.mark.parametrize("moved", [(0.15, 0.035, -0.077), (0.112, 0.035, -0.077), (0.09, -0.035, -0.077)])
def test_ball_movement_or_exit_from_foot_region_is_rejected(moved):
    _, arguments = _case()
    rgb, _ = _case(moved)
    assert not check_handoff(rgb, **arguments)["accepted"]


def test_large_movement_inside_same_foot_region_is_rejected():
    _, arguments = _case((0.077, 0.022, -0.045))
    rgb, _ = _case((0.102, 0.048, -0.103))
    assert not check_handoff(rgb, **arguments)["accepted"]


def test_image_clipped_ball_cannot_be_rescued_by_partial_arc():
    point = np.array([0.09, 0.035, -0.077])
    _, arguments = _case(point)
    camera_point = np.array([(2 - 159.5) / arguments["config"].focal_px * 0.237, 0, 0.237])
    camera_position = point - arguments["pose"][1] @ camera_point
    rgb, arguments = _case(point, camera_position=camera_position)
    assert np.any(rgb[:, 0, 0] == 230)
    assert not check_handoff(rgb, **arguments)["accepted"]


@pytest.mark.parametrize("field", ["position", "rotation", "lock"])
def test_nonfinite_pose_or_lock_is_rejected(field):
    rgb, arguments = _case()
    if field == "position":
        arguments["pose"][0][0] = np.nan
    elif field == "rotation":
        arguments["pose"][1][0, 0] = 2
    else:
        arguments["locked"]["center_body_m"][0] = np.nan
    assert not check_handoff(rgb, **arguments)["accepted"]


@pytest.mark.parametrize("field,value", [("sensor_source", "rgb_only"), ("tof_zone_count", 0)])
def test_no_partial_arc_without_tof_lock_provenance(field, value):
    rgb, arguments = _case()
    arguments["locked"][field] = value
    assert not check_handoff(_joined_red_foot(rgb), **arguments)["accepted"]


def test_colour_patch_with_only_a_short_curved_fragment_is_rejected():
    rgb, arguments = _case()
    point = np.asarray(arguments["locked"]["center_body_m"])
    uv = project_camera(arguments["config"], arguments["pose"][1].T @ (point - arguments["pose"][0]))
    # Leave less than a quarter of the true outline; closing the mask creates
    # straight internal edges, which must not become independent ball evidence.
    rgb[int(uv[1]) - 5 :, :] = 100
    rgb[:, : int(uv[0]) + 2] = 100
    assert not check_handoff(rgb, **arguments)["accepted"]


def _foot_pose():
    return dict(position_body_m=[0.01, 0.042, -0.117], rotation_body_from_foot=np.eye(3).tolist())


def test_encoder_foot_pose_accepts_reachable_ball_outside_legacy_body_box():
    rgb, arguments = _case((0.106, 0.0147, -0.082))
    assert not check_handoff(rgb, **arguments)["accepted"]
    result = check_handoff(rgb, **arguments, foot_pose=_foot_pose())
    assert result["accepted"], result
    np.testing.assert_allclose(result["center_foot_m"], [0.096, -0.0273, 0.035], atol=0.006)


def test_current_foot_pose_can_refuse_ball_inside_legacy_body_box():
    rgb, arguments = _case()
    assert check_handoff(rgb, **arguments)["accepted"]
    foot_pose = _foot_pose()
    foot_pose["position_body_m"][0] = -0.06
    assert not check_handoff(rgb, **arguments, foot_pose=foot_pose)["accepted"]


@pytest.mark.parametrize("kind", ["missing", "nan_position", "nan_rotation", "reflection", "scaled", "short"])
def test_optional_foot_pose_never_falls_back_after_invalid_fk(kind):
    rgb, arguments = _case()
    foot_pose = _foot_pose()
    if kind == "missing":
        foot_pose = {}
    elif kind == "nan_position":
        foot_pose["position_body_m"][0] = np.nan
    elif kind == "nan_rotation":
        foot_pose["rotation_body_from_foot"][0][0] = np.nan
    elif kind == "reflection":
        foot_pose["rotation_body_from_foot"] = np.diag([-1.0, 1.0, 1.0]).tolist()
    elif kind == "scaled":
        foot_pose["rotation_body_from_foot"] = (2 * np.eye(3)).tolist()
    else:
        foot_pose["position_body_m"] = [0, 0]
    assert not check_handoff(rgb, **arguments, foot_pose=foot_pose)["accepted"]


@pytest.mark.parametrize("age", [-0.01, 0.1201, np.nan, np.inf])
def test_encoder_foot_pose_never_rescues_old_or_nonfinite_rgb(age):
    rgb, arguments = _case()
    arguments["observation"]["sensor_age_s"] = age
    assert not check_handoff(rgb, **arguments, foot_pose=_foot_pose())["accepted"]


@pytest.mark.parametrize("stationary", [False, None, 1, np.nan, "yes"])
def test_encoder_foot_pose_preserves_strict_stationary_guard(stationary):
    rgb, arguments = _case()
    arguments["observation"]["stationary"] = stationary
    assert not check_handoff(rgb, **arguments, foot_pose=_foot_pose())["accepted"]
