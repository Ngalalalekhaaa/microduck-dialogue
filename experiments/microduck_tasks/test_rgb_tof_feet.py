"""Foot kinematics matches model sites while ignoring live roots and balls."""

import numpy as np
import pytest

from .rgb_tof_feet import FootPose, feet_from_encoders


@pytest.fixture(scope="module")
def model():
    from .bootstrap import ASSETS, configure

    try:
        configure()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_path(str(ASSETS / "scene_ball.xml"))


def test_home_sites_are_symmetric_and_transform_round_trip(model):
    from microduck_deploy.policy import HOME

    feet = feet_from_encoders(model, HOME)
    for foot, sign in (("left", 1), ("right", -1)):
        pose = feet[foot]
        np.testing.assert_allclose(
            pose.position_body_m, [0.000015921845622, sign * 0.041820958697876, -0.117127771906494]
        )
        np.testing.assert_allclose(pose.rotation_body_from_foot[:, 0], [1, 0, 0], atol=1e-12)
        assert pose.rotation_body_from_foot[1, 1] > 0.99
        assert pose.rotation_body_from_foot[2, 2] > 0.99
        point = np.array([0.09, sign * 0.035, -0.075])
        np.testing.assert_allclose(pose.to_body(pose.to_foot(point)), point, atol=1e-12)
        np.testing.assert_allclose(pose.to_foot(pose.position_body_m), np.zeros(3), atol=1e-12)


def test_nonzero_leg_pose_matches_independent_live_fk_for_arbitrary_roots(model):
    import mujoco
    from microduck_deploy.policy import HOME, JOINT_NAMES

    encoders = HOME.copy()
    encoders[JOINT_NAMES.index("left_knee")] += 0.25
    encoders[JOINT_NAMES.index("right_hip_pitch")] -= 0.2
    encoders[JOINT_NAMES.index("head_yaw")] += 0.4
    original_encoders, original_defaults = encoders.copy(), model.qpos0.copy()
    predicted = feet_from_encoders(model, encoders)
    data = mujoco.MjData(model)
    root = int(model.joint("trunk_base_freejoint").qposadr[0])
    ball = int(model.joint("ball_free").qposadr[0])
    trunk = model.body("trunk_base").id
    qi = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    for offset in (0.0, 0.8, -0.45):
        axis = np.array([0.3, 0.4, 0.5])
        axis /= np.linalg.norm(axis)
        data.qpos[root : root + 7] = [offset, -2 * offset, 0.3 + offset, np.cos(offset), *(np.sin(offset) * axis)]
        data.qpos[ball : ball + 3] = [5 * offset, 2 - offset, 0.5]
        data.qpos[qi] = encoders
        mujoco.mj_kinematics(model, data)
        body_rotation = data.xmat[trunk].reshape(3, 3)
        live_before = data.qpos.copy()
        repeated = feet_from_encoders(model, encoders)
        for foot in predicted:
            site = model.site(f"{foot}_foot").id
            expected_position = body_rotation.T @ (data.site_xpos[site] - data.xpos[trunk])
            expected_rotation = body_rotation.T @ data.site_xmat[site].reshape(3, 3)
            np.testing.assert_allclose(repeated[foot].position_body_m, expected_position, atol=1e-12)
            np.testing.assert_allclose(repeated[foot].rotation_body_from_foot, expected_rotation, atol=1e-12)
            np.testing.assert_allclose(repeated[foot].position_body_m, predicted[foot].position_body_m)
        np.testing.assert_array_equal(data.qpos, live_before)
    np.testing.assert_array_equal(encoders, original_encoders)
    np.testing.assert_array_equal(model.qpos0, original_defaults)


def test_rejects_wrong_encoder_shapes_and_nonfinite_values(model):
    for value in (np.zeros(13), np.zeros((1, 14)), np.full(14, np.nan), np.full(14, np.inf)):
        with pytest.raises(ValueError, match="Joint encoders"):
            feet_from_encoders(model, value)


def test_foot_pose_rejects_invalid_geometry_and_points():
    for rotation in (np.zeros((3, 3)), np.diag([1, 1, -1]), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError, match="rotation"):
            FootPose("left", np.zeros(3), rotation)
    with pytest.raises(ValueError, match="Choose foot"):
        FootPose("middle", np.zeros(3), np.eye(3))
    pose = FootPose("right", np.zeros(3), np.eye(3))
    for point in ([0, 0], [0, np.nan, 0], [0, 0, np.inf]):
        with pytest.raises(ValueError, match="finite"):
            pose.to_foot(point)
        with pytest.raises(ValueError, match="finite"):
            pose.to_body(point)
    with pytest.raises(ValueError):
        pose.position_body_m[0] = 1
