"""Sensor-only body estimation, including motion that is not a yaw rotation."""

import math

import numpy as np
import pytest

from .rgb_tof_body import GRAVITY, BodyEstimator, _gravity, _orientation


def test_stationary_tilt_and_quaternion_agree_without_absolute_heading():
    estimator = BodyEstimator()
    angle = math.radians(23)
    gravity = np.array((math.sin(angle), 0, -math.cos(angle)))
    for index in range(31):
        state = estimator.update(index * 0.02, [0, 0, 0], -GRAVITY * gravity, [0, 0])
        if index < 15:
            assert not state.stationary
    assert state.stationary
    assert state.confidence == 1
    assert state.yaw == pytest.approx(0)
    assert state.tilt_deg == pytest.approx(23)
    assert state.projected_gravity == pytest.approx(gravity)
    assert _gravity(state.orientation_wxyz) == pytest.approx(gravity)
    assert state.is_fresh(0.65)
    assert not state.is_fresh(0.71)
    assert not state.is_fresh(0.59)


def test_yaw_uses_body_gyro_not_just_gyro_z_when_tilted():
    estimator = BodyEstimator()
    roll = math.radians(30)
    gravity = np.array((0, -math.sin(roll), -math.cos(roll)))
    gyro = -gravity * 0.5  # Constant world-up rotation expressed in the tilted body.
    estimator.update(0, gyro, -GRAVITY * gravity, [0])
    for index in range(1, 101):
        state = estimator.update(index * 0.01, gyro, -GRAVITY * gravity, [0])
    assert state.yaw == pytest.approx(0.5, abs=1e-10)
    assert state.tilt_deg == pytest.approx(30, abs=1e-10)
    assert not state.stationary


def test_roll_rotation_and_rejected_acceleration_propagate_gyro():
    estimator = BodyEstimator()
    estimator.update(0, [0, 0, 0], [0, 0, GRAVITY], [0])
    for index in range(1, 101):
        state = estimator.update(index * 0.01, [0.4, 0, 0], [0, 0, 2 * GRAVITY], [0])
    assert state.tilt_deg == pytest.approx(math.degrees(0.4))
    assert state.yaw == pytest.approx(0)
    assert not state.accel_correction_used
    assert state.confidence == 0.25
    assert not state.stationary


@pytest.mark.parametrize("motion", ["gyro", "encoder", "acceleration"])
def test_motion_rejects_stationary_and_needs_a_new_settled_window(motion):
    estimator = BodyEstimator()
    joint = 0
    for index in range(31):
        gyro = [0, 0, 0.8] if motion == "gyro" else [0, 0, 0]
        joint += 0.02 if motion == "encoder" else 0
        accel = [1.5 * (-1) ** index, 0, GRAVITY] if motion == "acceleration" else [0, 0, GRAVITY]
        state = estimator.update(index * 0.02, gyro, accel, [joint])
    assert not state.stationary
    state = estimator.update(0.62, [0, 0, 0], [0, 0, GRAVITY], [joint])
    assert not state.stationary
    for index in range(32, 50):
        state = estimator.update(index * 0.02, [0, 0, 0], [0, 0, GRAVITY], [joint])
    assert state.stationary


def test_invalid_inputs_leave_previous_state_unchanged():
    estimator = BodyEstimator()
    with pytest.raises(ValueError, match="standing"):
        estimator.update(0, [0, 0, 0], [0, 0, 0], [0])
    prior = estimator.update(0, [0, 0, 0], [0, 0, GRAVITY], [0])
    for timestamp in (0, -0.1, 0.2, math.nan):
        with pytest.raises(ValueError, match="interval|finite"):
            estimator.update(timestamp, [0, 0, 0], [0, 0, GRAVITY], [0])
        assert estimator.state is prior
    for gyro, accel, joint in (
        ([0, math.inf, 0], [0, 0, GRAVITY], [0]),
        ([0, 0, 0], [0, math.nan, GRAVITY], [0]),
        ([0, 0, 0], [0, 0, GRAVITY], [math.nan]),
    ):
        with pytest.raises(ValueError, match="finite"):
            estimator.update(0.02, gyro, accel, joint)
        assert estimator.state is prior
    with pytest.raises(ValueError, match="count changed"):
        estimator.update(0.02, [0, 0, 0], [0, 0, GRAVITY], [0, 0])
    estimator.reset()
    assert not estimator.update(1, [0, 0, 0], [0, 0, GRAVITY], [0, 0]).stationary


def test_combined_tilt_reconstructs_expected_projected_gravity():
    gravity = np.array([0.3, -0.4, -math.sqrt(0.75)])
    quat = _orientation(gravity, 0.7)
    assert np.linalg.norm(quat) == pytest.approx(1)
    assert _gravity(quat) == pytest.approx(gravity)
