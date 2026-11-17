"""Counterexamples for kick attribution, setup quality, and recovery acceptance."""

import numpy as np
import pytest
from evaluation import assess


def episode(*, launch=True):
    time = np.arange(276) * 0.02
    count = len(time)
    ball = np.tile([0.09, 0.042, 0.035], (count, 1))
    if launch:
        ball[:, 0] += np.interp(time, [0, 2.1, 2.5, 5.5], [0, 0, 0.16, 0.36])
    return {
        "time": time,
        "ball_xyz": ball,
        "trunk_velocity": np.zeros((count, 3)),
        "tilt_deg": np.zeros(count),
        "trunk_xyz": np.tile([0, 0, 0.12], (count, 1)),
        "gyro": np.zeros((count, 3)),
    }


def contact(time=2.1, geom="left_foot_collision", impulse=0.02):
    return {"time_s": time, "geom": geom, "normal_impulse_ns": impulse}


def evaluate(arrays, contacts, *, foot="left", lag_seconds=0.02, failure=None, control=False):
    return assess(
        arrays,
        contacts,
        foot=foot,
        kick_seconds=0.5,
        warmup_seconds=2.0,
        recovery_seconds=3.0,
        lag_seconds=lag_seconds,
        failure=failure,
        control=control,
    )


@pytest.mark.parametrize("foot", ["left", "right"])
def test_designated_foot_launch_and_stable_recovery_succeed(foot):
    result = evaluate(episode(), [contact(geom=f"{foot}_foot_collision")], foot=foot)
    assert result["status"] == "kick_success"
    assert result["success"]
    assert result["ball_forward_m"] == pytest.approx(0.36)
    assert result["designated_foot_hit"]
    assert result["recovered_stable"]


def test_standing_without_touching_or_moving_ball_passes_control():
    result = evaluate(episode(launch=False), [], control=True)
    assert result["status"] == "control_stable"
    assert not result["success"]


@pytest.mark.parametrize("axis", [0, 1], ids=["forward", "sideways"])
def test_control_rejects_ball_excursion_even_if_it_returns_to_start(axis):
    arrays = episode(launch=False)
    excursion = (arrays["time"] >= 3) & (arrays["time"] < 4)
    arrays["ball_xyz"][excursion, axis] += 0.5
    result = evaluate(arrays, [], control=True)
    assert result["status"] == "control_failed"
    assert result["maximum_ball_xy_drift_m"] == pytest.approx(0.5)


def test_control_rejects_robot_contact_even_without_visible_ball_motion():
    result = evaluate(episode(launch=False), [contact(3.0)], control=True)
    assert result["status"] == "control_failed"


def test_tiny_foot_touch_then_body_push_does_not_count_as_kick():
    arrays = episode(launch=False)
    arrays["ball_xyz"][:, 0] += np.interp(arrays["time"], [0, 3, 4, 5.5], [0, 0, 0.5, 0.5])
    contacts = [contact(impulse=1e-6), contact(3.0, "trunk_collision", impulse=1.0)]
    result = evaluate(arrays, contacts)
    assert not result["success"]
    assert result["status"] == "other_or_late_robot_contact"
    assert result["ball_forward_at_strike_end_m"] == 0
    assert result["non_designated_foot_contacts"] == 1


def test_ball_moving_only_after_kick_window_is_not_a_launch():
    arrays = episode(launch=False)
    arrays["ball_xyz"][arrays["time"] >= 3, 0] += 0.5
    result = evaluate(arrays, [contact(impulse=1e-6)])
    assert not result["success"]
    assert result["status"] == "no_forward_launch_during_kick"


def test_contact_after_policy_switch_is_attributed_using_actual_motor_delay():
    arrays = episode(launch=False)
    arrays["ball_xyz"][:, 0] += np.interp(arrays["time"], [0, 2.56, 2.60, 5.5], [0, 0, 0.08, 0.4])
    contacts = [contact(2.56)]
    delayed = evaluate(arrays, contacts, lag_seconds=0.1)
    nominal = evaluate(arrays, contacts, lag_seconds=0.02)
    assert delayed["success"]
    assert delayed["strike_window_end_s"] == pytest.approx(2.6)
    assert not nominal["success"]
    assert nominal["status"] == "missed_ball"


def test_zero_latency_has_no_assumed_twenty_millisecond_grace_period():
    arrays = episode(launch=False)
    arrays["ball_xyz"][:, 0] += np.interp(arrays["time"], [0, 2.5, 2.52, 5.5], [0, 0, 0.06, 0.4])
    contacts = [contact(2.505)]
    assert evaluate(arrays, contacts, lag_seconds=0.02)["success"]
    assert not evaluate(arrays, contacts, lag_seconds=0)["success"]


@pytest.mark.parametrize("contamination", ["contact", "temporary_drift"])
def test_preparation_contamination_disqualifies_an_otherwise_successful_kick(contamination):
    arrays = episode()
    contacts = [contact()]
    if contamination == "contact":
        contacts.insert(0, contact(1.0))
    else:
        arrays["ball_xyz"][(arrays["time"] >= 0.8) & (arrays["time"] < 1.2), 1] += 0.02
    result = evaluate(arrays, contacts)
    assert result["status"] == "setup_contaminated"
    assert not result["setup_clean"]
    assert not result["success"]


@pytest.mark.parametrize("failure", ["fallen", "nonfinite_state", "invalid_policy_output"])
def test_recorded_failure_cannot_be_overridden_by_apparent_ball_success(failure):
    result = evaluate(episode(), [contact()], failure=failure)
    assert result["status"] == failure
    assert not result["success"]


def test_incomplete_recovery_does_not_pass_using_a_short_stable_tail():
    arrays = {name: value[:-30] for name, value in episode().items()}
    result = evaluate(arrays, [contact()])
    assert result["status"] == "unstable_recovery"
    assert not result["recovered_stable"]
    assert not result["success"]


def test_late_body_motion_disqualifies_recovery_even_if_final_pose_is_still():
    arrays = episode()
    arrays["trunk_velocity"][-5, 1] = 0.2
    result = evaluate(arrays, [contact()])
    assert result["final_xy_speed_m_s"] == 0
    assert result["status"] == "unstable_recovery"
    assert not result["success"]


def test_ball_movement_without_designated_foot_contact_is_not_a_kick():
    result = evaluate(episode(), [contact(geom="right_foot_collision")])
    assert result["status"] == "missed_ball"
    assert not result["success"]


def test_other_body_contact_disqualifies_even_a_valid_foot_launch():
    result = evaluate(episode(), [contact(), contact(2.2, "right_foot_collision")])
    assert result["status"] == "other_or_late_robot_contact"
    assert not result["success"]
