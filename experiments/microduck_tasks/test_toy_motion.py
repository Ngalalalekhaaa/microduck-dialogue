"""Physical semantic completion and cancellation tests for dialogue motion."""

import math

import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def configured():
    from .bootstrap import configure

    try:
        configure()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    for module in ("mujoco", "onnxruntime", "bam"):
        pytest.importorskip(module)


@pytest.fixture
def runtime(tmp_path):
    from .toy_motion import ToyRuntime

    robot = ToyRuntime(tmp_path, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
    robot.reset()
    try:
        yield robot
    finally:
        robot.close()


def test_full_turns_cross_yaw_wrap_and_measure_both_directions(runtime):
    from .toy_motion import ToyMotion

    motion = ToyMotion(runtime)
    for direction, sign in (("left", 1), ("right", -1)):
        start = len(runtime.rows) - 1
        result = motion.execute({"type": "turn", "direction": direction, "angle_deg": 360})
        assert result["success"], result
        measured = np.rad2deg(np.unwrap([row[5] for row in runtime.rows[start:]]))
        actual = float(measured[-1] - measured[0])
        assert abs(actual - sign * 360) < 8
        assert abs(actual - result["metrics"]["angle_deg"]) < 0.01
        assert runtime.snapshot()["tilt_deg"] < 15


@pytest.mark.parametrize("color", ["yellow", "blue"])
def test_navigation_stops_near_target_without_kicking(runtime, color):
    from .toy_motion import ToyMotion

    before = runtime.truth()
    result = ToyMotion(runtime).execute(
        {
            "type": "navigate",
            "target": {"category": "ball", "attributes": [{"name": "color", "value": color}], "entity_id": None},
            "standoff_m": 0.25,
        }
    )
    assert result["success"], result
    with runtime.lock:
        trunk = runtime.data.qpos[runtime.qa : runtime.qa + 2].copy()
    assert abs(np.linalg.norm(runtime.truth()[color][:2] - trunk) - 0.25) < 0.045
    assert all(not row[3].startswith("kick_") for row in runtime.rows)
    assert all(np.linalg.norm(runtime.truth()[c] - before[c]) < 0.005 for c in before)
    assert not runtime.contacts


def test_sitting_holds_when_stopped_then_stands_from_measured_pose(runtime):
    from .toy_motion import ToyMotion

    motion = ToyMotion(runtime)
    assert motion.execute({"type": "posture", "pose": "sitting"})["success"]
    assert runtime.posture() == "sitting"
    runtime.request_stop()
    # The real-time loop performs these same idle ticks even with cancellation latched.
    for _ in range(100):
        runtime._tick("stand")
    assert runtime.cancel.is_set()
    assert runtime.posture() == "sitting"
    assert runtime.metrics[-1]["trunk_z"] < 0.08
    runtime.cancel.clear()
    assert motion.execute({"type": "posture", "pose": "standing"})["success"]
    assert runtime.posture() == "standing"
    assert runtime.policy.sit_level == 0
    assert runtime.metrics[-1]["trunk_z"] > 0.09


def test_cancel_stops_before_observation_or_motion(runtime):
    from .toy_motion import ToyMotion

    runtime.request_stop()
    start = runtime.snapshot()["time_s"]
    motion = ToyMotion(runtime, lambda: pytest.fail("Cancelled task must not acquire a frame."))
    result = motion.execute({"type": "turn", "direction": "left", "angle_deg": 360})
    assert not result["success"] and result["reason"] == "cancelled"
    assert runtime.snapshot()["time_s"] == start


def test_stale_and_nonfinite_observations_cannot_move(runtime):
    from .toy_motion import ToyMotion

    observation = runtime.observe()[0]
    goal = {"type": "turn", "direction": "left", "angle_deg": 90}
    start = runtime.snapshot()["time_s"]
    observation["time_s"] -= 1
    with pytest.raises(ValueError, match="Stale"):
        ToyMotion(runtime, lambda: observation).execute(goal)
    observation["time_s"], observation["yaw"] = start, math.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        ToyMotion(runtime, lambda: observation).execute(goal)
    assert runtime.snapshot()["time_s"] == start
