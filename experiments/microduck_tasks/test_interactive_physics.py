"""Actual CPU physics ticks remain visible at 1x, while model inference freezes physics."""

import json
import threading
import time

import numpy as np
import pytest


@pytest.fixture
def runtime(tmp_path):
    from .bootstrap import configure

    try:
        configure()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    for module in ("mujoco", "onnxruntime", "bam"):
        pytest.importorskip(module)
    from .toy_motion import ToyRuntime

    value = ToyRuntime(tmp_path, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
    value.reset()
    value.enable_interactive_pacing()
    try:
        yield value
    finally:
        value.request_stop()
        value.close()


def test_live_roll_publishes_intermediate_motion_at_real_time(runtime):
    frames, errors, done = [], [], threading.Event()
    started, initial = time.monotonic(), runtime.display_snapshot()

    def perform():
        try:
            assert runtime.perform("roll")
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=perform, daemon=True)
    thread.start()
    while not done.wait(1 / 30):
        frames.append(runtime.display_snapshot())
    thread.join(2)
    assert not errors, errors
    final, wall_seconds = runtime.display_snapshot(), time.monotonic() - started
    simulated_seconds = final["time_s"] - initial["time_s"]
    distinct = {frame["display_frame_id"]: frame for frame in frames}
    rolling = [frame for frame in distinct.values() if frame["skill"] == "roll"]
    assert simulated_seconds == pytest.approx(5.0)
    assert wall_seconds >= simulated_seconds * 0.98
    assert len(rolling) >= 30, "The viewer must receive the motion, not just before/after states."
    assert max(frame["tilt_deg"] for frame in rolling) > 150
    assert final["tilt_deg"] < 15 and runtime.worker is None
    summary = dict(
        wall_seconds=wall_seconds,
        simulated_seconds=simulated_seconds,
        display_ticks=final["display_frame_id"] - initial["display_frame_id"],
        distinct_polled_frames=len(distinct),
        distinct_roll_frames=len(rolling),
        observed_roll_max_tilt_deg=max(frame["tilt_deg"] for frame in rolling),
        recovered_tilt_deg=final["tilt_deg"],
        poll_hz=30,
    )
    (runtime.output / "interactive_physics_summary.json").write_text(json.dumps(summary, indent=2))


def test_display_reads_continue_while_inference_holds_world_lock_and_physics_stays_frozen(runtime):
    done, results, errors = threading.Event(), [], []
    initial = runtime.display_snapshot()

    def move():
        try:
            results.append(runtime.execute("move_forward", 0.6))
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=move, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while runtime.display_snapshot()["display_frame_id"] < initial["display_frame_id"] + 2:
            assert time.monotonic() < deadline and not errors
            done.wait(0.005)
        # This is the lock held by a lockstep model call, while the action thread is alive.
        with runtime.lock:
            before = runtime.display_snapshot()
            observed = []
            for _ in range(10):
                observed.append(runtime.display_snapshot())
                time.sleep(1 / 30)
            assert not done.is_set(), "The action must remain blocked until inference releases the world."
            assert {frame["display_frame_id"] for frame in observed} == {before["display_frame_id"]}
            assert {frame["time_s"] for frame in observed} == {before["time_s"]}
            for frame in observed:
                np.testing.assert_array_equal(frame["qpos"], before["qpos"])
        continued = time.monotonic()
        assert done.wait(3) and results == [True] and not errors
        final = runtime.display_snapshot()
        remaining_sim = final["time_s"] - before["time_s"]
        assert time.monotonic() - continued >= remaining_sim * 0.95
        assert final["time_s"] - initial["time_s"] == pytest.approx(0.6)
        assert runtime.worker is None
    finally:
        runtime.request_stop()
        thread.join(2)
