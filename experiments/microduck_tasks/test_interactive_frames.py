"""Presentation never borrows the physics lock or catches up a delayed inference."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from .interactive_frames import InteractiveFrames
from .test_toy_stop_entry import held_elsewhere


def test_cached_frame_is_complete_and_cannot_mutate_live_or_previous_data():
    frames = InteractiveFrames(clock=lambda: 12.0)
    source = dict(time_s=0.02, skill="roll", qpos=np.arange(7.0), qvel=np.zeros(6))
    frames.publish(source)
    source["qpos"][:] = -10
    first = frames.snapshot()
    np.testing.assert_array_equal(first["qpos"], np.arange(7.0))
    first["qpos"][:] = -20
    assert frames.snapshot()["qpos"][0] == 0
    assert first["display_frame_id"] == 1 and first["display_wall_time_s"] == 12
    frames.publish(dict(source, time_s=0.04))
    second = frames.snapshot()
    assert second["display_frame_id"] == 2 and second["time_s"] == 0.04
    assert first["time_s"] == 0.02


def test_pacing_accounts_for_tick_compute_and_resets_after_inference_gap():
    now, waits = [10.003], []
    frames = InteractiveFrames(clock=lambda: now[0])
    cancelled = SimpleNamespace(wait=waits.append)
    frames.wait_after_tick(10.0, cancelled)
    assert waits[-1] == pytest.approx(0.017)
    now[0] = 15.003
    frames.wait_after_tick(15.0, cancelled)
    assert waits[-1] == pytest.approx(0.017), "Idle/model time must not become a catch-up debt."
    now[0] = 15.05
    frames.wait_after_tick(15.0, cancelled)
    assert len(waits) == 2, "A slow tick adds no extra delay, but no historical deadline is carried forward."


@pytest.mark.parametrize("period", [0, -1, float("nan"), float("inf")])
def test_nonfinite_or_nonpositive_pacing_period_is_refused(period):
    with pytest.raises(ValueError, match="finite, positive"):
        InteractiveFrames(period)


@pytest.fixture
def runtime():
    from .bootstrap import configure

    try:
        configure()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    for module in ("mujoco", "onnxruntime"):
        pytest.importorskip(module)
    from .runtime import Runtime

    value = object.__new__(Runtime)
    value.lock = threading.RLock()
    value.cancel, value.shutdown, value.pulse_done = threading.Event(), threading.Event(), threading.Event()
    value.remaining, value.active_skill, value.worker, value.failure = 0, "stand", None, None
    value._interactive_frames = None
    value.data = SimpleNamespace(time=0.0)
    value.supports_skill = lambda skill: True
    value._tick = lambda skill: setattr(value.data, "time", value.data.time + 0.02)
    return value


def test_default_runtime_does_not_wait_or_create_a_background_worker(runtime):
    assert runtime.execute("roll", 0.06)
    assert runtime.data.time == pytest.approx(0.06)
    assert runtime.worker is None and runtime._interactive_frames is None
    with pytest.raises(RuntimeError, match="enable_interactive_pacing"):
        runtime.display_snapshot()


def test_pacing_wait_occurs_outside_world_lock_and_only_after_completed_tick(runtime):
    checks = []

    def wait_after_tick(start, cancel):
        acquired = threading.Event()

        def acquire():
            with runtime.lock:
                acquired.set()

        thread = threading.Thread(target=acquire)
        thread.start()
        assert acquired.wait(0.5), "Sleeping while owning the world lock would starve the live viewer."
        thread.join(1)
        checks.append(runtime.data.time)

    runtime._interactive_frames = SimpleNamespace(clock=lambda: 0.0, wait_after_tick=wait_after_tick)
    assert runtime.execute("roll", 0.06)
    assert checks == pytest.approx([0.02, 0.04, 0.06])
    assert runtime.worker is None


def test_display_snapshot_never_waits_for_busy_world_lock_or_advances_physics(runtime):
    frames = runtime._interactive_frames = InteractiveFrames()
    frames.publish(dict(time_s=1.0, qpos=np.zeros(7), qvel=np.zeros(6)))
    with held_elsewhere(runtime.lock):
        for _ in range(10):
            assert runtime.display_snapshot()["time_s"] == 1.0
    assert runtime.data.time == 0.0 and runtime.worker is None


def test_cancel_during_pacing_never_advances_another_motion_tick(runtime):
    runtime._interactive_frames = SimpleNamespace(clock=lambda: 0.0, wait_after_tick=lambda start, cancel: cancel.set())
    assert not runtime.execute("roll", 0.06)
    assert runtime.data.time == pytest.approx(0.02)
    assert runtime.last_execution["started"] and not runtime.last_execution["completed"]


def test_paced_lockstep_mode_refuses_a_background_physics_worker(runtime):
    runtime._interactive_frames = InteractiveFrames()
    with pytest.raises(RuntimeError, match="lockstep"):
        runtime.start_realtime()
    assert runtime.worker is None
