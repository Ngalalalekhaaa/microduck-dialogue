"""Observation freshness follows the simulated scene clock, including render delay."""

import itertools

import pytest

from . import plan_tasks
from .plan_tasks import PlanTaskSession
from .test_plan_tasks import Runtime
from .test_skill_tasks import ScriptedBackend
from .test_tasks import MemoryRecorder


def prepared(runtime):
    session = PlanTaskSession(runtime, MemoryRecorder(), ScriptedBackend(), verbose=False)
    session.last_trace = {}
    session._execute = lambda observation, skill, seconds, trace=None: runtime.execute(skill, seconds)
    return session


def test_slow_render_of_frozen_world_does_not_make_current_observation_stale(monkeypatch):
    clock = itertools.count(step=0.5)
    monkeypatch.setattr(plan_tasks.time, "monotonic", lambda: next(clock))
    runtime = Runtime()
    result = prepared(runtime)._step({"skill": "turn_left", "seconds": 0.2}, None)
    assert result == "completed"
    assert runtime.calls == [("turn_left", 0.2)]


@pytest.mark.parametrize("age", [0.301, -0.001, float("nan"), float("inf")])
def test_stale_future_or_invalid_camera_clock_cannot_move(age):
    class StaleRuntime(Runtime):
        def observe(self):
            observation, images, depths = super().observe()
            observation["time_s"] = self.time_s - age
            return observation, images, depths

    runtime = StaleRuntime()
    result = prepared(runtime)._step({"skill": "turn_left", "seconds": 0.2}, None)
    assert result == "camera_deadline_exceeded"
    assert runtime.calls == []


def test_world_advancing_while_camera_records_still_hits_same_deadline():
    class MovingRecorder(MemoryRecorder):
        def record_observation(self, *args):
            observation = super().record_observation(*args)
            runtime.time_s += 0.31
            return observation

    runtime = Runtime()
    session = prepared(runtime)
    session.recorder = MovingRecorder()
    assert session._step({"skill": "turn_left", "seconds": 0.2}, None) == "camera_deadline_exceeded"
    assert runtime.calls == []
