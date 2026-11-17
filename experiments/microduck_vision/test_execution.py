"""Exercise the real executor loop with a fake tick; no physics or model loading."""

import importlib
import math

import pytest
from bootstrap import MICRODUCK
from planner import SkillCommand


@pytest.fixture
def executor(monkeypatch):
    monkeypatch.syspath_prepend(str(MICRODUCK / "scripts"))
    monkeypatch.syspath_prepend(str(MICRODUCK / "deploy_hd1910"))
    runtime_type = importlib.import_module("runtime").Runtime
    instance = object.__new__(runtime_type)
    calls = []
    instance.tick = lambda twist, label: calls.append((twist, label))
    return instance, calls


@pytest.mark.parametrize(
    ("skill", "twist"),
    [
        ("move_forward", (0.2, 0.0, 0.0)),
        ("move_backward", (-0.2, 0.0, 0.0)),
        ("move_left", (0.0, 0.2, 0.0)),
        ("move_right", (0.0, -0.2, 0.0)),
    ],
)
@pytest.mark.parametrize("duration", [0.12, 0.32, 0.8])
def test_pulse_keeps_control_running_then_stands(executor, skill, twist, duration):
    runtime, calls = executor
    runtime.execute(SkillCommand(skill, duration))
    pulse_ticks = round(duration * 50)
    assert calls[:pulse_ticks] == [(twist, skill)] * pulse_ticks
    assert calls[pulse_ticks:] == [((0.0, 0.0, 0.0), "stand")] * 50


def test_stand_continues_fifty_control_ticks(executor):
    runtime, calls = executor
    runtime.execute(SkillCommand("stand"))
    assert calls == [((0.0, 0.0, 0.0), "stand")] * 50


@pytest.mark.parametrize(
    "command",
    [
        SkillCommand("unknown", 0.2),
        SkillCommand("move_forward", math.nan),
        SkillCommand("move_forward", math.inf),
        SkillCommand("move_forward", 0.13),
        SkillCommand("move_forward", 0.82),
        SkillCommand("stand", 0.2),
    ],
)
def test_invalid_command_is_refused_before_any_tick(executor, command):
    runtime, calls = executor
    with pytest.raises(ValueError):
        runtime.execute(command)
    assert calls == []


def test_tick_failure_stops_the_execution_loop(executor):
    runtime, calls = executor

    def failing_tick(twist, label):
        calls.append((twist, label))
        if len(calls) == 3:
            raise RuntimeError("fallen")

    runtime.tick = failing_tick
    with pytest.raises(RuntimeError, match="fallen"):
        runtime.execute(SkillCommand("move_forward", 0.32))
    assert len(calls) == 3
