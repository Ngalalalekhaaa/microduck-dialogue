"""CPU MuJoCo checks: safe pause cannot cut a full motion or a continuous gait segment."""

import json

import pytest


@pytest.fixture
def physical_session(tmp_path):
    from .bootstrap import configure

    try:
        configure()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    for module in ("mujoco", "onnxruntime", "bam"):
        pytest.importorskip(module)
    from .toy_motion import ToyRuntime
    from .toy_session import ToySession

    output = tmp_path / "episode"
    output.mkdir()
    runtime = ToyRuntime(output, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
    runtime.reset()
    session = ToySession(runtime)
    try:
        yield session
    finally:
        session.close()
        runtime.close()


@pytest.mark.parametrize(("command", "skill"), [("先前滚翻，再点头", "roll"), ("先向左转90度，再点头", "turn_left")])
def test_finite_action_finishes_and_stabilizes_before_retaining_tail(physical_session, monkeypatch, command, skill):
    session, ticks, requested = physical_session, [], []
    runtime = session.runtime
    tick = runtime._tick

    def pause_during_motion(active):
        tick(active)
        if active == skill:
            ticks.append(active)
            if len(ticks) == 5:
                requested.append(runtime.snapshot()["time_s"])
                session.request_pause()
        assert not runtime.cancel.is_set(), "Safe pause must never cancel a motion tick."

    monkeypatch.setattr(runtime, "_tick", pause_during_motion)
    session.run(command)
    assert requested and len(ticks) > 5
    assert session.execution["state"] == "paused"
    assert session.execution["active"] == {"type": "perform", "action": "nod", "repeat": 1}
    assert len(session.results) == 1 and session.results[0]["status"] == "succeeded"
    recovery = session.results[0]["metrics"]["pause_recovery"]
    assert recovery["success"] and recovery["metrics"]["stable_observations"] == 3
    paused = runtime.snapshot()
    assert paused["stationary"] and paused["tilt_deg"] < 15
    if skill == "roll":
        assert session.results[0]["metrics"]["checks"][0]["max_tilt_deg"] > 150
    else:
        assert abs(session.results[0]["metrics"]["angle_deg"] - 90) < 8
    session.run("继续")
    assert len(session.results) == 2 and all(row["status"] == "succeeded" for row in session.results)
    assert session.execution["state"] == "idle"
    summary = dict(
        command=command,
        pause_requested_time_s=requested[0],
        paused_time_s=paused["time_s"],
        completed_action_ticks=len(ticks),
        pause_recovery=recovery,
        result_statuses=[row["status"] for row in session.results],
    )
    (runtime.output / "physical_pause_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def test_continuous_walk_finishes_short_segment_then_stands_and_resumes(physical_session, monkeypatch):
    session, ticks = physical_session, []
    runtime = session.runtime
    tick = runtime._tick

    def pause_during_segment(skill):
        tick(skill)
        ticks.append(skill)
        if ticks.count("move_forward") == 3 and not session.pause_requested:
            session.request_pause()
        assert not runtime.cancel.is_set()

    monkeypatch.setattr(runtime, "_tick", pause_during_segment)
    session.run("一直前进")
    session.wait(60)
    assert session.execution["state"] == "paused" and session.pause_requested
    assert ticks[:10] == ["move_forward"] * 10, "Finish the full 0.2-second segment after the request at tick 3."
    assert all(skill == "stand" for skill in ticks[10:])
    assert session.results[0]["status"] == "blocked" and session.results[0]["reason"] == "safe_pause_boundary"
    assert session.results[0]["metrics"]["stable_observations"] == 3
    paused = runtime.snapshot()
    assert paused["stationary"] and paused["tilt_deg"] < 15
    first_length = len(ticks)
    ticks.clear()
    session.run("继续")
    session.wait(60)
    assert session.execution["state"] == "paused"
    assert ticks[:10] == ["move_forward"] * 10
    assert len(session.results) == 2 and all(row["status"] == "blocked" for row in session.results)
    summary = dict(
        first_attempt_ticks=first_length,
        second_attempt_ticks=len(ticks),
        paused_tilt_deg=paused["tilt_deg"],
        attempts=[row["metrics"] for row in session.results],
        result_statuses=[row["status"] for row in session.results],
    )
    (runtime.output / "physical_pause_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
