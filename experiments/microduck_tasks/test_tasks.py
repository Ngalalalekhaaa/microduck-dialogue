"""Closed-loop task, execution-log and independent kick acceptance regressions."""

import copy
import json
import math
import threading

import numpy as np
import pytest

from experiments.microduck_tasks.language import Intent
from experiments.microduck_tasks.planner import KickPlanner
from experiments.microduck_tasks.tasks import TaskSession, evaluate_kick


def ball(color="red", x=0.4, y=0.0, confidence=0.9):
    return {"color": color, "center_body_m": [x, y, -0.08], "confidence": confidence}


def observed(detections=(), yaw=0.0, time_s=0.0, stationary=True):
    return {
        "frame_id": "frame_00001",
        "time_s": time_s,
        "yaw": yaw,
        "stationary": stationary,
        "tilt_deg": 2.0,
        "detections": list(detections),
    }


@pytest.mark.parametrize("color", ["red", "yellow", "blue"])
def test_search_uses_visibility_of_requested_color_only(color):
    planner = KickPlanner(color)
    others = [ball(c, x=0.0895, y=0.0411) for c in ("red", "yellow", "blue") if c != color]
    decision = planner.decide(observed(others))
    assert (decision.phase, decision.skill) == ("search", "turn_left")


def test_search_counts_measured_rotation_across_wrapped_yaw():
    planner = KickPlanner("red")
    for index in range(18):
        unwrapped = 3.0 + index * 0.4
        yaw = math.atan2(math.sin(unwrapped), math.cos(unwrapped))
        decision = planner.decide(observed(yaw=yaw, time_s=index * 2.0))
        if decision is None:
            break
    assert decision is None
    assert planner.status == "not_found"
    assert index >= 16


def test_stalled_robot_does_not_claim_full_search_from_commanded_turns():
    planner = KickPlanner("red")
    for index in range(40):
        decision = planner.decide(observed(yaw=0.0, time_s=index))
        assert decision.phase == "search"
    assert planner.search_angle == 0.0


def test_yaw_oscillation_does_not_count_as_covering_a_full_circle():
    planner = KickPlanner("red")
    for index in range(40):
        decision = planner.decide(observed(yaw=0.2 if index % 2 else -0.2, time_s=index))
        assert decision is not None
        assert decision.phase == "search"
    assert planner.status == "running"


def test_search_stops_when_target_enters_view():
    planner = KickPlanner("blue")
    assert planner.decide(observed()).phase == "search"
    decision = planner.decide(observed([ball("blue", 0.4, 0.0)], yaw=0.5, time_s=2))
    assert decision.phase == "approach"
    assert decision.skill == "move_forward"
    assert planner.search_angle == 0


def test_search_has_time_and_decision_budgets():
    planner = KickPlanner("red")
    planner.decide(observed(time_s=2))
    assert planner.decide(observed(time_s=183)) is None
    assert planner.status == "timeout"
    planner = KickPlanner("red")
    for _ in range(121):
        decision = planner.decide(observed())
    assert decision is None
    assert planner.status == "timeout"


@pytest.mark.parametrize(("y", "foot"), [(0.0411, "left"), (-0.0429, "right")])
def test_kick_requires_reobserved_alignment_and_stationary_body(y, foot):
    planner = KickPlanner("red")
    obs = observed([ball("red", 0.089, y)])
    assert planner.decide(obs).skill == "stand"
    obs["stationary"] = False
    assert planner.decide(obs).skill == "stand"
    obs["stationary"] = True
    assert planner.decide(obs).skill == f"kick_{foot}"


def test_lost_target_cancels_previous_alignment():
    planner = KickPlanner("red")
    aligned = observed([ball("red", 0.0895, 0.0411)])
    assert planner.decide(aligned).phase == "settle"
    assert planner.decide(observed()).phase == "search"
    assert planner.decide(aligned).phase == "settle"


def test_other_ball_is_an_obstacle_and_never_substitutes_for_target():
    planner = KickPlanner("red")
    obs = observed([ball("red", 0.4, 0), ball("blue", 0.15, 0.01)])
    decision = planner.decide(obs)
    assert (decision.phase, decision.skill) == ("avoid", "move_right")


@pytest.mark.parametrize(("x", "y"), [(math.nan, 0), (0.2, math.inf), (-math.inf, 0)])
def test_nonfinite_target_never_produces_action(x, y):
    planner = KickPlanner("red")
    assert planner.decide(observed([ball(x=x, y=y)])) is None
    assert planner.status == "invalid_vision"


def trace_row(time_s=11, tilt=3, speed=0, height=0.12):
    qvel = np.zeros(32)
    qvel[0] = speed
    return (
        time_s,
        np.zeros(35),
        qvel,
        "stand",
        tilt,
        0,
        np.zeros((3, 3)),
        np.array([0, 0, height]),
        np.zeros(14),
        np.zeros(14),
    )


def contact(color="red", time_s=10.2, geom="left_foot_collision", other_color=None):
    return {"color": color, "time_s": time_s, "geom": geom, "other_color": other_color, "skill": "kick_left"}


def evaluate(contacts=None, rows=None, displacement=0.2):
    before = {c: np.array([0.1, 0, 0.035]) for c in ("red", "yellow", "blue")}
    after = {c: value.copy() for c, value in before.items()}
    after["red"][0] += displacement
    return evaluate_kick(
        "red",
        "left",
        before,
        after,
        10.0,
        [contact()] if contacts is None else contacts,
        [trace_row()] if rows is None else rows,
    )


def test_acceptance_requires_real_target_foot_contact_and_motion():
    result = evaluate()
    assert result["success"] is True
    assert result["target_foot_contact"] is True
    assert result["target_displacement_m"] == pytest.approx(0.2)
    assert evaluate(contacts=[])["success"] is False
    assert evaluate(displacement=0.01)["success"] is False


@pytest.mark.parametrize(
    "contacts",
    [
        [contact(geom="right_foot_collision")],
        [contact(geom="left_knee_collision")],
        [contact(time_s=11)],
        [contact(), contact(time_s=9.5)],
        [contact(), contact(color="blue")],
        [contact(), contact(color="yellow")],
        [contact(other_color="blue")],
    ],
)
def test_acceptance_rejects_wrong_foot_wrong_ball_early_push_and_missing_kick_contact(contacts):
    assert evaluate(contacts=contacts)["success"] is False


@pytest.mark.parametrize("rows", [[], [trace_row(tilt=51)], [trace_row(), trace_row(tilt=math.nan)]])
def test_acceptance_rejects_falls_and_invalid_motion_trace(rows):
    assert evaluate(rows=rows)["success"] is False


@pytest.mark.parametrize("rows", [[trace_row(speed=1.0)], [trace_row(height=0.04)]])
def test_acceptance_requires_recovered_stationary_standing_body(rows):
    assert evaluate(rows=rows)["success"] is False


def test_acceptance_rejects_rotation_and_transient_collapse_even_if_final_pose_is_upright():
    spinning = trace_row()
    spinning[2][5] = 0.5
    assert evaluate(rows=[spinning])["success"] is False
    assert evaluate(rows=[trace_row(height=0.05), trace_row()])["success"] is False


def test_invalid_acceptance_metrics_remain_json_serializable():
    for rows in ([], [trace_row(tilt=math.nan)]):
        result = evaluate(rows=rows)
        assert result["success"] is False
        json.dumps(result, allow_nan=False)


class MemoryRecorder:
    def __init__(self):
        self.events = []

    def record_observation(self, frame_id, time_s, images, depths, perception):
        return {
            "frame_id": frame_id,
            "time_s": time_s,
            "images": {name: f"{name}.png" for name in images},
            "depths": {},
            "detections": copy.deepcopy(perception),
        }

    def append_event(self, event):
        self.events.append(copy.deepcopy(event))


class FakeRuntime:
    def __init__(self, mode="complete", detections=()):
        self.time_s = 2.0
        self.cancel = threading.Event()
        self.calls, self.contacts, self.rows = [], [], []
        self.stop_calls = 0
        self.mode, self.detections = mode, list(detections)
        self.last_execution = {"started": False, "completed": False, "start_time_s": None, "end_time_s": None}

    def snapshot(self):
        return {"time_s": self.time_s}

    def execute(self, skill, seconds):
        self.calls.append((skill, seconds))
        self.last_execution = {"started": False, "completed": False, "start_time_s": None, "end_time_s": None}
        if self.mode in ("unstarted", "idle_advanced"):
            if self.mode == "idle_advanced":
                self.time_s += 0.02
                self.cancel.set()
            return False
        self.last_execution.update(started=True, start_time_s=self.time_s)
        self.time_s += seconds if self.mode == "complete" else 0.04
        self.last_execution["end_time_s"] = self.time_s
        if self.mode == "error":
            raise RuntimeError("test failure after two ticks")
        if self.mode == "partial":
            self.cancel.set()
            return False
        self.last_execution["completed"] = True
        return True

    def request_stop(self):
        self.stop_calls += 1
        self.cancel.set()

    def observe(self):
        obs = observed(self.detections, time_s=self.time_s)
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        return obs, {"front": image}, {"front": np.ones((12, 16))}

    def truth(self):
        raise AssertionError("A non-kick task must not read hidden ball positions.")


def prepared(mode="complete"):
    runtime, recorder = FakeRuntime(mode), MemoryRecorder()
    session = TaskSession(runtime, recorder, verbose=False)
    session.instruction = "往前走"
    session.intent = Intent("move", direction="forward")
    session.task_context = {}
    return session, runtime, recorder


@pytest.mark.parametrize(
    ("mode", "started", "completed"),
    [
        ("complete", True, True),
        ("partial", True, False),
        ("unstarted", False, False),
    ],
)
def test_execution_log_distinguishes_completed_partial_and_unstarted_actions(mode, started, completed):
    session, _, recorder = prepared(mode)
    assert session.perform(observed(), "direct", "move_forward", 0.32) is completed
    event = recorder.events[-1]
    assert event["execution"]["started"] is started
    assert event["execution"]["completed"] is completed
    if started:
        assert event["execution"]["end_time_s"] > event["execution"]["start_time_s"]
    else:
        assert event["execution"]["start_time_s"] is None
        assert event["execution"]["end_time_s"] is None


def test_exception_still_records_partial_execution():
    session, _, recorder = prepared("error")
    with pytest.raises(RuntimeError, match="two ticks"):
        session.perform(observed(), "direct", "move_forward", 0.32)
    assert recorder.events[-1]["execution"] == {
        "started": True,
        "completed": False,
        "start_time_s": 2.0,
        "end_time_s": 2.04,
    }


def test_cancel_before_execution_records_unstarted_and_calls_no_skill():
    session, runtime, recorder = prepared()
    runtime.cancel.set()
    assert session.perform(observed(), "direct", "move_forward", 0.32) is False
    assert runtime.calls == []
    assert recorder.events[-1]["execution"]["started"] is False


def test_background_standing_time_does_not_mark_a_cancelled_zero_tick_pulse_started():
    session, runtime, recorder = prepared("idle_advanced")
    before = runtime.snapshot()["time_s"]
    assert session.perform(observed(), "direct", "move_forward", 0.32) is False
    assert runtime.snapshot()["time_s"] > before
    assert recorder.events[-1]["execution"] == {
        "started": False,
        "completed": False,
        "start_time_s": None,
        "end_time_s": None,
    }


@pytest.mark.parametrize("text", ["你现在看到了什么", "你能踢红球吗", "踢它"])
def test_dialogue_and_ambiguity_never_read_truth_or_execute_motion(text):
    runtime, recorder = FakeRuntime(), MemoryRecorder()
    result = TaskSession(runtime, recorder, verbose=False).run(text)
    assert result["status"] in ("success", "clarification")
    assert runtime.calls == []
    assert recorder.events[-1]["done"] is True


def test_stop_calls_runtime_stop_and_never_reports_motion_execution():
    runtime, recorder = FakeRuntime(), MemoryRecorder()
    result = TaskSession(runtime, recorder, verbose=False).run("停下")
    assert result["status"] == "success"
    assert runtime.stop_calls == 1
    assert runtime.cancel.is_set()
    assert not any(event["execution"]["started"] for event in recorder.events)


def test_partial_task_reports_cancelled_and_does_not_attempt_settling_after_cancel():
    runtime, recorder = FakeRuntime("partial"), MemoryRecorder()
    result = TaskSession(runtime, recorder, verbose=False).run("往前走")
    assert result["status"] == "cancelled"
    assert runtime.calls == [("move_forward", 0.32)]
    assert recorder.events[-1]["execution"]["started"] is False


def test_motion_failure_stops_runtime_and_preserves_partial_event():
    runtime, recorder = FakeRuntime("error"), MemoryRecorder()
    result = TaskSession(runtime, recorder, verbose=False).run("往前走")
    assert result["status"] == "error"
    assert runtime.stop_calls == 1
    assert any(event["execution"]["started"] and not event["execution"]["completed"] for event in recorder.events)
