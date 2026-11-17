"""Teacher/student request parity and rejection of fabricated training history."""

import copy

import pytest

from .head_features import payload
from .plan_protocol import parse_plan
from .plan_tasks import PlanTaskSession
from .plan_teacher import PlanTeacherBackend, PlanTeacherSession
from .plan_teacher_data import task_rows
from .test_plan_tasks import Runtime
from .test_skill_tasks import ScriptedBackend
from .test_tasks import MemoryRecorder, ball


def direct_run():
    runtime, recorder = Runtime(), MemoryRecorder()
    result = PlanTeacherSession(runtime, recorder, PlanTeacherBackend(), verbose=False).run("往前走一点")
    return runtime, recorder, result


def test_export_replays_identical_inputs_through_student_executor():
    import json

    teacher_runtime, recorder, result = direct_run()
    rows = task_rows(recorder.events, result)
    backend = ScriptedBackend(*(parse_plan(json.dumps(row["target"])) for _, row in rows))
    backend.supports_continuation = False
    student_runtime = Runtime()
    session = PlanTaskSession(student_runtime, MemoryRecorder(), backend, verbose=False)
    session.run("往前走一点")
    assert student_runtime.calls == teacher_runtime.calls == [("move_forward", 0.2), ("move_forward", 0.12)]
    assert len(rows) == len(backend.requests) == 2
    for (event, row), observed in zip(rows, backend.requests, strict=True):
        assert row["state"]["observation_id"] == event["observation"]["frame_id"]
        assert row["state"] == observed.state
        assert row["context"] == observed.context
        assert payload(row["instruction"], row["state"], row["context"]) == payload(
            observed.instruction, observed.state, observed.context
        )


def test_semantic_history_retains_early_reach_and_waits_for_kick_recovery():
    class ReachingRuntime(Runtime):
        def execute(self, skill, seconds):
            completed = super().execute(skill, seconds)
            if skill == "move_forward":
                self.detections = [ball(x=0.0895, y=0.0411)]
            return completed

    runtime, recorder = ReachingRuntime([ball(x=0.4)]), MemoryRecorder()
    result = PlanTeacherSession(runtime, recorder, PlanTeacherBackend(), verbose=False).run("踢红球")
    result["evaluation"] = {"success": True}  # A fixture for the independent acceptance gate.
    rows = task_rows(recorder.events, result)
    assert [r["target"]["steps"][0]["skill"] for _, r in rows[:-1]] == ["approach_target", "kick_left"]
    approach = rows[1][1]["context"]["recent_outcomes"][-1]
    assert approach["step"] == {"skill": "approach_target", "seconds": 2.0}
    assert approach["outcome"] == "target_reachable"
    assert runtime.calls == [("move_forward", 0.3), ("kick_left", 0.5), ("stand", 3.0)]
    assert rows[-1][1]["state"]["task_elapsed_s"] == pytest.approx(3.8)
    assert rows[-1][1]["context"]["recent_outcomes"][-1]["outcome"] == "completed"


@pytest.mark.parametrize("mutation", ["primitive_history", "fake_result", "wrong_frame", "future_history"])
def test_export_refuses_misaligned_requests(mutation):
    _, recorder, result = direct_run()
    events = copy.deepcopy(recorder.events)
    plans = [e for e in events if e["phase"] == "model_plan"]
    context = plans[-1]["teacher_trace"]["input_context"]
    if mutation == "primitive_history":
        context["recent_outcomes"][0]["step"]["seconds"] = 0.2
    elif mutation == "fake_result":
        context["recent_outcomes"][0]["outcome"] = "target_reachable"
    elif mutation == "wrong_frame":
        plans[-1]["teacher_trace"]["input_state"]["observation_id"] = "future_frame"
    else:
        plans[0]["teacher_trace"]["input_context"]["recent_outcomes"] = copy.deepcopy(context["recent_outcomes"])
    with pytest.raises(ValueError, match="request_does_not_match"):
        task_rows(events, result)


def test_model_source_cannot_become_teacher_labels():
    _, recorder, result = direct_run()
    recorder.events[0]["controller_source"] = "frozen_vlm_trained_skill_head"
    with pytest.raises(ValueError, match="not_a_rule"):
        task_rows(recorder.events, result)


def test_teacher_does_not_treat_a_rejected_kick_as_completed_task():
    from .plan_inputs import feedback, request
    from .test_plan_protocol import TARGET
    from .test_tasks import observed

    history = [feedback({"skill": "kick_left", "seconds": 0.5}, "kick_target_outside_measured_region", TARGET)]
    current = request("踢红球", observed([ball(x=0.4)]), {}, "task_1", 0, history)
    plan = PlanTeacherBackend().decide(current)
    assert plan.steps == [{"skill": "approach_target", "seconds": 2.0}]


def test_feature_extraction_refuses_legacy_inputs_before_loading_model(tmp_path, monkeypatch):
    import sys

    from .extract_head_features import main

    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["extract", "--dataset", str(tmp_path), "--output", str(tmp_path / "features")])
    with pytest.raises(ValueError, match="semantic boundary"):
        main()
    assert not (tmp_path / "features").exists()


def test_head_refuses_changed_executor_contract_before_loading_backbone(tmp_path):
    torch = pytest.importorskip("torch")
    from .head_backend import SkillHeadBackend
    from .plan_inputs import BOUNDARY_CONTRACT

    checkpoint = tmp_path / "head.pt"
    torch.save(dict(input_contract=BOUNDARY_CONTRACT, input_contract_sha256="changed"), checkpoint)
    with pytest.raises(ValueError, match="input/executor contract changed"):
        SkillHeadBackend(checkpoint, tmp_path)
