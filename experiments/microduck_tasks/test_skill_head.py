"""Input isolation, raw prediction fidelity, and head-specific execution semantics."""

import copy
import math

import numpy as np
import pytest

from .head_features import messages, payload
from .plan_protocol import CATALOG, VERSION
from .skill_vlm import SkillInput


def request():
    state = dict(
        yaw=0.0,
        tilt_deg=1.0,
        stationary=True,
        entities=[],
        observation_id="frame_1",
        task_elapsed_s=0.0,
        available_skills=CATALOG,
        schema_version=VERSION,
    )
    context = dict(task_id="task_1", committed_steps=[], recent_outcomes=[])
    return SkillInput("踢红球", {}, state, context)


def test_identifiers_and_elapsed_counters_do_not_change_model_input():
    one, two = request(), copy.deepcopy(request())
    two.state.update(observation_id="frame_999", task_elapsed_s=999.0)
    two.context["task_id"] = "task_999"
    assert payload(one.instruction, one.state, one.context) == payload(two.instruction, two.state, two.context)
    assert all(m["role"] != "assistant" for m in messages(one.instruction, one.state, one.context))


def test_teacher_and_continuation_information_is_not_silently_accepted():
    one = request()
    one.state["teacher_next_skill"] = "kick_left"
    with pytest.raises(ValueError):
        payload(one.instruction, one.state, one.context)
    one = request()
    one.context["committed_steps"] = [dict(skill="move_forward", seconds=0.2)]
    with pytest.raises(ValueError, match="decision boundaries"):
        payload(one.instruction, one.state, one.context)


def test_missing_target_refuses_predicted_kick_without_rule_replacement():
    pytest.importorskip("torch")
    from .head_backend import to_plan

    with pytest.raises(ValueError, match="cannot be bound"):
        to_plan("kick_left", request())
    plan = to_plan("move_backward", request())
    assert plan.steps == [dict(skill="move_backward", seconds=0.32)]
    assert plan.target is None


def test_head_waits_for_completed_action_before_next_observation():
    from .plan_tasks import PlanTaskSession
    from .test_plan_tasks import Runtime, action_plan, finish
    from .test_skill_tasks import ScriptedBackend
    from .test_tasks import MemoryRecorder

    runtime = Runtime()

    class Backend(ScriptedBackend):
        supports_continuation = False

        def decide(self, request):
            assert not request.context["committed_steps"]
            if self.requests:
                assert len(runtime.calls) == 4
                assert request.context["recent_outcomes"]
            return super().decide(request)

    backend = Backend(action_plan(), finish())
    assert (
        PlanTaskSession(runtime, MemoryRecorder(), backend, verbose=False).run("继续接近目标")["status"]
        == "model_reported"
    )


def test_macro_f1_does_not_hide_unsupported_classes():
    pytest.importorskip("torch")
    from .train_skill_head import metrics

    result = metrics([0, 0, 1], [0, 0, 0], ["approach", "kick", "finish"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(0.8 / 3)
    assert result["per_class"]["kick"]["recall"] == 0


def test_direct_acceptance_measures_body_frame_motion():
    from .evaluate_skill_head import direct_acceptance

    first = [None] * 8
    first[5], first[7] = math.pi / 2, np.array([0.0, 0.0, 0.12])
    last = list(first)
    last[7] = np.array([0.0, 0.01, 0.12])
    assert direct_acceptance([first, last], "move_forward")["success"]
    assert not direct_acceptance([first, last], "move_backward")["success"]


def test_direct_acceptance_wraps_yaw_across_pi():
    from .evaluate_skill_head import direct_acceptance

    first = [None] * 8
    first[5], first[7] = math.pi - 0.03, np.array([0.0, 0.0, 0.12])
    last = list(first)
    last[5] = -math.pi + 0.07
    assert direct_acceptance([first, last], "turn_left")["success"]
    assert not direct_acceptance([first, last], "turn_right")["success"]
