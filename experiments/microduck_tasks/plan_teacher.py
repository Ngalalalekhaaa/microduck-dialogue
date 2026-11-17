"""Sensor-only rule selector using exactly the deployed semantic skill executor."""

import copy

from .kick_region import choose_foot
from .language import parse_instruction
from .plan_inputs import BOUNDARY_CONTRACT
from .plan_protocol import validate_input
from .plan_tasks import PlanTaskSession
from .selection_plan import bind_target, to_plan


class PlanTeacherBackend:
    name = "rgbd_rule_teacher"
    supports_continuation = False

    def __init__(self):
        self.metadata = dict(
            backend=self.name,
            teacher="efficient_plan",
            robot_finetuned=False,
            input_contract=BOUNDARY_CONTRACT,
            supports_continuation=False,
        )
        self.last_trace = {}

    def decide(self, request):
        validate_input(request.state, request.context)
        if request.context["committed_steps"]:
            raise ValueError("Collect this teacher only at completed semantic skill boundaries.")
        intent = parse_instruction(request.instruction)
        history = request.context["recent_outcomes"]
        if intent.kind == "describe":
            selected = "answer"
        elif intent.kind in ("move", "turn"):
            selected = f"{intent.kind}_{intent.direction}"
            if any(o["step"]["skill"] == selected and o["outcome"] == "completed" for o in history):
                selected = "finish"
        elif intent.kind == "kick":
            target = bind_target(request.instruction, request.state)
            kicked = any(
                o["step"]["skill"].startswith("kick_")
                and o["outcome"] == "completed"
                and o["target"] is not None
                and o["target"]["id"] == f"ball:{intent.color}"
                for o in history
            )
            if kicked:
                selected = "finish"
            elif target is None:
                selected = "turn_left"
            else:
                entity = next(e for e in request.state["entities"] if e["id"] == target["id"])
                foot = choose_foot(entity["position_body_m"])
                selected = f"kick_{foot}" if foot and request.state["stationary"] else "approach_target"
        else:
            selected = "clarify"
        self.last_trace = copy.deepcopy(
            dict(
                input_contract=BOUNDARY_CONTRACT,
                input_state=request.state,
                input_context=request.context,
                selected_skill=selected,
            )
        )
        return to_plan(selected, request)


class PlanTeacherSession(PlanTaskSession):
    trace_field = "teacher_trace"

    @property
    def controller_source(self):
        return "rule"
