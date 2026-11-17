"""Export exact recorded requests, with causal semantic feedback checks."""

import copy

from .dataset import _json
from .language import parse_instruction
from .plan_inputs import BOUNDARY_CONTRACT, request
from .plan_protocol import SYSTEM, parse_plan
from .training_data import _completed


def task_rows(events, result):
    if not events or not events[-1]["done"] or sum(e["done"] for e in events) != 1:
        raise ValueError("requires_one_terminal_event")
    if not result or result["status"] not in ("success", "model_reported", "answered", "clarification"):
        raise ValueError("task_not_successful")
    if result.get("evaluation") is not None and not result["evaluation"].get("success"):
        raise ValueError("task_not_independently_verified")
    if parse_instruction(events[0]["instruction"]).kind == "kick" and not (result.get("evaluation") or {}).get(
        "success"
    ):
        raise ValueError("kick_not_independently_verified")
    if any(e.get("model_trace") or e.get("controller_source") not in ("rule", "executor") for e in events):
        raise ValueError("not_a_rule_demonstration")
    if any(not _completed(e) for e in events if e["decision"] is not None):
        raise ValueError("incomplete_action_or_recovery")
    history, rows, pending = [], [], None
    started = None
    for event in events:
        trace = event.get("teacher_trace") or {}
        if event["phase"] == "model_plan":
            if pending is not None:
                raise ValueError("missing_semantic_step_result")
            if trace.get("input_contract") != BOUNDARY_CONTRACT:
                raise ValueError("requires_semantic_boundary_contract")
            state, context = trace["input_state"], trace["input_context"]
            obs = event["observation"]
            if started is None:
                started = obs["time_s"] - state["task_elapsed_s"]
            expected = request(event["instruction"], obs, {}, context["task_id"], started, history)
            if state != expected.state or context != expected.context:
                raise ValueError("request_does_not_match_observation_and_completed_history")
            plan = parse_plan(_json(trace["plan"]))
            if len(plan.steps) > 1:
                raise ValueError("requires_one_semantic_step_per_decision")
            row = dict(
                format_version="microduck-plan-sft-2",
                input_contract=BOUNDARY_CONTRACT,
                instruction=event["instruction"],
                state=copy.deepcopy(state),
                context=copy.deepcopy(context),
                target=plan.to_dict(),
                system_prompt=SYSTEM,
            )
            if plan.steps:
                pending = (event, row)
            else:
                rows.append((event, row))
        elif event["phase"] == "plan_step_result":
            if pending is None:
                raise ValueError("result_without_submitted_step")
            source, row = pending
            plan = row["target"]
            if trace.get("step") != plan["steps"][0] or trace.get("target") != plan["target"]:
                raise ValueError("feedback_does_not_match_submitted_step")
            history.append(copy.deepcopy(trace))
            # Failed proposals remain in raw logs/history, never become positive action labels.
            if trace["outcome"] in ("completed", "target_reachable"):
                rows.append((source, row))
            pending = None
    if pending is not None or not rows or rows[-1][1]["target"]["status"] == "continue":
        raise ValueError("missing_terminal_plan")
    if any(r["target"]["steps"] and r["target"]["steps"][0]["skill"].startswith("kick_") for _, r in rows):
        if not (result.get("evaluation") or {}).get("success"):
            raise ValueError("kick_not_independently_verified")
    return rows
