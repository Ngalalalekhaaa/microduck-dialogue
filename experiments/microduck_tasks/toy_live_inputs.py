"""Construct pilot inputs from current observations and executor-owned state."""

import copy

from .plan_protocol import entities
from .toy_catalog import SELF_ACTIONS
from .toy_inputs import validate_input
from .toy_protocol import DEFAULT_MAX_GOALS, GOALS, VERSION


def capabilities(runtime):
    return dict(
        goal_types=list(GOALS),
        self_actions=list(SELF_ACTIONS),
        continuous_move=True,
        sit_hold=bool(getattr(runtime, "supports_sit_hold", False)),
        max_goals=DEFAULT_MAX_GOALS,
        queue_control=True,
        search_targets=[
            dict(category="ball", attributes=[dict(name="color", value=color)], entity_id=None)
            for color in ("red", "yellow", "blue")
        ],
    )


def request(runtime, observation, *, text, kind, history, execution, name="小鸭"):
    with runtime.lock:
        age = max(0, float(runtime.data.time) - observation["time_s"])
        height = float(runtime.data.qpos[runtime.qa + 2])
        tilt = observation["tilt_deg"]
        posture = (
            "fallen"
            if runtime.failure or tilt > 50
            else "sitting"
            if height < 0.08
            else "standing"
            if height > 0.10
            else "transitioning"
        )
        if hasattr(runtime, "posture"):
            posture = runtime.posture()
    value = dict(
        schema_version=VERSION,
        identity=dict(name=name),
        event=dict(kind=kind, text=text),
        history=copy.deepcopy(history[-12:]),
        observation=dict(
            age_s=age,
            entities=[
                {k: e[k] for k in ("id", "category", "attributes", "position_body_m", "confidence")}
                for e in entities(observation)
            ],
            posture=posture,
            stationary=observation["stationary"],
            environment=dict(
                surface="simulation_floor", motion_allowed=not bool(runtime.failure), edge_guard="unavailable"
            ),
        ),
        execution=copy.deepcopy(execution),
        capabilities=capabilities(runtime),
    )
    validate_input(value)
    return value
