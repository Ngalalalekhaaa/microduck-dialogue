"""The same draft input envelope for reviewed examples and future online calls."""

from .toy_catalog import SELF_ACTIONS
from .toy_protocol import GOALS, VERSION, attributes, goal, keys, number, sequence, target, text, validate_capacity


def validate_input(value):
    keys(value, "schema_version identity event history observation execution capabilities", "dialogue input")
    if value["schema_version"] != VERSION:
        raise ValueError("Wrong dialogue contract version.")
    keys(value["identity"], "name", "identity")
    text(value["identity"]["name"], "name", 40)
    event = value["event"]
    keys(event, "kind text", "event")
    if event["kind"] not in ("user", "feedback"):
        raise ValueError("Use user or trusted executor feedback events.")
    text(event["text"], "event text", 1000)
    sequence(value["history"], 12, "history")
    for item in value["history"]:
        keys(item, "role text", "public dialogue turn")
        if item["role"] not in ("user", "assistant"):
            raise ValueError("History contains only public dialogue.")
        text(item["text"], "history text", 1000)
    obs = value["observation"]
    keys(obs, "age_s entities posture stationary environment", "observation")
    number(obs["age_s"], 0, 3600, "observation age")
    if obs["posture"] not in ("standing", "sitting", "transitioning", "fallen", "unknown"):
        raise ValueError("Unknown observed posture.")
    if type(obs["stationary"]) is not bool:
        raise ValueError("stationary must be a boolean.")
    env = obs["environment"]
    keys(env, "surface motion_allowed edge_guard", "environment")
    if env["surface"] not in ("simulation_floor", "desktop"):
        raise ValueError("Specify simulation_floor or desktop.")
    if type(env["motion_allowed"]) is not bool or env["edge_guard"] not in ("available", "unavailable", "triggered"):
        raise ValueError("Invalid local motion guard observation.")
    sequence(obs["entities"], 20, "entities")
    ids = []
    for entity in obs["entities"]:
        keys(entity, "id category attributes position_body_m confidence", "observed entity")
        text(entity["id"], "entity id", 100)
        text(entity["category"], "category", 40)
        attributes(entity["attributes"])
        sequence(entity["position_body_m"], 3, "position")
        if len(entity["position_body_m"]) != 3:
            raise ValueError("Body-frame positions need three coordinates in metres.")
        for coordinate in entity["position_body_m"]:
            number(coordinate, -10, 10, "position")
        number(entity["confidence"], 0, 1, "confidence")
        ids.append(entity["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Entity ids must be unique within the observation.")
    execution = value["execution"]
    keys(execution, "state revision active pending last_result", "execution")
    if execution["state"] not in ("idle", "running", "paused", "stopping", "failed"):
        raise ValueError("Invalid execution state.")
    if type(execution["revision"]) is not int or execution["revision"] < 0:
        raise ValueError("Use a nonnegative executor-owned revision.")
    if execution["active"] is not None:
        goal(execution["active"])
    sequence(execution["pending"], None, "pending goals")
    for item in execution["pending"]:
        goal(item)
    if execution["state"] in ("running", "paused") and execution["active"] is None:
        raise ValueError("Running/paused state needs an active goal.")
    if execution["state"] in ("idle", "failed") and (execution["active"] is not None or execution["pending"]):
        raise ValueError("Idle/failed states cannot carry runnable goals.")
    result = execution["last_result"]
    if result is not None:
        keys(result, "goal status reason verified", "execution result")
        goal(result["goal"])
        if (
            result["status"] not in ("succeeded", "failed", "cancelled", "blocked")
            or type(result["verified"]) is not bool
        ):
            raise ValueError("Use an executor result with explicit verification.")
        if result["status"] == "succeeded" and result["verified"] is not True:
            raise ValueError("Success requires independent completion verification.")
        text(result["reason"], "result reason", 200)
    if event["kind"] == "feedback" and result is None:
        raise ValueError("Feedback must include the actual executor result.")
    caps = value["capabilities"]
    keys(
        caps, "goal_types self_actions continuous_move sit_hold search_targets max_goals queue_control", "capabilities"
    )
    capacity = validate_capacity(caps["max_goals"])
    queued = int(execution["active"] is not None) + len(execution["pending"])
    if capacity is not None and queued > capacity:
        raise ValueError("Existing active and pending queue exceeds the executor's goal capacity.")
    sequence(caps["search_targets"], 20, "search_targets")
    for item in caps["search_targets"]:
        target(item)
        if item["entity_id"] is not None:
            raise ValueError("Search capabilities describe attributes, not observed entity ids.")
    for name, allowed in (("goal_types", GOALS), ("self_actions", SELF_ACTIONS)):
        sequence(caps[name], len(allowed), name)
        if any(not isinstance(s, str) or s not in allowed for s in caps[name]) or len(caps[name]) != len(
            set(caps[name])
        ):
            raise ValueError("Unknown or duplicate capability.")
    if any(type(caps[name]) is not bool for name in ("continuous_move", "sit_hold", "queue_control")):
        raise ValueError("Capability flags must be booleans from the executor.")
    return value


def model_payload(value):
    """Shared serialization, excluding opaque executor revisions from learning."""
    import copy

    validate_input(value)
    result = copy.deepcopy(value)
    del result["execution"]["revision"]
    return result
