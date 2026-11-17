"""Bounded simulation rollouts that keep evaluator expectations outside model inputs."""

import copy
import json
import threading
import time
from collections import Counter


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def raw_plan_exact(trace, expected, *, selector_only=False):
    """Score raw model plans, including rejected output, and exclude host control."""
    sources = ("learned_selector",) if selector_only else ("learned_selector", "vlm_generation")
    if trace and trace.get("decision_source") in sources:
        return trace.get("raw_prediction") == expected
    return None


def _active_continuous(session):
    with session.state_lock:
        goal = session.execution["active"]
        return bool(goal and goal["type"] == "move" and goal["limit"]["kind"] == "until_cancelled")


def _wait_step(session, step, interventions, expired):
    start = session.runtime.snapshot()["time_s"]
    continuous_at = None
    while session.worker is not None and session.worker.is_alive():
        if expired:
            raise TimeoutError(expired[0])
        now = session.runtime.snapshot()["time_s"]
        if step["wait"] == "advance" and now - start >= step["seconds"]:
            return
        if step["wait"] != "advance" and _active_continuous(session):
            continuous_at = now if continuous_at is None else continuous_at
            if now - continuous_at >= 0.3:
                interventions.append(dict(reason="unexpected_unbounded_prediction", simulation_time_s=now))
                session.run("停下", wait=False)
                break
        time.sleep(0.002)
    session.wait(timeout=2)


def _watchdog(runtime, start_sim, wall_limit, sim_limit, done, expired):
    started = time.monotonic()
    while not done.wait(0.01):
        reason = "case_wall_time_limit" if time.monotonic() - started > wall_limit else None
        # Model backends own this lock during inference; never wait for it here.
        if runtime.lock.acquire(blocking=False):
            try:
                if float(runtime.data.time) - start_sim > sim_limit:
                    reason = "case_simulation_time_limit"
            finally:
                runtime.lock.release()
        if reason:
            expired.append(reason)
            runtime.cancel.set()
            return


def run_case(case, backend, output, *, seed, wall_limit=120, sim_limit=100):
    from .scene import sample_layout
    from .toy_motion import ToyRuntime
    from .toy_session import ToySession

    output.mkdir(parents=True, exist_ok=False)
    layout = sample_layout(seed)
    runtime = ToyRuntime(output, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
    session = None
    done, expired, interventions, turns = threading.Event(), [], [], []
    started = time.monotonic()
    watchdog = None
    error = None
    try:
        runtime.reset(layout)
        start_sim = runtime.snapshot()["time_s"]
        session = ToySession(runtime, seed=seed, split_group=f"selector_eval_{case['id']}", backend=backend)
        session.recorder.metadata.update(
            training_ready=False,
            evaluation_only=True,
            initial_layout_world_xy_m=layout,
            simulation_clock="lockstep",
            case_id=case["id"],
        )
        session.recorder.save_metadata()
        watchdog = threading.Thread(
            target=_watchdog,
            args=(runtime, start_sim, wall_limit, sim_limit, done, expired),
            name="selector-evaluation-bound",
            daemon=True,
        )
        watchdog.start()
        for step in case["steps"]:
            if expired:
                raise TimeoutError(expired[0])
            previous_events = len(session.recorder.events)
            response = session.run(step["text"], wait=False)
            actual = {key: copy.deepcopy(response[key]) for key in ("decision", "goals")}
            with session.recorder.lock:
                user = next(
                    row
                    for row in session.recorder.events[previous_events:]
                    if row["event"] == "dialogue" and row["request"]["event"]["kind"] == "user"
                )
                trace = copy.deepcopy(user.get("model_trace"))
                before = copy.deepcopy(user["request"]["execution"])
            with session.state_lock:
                after = copy.deepcopy(session.execution)
            turns.append(
                dict(
                    text=step["text"],
                    expected=copy.deepcopy(step["expected"]),
                    prediction=actual,
                    response=copy.deepcopy(response),
                    semantic_exact=actual == step["expected"],
                    decision_exact=actual["decision"] == step["expected"]["decision"],
                    model_trace=trace,
                    raw_model_exact=raw_plan_exact(trace, step["expected"]),
                    raw_selector_exact=raw_plan_exact(trace, step["expected"], selector_only=True),
                    execution_before=before,
                    execution_after=after,
                )
            )
            if expired:
                runtime.cancel.set()
                raise TimeoutError(expired[0])
            _wait_step(session, step, interventions, expired)
        if expired:
            raise TimeoutError(expired[0])
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        runtime.cancel.set()
    finally:
        done.set()
        if watchdog is not None:
            watchdog.join(1)
        if session is not None:
            session.close()
        runtime.save_trace()
        runtime.close()
    results = copy.deepcopy(session.results) if session is not None else []
    statuses = dict(Counter(row["status"] for row in results))
    semantic = len(turns) == len(case["steps"]) and all(row["semantic_exact"] for row in turns)
    finite_goals = [
        goal
        for step in case["steps"]
        for goal in step["expected"]["goals"]
        if not (goal["type"] == "move" and goal["limit"]["kind"] == "until_cancelled")
    ]
    finite_verified = None
    if finite_goals:
        finite_verified = (
            semantic
            and [row["goal"] for row in results] == finite_goals
            and all(row["status"] == "succeeded" for row in results)
        )
    control_checks = {}
    if case["id"] == "continuous_dialogue_control" and len(turns) == 6:
        visual, pause, chat, resume, stop = turns[1:]
        control_checks = dict(
            visual_preserved_running_queue=(
                visual["execution_before"]["state"] == "running"
                and visual["execution_before"] == visual["execution_after"]
            ),
            paused=pause["execution_after"]["state"] == "paused",
            chat_preserved_paused_queue=(
                chat["execution_before"]["state"] == "paused" and chat["execution_before"] == chat["execution_after"]
            ),
            resumed=resume["execution_after"]["state"] == "running",
            stopped=stop["execution_after"]["state"] == "idle" and stop["execution_after"]["active"] is None,
        )
    report = dict(
        case_id=case["id"],
        seed=seed,
        expected_source="hand_authored_evaluation_only_never_model_input",
        backend=backend.name,
        turns=turns,
        semantic_all_correct=semantic,
        expected_finite_plan_verified=finite_verified,
        control_checks=control_checks,
        goal_results=results,
        goal_status=statuses,
        evaluator_interventions=interventions,
        error=error,
        timed_out=bool(expired or (error and error["type"] == "TimeoutError")),
        runtime_failure=runtime.failure,
        elapsed_seconds=time.monotonic() - started,
        simulation_seconds=float(runtime.data.time),
        output=str(output.resolve()),
    )
    if session is not None:
        session.recorder.metadata.update(physics=runtime.physics, policy_sha256=runtime.policy.hashes)
        session.recorder.save_metadata()
    write_json(output / "evaluation.json", report)
    return report
