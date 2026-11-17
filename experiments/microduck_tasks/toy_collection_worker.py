"""Collect and independently audit one curriculum episode in an isolated process."""

import copy
import gc
import json
import re
import shutil
import time
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


class ExpectedTeacher:
    """Reject a wrong semantic label before it reaches the recorder or executor."""

    def __init__(self, teacher, spec):
        self.teacher, self.spec, self.index = teacher, spec, 0
        self.metadata = getattr(teacher, "metadata", {})

    def decide(self, request):
        response = self.teacher.decide(request)
        if request["event"]["kind"] == "user":
            self.verify_turn(request["event"]["text"], response)
        return response

    def verify_turn(self, text, response):
        """Also audit explicit host controls that intentionally bypass the teacher."""
        index = self.index
        if index >= len(self.spec["commands"]) or text != self.spec["commands"][index]:
            raise ValueError("User turn differs from the independently reviewed curriculum.")
        actual = {key: response[key] for key in ("decision", "goals")}
        if actual != self.spec["expected"][index]:
            raise ValueError(f"Semantic label mismatch at turn {index}: {actual!r} != {self.spec['expected'][index]!r}")
        self.index += 1


def _execution(session):
    with session.state_lock:
        return copy.deepcopy(session.execution)


def _wait_sim(session, seconds, timeout=30):
    start = session.runtime.snapshot()["time_s"]
    deadline = time.monotonic() + timeout
    while session.runtime.snapshot()["time_s"] - start < seconds:
        if session.worker is None or not session.worker.is_alive():
            session.wait(timeout=1)
            raise RuntimeError("Active control goal finished before its scheduled interruption.")
        if time.monotonic() > deadline:
            raise TimeoutError("Simulation did not advance to the scheduled control utterance.")
        time.sleep(0.001)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_goals(events, accepted):
    """Every real start must refer to an accepted goal in the original order."""
    previous, started, completed, failed, terminal = {}, {}, set(), set(), {}
    resumable = set()
    for event in events:
        kind, task = event["event"], event.get("task_id")
        if kind == "dialogue" and event["response"]["decision"] == "resume":
            resumable.add(task)
        if kind == "goal_start":
            index, goal = event["goal_index"], event["goal"]
            _require(task in accepted and 0 <= index < len(accepted[task]), "Goal start has no accepted plan position.")
            _require(goal == accepted[task][index], "Goal start differs from the accepted semantic goal.")
            _require(task not in failed, "A failed goal incorrectly advanced its pending queue.")
            last = previous.get(task, -1)
            if index == last:
                _require(task in resumable, "A goal repeated without an explicit user resume.")
                resumable.remove(task)
            else:
                _require(index == last + 1, "Goal execution skipped or reordered accepted tasks.")
            previous[task] = index
            key = (task, index, event["revision"])
            _require(key not in started, "Duplicate goal start in the same executor revision.")
            started[key] = goal
        elif kind == "goal_result":
            key = (task, event["goal_index"], event["revision"])
            _require(started.get(key) == event["goal"], "Goal result has no matching earlier start.")
            _require(key not in completed, "Duplicate result for one goal attempt.")
            _require(event["reason"] != "execution_exception", "Execution exceptions must be quarantined.")
            completed.add(key)
            terminal[task] = event["status"]
            if event["status"] == "failed":
                failed.add(task)
    _require(set(started) == completed, "An accepted start is missing its actual terminal result.")
    for task, goals in accepted.items():
        _require(task in previous, "An accepted task never reached the executor.")
        if terminal[task] == "succeeded":
            _require(previous[task] == len(goals) - 1, "An accepted queue silently dropped its unfinished tail.")
    return {"accepted_plan_tasks": len(accepted), "goal_start_order_verified": True, "goal_attempts": len(started)}


def _run_script(session, spec):
    accepted, checks = {}, {}

    def utter(index, *, wait=True):
        response = session.run(spec["commands"][index], wait=wait)
        if session.backend.index == index:
            session.backend.verify_turn(spec["commands"][index], response)
        if response["decision"] == "replace":
            accepted[session.task_id] = copy.deepcopy(response["goals"])
        elif response["decision"] == "append":
            accepted[session.task_id].extend(copy.deepcopy(response["goals"]))
        return response

    kind = spec["kind"]
    if kind == "continuous_control":
        _require(len(spec["commands"]) == 6, "Continuous control requires its reviewed six-turn script.")
        utter(0, wait=False)
        _wait_sim(session, 0.6)
        before = _execution(session)
        utter(1, wait=False)
        after = _execution(session)
        _require(after == before and after["state"] == "running", "Visual chat interrupted continuous movement.")
        utter(2, wait=False)
        session.wait()
        paused = _execution(session)
        _require(paused["state"] == "paused" and paused["active"] == before["active"], "Pause lost the active goal.")
        utter(3, wait=False)
        _require(_execution(session) == paused, "Greeting unexpectedly resumed a paused goal.")
        utter(4, wait=False)
        _wait_sim(session, 0.4)
        utter(5, wait=False)
        checks.update(visual_chat_preserved_queue=True, pause_chat_preserved_queue=True, resume_executed=True)
    elif kind == "append_cancel":
        _require(len(spec["commands"]) == 4, "Append/cancel requires its reviewed four-turn script.")
        utter(0, wait=False)
        _wait_sim(session, 0.2)
        before = _execution(session)
        utter(1, wait=False)
        appended = _execution(session)
        _require(appended["state"] == "running" and appended["active"] == before["active"], "Move ended before append.")
        _require(appended["pending"] == spec["expected"][1]["goals"], "Append changed the wrong queue position.")
        utter(2, wait=False)
        _require(_execution(session) == appended, "Visual chat changed the active or appended goal.")
        utter(3, wait=False)
        _require(
            all(row["goal_index"] == 0 for row in session.results), "An appended goal executed before cancellation."
        )
        checks.update(append_order_verified=True, visual_chat_preserved_queue=True, appended_goal_never_executed=True)
    else:
        for index in range(len(spec["commands"])):
            utter(index)
    session.wait(timeout=1)
    state = _execution(session)
    _require(
        state["state"] in ("idle", "failed") and state["active"] is None and not state["pending"], "Queue did not end."
    )
    if kind == "missing_target":
        _require(
            len(session.results) == 1 and session.results[0]["status"] == "failed",
            "Missing target was not a single failed goal.",
        )
        _require(session.results[0]["goal_index"] == 0, "Missing target advanced to a later goal.")
        checks["missing_target_cancelled_pending_goals"] = True
    checks.update(audit_goals(session.recorder.events, accepted))
    checks["all_expected_user_labels_verified"] = session.backend.index == len(spec["commands"])
    _require(checks["all_expected_user_labels_verified"], "Some reviewed user turns were never collected.")
    return checks


def _layout(spec):
    from .scene import sample_layout

    goals = [goal for expected in spec["expected"] for goal in expected["goals"]]
    object_task = any(goal["type"] in ("kick", "navigate") for goal in goals)
    layout = sample_layout(spec["seed"], **({} if object_task else {"radius_min": 0.75, "radius_max": 1.1}))
    missing = spec.get("metadata", {}).get("missing_color")
    if missing is not None:
        layout[missing] = [10.0, 10.0]
    return layout


def collect_one(spec, output_root, keep_trajectory=True):
    """Return one audited result; preserve invalid recordings only in quarantine."""
    episode_id = spec["episode_id"]
    if not isinstance(episode_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", episode_id):
        raise ValueError("Curriculum episode_id must be a nonempty path-safe string.")
    output_root = Path(output_root).resolve()
    output = output_root / "episodes" / episode_id
    output.mkdir(parents=True, exist_ok=False)
    runtime = session = None
    errors, checks, rows, results, execution = [], {}, [], [], None
    metadata = {"episode_id": episode_id, "collection_validated": False, "training_ready": False, "curriculum": spec}
    started = time.monotonic()
    with (output / "worker.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            from .bootstrap import configure

            configure()
            from .toy_dataset import read_episode
            from .toy_motion import ToyRuntime
            from .toy_session import ToySession
            from .toy_teacher import ToyTeacher

            layout = _layout(spec)
            runtime = ToyRuntime(output, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
            runtime.reset(layout)
            teacher = ExpectedTeacher(ToyTeacher(), spec)
            session = ToySession(runtime, seed=spec["seed"], split_group=spec["split_group"], backend=teacher)
            session.recorder.metadata.update(
                collection_kind=spec["kind"],
                curriculum=spec,
                collection_validated=False,
                training_ready=False,
                initial_layout_world_xy_m=layout,
            )
            session.recorder.save_metadata()
            checks = _run_script(session, spec)
            _require(not runtime.failure, f"Runtime physics failure: {runtime.failure}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            if session is not None:
                execution = _execution(session)
                try:
                    session.close()
                except Exception as exc:
                    errors.append(f"Session close: {type(exc).__name__}: {exc}")
                results = copy.deepcopy(session.results)
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as exc:
                    errors.append(f"Runtime close: {type(exc).__name__}: {exc}")
                try:
                    if keep_trajectory:
                        runtime.save_trace()
                    if session is not None:
                        session.recorder.metadata.update(physics=runtime.physics, policy_sha256=runtime.policy.hashes)
                except Exception as exc:
                    errors.append(f"Trace save: {type(exc).__name__}: {exc}")
            if session is not None:
                metadata = session.recorder.metadata
            metadata.update(collection_validated=not errors, training_ready=not errors)
            (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2))
            if not errors:
                try:
                    _, rows = read_episode(output)
                    checks["causal_history_images_contract_hash_verified"] = True
                except Exception as exc:
                    errors.append(f"Dataset audit: {type(exc).__name__}: {exc}")
                    metadata.update(collection_validated=False, training_ready=False)
                    (output / "metadata.json").write_text(
                        json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2)
                    )
            summary = {
                "episode_id": episode_id,
                "valid": not errors,
                "error": "; ".join(errors) or None,
                "dialogue_rows": len(rows),
                "goal_status": dict(Counter(row["status"] for row in results)),
                "results": results,
                "checks": checks,
                "metadata": metadata,
                "execution": execution,
                "wall_seconds": time.monotonic() - started,
                "simulation_seconds": runtime.snapshot()["time_s"] if runtime is not None else 0,
            }
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
    if errors:
        quarantine = output_root / "quarantine" / episode_id
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(quarantine))
        output = quarantine
    answer = {
        key: summary[key]
        for key in ("episode_id", "valid", "error", "dialogue_rows", "goal_status", "checks", "wall_seconds")
    } | {"output": str(output)}
    if session is not None:
        session.runner.session = None
        session.runner = None
    session = runtime = None
    gc.collect()
    return answer
