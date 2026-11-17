"""Audit the fixed collection plan before starting expensive simulation."""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from . import collect_training
from .collect_training import _direct_acceptance, _executor_recovery, make_plan
from .language import parse_instruction


def test_scene_split_cannot_leak_across_color_or_direct_tasks():
    splits, plan = make_plan()
    assert [len(splits[split]) for split in ("train", "val", "test")] == [80, 10, 10]
    assert len(plan) == 400
    assert len({case["case_id"] for case in plan}) == len(plan)
    seed_split = {}
    for case in plan:
        seed_split.setdefault(case["seed"], case["split"])
        assert seed_split[case["seed"]] == case["split"]
        assert case["seed"] in splits[case["split"]]
    assert len(seed_split) == 100


def test_all_planned_commands_parse_to_the_intended_task():
    _, plan = make_plan()
    for case in plan:
        instructions = case["instructions"]
        if case["kind"] == "kick":
            before, kick, after = (parse_instruction(text) for text in instructions)
            assert before.kind == after.kind == "describe"
            assert kick.kind == "kick"
            assert kick.color == case["target_color"]
        else:
            intents = [parse_instruction(text) for text in instructions]
            assert [intent.kind for intent in intents] == [
                "describe",
                "move",
                "move",
                "move",
                "move",
                "turn",
                "turn",
                "stop",
                "describe",
                "clarify",
            ]
            assert [intent.direction for intent in intents[1:7]] == [
                "forward",
                "backward",
                "left",
                "right",
                "left",
                "right",
            ]
            assert all(intent.seconds <= 0.6 for intent in intents[1:7])


def test_held_out_kick_and_movement_phrasings_are_not_training_phrasings():
    _, plan = make_plan()
    phrases = {split: set() for split in ("train", "val", "test")}
    for case in plan:
        selected = case["instructions"][1:2] if case["kind"] == "kick" else case["instructions"][1:7]
        phrases[case["split"]].update(selected)
    assert not phrases["train"] & phrases["val"]
    assert not phrases["train"] & phrases["test"]
    assert not phrases["val"] & phrases["test"]


@pytest.mark.parametrize("counts", [(0, 10, 10), (80, 0, 10), (80, 10, 0)])
def test_empty_splits_are_refused(counts):
    with pytest.raises(ValueError, match="at least one"):
        make_plan(1000, *counts)


def test_teacher_stabilization_is_not_executor_recovery():
    event = {"phase": "settle", "decision": {"skill": "stand", "seconds": 0.6}}
    assert not _executor_recovery(event)
    event["decision"]["seconds"] = 1.0
    assert _executor_recovery(event)


@pytest.fixture
def stable_direct_trace(tmp_path):
    _, plan = make_plan(1, 1, 1, 1)
    case = next(case for case in plan if case["kind"] == "direct_dialogue")
    results = []
    for instruction in case["instructions"]:
        intent = parse_instruction(instruction)
        results.append(
            {"intent": intent.to_dict(), "status": "clarification" if intent.kind == "clarify" else "success"}
        )
    arrays = {
        "qpos": np.zeros((7, 7)),
        "qvel": np.zeros((7, 6)),
        "tilt_deg": np.zeros(7),
        "trunk_xyz": np.array([[0, 0, 0.12]] * 7),
        "yaw": np.array([0, 0, 0, 0, 0, 0.5, 0]),
        "time": np.arange(7, dtype=float),
    }
    arrays["trunk_xyz"][1, 0] = 0.02
    arrays["trunk_xyz"][3, 1] = 0.02
    np.savez_compressed(tmp_path / "trajectory.npz", **arrays)
    (tmp_path / "contacts.json").write_text("[]")
    events = []
    for index, skill in enumerate(
        ("move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right")
    ):
        task_id = f"task_{index}"
        events.extend(
            [
                {
                    "task_id": task_id,
                    "phase": "direct",
                    "decision": {"skill": skill},
                    "execution": {"completed": True, "start_time_s": float(index)},
                },
                {"task_id": task_id, "phase": "done", "done": True, "observation": {"time_s": float(index + 1)}},
            ]
        )
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    return tmp_path, {"results": results}, arrays


def test_direct_success_requires_independent_trace_not_only_task_status(stable_direct_trace):
    directory, summary, arrays = stable_direct_trace
    assert _direct_acceptance(directory, summary)["success"]
    arrays["tilt_deg"][1] = 40
    np.savez_compressed(directory / "trajectory.npz", **arrays)
    assert not _direct_acceptance(directory, summary)["success"]


def test_direct_ball_contact_is_excluded(stable_direct_trace):
    directory, summary, _ = stable_direct_trace
    (directory / "contacts.json").write_text(json.dumps([{"color": "red", "other_color": None}]))
    result = _direct_acceptance(directory, summary)
    assert not result["success"]
    assert result["robot_ball_contacts"] == 1


def test_direct_misparsed_task_is_excluded(stable_direct_trace):
    directory, summary, _ = stable_direct_trace
    summary["results"][1]["intent"]["direction"] = "backward"
    assert not _direct_acceptance(directory, summary)["success"]


def test_direct_wrong_actual_motion_is_excluded(stable_direct_trace):
    directory, summary, arrays = stable_direct_trace
    arrays["trunk_xyz"][1, 0] = -0.02
    np.savez_compressed(directory / "trajectory.npz", **arrays)
    result = _direct_acceptance(directory, summary)
    assert not result["success"]
    assert result["actual_directions"]["measurements"][0]["signed_progress"] < 0


def test_parallel_collection_resumes_without_duplicates_and_serializes_manifest(tmp_path, monkeypatch):
    _, plan = make_plan(1, 1, 1, 1)
    manifest = {"plan": plan, "plan_sha256": "test", "cases": []}
    main_thread = threading.get_ident()
    launched = []
    lock = threading.Lock()
    original_json = collect_training._json

    def write_json(path, value):
        assert threading.get_ident() == main_thread
        original_json(path, value)

    def fake_run(case, output, plan_sha256, timeout):
        with lock:
            launched.append(case["case_id"])
        time.sleep(0.001)
        return {
            **case,
            "success": True,
            "status": "success",
            "searched": False,
            "selected_kick": "kick_left" if case["kind"] == "kick" else None,
            "bytes_on_disk": 0,
            "wall_seconds": 0.001,
        }

    monkeypatch.setattr(collect_training, "_json", write_json)
    monkeypatch.setattr(collect_training, "_run_case", fake_run)
    monkeypatch.setattr(collect_training.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    args = SimpleNamespace(limit=5, workers=3, min_free_gib=8, timeout=1)
    collect_training._collect_pending(args, tmp_path, manifest)
    assert len(manifest["cases"]) == 5
    args.limit = None
    collect_training._collect_pending(args, tmp_path, manifest)
    assert len(manifest["cases"]) == len(plan)
    assert len(set(launched)) == len(launched) == len(plan)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert len(saved["cases"]) == len(plan)
