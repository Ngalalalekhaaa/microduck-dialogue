"""Task provenance, executor separation and inference-compatible SFT inputs."""

import copy
import json

import numpy as np
import pytest

from .dataset import Recorder
from .skill_vlm import SYSTEM_PROMPT, _input_context, _input_state, parse_proposal
from .training_data import execution_feedback, export_training_dataset, parse_seed_list


def make_episode(tmp_path, *, seed=1, name="episode", kind="kick", status="success", metadata=None):
    recorder = Recorder(tmp_path / name, name, seed, metadata or {"backend": "rgbd_rule_teacher"})
    time = 0.0
    events = []

    def add(phase, skill=None, seconds=0, *, done=False, reply="", context=None):
        nonlocal time
        index = len(events)
        observation = recorder.record_observation(
            f"frame_{index}",
            time,
            {"front": np.zeros((8, 10, 3), np.uint8), "near": np.full((8, 10, 3), 100, np.uint8)},
            {},
            [{"color": "red", "center_body_m": [0.0895, 0.0411, 0.035], "camera": "near"}],
        )
        observation.update(yaw=0.01, stationary=True, tilt_deg=1.2, oracle_world_positions={"red": [1, 2, 3]})
        event = {
            "event_id": f"event_{index}",
            "task_id": "task_001",
            "instruction": {"kick": "请踢红球", "move": "向前走一点", "describe": "你看到什么", "clarify": "踢球"}.get(
                kind, "停下"
            ),
            "intent": {"kind": kind, "color": "red" if kind == "kick" else None},
            "context": context or {"visible_colors": ["red"], "injected_answer": "not_an_input"},
            "observation": observation,
            "phase": phase,
            "decision": {"skill": skill, "seconds": seconds, "target_color": "red"} if skill else None,
            "execution": {
                "started": skill is not None,
                "completed": skill is not None,
                "start_time_s": time if skill else None,
                "end_time_s": time + seconds if skill else None,
            },
            "done": done,
            "status": status if done else "running",
            "reply": reply,
        }
        recorder.append_event(event)
        events.append(event)
        time += seconds
        return event

    def finish():
        add("done", done=True, reply="我看到红球。" if kind == "describe" else "已完成。")
        summary = {
            "results": [{"task_id": "task_001", "status": status, "evaluation": {"success": status == "success"}}]
        }
        (recorder.root / "summary.json").write_text(json.dumps(summary))

    return recorder, events, add, finish


def read_rows(path):
    return [
        json.loads(line)
        for split in ("train", "val", "test")
        for line in (path / f"{split}.jsonl").read_text().splitlines()
    ]


def successful_kick(tmp_path, **options):
    recorder, events, add, finish = make_episode(tmp_path, **options)
    add("interpret")
    add("search", "turn_left", 1.0)
    add("settle", "stand", 1.0)
    add("settle", "stand", 0.6)
    add("kick", "kick_left", 0.5)
    add("recover", "stand", 3.0)
    finish()
    return recorder, events


def test_executor_recovery_removed_but_planner_alignment_stand_kept(tmp_path):
    successful_kick(tmp_path / "raw")
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data", seed_splits={1: "train"})
    assert "training_started" not in manifest
    assert manifest["training_started_by_exporter"] is False
    assert manifest["export_only"] is True
    assert manifest["note_training_status_in"] == "training run directory/run_status.json"
    rows = read_rows(tmp_path / "data")
    assert [row["target"]["skill"] for row in rows] == ["turn_left", "stand", "kick_left", None]
    assert rows[1]["target"]["seconds"] == 0.6
    assert manifest["ignored_event_counts"]["executor_recovery"] == 2
    assert rows[-1]["target"]["type"] == "finish"
    assert "独立验收" in rows[-1]["target"]["reply"]
    assert "已踢动" not in rows[-1]["target"]["reply"]


def test_inputs_match_runtime_with_no_rule_or_current_target_leakage(tmp_path):
    successful_kick(tmp_path / "raw")
    export_training_dataset(tmp_path / "raw", tmp_path / "data")
    rows = read_rows(tmp_path / "data")
    for row in rows:
        assert row["messages"][0]["content"] == SYSTEM_PROMPT
        payload = json.loads(row["messages"][1]["content"][-1]["text"])
        assert payload == {key: row[key] for key in ("instruction", "state", "context")}
        assert _input_state(row["state"]) == row["state"]
        assert _input_context(row["context"]) == row["context"]
        assert parse_proposal(row["messages"][-1]["content"]).to_dict() == row["label"] == row["target"]
        assert not {"intent", "phase", "oracle_world_positions", "visible_colors", "injected_answer"} & payload.keys()
        assert "oracle_world_positions" not in json.dumps(payload)
        assert "not_an_input" not in json.dumps(payload)
        assert "task_id" not in payload["context"]
        for relative in row["images"].values():
            assert (tmp_path / "data" / relative).is_file()
    assert rows[0]["context"]["previous_action"] is None
    assert "selected_color" not in rows[0]["context"]
    assert rows[1]["context"]["selected_color"] == "red"
    assert rows[1]["context"]["previous_action"] == rows[0]["target"]
    assert rows[1]["context"]["previous_feedback"] == execution_feedback(rows[0]["target"])
    assert rows[-1]["context"]["step"] == 3


def test_failed_kick_does_not_generate_any_positive_training_samples(tmp_path):
    successful_kick(tmp_path / "raw", status="verification_failed")
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert not read_rows(tmp_path / "data")
    assert manifest["excluded_counts"] == {"task_not_successful:verification_failed": 1}


def test_kick_requires_independent_acceptance_even_if_event_claims_success(tmp_path):
    recorder, _ = successful_kick(tmp_path / "raw")
    (recorder.root / "summary.json").unlink()
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert not read_rows(tmp_path / "data")
    assert manifest["excluded_counts"] == {"kick_missing_independent_success_evaluation": 1}


def test_current_model_predictions_are_never_promoted_to_correct_labels(tmp_path):
    successful_kick(tmp_path / "raw", metadata={"backend": "local_pretrained_vlm", "is_trained_vlm": True})
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert not read_rows(tmp_path / "data")
    assert manifest["excluded_counts"] == {"model_prediction_not_demonstration": 1}


@pytest.mark.parametrize(
    "kind,status,target_type", [("describe", "success", "answer"), ("clarify", "clarification", "clarify")]
)
def test_answer_and_clarification_need_no_fabricated_action(tmp_path, kind, status, target_type):
    _, _, add, finish = make_episode(tmp_path / "raw", kind=kind, status=status)
    add("interpret")
    finish()
    export_training_dataset(tmp_path / "raw", tmp_path / "data")
    row = read_rows(tmp_path / "data")[0]
    assert row["target"]["type"] == target_type
    assert row["target"]["skill"] is None and row["target"]["seconds"] == 0
    assert row["target"]["target_color"] is None
    assert row["context"]["history"] == []


def test_single_movement_has_action_and_observed_finish(tmp_path):
    _, _, add, finish = make_episode(tmp_path / "raw", kind="move")
    add("interpret")
    add("direct", "move_forward", 0.32)
    add("settle", "stand", 1.0)
    finish()
    export_training_dataset(tmp_path / "raw", tmp_path / "data")
    action, terminal = read_rows(tmp_path / "data")
    assert action["target"]["skill"] == "move_forward"
    assert action["target"]["target_color"] is None
    assert terminal["target"]["type"] == "finish"
    assert terminal["state"]["sim_time_s"] == 1.32
    assert terminal["context"]["previous_action"] == action["target"]


def test_incomplete_recovery_or_out_of_interface_duration_excludes_entire_task(tmp_path):
    _, _, add, finish = make_episode(tmp_path / "raw", kind="move", name="no_recovery")
    add("direct", "move_forward", 0.32)
    finish()
    _, _, add, finish = make_episode(tmp_path / "raw", kind="move", name="too_long")
    add("direct", "move_forward", 0.8)
    add("settle", "stand", 1.0)
    finish()
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert not read_rows(tmp_path / "data")
    assert "missing_completed_executor_recovery" in manifest["excluded_counts"]
    assert any("duration must" in key for key in manifest["excluded_counts"])


def test_same_seed_all_tasks_and_paraphrases_stay_in_explicit_split(tmp_path):
    for index, seed in enumerate((1, 1, 2, 3)):
        successful_kick(tmp_path / "raw", name=f"episode_{index}", seed=seed)
    manifest = export_training_dataset(
        tmp_path / "raw", tmp_path / "data", seed_splits={1: "train", 2: "val", 3: "test"}
    )
    assert {split: info["samples"] for split, info in manifest["counts"].items()} == {"train": 8, "val": 4, "test": 4}
    for split, seed in (("train", 1), ("val", 2), ("test", 3)):
        rows = [json.loads(line) for line in (tmp_path / "data" / f"{split}.jsonl").read_text().splitlines()]
        assert {row["scene_seed"] for row in rows} == {seed}


def test_partial_explicit_seed_mapping_fails_before_writing_dataset(tmp_path):
    successful_kick(tmp_path / "raw")
    with pytest.raises(ValueError, match="missing scene seed"):
        export_training_dataset(tmp_path / "raw", tmp_path / "data", seed_splits={2: "train"})
    assert not (tmp_path / "data").exists()


def test_history_max_six_and_selected_color_from_prior_context_only(tmp_path):
    _, _, add, finish = make_episode(tmp_path / "raw")
    add("interpret", context={"selected_color": "blue", "intent": "do not copy"})
    for _ in range(9):
        add("search", "turn_left", 1.0)
        add("settle", "stand", 1.0)
    add("kick", "kick_left", 0.5)
    add("recover", "stand", 3.0)
    finish()
    export_training_dataset(tmp_path / "raw", tmp_path / "data")
    rows = read_rows(tmp_path / "data")
    assert rows[0]["context"]["selected_color"] == "blue"
    assert rows[1]["context"]["selected_color"] == "red"
    assert len(rows[-1]["context"]["history"]) == 6
    assert rows[-1]["context"]["step"] == 10


def test_rgb_only_omits_measurements_consistently(tmp_path):
    successful_kick(tmp_path / "raw")
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data", rgbd_assist=False)
    assert manifest["rgbd_assist"] is False
    assert all("detections_rgbd" not in row["state"] for row in read_rows(tmp_path / "data"))


def test_rerun_refuses_existing_dataset_and_never_changes_raw_logs(tmp_path):
    recorder, _ = successful_kick(tmp_path / "raw")
    before = (recorder.root / "events.jsonl").read_bytes()
    export_training_dataset(tmp_path / "raw", tmp_path / "data")
    with pytest.raises(FileExistsError):
        export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert (recorder.root / "events.jsonl").read_bytes() == before


def test_source_ids_cannot_collide_across_roots(tmp_path):
    successful_kick(tmp_path / "raw_a")
    successful_kick(tmp_path / "raw_b")
    with pytest.raises(ValueError, match="Duplicate episode_id"):
        export_training_dataset([tmp_path / "raw_a", tmp_path / "raw_b"], tmp_path / "data")


def test_input_whitelist_refuses_privileged_detection_fields(tmp_path):
    recorder, events = successful_kick(tmp_path / "raw")
    changed = copy.deepcopy(events)
    changed[1]["observation"]["detections"][0]["world_position"] = [1, 2, 3]
    (recorder.root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in changed))
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert not read_rows(tmp_path / "data")
    assert any("RGB-D detection accepts only" in reason for reason in manifest["excluded_counts"])


@pytest.mark.parametrize("mutation", ["moving", "unaligned", "missing"])
def test_kick_label_must_pass_the_deployed_execution_guard(tmp_path, mutation):
    recorder, events = successful_kick(tmp_path / "raw")
    kick = next(event for event in events if event["phase"] == "kick")
    if mutation == "moving":
        kick["observation"]["stationary"] = False
    elif mutation == "unaligned":
        kick["observation"]["detections"][0]["center_body_m"][0] = 0.8
    else:
        kick["observation"]["detections"] = []
    (recorder.root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert not read_rows(tmp_path / "data")
    assert any(reason.startswith("deployed_guard_rejected:") for reason in manifest["excluded_counts"])


@pytest.mark.parametrize("verified", [True, False, None])
def test_new_direct_collection_requires_independent_verification(tmp_path, verified):
    metadata = {
        "backend": "rgbd_rule_teacher",
        "collection_kind": "direct_dialogue",
        "collection_verified": verified,
        "training_split": "val",
    }
    _, _, add, finish = make_episode(tmp_path / "raw", kind="move", metadata=metadata)
    add("direct", "move_forward", 0.32)
    add("settle", "stand", 1.0)
    finish()
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert manifest["seed_splits"] == {"1": "val"}
    assert len(read_rows(tmp_path / "data")) == (2 if verified else 0)
    if not verified:
        assert manifest["excluded_counts"] == {"direct_collection_not_independently_verified": 1}


def test_declared_collection_split_cannot_be_overridden_silently(tmp_path):
    successful_kick(tmp_path / "raw", metadata={"backend": "rule", "training_split": "test"})
    with pytest.raises(ValueError, match="disagrees with collection metadata"):
        export_training_dataset(tmp_path / "raw", tmp_path / "data", seed_splits={1: "train"})
    assert not (tmp_path / "data").exists()


def test_missing_kick_evaluation_object_excludes_without_crashing(tmp_path):
    recorder, _ = successful_kick(tmp_path / "raw")
    (recorder.root / "summary.json").write_text(
        json.dumps({"results": [{"task_id": "task_001", "status": "success", "evaluation": None}]})
    )
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert manifest["excluded_counts"] == {"kick_missing_independent_success_evaluation": 1}


def test_executed_duration_must_match_the_proposed_duration(tmp_path):
    recorder, events = successful_kick(tmp_path / "raw")
    events[1]["execution"]["end_time_s"] = 0.5
    (recorder.root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert manifest["excluded_counts"] == {"executed_duration_differs_from_proposal": 1}


def test_active_collection_exports_only_the_completed_manifest_snapshot(tmp_path):
    recorder, _ = successful_kick(tmp_path / "raw", name="completed")
    active = tmp_path / "raw" / "active"
    active.mkdir()
    (active / "episode.json").write_text("unfinished json")
    summary = {
        "format_version": "microduck-rule-training-collection-1",
        "planned_total": 2,
        "cases": [{"output": str(recorder.root.resolve())}],
    }
    (tmp_path / "raw" / "summary.json").write_text(json.dumps(summary))
    manifest = export_training_dataset(tmp_path / "raw", tmp_path / "data")
    assert len(read_rows(tmp_path / "data")) == 4
    assert manifest["collection_snapshots"][0]["completed_cases"] == 1
    assert manifest["collection_snapshots"][0]["planned_cases"] == 2


def test_collection_manifest_cannot_pull_recordings_from_another_root(tmp_path):
    recorder, _ = successful_kick(tmp_path / "outside")
    root = tmp_path / "raw"
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps(
            {
                "format_version": "microduck-rule-training-collection-1",
                "cases": [{"output": str(recorder.root.resolve())}],
            }
        )
    )
    with pytest.raises(ValueError, match="escapes its recording root"):
        export_training_dataset(root, tmp_path / "data")


def test_episode_clock_offsets_and_task_ids_do_not_change_student_inputs(tmp_path):
    successful_kick(tmp_path / "raw", name="original")
    recorder, events = successful_kick(tmp_path / "raw", name="shifted")
    for event in events:
        event["task_id"] = "task_987"
        event["observation"]["time_s"] += 137.281
        for key in ("start_time_s", "end_time_s"):
            if event["execution"][key] is not None:
                event["execution"][key] += 137.281
    (recorder.root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    summary = json.loads((recorder.root / "summary.json").read_text())
    summary["results"][0]["task_id"] = "task_987"
    (recorder.root / "summary.json").write_text(json.dumps(summary))
    export_training_dataset(tmp_path / "raw", tmp_path / "data")
    rows = read_rows(tmp_path / "data")
    original = [row for row in rows if row["episode_id"] == "original"]
    shifted = [row for row in rows if row["episode_id"] == "shifted"]
    assert len(original) == len(shifted) == 4
    for first, second in zip(original, shifted, strict=True):
        assert first["state"] == second["state"]
        assert first["context"] == second["context"]
        assert first["target"] == second["target"]
        assert first["messages"][1]["content"][-1] == second["messages"][1]["content"][-1]
        assert first["task_id"] == "task_001"
        assert second["task_id"] == "task_987"


@pytest.mark.parametrize("value,expected", [("0,2-4,8", {0, 2, 3, 4, 8}), ("", set()), ("7", {7})])
def test_seed_range_parser(value, expected):
    assert parse_seed_list(value) == expected


@pytest.mark.parametrize("value", ["4-2", "1-100002", "a", "1-2-3"])
def test_bad_seed_ranges_rejected(value):
    with pytest.raises(ValueError):
        parse_seed_list(value)
