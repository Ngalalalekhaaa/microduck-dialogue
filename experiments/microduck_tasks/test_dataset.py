"""Dataset checks focus on execution provenance and scene-level leakage."""

import json

import numpy as np
import pytest

from .dataset import Recorder, export_vlm_dataset, split_for_seed


def recorder(tmp_path, episode_id="episode_000", seed=42, metadata=None):
    rec = Recorder(tmp_path / episode_id, episode_id, seed, {"backend": "rule"} if metadata is None else metadata)
    observation = rec.record_observation(
        "000",
        0.0,
        {"navigation": np.zeros((8, 10, 3), np.uint8)},
        {"navigation": np.ones((8, 10), np.float32)},
        [{"color": "red", "forward_m": 0.4}],
    )
    return rec, observation


def event(observation, event_id="000", task_id="task_000", executed=True, done=False, status="running"):
    return {
        "event_id": event_id,
        "task_id": task_id,
        "instruction": "踢一下红色的球",
        "intent": {"kind": "kick", "color": "red"},
        "observation": observation,
        "phase": "approach",
        "decision": {"skill": "move_forward", "seconds": 0.32},
        "execution": {
            "started": executed,
            "completed": executed,
            "start_time_s": 0.0 if executed else None,
            "end_time_s": 1.32 if executed else None,
        },
        "done": done,
        "status": status,
        "reply": "",
    }


def rows(output):
    return [json.loads(line) for path in output.glob("*.jsonl") for line in path.read_text().splitlines()]


def test_only_executed_successful_task_actions_become_imitation(tmp_path):
    rec, obs = recorder(tmp_path / "raw")
    rec.append_event(event(obs))
    terminal = event(obs, "001", executed=False, done=True, status="success")
    terminal["decision"] = None
    terminal["reply"] = "已经踢动红球。"
    rec.append_event(terminal)
    rec.append_event(event(obs, "002", "task_failed", done=True, status="fallen"))
    rec.append_event(event(obs, "003", "task_unstarted", executed=False, done=True, status="success"))
    export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    actions = [row for row in rows(tmp_path / "dataset") if row["kind"] == "action"]
    assert len(actions) == 1
    assert actions[0]["task_id"] == "task_000"
    assert ":000:action" in actions[0]["sample_id"]
    assert len([row for row in rows(tmp_path / "dataset") if row["kind"] == "intent"]) == 3


def test_incomplete_and_zero_elapsed_executions_are_excluded(tmp_path):
    rec, obs = recorder(tmp_path / "raw")
    interrupted = event(obs)
    interrupted["execution"].update(completed=False, end_time_s=0.1)
    rec.append_event(interrupted)
    instantaneous = event(obs, "001")
    instantaneous["execution"]["end_time_s"] = 0.0
    rec.append_event(instantaneous)
    rec.append_event(event(obs, "002", executed=False, done=True, status="success"))
    export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    assert not any(row["kind"] == "action" for row in rows(tmp_path / "dataset"))


def test_same_seed_and_entire_episode_remain_in_one_split(tmp_path):
    for index, seed in enumerate((13, 13, 14, 15)):
        rec, obs = recorder(tmp_path / "raw", f"episode_{index}", seed)
        rec.append_event(event(obs))
        terminal = event(obs, "001", executed=False, done=True, status="success")
        terminal["reply"] = "完成。"
        rec.append_event(terminal)
    manifest = export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    episodes = manifest["episodes"]
    assert episodes[0]["split"] == episodes[1]["split"] == split_for_seed(13)
    observed = {}
    for path in (tmp_path / "dataset").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            observed.setdefault(row["episode_id"], set()).add(path.stem)
            assert row["scene_seed"] in (13, 14, 15)
    assert all(len(splits) == 1 for splits in observed.values())


def test_export_contains_portable_images_without_privileged_input(tmp_path):
    rec, obs = recorder(tmp_path / "raw")
    rec.append_event(event(obs, done=True, status="success"))
    result = export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    assert result["training_started"] is False
    for row in rows(tmp_path / "dataset"):
        user = row["messages"][1]["content"]
        for block in user:
            if block["type"] == "image":
                assert (tmp_path / "dataset" / block["image"]).is_file()
            else:
                assert "forward_m" not in block["text"]
                assert "depth" not in block["text"]
        assert row["teacher"]["uses_depth"]
        json.loads(row["messages"][-1]["content"][0]["text"])


def test_missing_image_fails_before_output_is_created(tmp_path):
    rec, obs = recorder(tmp_path / "raw")
    rec.append_event(event(obs, done=True, status="success"))
    (rec.root / obs["images"]["navigation"]).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_description_keeps_reply_without_fabricating_execution(tmp_path):
    rec, obs = recorder(tmp_path / "raw")
    query = event(obs, executed=False, done=True, status="success")
    query.update(
        instruction="现在能看到什么？",
        intent={"kind": "describe"},
        decision={"skill": "describe", "seconds": 0.0},
        reply="我看到前方有红球。",
    )
    rec.append_event(query)
    export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    assert {row["kind"] for row in rows(tmp_path / "dataset")} == {"intent", "reply"}


def test_intent_preserves_pronoun_context_and_native_schema(tmp_path):
    rec, obs = recorder(tmp_path / "raw")
    entry = event(obs, done=True, status="success")
    entry.update(instruction="踢它", context={"selected_color": "red"})
    rec.append_event(entry)
    export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    sample = next(row for row in rows(tmp_path / "dataset") if row["kind"] == "intent")
    user_text = sample["messages"][1]["content"][-1]["text"]
    assert json.loads(user_text)["context"] == {"selected_color": "red"}
    assert json.loads(sample["messages"][-1]["content"][0]["text"]) == entry["intent"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda entry: entry.pop("execution"),
        lambda entry: entry["execution"].update(started=False),
        lambda entry: entry["execution"].update(end_time_s=None),
        lambda entry: entry["execution"].update(end_time_s=-1),
        lambda entry: entry["decision"].update(seconds=float("nan")),
        lambda entry: entry.update(decision=None),
    ],
)
def test_invalid_execution_provenance_refused(tmp_path, mutation):
    rec, obs = recorder(tmp_path)
    entry = event(obs)
    mutation(entry)
    with pytest.raises(ValueError):
        rec.append_event(entry)
    assert not (rec.root / "events.jsonl").read_text()


def test_duplicate_frame_or_event_refused(tmp_path):
    rec, obs = recorder(tmp_path)
    rec.append_event(event(obs))
    with pytest.raises(ValueError, match="Duplicate"):
        rec.append_event(event(obs))
    with pytest.raises(ValueError, match="already recorded"):
        rec.record_observation("000", 0.0, {"navigation": np.zeros((8, 10, 3), np.uint8)}, {}, [])
    with pytest.raises(FileExistsError):
        Recorder(rec.root, "episode_000", 42, {})


def test_path_escape_and_nonfinite_depth_refused(tmp_path):
    rec, obs = recorder(tmp_path)
    with pytest.raises(ValueError, match="letters"):
        rec.record_observation("../escape", 0.0, {}, {}, [])
    obs["images"]["navigation"] = "../escape.png"
    with pytest.raises(ValueError, match="escapes"):
        rec.append_event(event(obs))
    with pytest.raises(ValueError, match="finite"):
        rec.record_observation(
            "001",
            0.0,
            {"navigation": np.zeros((8, 10, 3), np.uint8)},
            {"navigation": np.full((8, 10), float("nan"), np.float32)},
            [],
        )


def test_export_never_overwrites_existing_dataset(tmp_path):
    rec, obs = recorder(tmp_path / "raw")
    rec.append_event(event(obs, done=True, status="success"))
    export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    with pytest.raises(FileExistsError):
        export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")


def test_successful_vlm_predictions_stay_in_raw_logs_without_supervision(tmp_path):
    rec, obs = recorder(
        tmp_path / "raw",
        metadata={"backend": "local_pretrained_vlm", "is_trained_vlm": True, "robot_finetuned": False},
    )
    entry = event(obs, done=True, status="success")
    entry.update(
        intent={},
        controller_source="pretrained_vlm",
        reply="已经完成。",
        model_trace={
            "model_proposal": {"skill": "move_forward", "seconds": 0.32},
            "guard_result": {"accepted": True},
            "raw_response": '{"skill":"move_forward","seconds":0.32}',
            "latency_s": 0.75,
            "model": "local-test-model",
            "endpoint": "local",
            "prompt_version": "test-1",
            "input_state": {"stationary": True},
            "input_context": {"previous_action": None},
        },
    )
    rec.append_event(entry)
    raw = (rec.root / "events.jsonl").read_text()
    manifest = export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    assert rows(tmp_path / "dataset") == []
    assert not (tmp_path / "dataset" / "images").exists()
    assert manifest["excluded"] == [
        {
            "episode_id": "episode_000",
            "backend": "local_pretrained_vlm",
            "reason": "model_prediction_not_demonstration",
            "events": 1,
        }
    ]
    assert manifest["source_backends"]["local_pretrained_vlm"] == {
        "episodes": 1,
        "events": 1,
        "exported_samples": 0,
        "excluded_episodes": 1,
    }
    assert not manifest["episodes"][0]["supervision_eligible"]
    assert (rec.root / "events.jsonl").read_text() == raw
    assert json.loads(raw)["model_trace"] == entry["model_trace"]


@pytest.mark.parametrize(
    "provenance",
    [
        {"controller_source": "pretrained_vlm"},
        {"controller_source": "finetuned_vlm"},
        {"controller_source": "frozen_vlm_trained_skill_head"},
        {"model_trace": {"model_proposal": {"skill": "move_forward"}}},
    ],
)
def test_model_event_cannot_be_exported_under_legacy_rule_metadata(tmp_path, provenance):
    rec, obs = recorder(tmp_path / "raw")
    entry = event(obs, done=True, status="success")
    entry.update(provenance)
    rec.append_event(entry)
    manifest = export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    assert rows(tmp_path / "dataset") == []
    assert manifest["excluded"][0]["reason"] == "model_prediction_not_demonstration"


def test_phase_is_auditable_but_not_a_student_input(tmp_path):
    rec, obs = recorder(tmp_path / "raw")
    first = event(obs)
    first["phase"] = "turn_to_target"
    rec.append_event(first)
    second = event(obs, "001", done=True, status="success")
    second.update(phase="kick_left", reply="完成。")
    rec.append_event(second)
    export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    for sample in rows(tmp_path / "dataset"):
        student_input = json.loads(sample["messages"][1]["content"][-1]["text"])
        assert "phase" not in student_input
        assert sample["source_event"]["phase"] in {"turn_to_target", "kick_left"}
        if sample["kind"] != "intent":
            assert student_input["previous_action"] == (
                first["decision"] if sample["source_event"]["event_id"] == "001" else None
            )


@pytest.mark.parametrize("metadata,expected", [({}, "rule"), ({"backend": "rgbd_rule_teacher"}, "rgbd_rule_teacher")])
def test_legacy_rule_export_and_exact_teacher_backend(tmp_path, metadata, expected):
    rec, obs = recorder(tmp_path / "raw", metadata=metadata)
    rec.append_event(event(obs, done=True, status="success"))
    manifest = export_vlm_dataset(tmp_path / "raw", tmp_path / "dataset")
    exported = rows(tmp_path / "dataset")
    assert {sample["kind"] for sample in exported} == {"intent", "action"}
    assert {sample["teacher"]["backend"] for sample in exported} == {expected}
    assert manifest["excluded"] == []
    assert manifest["source_backends"][expected]["exported_samples"] == len(exported)


@pytest.mark.parametrize(
    "provenance",
    [
        {"controller_source": "unknown"},
        {"controller_source": {}},
        {"model_trace": "model selected forward"},
        {"model_trace": {"latency_s": -1}},
        {"teacher_trace": "rule selected forward"},
        {"teacher_trace": {"latency_s": -1}},
    ],
)
def test_invalid_optional_model_provenance_is_refused(tmp_path, provenance):
    rec, obs = recorder(tmp_path)
    entry = event(obs)
    entry.update(provenance)
    with pytest.raises(ValueError):
        rec.append_event(entry)
    assert not (rec.root / "events.jsonl").read_text()
