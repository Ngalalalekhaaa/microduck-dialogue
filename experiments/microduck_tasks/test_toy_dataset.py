"""Causal trace validation; these tiny test images are not training recordings."""

import copy
import json

import pytest
from PIL import Image

from .toy_collection_audit import _numeric_identity, audit
from .toy_dataset import FORMAT_VERSION, SOURCE, TRACE_VERSION, contract_digest, digest, export, read_episode
from .toy_fixtures import perform, request
from .toy_protocol import VERSION


def trace(root, *, episode="episode0", seed=1, family="greeting", feedback=False):
    root.mkdir(parents=True)
    for camera in ("front", "near"):
        Image.new("RGB", (8, 8)).save(root / f"{camera}.png")
    metadata = {
        "format_version": TRACE_VERSION,
        "schema_version": VERSION,
        "episode_id": episode,
        "seed": seed,
        "split_group": family,
        "controller_source": SOURCE,
        "recording_source": "runtime",
        "training_ready": True,
        "contract_sha256": contract_digest(),
    }
    state = request("你好")
    state["observation"]["age_s"] = 0
    response = {"schema_version": VERSION, "reply": "你好，我叫小鸭。", "decision": "respond", "goals": []}
    event = {
        "event": "dialogue",
        "event_id": "event0",
        "turn_id": "turn0",
        "observation_id": "obs0",
        "task_id": "task0",
        "revision": 0,
        "time_s": 0,
        "observation_time_s": 0,
        "request": state,
        "response": response,
        "controller_source": SOURCE,
        "media": {camera: f"{camera}.png" for camera in ("front", "near")},
    }
    events = [event]
    if feedback:
        performed = perform("nod")
        response.update(reply="好，我点一下头。", decision="replace", goals=[performed])
        state["event"]["text"] = "点头"
        events.extend(
            [
                {
                    "event": "goal_start",
                    "event_id": "event1",
                    "time_s": 0.1,
                    "task_id": "task0",
                    "goal_index": 0,
                    "goal": performed,
                },
                {
                    "event": "goal_result",
                    "event_id": "event2",
                    "time_s": 4,
                    "task_id": "task0",
                    "goal_index": 0,
                    "goal": performed,
                    "success": True,
                    "status": "succeeded",
                    "reason": "verified head pitch excursion and stable recovery",
                    "verified": True,
                },
            ]
        )
        result = {name: events[-1][name] for name in ("goal", "status", "reason", "verified")}
        feedback_event = copy.deepcopy(event)
        feedback_event.update(event_id="event3", turn_id="turn1", time_s=4, observation_time_s=4, result_ref="event2")
        feedback_event["request"]["history"] = [
            {"role": "user", "text": "点头"},
            {"role": "assistant", "text": "好，我点一下头。"},
        ]
        feedback_event["request"]["event"] = {"kind": "feedback", "text": "目标执行结束。"}
        feedback_event["request"]["execution"]["last_result"] = result
        feedback_event["response"].update(reply="点头完成了。", decision="respond", goals=[])
        events.append(feedback_event)
    save(root, metadata, events)
    return metadata, events


def save(root, metadata, events):
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))


def test_exports_full_causal_request_and_independent_episode_holdout(tmp_path):
    inputs, output = tmp_path / "input", tmp_path / "output"
    trace(inputs / "a", episode="a", seed=1, feedback=True)
    trace(inputs / "b", episode="b", seed=2, family="visual")
    result = export(inputs, output)
    assert result["training_ready"] and not result["deployment_ready"]
    assert result["episodes"] == 2 and sum(result["rows"].values()) == 3
    assert set(result["episode_splits"].values()) == {"train", "val"}
    rows = [json.loads(line) for path in output.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert all(row["format_version"] == FORMAT_VERSION for row in rows)
    assert all((output / row["images"]["front"]).is_file() for row in rows)
    feedback = next(row for row in rows if row["request"]["event"]["kind"] == "feedback")
    assert feedback["request"]["execution"]["last_result"]["verified"]
    with pytest.raises(FileExistsError):
        export(inputs, output)


def test_shared_seed_and_paraphrase_family_never_cross_splits(tmp_path):
    trace(tmp_path / "input/a", episode="a", seed=1, family="one")
    trace(tmp_path / "input/b", episode="b", seed=1, family="two")
    trace(tmp_path / "input/c", episode="c", seed=2, family="two")
    trace(tmp_path / "input/d", episode="d", seed=3, family="three")
    result = export(tmp_path / "input", tmp_path / "output")
    split = result["episode_splits"]
    assert split["a"] == split["b"] == split["c"]
    assert split["d"] != split["a"]


def test_three_way_export_preserves_groups_and_balances_episode_counts(tmp_path):
    for index in range(20):
        trace(tmp_path / "input" / str(index), episode=f"episode{index}", seed=index, family=f"family{index}")
    result = export(tmp_path / "input", tmp_path / "output", three_way=True)
    assert result["split_episode_counts"] == {"train": 16, "val": 2, "test": 2}
    assert result["independent_split_groups"] == 20
    assert set(result["files_sha256"]) == {"train", "val", "test"}
    assert "80/10/10" in result["split_rule"]
    repeated = export(tmp_path / "input", tmp_path / "repeat", three_way=True)
    assert repeated["episode_splits"] == result["episode_splits"]


def test_three_way_handles_transitive_scene_and_family_connections(tmp_path):
    for name, seed, family in (("a", 1, "one"), ("b", 2, "two"), ("c", 1, "two"), ("d", 3, "three"), ("e", 4, "four")):
        trace(tmp_path / "input" / name, episode=name, seed=seed, family=family)
    result = export(tmp_path / "input", tmp_path / "output", three_way=True)
    splits = result["episode_splits"]
    assert splits["a"] == splits["b"] == splits["c"]
    assert set(splits.values()) == {"train", "val", "test"}
    assert result["independent_split_groups"] == 3


def test_three_way_rejects_unseparable_inputs_without_creating_output(tmp_path):
    for index in range(4):
        trace(tmp_path / "input" / str(index), episode=f"episode{index}", seed=index, family="same")
    with pytest.raises(ValueError, match="three independent"):
        export(tmp_path / "input", tmp_path / "output", three_way=True)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("value", [False, None, "true"])
def test_collection_not_validated_is_quarantined(tmp_path, value):
    root = tmp_path / "episode"
    metadata, events = trace(root)
    metadata["collection_validated"] = value
    save(root, metadata, events)
    with pytest.raises(ValueError, match="quarantine"):
        read_episode(root)


def test_audit_reports_real_repetitions_and_goal_results_without_inflating_diversity(tmp_path):
    for index in range(3):
        trace(
            tmp_path / "input" / str(index),
            episode=f"episode{index}",
            seed=index,
            family=f"family{index}",
            feedback=True,
        )
    export(tmp_path / "input", tmp_path / "output", three_way=True)
    result = audit(tmp_path / "output", source_root=tmp_path / "input")
    assert result["split_isolation_passed"]
    assert result["overall"]["rows"] == 6
    assert result["overall"]["unique_user_utterances"] == 1
    assert result["overall"]["repeated_normalized_user_rows"] == 2
    assert result["overall"]["unique_goal_sequences"] == 1
    assert result["media"]["camera_frame_references"] == 12
    assert result["media"]["unique_png_sha256"] == 1
    assert result["media"]["repeated_exact_frame_references"] == 11
    assert result["media"]["image_hashes_shared_across_splits"] == 1
    assert result["source"]["goal_results"] == {"succeeded": 3}
    assert set(result["splits"]) == {"train", "val", "test"}


def test_audit_counts_equivalent_numeric_plans_once_without_rewriting_samples(tmp_path):
    for index, angle in enumerate((360, 360.0, 180, 180.0)):
        root = tmp_path / "input" / str(index)
        metadata, events = trace(root, episode=f"episode{index}", seed=index, family=f"family{index}")
        events[0]["request"]["capabilities"]["goal_types"].append("turn")
        events[0]["request"]["event"]["text"] = f"向左转{angle}度"
        events[0]["response"].update(
            reply="好的，向左转。",
            decision="replace",
            goals=[{"type": "turn", "direction": "left", "angle_deg": angle}],
        )
        save(root, metadata, events)
    output = tmp_path / "output"
    export(tmp_path / "input", output, three_way=True)
    before = {str(path): digest(path) for path in tmp_path.rglob("*") if path.is_file()}
    result = audit(output, source_root=tmp_path / "input")
    assert result["overall"]["unique_exact_plans"] == 2
    assert result["overall"]["unique_goal_sequences"] == 2
    assert result["overall"]["goal_sequences"] == {"turn:left:360": 2, "turn:left:180": 2}
    assert before == {str(path): digest(path) for path in tmp_path.rglob("*") if path.is_file()}


def test_numeric_plan_identity_keeps_booleans_distinct_and_preserves_source():
    source = {"enabled": True, "nested": [360.0, -0.0, 1.25, False, 1]}
    before = json.dumps(source)
    result = _numeric_identity(source)
    assert result["enabled"] is True and result["nested"][3] is False
    assert type(result["nested"][0]) is int
    assert result["nested"] == [360, 0, 1.25, False, 1]
    assert json.dumps(result) != json.dumps({"enabled": 1, "nested": [360, 0, 1.25, 0, 1]})
    assert json.dumps(source) == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_numeric_plan_identity_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="finite"):
        _numeric_identity({"angle_deg": value})


def test_audit_rejects_shared_scene_even_when_export_manifest_is_rehashed(tmp_path):
    for index in range(3):
        trace(tmp_path / "input" / str(index), episode=f"episode{index}", seed=index, family=f"family{index}")
    output = tmp_path / "output"
    export(tmp_path / "input", output, three_way=True)
    manifest = json.loads((output / "manifest.json").read_text())
    for split in ("train", "val", "test"):
        path = output / f"{split}.jsonl"
        row = json.loads(path.read_text())
        row["seed"] = 999
        path.write_text(json.dumps(row) + "\n")
        manifest["files_sha256"][split] = digest(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Split leakage"):
        audit(output)


def test_audit_still_rejects_non_png_or_changed_camera_frames(tmp_path):
    trace(tmp_path / "input/a", episode="a", seed=1, family="one")
    output = tmp_path / "output"
    export(tmp_path / "input", output)
    row = json.loads((output / "train.jsonl").read_text())
    Image.new("RGB", (8, 8), "red").save(output / row["images"]["front"], format="JPEG")
    with pytest.raises(ValueError, match="RGB PNG"):
        audit(output)


@pytest.mark.parametrize(
    "field,value", [("recording_source", "design"), ("training_ready", False), ("contract_sha256", "old")]
)
def test_rejects_draft_or_incompatible_provenance(tmp_path, field, value):
    metadata, events = trace(tmp_path / "episode")
    metadata[field] = value
    save(tmp_path / "episode", metadata, events)
    with pytest.raises(ValueError, match="runtime traces"):
        read_episode(tmp_path / "episode")


@pytest.mark.parametrize("problem", ["future_image", "history", "image_path", "image_hash", "revision", "duplicate"])
def test_rejects_input_and_media_corruption(tmp_path, problem):
    root = tmp_path / "episode"
    metadata, events = trace(root)
    event = events[0]
    if problem == "future_image":
        event["observation_time_s"] = 1
    elif problem == "history":
        event["request"]["history"] = [{"role": "assistant", "text": "虚构的过去。"}]
    elif problem == "image_path":
        event["media"]["front"] = "../outside.png"
    elif problem == "image_hash":
        event["media_sha256"] = {"front": "wrong", "near": "wrong"}
    elif problem == "revision":
        event["revision"] = 1
    else:
        events.append(copy.deepcopy(event))
    save(root, metadata, events)
    with pytest.raises(ValueError):
        read_episode(root)


@pytest.mark.parametrize("problem", ["future", "unverified", "no_start", "contradiction", "feedback_motion"])
def test_rejects_fabricated_or_mislabeled_completion(tmp_path, problem):
    root = tmp_path / "episode"
    metadata, events = trace(root, feedback=True)
    if problem == "future":
        events[-1]["result_ref"] = "future_result"
    elif problem == "unverified":
        events[2]["verified"] = False
    elif problem == "no_start":
        del events[1]
    elif problem == "contradiction":
        events[-1]["request"]["execution"]["last_result"]["reason"] = "Another result."
    else:
        events[-1]["response"].update(decision="replace", goals=[perform("nod")])
    save(root, metadata, events)
    with pytest.raises(ValueError):
        read_episode(root)


@pytest.mark.parametrize("status", ["failed", "cancelled", "blocked"])
def test_preserves_correct_failure_and_cancellation_feedback(tmp_path, status):
    root = tmp_path / "episode"
    metadata, events = trace(root, feedback=True)
    events[2].update(status=status, success=False, verified=False, reason="task did not finish")
    events[-1]["request"]["execution"]["last_result"] = {
        name: events[2][name] for name in ("goal", "status", "reason", "verified")
    }
    events[-1]["response"]["reply"] = "这次没有完成，任务已结束。"
    save(root, metadata, events)
    _, rows = read_episode(root)
    assert rows[-1]["request"]["execution"]["last_result"]["status"] == status
    assert "没有完成" in rows[-1]["target"]["reply"]
