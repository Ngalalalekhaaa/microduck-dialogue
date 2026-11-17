"""Check training data boundaries without loading the large pretrained model."""

import json

import numpy as np
import pytest
from PIL import Image

from experiments.microduck_tasks.train_vlm import SYSTEM_PROMPT, encode, messages_for, read_records, validation_indices


@pytest.fixture
def dataset(tmp_path):
    for name in ("front.png", "near.png"):
        Image.new("RGB", (32, 24), "blue").save(tmp_path / name)
    row = {
        "images": {"front": "front.png", "near": "near.png"},
        "instruction": "踢红球",
        "state": {"yaw": 0.0, "stationary": True, "tilt_deg": 0.0, "sim_time_s": 0.0},
        "context": {"step": 0, "selected_color": None},
        "target": {
            "type": "skill",
            "skill": "turn_left",
            "seconds": 1.0,
            "target_color": "red",
            "reply": "转身寻找红球。",
        },
    }
    return tmp_path, row


def save(root, row):
    path = root / "train.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_training_input_is_original_instruction_and_whitelisted_observation(dataset):
    root, row = dataset
    row["context"]["task_id"] = "log-only-task-id"
    records = read_records(save(root, row))
    messages = messages_for(records[0])
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert len([part for part in messages[1]["content"] if part["type"] == "image"]) == 2
    payload = json.loads(messages[1]["content"][-1]["text"])
    assert set(payload) == {"instruction", "state", "context"}
    assert payload["instruction"] == "踢红球"
    assert "target" not in payload
    assert "task_id" not in payload["context"]


@pytest.mark.parametrize("mutation", ["outside_image", "oracle_state", "illegal_label", "wrong_prompt"])
def test_reject_invalid_training_samples(dataset, mutation):
    root, row = dataset
    if mutation == "outside_image":
        row["images"]["front"] = "../outside.png"
    elif mutation == "oracle_state":
        row["state"]["ball_world_position"] = [1, 2, 3]
    elif mutation == "illegal_label":
        row["target"]["skill"] = "arbitrary_joint_command"
    else:
        row["messages"] = [{"role": "system", "content": "outdated prompt"}]
    with pytest.raises(ValueError):
        read_records(save(root, row))


class TinyProcessor:
    """A character tokenizer exposes exact supervision boundaries in the test."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "<end>")

    def __call__(self, text, images, return_tensors, padding):
        assert len(images) == 2
        return {"input_ids": np.array([[ord(character) for character in text[0]]])}


def test_assistant_boundary_excludes_entire_input_without_truncation(dataset):
    root, row = dataset
    processor = TinyProcessor()
    batch, boundary = encode(row, root, processor, 20000)
    assistant = "".join(chr(token) for token in batch["input_ids"][0, boundary:])
    assert assistant == json.dumps(row["target"], ensure_ascii=False, allow_nan=False) + "<end>"
    with pytest.raises(ValueError, match="no truncation"):
        encode(row, root, processor, 10)


def test_validation_subset_keeps_rare_skill_and_reply_classes():
    records = [{"sample_id": str(index), "target": {"type": "skill", "skill": "move_forward"}} for index in range(50)]
    records.extend(
        [
            {"sample_id": "rare-kick", "target": {"type": "skill", "skill": "kick_left"}},
            {"sample_id": "answer", "target": {"type": "answer", "skill": None}},
        ]
    )
    selected = validation_indices(records, 3)
    categories = {records[index]["target"]["skill"] or records[index]["target"]["type"] for index in selected}
    assert categories == {"move_forward", "kick_left", "answer"}
    assert selected == validation_indices(records, 3)
