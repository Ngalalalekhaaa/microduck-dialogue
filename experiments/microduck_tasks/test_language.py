"""Language intent, negation and actual image/observation backend contracts."""

import math

import numpy as np
import pytest

from experiments.microduck_tasks.language import Intent, describe_view, intent_from_dict, parse_instruction
from experiments.microduck_tasks.vlm import RuleVLMBackend, VisionLanguageInput


@pytest.mark.parametrize("text", ["踢红球", "把红色那个踢走", "踢一下红色的球", "请帮我踢红球", "红球踢一下"])
def test_kick_paraphrases(text):
    assert parse_instruction(text) == Intent("kick", color="red")


@pytest.mark.parametrize(("color", "name"), [("red", "红"), ("yellow", "黄"), ("blue", "蓝")])
def test_three_colors(color, name):
    assert parse_instruction(f"请踢一下{name}色的球").color == color


@pytest.mark.parametrize(
    ("text", "direction"),
    [
        ("往前走", "forward"),
        ("向前挪一下", "forward"),
        ("前进", "forward"),
        ("往后走", "backward"),
        ("后退", "backward"),
        ("退后一点", "backward"),
        ("向左移动", "left"),
        ("往左挪一下", "left"),
        ("右移", "right"),
        ("朝右走", "right"),
    ],
)
def test_move_paraphrases(text, direction):
    assert parse_instruction(text) == Intent("move", direction=direction)


@pytest.mark.parametrize(("text", "direction"), [("向左转", "left"), ("右转一下", "right"), ("转向左边", "left")])
def test_turn_is_not_translation(text, direction):
    assert parse_instruction(text) == Intent("turn", direction=direction, seconds=0.6)


@pytest.mark.parametrize(
    "text",
    [
        "你现在视野中有什么",
        "能看到什么",
        "你看到了什么",
        "看看你看到了什么",
        "能看到哪些球",
        "眼前有什么",
        "描述一下眼前的东西",
    ],
)
def test_describe_paraphrases(text):
    assert parse_instruction(text).kind == "describe"


@pytest.mark.parametrize("text", ["停下", "别走了", "先停一停", "不要动", "停止", "站住"])
def test_stop_paraphrases(text):
    assert parse_instruction(text).kind == "stop"


@pytest.mark.parametrize(
    "text",
    ["别踢红球踢蓝球", "不要踢红色的球，踢蓝色的球", "别把红球踢走，踢蓝球", "红球别踢，踢蓝球", "不要红球，踢蓝球"],
)
def test_negative_target_is_not_selected(text):
    assert parse_instruction(text) == Intent("kick", color="blue")


@pytest.mark.parametrize(
    "text",
    ["不要踢红球", "别踢", "不许踢蓝球", "请不要踢红球", "不要踢红色的球了", "我不想踢红球", "不要帮我踢红球"],
)
def test_negative_command_never_kicks(text):
    assert parse_instruction(text).kind == "stop"


@pytest.mark.parametrize(
    "text",
    [
        "你能踢红球吗",
        "能不能踢蓝球",
        "是否可以往前走",
        "踢红球？",
        "红球还是蓝球",
        "踢红球和蓝球",
        "踢球",
        "踢它",
        "向左转然后往前走",
        "往前走踢红球",
        "如果看到红球就踢它",
        "向前走10米",
        "踢红球两次",
        "转一圈",
        "我想知道怎么踢红球",
        "刚才踢红球了",
        "唱首歌",
        "",
        None,
    ],
)
def test_ambiguous_or_unsupported_commands_do_not_move(text):
    assert parse_instruction(text).kind == "clarify"


def test_explicit_context_resolves_pronoun():
    assert parse_instruction("踢它", {"target_color": "yellow"}) == Intent("kick", color="yellow")
    assert parse_instruction("踢那个", {"visible_colors": ["red"]}) == Intent("kick", color="red")
    assert parse_instruction("踢那个", {"visible_colors": ["red", "blue"]}).kind == "clarify"


def test_description_uses_visible_measurements_and_deduplicates_cameras():
    detections = [
        {"color": "red", "center_body_m": [0.4, 0.1, 0.02], "camera": "front"},
        {"color": "red", "center_body_m": [0.4, 0.1, 0.02], "camera": "near"},
        {"color": "blue", "center_body_m": [0.3, -0.1, 0.02]},
        {"color": "yellow", "center_body_m": [0.3, 0, 0.02], "visible": False},
    ]
    reply = describe_view(detections)
    assert reply == "我现在看到左前方有红球，右前方有蓝球。"
    assert "没有检测到" in describe_view([])
    assert "没有检测到" in describe_view([{"color": "red", "center_body_m": [math.nan, 0, 0]}])


def request(instruction="踢红球", **changes):
    values = dict(
        instruction=instruction,
        images={"front": np.zeros((20, 30, 3), dtype=np.uint8)},
        observation={"detections": []},
        context={},
    )
    values.update(changes)
    return VisionLanguageInput(**values)


def test_rule_backend_requires_actual_rgb_and_observations():
    backend = RuleVLMBackend()
    assert backend.interpret(request()) == Intent("kick", color="red")
    assert backend.interpret(request(images={})).kind == "clarify"
    assert backend.interpret(request(images={"front": "some-image.png"})).kind == "clarify"
    assert backend.interpret(request(images={"front": np.zeros((10, 10))})).kind == "clarify"
    assert backend.interpret(request("你看到了什么", observation={})).kind == "clarify"
    assert backend.is_trained_vlm is False


def test_backend_description_does_not_read_hidden_world_or_context():
    reply = RuleVLMBackend().interpret(request("你看到了什么", context={"world_balls": [{"color": "red"}]})).reply
    assert "没有检测到" in reply


def test_intents_roundtrip_and_validate_model_output():
    intent = Intent("kick", color="blue")
    assert intent_from_dict(intent.to_dict()) == intent
    with pytest.raises(TypeError):
        intent_from_dict({"kind": "stop", "unexpected": "field"})
    for invalid in [
        {"kind": "execute_python"},
        {"kind": "kick"},
        {"kind": "kick", "color": "green"},
        {"kind": "move", "direction": "forward", "seconds": math.nan},
        {"kind": "move", "direction": "forward", "seconds": True},
        {"kind": "turn", "direction": "forward"},
    ]:
        with pytest.raises(ValueError):
            intent_from_dict(invalid)
