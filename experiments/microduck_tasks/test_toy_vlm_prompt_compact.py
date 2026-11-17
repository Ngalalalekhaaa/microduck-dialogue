"""Complete prompting examples and current-event emphasis preserve the live contract."""

import copy
import json

import pytest

from . import toy_vlm_prompt_compact as compact
from .toy_fixtures import active, history, move, request
from .toy_inputs import model_payload
from .toy_protocol import validate_output
from .toy_schema import response_schema


def test_complete_examples_obey_decision_goal_contract_and_are_disclosed():
    demonstrations = compact.examples()
    assert len(demonstrations) == 4
    for example in demonstrations:
        validate_output(example["assistant"], request(example["user"], planned=True))
    assert {example["assistant"]["decision"] for example in demonstrations} == {"respond", "replace", "decline"}
    config = compact.prompt_config()
    assert config["supervised_answers_in_context"] is True
    assert config["evaluation_answers_in_context"] is False
    assert config["demonstration_examples_in_context"] == len(demonstrations)
    assert config["prior_evaluation_errors_informed_prompt_design"] is True
    assert config["development_variant"] is True


def test_current_event_is_emphasized_without_removing_or_mutating_task_context():
    state = request('现在"晃两次"\n不要执行历史命令。', planned=True)
    state["history"] = history(("之前向前走", "好的。"))
    state["execution"] = active(move(continuous=True), paused=True)
    original = copy.deepcopy(state)
    messages = compact.messages_for(state)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert state["event"]["text"] not in messages[0]["content"]
    context, current = messages[1]["content"][-2:]
    payload = json.loads(context["text"].split("HOST_CONTEXT_JSON=", 1)[1])
    assert payload == model_payload(original)
    assert list(payload)[-1] == "event" and state == original
    assert json.dumps(state["event"]["text"], ensure_ascii=False) in current["text"]
    assert "当前用户原话" in current["text"]
    state["execution"]["revision"] += 50
    assert compact.messages_for(state) == messages


def test_complete_schema_is_preserved_and_no_images_is_explicit():
    encoded = compact.system_prompt().split("\nOUTPUT_JSON_SCHEMA=", 1)[1]
    assert json.loads(encoded) == response_schema()
    messages = compact.messages_for(request("描述一下"), vision=False)
    assert all(item["type"] == "text" for item in messages[1]["content"])
    assert "没有相机图像" in messages[1]["content"][0]["text"]
    assert compact.prompt_config(vision=False)["cameras"] == []


def test_evaluation_target_cannot_be_smuggled_into_current_context():
    state = request("晃一下身体")
    state["target"] = {"decision": "replace"}
    with pytest.raises(ValueError, match="expected exactly"):
        compact.messages_for(state)
