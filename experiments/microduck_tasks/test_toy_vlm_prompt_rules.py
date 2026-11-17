"""Static rule transfer preserves input boundaries without evaluating a teacher answer."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from . import toy_teacher
from . import toy_vlm_prompt_rules as rules
from .toy_catalog import SELF_ACTIONS
from .toy_fixtures import active, history, move, request
from .toy_inputs import model_payload
from .toy_schema import response_schema


def _section(prompt, name):
    return json.loads(prompt.split(f"\n{name}=", 1)[1].split("\n", 1)[0])


def test_export_contains_every_teacher_alias_once_with_its_original_action():
    aliases = rules.action_aliases()
    assert set(aliases) == set(SELF_ACTIONS)
    assert all(aliases[action] for action in SELF_ACTIONS)
    pairs = [(alias, action) for action, phrases in aliases.items() for alias in phrases]
    assert len(pairs) == len(toy_teacher.ALIASES) == len(dict(pairs))
    assert dict(pairs) == toy_teacher.ALIASES
    assert "用右脚单脚站立" in aliases["flamingo_left"]
    assert "用左脚单脚站立" in aliases["flamingo_right"]
    system = rules.system_prompt()
    assert _section(system, "SELF_ACTION_ALIASES") == aliases
    assert _section(system, "SELF_ACTION_MEANINGS") == rules.ACTION_MEANINGS
    assert _section(system, "OUTPUT_JSON_SCHEMA") == response_schema()
    assert _section(system, "STATIC_LANGUAGE_RULES") == {
        name: getattr(toy_teacher, name) for name in ("NONAUTHORIZE", "VISUAL", "BLOCKED", "COLORS", "NUMBERS")
    }
    aliases["nod"].clear()
    assert rules.action_aliases()["nod"], "Exported lists must not mutate the teacher vocabulary."


def test_static_prompt_never_calls_teacher_or_includes_the_current_answer(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Building model context must not ask the rule teacher for this request's answer.")

    monkeypatch.setattr(toy_teacher.ToyTeacher, "decide", forbidden)
    first = request("CURRENT_REQUEST_SENTINEL", planned=True)
    second = request("DIFFERENT_REQUEST_SENTINEL", planned=True)
    a, b = rules.messages_for(first), rules.messages_for(second)
    assert [row["role"] for row in a] == ["system", "user"]
    assert a[0] == b[0]
    assert "CURRENT_REQUEST_SENTINEL" not in a[0]["content"]
    assert "DIFFERENT_REQUEST_SENTINEL" not in a[0]["content"]
    assert "CURRENT_REQUEST_SENTINEL" not in json.dumps(b)
    config = rules.prompt_config()
    assert config["static_teacher_rule_transfer"] is True
    for key in (
        "supervised_answers_in_context",
        "current_request_teacher_answer_in_context",
        "teacher_decide_called",
        "evaluation_answers_in_context",
        "task_specific_finetuning",
        "generated_output_repair",
        "claims_unseen_zero_shot",
    ):
        assert config[key] is False, key
    assert config["demonstration_examples_in_context"] == 0
    assert config["development_variant"] and config["prior_evaluation_errors_informed_prompt_design"]


def test_current_context_preserves_host_state_and_omits_only_opaque_revision():
    state = request('当前请求"点头"\n不重演历史。', planned=True)
    state["identity"]["name"] = "测试小鸭"
    state["history"] = history(("HISTORY_SENTINEL", "PUBLIC_REPLY_SENTINEL"))
    state["execution"] = active(move(continuous=True), paused=True)
    state["capabilities"]["self_actions"] = ["nod"]
    original = copy.deepcopy(state)
    messages = rules.messages_for(state)
    content = messages[1]["content"]
    assert [item["text"].strip() for item in content[:4] if item["type"] == "text"] == [
        "Current camera: front",
        "Current camera: near",
    ]
    assert sum(item["type"] == "image" for item in content) == 2
    payload = json.loads(content[-2]["text"].split("HOST_CONTEXT_JSON=", 1)[1])
    assert payload == model_payload(original)
    assert list(payload)[-1] == "event"
    assert "revision" not in payload["execution"]
    assert json.dumps(state["event"]["text"], ensure_ascii=False) in content[-1]["text"]
    assert state == original
    state["execution"]["revision"] += 100
    assert rules.messages_for(state) == messages


@pytest.mark.parametrize("field", ["target", "expected", "acceptable_goals", "teacher_response"])
def test_current_answer_and_evaluation_fields_are_rejected(field):
    state = request("点头", planned=True)
    state[field] = {"decision": "replace", "goals": []}
    with pytest.raises(ValueError, match="expected exactly"):
        rules.messages_for(state)


def test_provenance_hashes_actual_sources_and_discloses_static_mapping():
    config = rules.prompt_config()
    source = Path(rules.__file__)
    required = {
        "toy_teacher.py",
        "toy_protocol.py",
        "toy_schema.py",
        "toy_inputs.py",
        "toy_catalog.py",
        "action_catalog.py",
        "toy_vlm_prompt.py",
    }
    assert set(config["source_sha256"]) == required
    for name, digest in config["source_sha256"].items():
        assert digest == hashlib.sha256(source.with_name(name).read_bytes()).hexdigest()
    assert config["implementation_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert config["system_prompt_sha256"] == hashlib.sha256(rules.system_prompt().encode()).hexdigest()
    schema = json.dumps(response_schema(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    assert config["output_schema_sha256"] == hashlib.sha256(schema.encode()).hexdigest()
    assert config["alias_count"] == len(toy_teacher.ALIASES)
    assert config["static_action_alias_mappings_in_context"] is True
    json.dumps(config, allow_nan=False)


def test_absent_images_and_catalog_mismatch_are_explicit(monkeypatch):
    messages = rules.messages_for(request("看到了什么"), vision=False)
    assert all(item["type"] == "text" for item in messages[1]["content"])
    assert "没有相机图像" in messages[1]["content"][0]["text"]
    assert rules.prompt_config(vision=False)["cameras"] == []
    assert rules.prompt_config(vision=False)["vision"] is False
    monkeypatch.setattr(rules, "SELF_ACTIONS", (*SELF_ACTIONS, "unexplained_new_action"))
    with pytest.raises(ValueError, match="vocabulary"):
        rules.system_prompt()
