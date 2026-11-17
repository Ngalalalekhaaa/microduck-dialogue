"""Malformed model text is rejected, never repaired into a different action plan."""

import math

import pytest

from .toy_vlm_json import parse_json_object


def test_parser_preserves_fields_and_leaves_semantic_validation_to_backend():
    raw = ' \n {"decision":"unknown","goals":[{"seconds":true}],"reply":"你好","extra":null} \t '
    assert parse_json_object(raw) == {
        "decision": "unknown",
        "goals": [{"seconds": True}],
        "reply": "你好",
        "extra": None,
    }


def test_parser_preserves_finite_numbers_and_strings_containing_nonfinite_names():
    result = parse_json_object('{"x":[1,1.25,1e308,-0.0],"names":["NaN","Infinity"]}')
    assert result["x"] == [1, 1.25, 1e308, -0.0]
    assert type(result["x"][0]) is int
    assert math.copysign(1, result["x"][3]) == -1
    assert result["names"] == ["NaN", "Infinity"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"decision":"respond","decision":"replace"}',
        '{"goals":[{"type":"move","type":"kick"}]}',
        '{"a":0,"\\u0061":1}',
    ],
)
def test_duplicate_keys_are_rejected_at_every_depth(raw):
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        parse_json_object(raw)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_nonfinite_numbers_and_overflow_are_rejected_at_every_depth(value):
    with pytest.raises(ValueError, match="[Nn]onfinite|finite numeric range"):
        parse_json_object('{"goals":[{"seconds":' + value + "}]}")


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"decision":"respond"}\n```',
        'Here is the result: {"decision":"respond"}',
        '{"decision":"respond"}\nDone.',
        '{"decision":"respond"}{"decision":"replace"}',
        '{"decision":"respond",}',
        "{'decision':'respond'}",
        '{"decision":"respond"',
        '<think>reason</think>{"decision":"respond"}',
        "",
    ],
)
def test_invalid_json_is_not_repaired(raw):
    with pytest.raises(ValueError):
        parse_json_object(raw)


@pytest.mark.parametrize("raw", ["[]", "[{}]", "null", "true", "12", '"{}"'])
def test_top_level_must_be_one_object(raw):
    with pytest.raises(ValueError, match="exactly one object"):
        parse_json_object(raw)


@pytest.mark.parametrize("raw", [None, {}, b"{}"])
def test_nontext_input_is_rejected(raw):
    with pytest.raises(ValueError, match="must be text"):
        parse_json_object(raw)
