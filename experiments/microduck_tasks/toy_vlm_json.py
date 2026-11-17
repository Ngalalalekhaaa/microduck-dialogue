"""Strict JSON parsing for generated replies, without changing their contents."""

import json
import math
from typing import Any


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON number is not allowed: {value}.")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"JSON number exceeds finite numeric range: {value}.")
    return result


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse exactly one finite JSON object; schema and context checks happen later."""
    if not isinstance(raw, str):
        raise ValueError("Generated JSON must be text.")
    result = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)
    if not isinstance(result, dict):
        raise ValueError("Generated JSON must contain exactly one object.")
    return result
