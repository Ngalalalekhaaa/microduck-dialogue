"""Strict, local-only request decoding for the Microduck adapter service."""

import base64
import binascii
import io
import json

from experiments.microduck_tasks.skill_vlm import (
    PROPOSAL_SCHEMA,
    SYSTEM_PROMPT,
    _input_context,
    _input_state,
    _loads,
)

MAX_BODY_BYTES = 8 * 1024 * 1024


def decode_image(part):
    from PIL import Image

    if not isinstance(part, dict) or set(part) != {"type", "image_url"} or part["type"] != "image_url":
        raise ValueError("Each named camera must be followed by one image_url object.")
    value = part["image_url"]
    if not isinstance(value, dict) or set(value) != {"url"}:
        raise ValueError("image_url accepts only a base64 PNG url field.")
    url = value["url"]
    prefix = "data:image/png;base64,"
    if not isinstance(url, str) or not url.startswith(prefix):
        raise ValueError("Supply inline data:image/png;base64 images; files and network URLs are refused.")
    try:
        binary = base64.b64decode(url[len(prefix) :], validate=True)
        with Image.open(io.BytesIO(binary)) as image:
            if image.format != "PNG" or min(image.size) < 1 or max(image.size) > 4096:
                raise ValueError("Camera images must be PNG with dimensions 1..4096.")
            if image.width * image.height > 4_194_304:
                raise ValueError("Keep each camera image at or below 4 megapixels.")
            image.load()
            return image.convert("RGB")
    except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Camera PNG decoding failed; supply a valid base64 RGB image.") from exc


def decode_request(raw, model_alias, *, protocol="v1"):
    """Return the same processor message layout and image order used by training."""
    value = _loads(raw.decode("utf-8"))
    system_prompt, schema, schema_name = SYSTEM_PROMPT, PROPOSAL_SCHEMA, "microduck_skill"
    if protocol == "v2":
        from .plan_protocol import SCHEMA, SYSTEM, validate_input

        system_prompt, schema, schema_name = SYSTEM, SCHEMA, "skill_plan_v2"
    elif protocol != "v1":
        raise ValueError("Select protocol v1 or v2.")
    allowed = {"model", "messages", "temperature", "max_tokens", "stream", "response_format", "chat_template_kwargs"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Request accepts only model, messages, temperature, max_tokens, stream and format settings.")
    if value.get("model") != model_alias:
        raise ValueError(f"Use model={model_alias!r}; this server loads only the selected robot adapter.")
    if value.get("temperature", 0) != 0 or value.get("stream", False) is not False:
        raise ValueError("Only temperature=0 and stream=false are supported.")
    max_tokens = value.get("max_tokens", 256)
    maximum = 512 if protocol == "v2" else 256
    if type(max_tokens) is not int or not 1 <= max_tokens <= maximum:
        raise ValueError(f"max_tokens must be an integer in 1..{maximum}.")
    if value.get("chat_template_kwargs", {"enable_thinking": False}) != {"enable_thinking": False}:
        raise ValueError("Only chat_template_kwargs={enable_thinking: false} is supported.")
    if "response_format" in value:
        expected = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
        if value["response_format"] != expected:
            raise ValueError("Use the existing Microduck skill JSON schema; arbitrary schemas are not supported.")
    messages = value.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Supply exactly the trained system prompt and one user message.")
    if messages[0] != {"role": "system", "content": system_prompt}:
        raise ValueError("System prompt differs from training; use the current LocalVLMBackend.")
    user = messages[1]
    if not isinstance(user, dict) or set(user) != {"role", "content"} or user["role"] != "user":
        raise ValueError("Second message must contain only role=user and content.")
    content = user["content"]
    if not isinstance(content, list) or len(content) != 5:
        raise ValueError("Supply front label/image, near label/image and one instruction/state/context text.")
    images, canonical = [], []
    for offset, camera in ((0, "front"), (2, "near")):
        label = {"type": "text", "text": f"Current camera: {camera}"}
        if content[offset] != label:
            raise ValueError("Camera order must be front then near, each with its Current camera label.")
        images.append(decode_image(content[offset + 1]))
        canonical.extend([label, {"type": "image"}])
    text = content[4]
    if not isinstance(text, dict) or set(text) != {"type", "text"} or text["type"] != "text":
        raise ValueError("Last content item must be one text object.")
    if not isinstance(text["text"], str):
        raise ValueError("Instruction/state/context must be JSON encoded as text.")
    payload = _loads(text["text"])
    if not isinstance(payload, dict) or set(payload) != {"instruction", "state", "context"}:
        raise ValueError("Input JSON requires exactly instruction, state and context.")
    instruction = payload["instruction"]
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4000:
        raise ValueError("Instruction must be a nonempty string of at most 4000 characters.")
    if protocol == "v2":
        validate_input(payload["state"], payload["context"])
    else:
        payload = {
            "instruction": instruction,
            "state": _input_state(payload["state"]),
            "context": _input_context(payload["context"]),
        }
    canonical.append({"type": "text", "text": json.dumps(payload, ensure_ascii=False, allow_nan=False)})
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": canonical}], images, max_tokens
