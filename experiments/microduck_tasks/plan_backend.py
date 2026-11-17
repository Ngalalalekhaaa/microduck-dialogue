"""V2 local HTTP client. Old v1 adapters must explicitly remain on the v1 backend."""

import base64
import io
import json
import time
import urllib.request

import numpy as np
from PIL import Image

from .plan_protocol import SCHEMA, SYSTEM, VERSION, parse_plan, validate_input
from .skill_vlm import LocalVLMBackend


class LocalPlanBackend(LocalVLMBackend):
    def __init__(self, endpoint, model, timeout=45):
        super().__init__(endpoint, model, timeout)
        if model in ("microduck-qlora", "microduck-onnx"):
            raise ValueError("The existing adapter was trained with v1. Use --backend vlm, or train a v2 adapter.")
        self.metadata["prompt_version"] = VERSION

    def _accept_model_metadata(self, value):
        if not isinstance(value, dict) or value.get("prompt_version") != VERSION:
            raise ValueError("This service was not trained for the v2 protocol.")
        return super()._accept_model_metadata(value)

    def decide(self, request):
        validate_input(request.state, request.context)
        started = time.monotonic()
        self.last_trace = {
            "model": self.model,
            "prompt_version": VERSION,
            "raw_response": None,
            "input_state": request.state,
            "input_context": request.context,
        }
        content = []
        if set(request.images) != {"front", "near"}:
            raise ValueError("Supply current robot front and near cameras.")
        for name in ("front", "near"):
            rgb = request.images[name]
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("Supply uint8 RGB images.")
            stream = io.BytesIO()
            Image.fromarray(rgb).save(stream, format="PNG")
            content.extend(
                [
                    {"type": "text", "text": f"Current camera: {name}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()},
                    },
                ]
            )
        content.append(
            {
                "type": "text",
                "text": json.dumps(
                    {"instruction": request.instruction, "state": request.state, "context": request.context},
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            }
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "skill_plan_v2", "strict": True, "schema": SCHEMA},
            },
        }
        http = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(http, timeout=self.timeout) as response:
                raw = response.read(131073)
            if len(raw) > 131072:
                raise ValueError("Plan response exceeds 128 KiB.")
            value = json.loads(raw)
            if "microduck_model_metadata" in value:
                self._accept_model_metadata(value["microduck_model_metadata"])
            text = value["choices"][0]["message"]["content"]
            self.last_trace["raw_response"] = text
            return parse_plan(text)
        finally:
            self.last_trace["latency_s"] = time.monotonic() - started
