"""Simulation-only Q1 continuation: generated replies with validated task plans."""

import copy
import json
import tempfile
import time
from pathlib import Path

from .q1r_prompt import messages_for, metadata, validate_prediction
from .toy_inputs import validate_input
from .toy_q1_backend import ToyQ1Backend
from .toy_selector_backend import _feedback, _response
from .toy_teacher import is_cancel, is_pause, is_resume
from .toy_vlm_json import parse_json_object


class Q1RGenerator:
    def __init__(self, model_dir, checkpoint):
        from .q0_model import load_model, weights_digest
        from .q1r_evaluate import generate

        checkpoint = Path(checkpoint).expanduser().resolve()
        identity = json.loads((checkpoint / "identity.json").read_text())
        if identity.get("prompt") != metadata():
            raise ValueError("Use a Q1R continuation checkpoint with the matching dialogue prompt.")
        results = json.loads((checkpoint.parent / "results.json").read_text())
        if not results.get("reload_pass") or results["selected_checkpoint"] != checkpoint.name:
            raise ValueError("Wait for Q1R validation and selected-checkpoint reload before interactive testing.")
        with tempfile.TemporaryDirectory(prefix="microduck-q1r-") as directory:
            self.model, self.processor, _ = load_model(model_dir, Path(directory))
        result = self.model.load_adapter(
            str(checkpoint), adapter_name="default", is_trainable=True, local_files_only=True
        )
        self.model.set_adapter("default")
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        actual = weights_digest(params)
        if result.missing_keys or result.unexpected_keys or actual != identity["weights_sha256"]:
            raise ValueError("Interactive Q1R weights differ from the selected checkpoint identity.")
        self.model.requires_grad_(False)
        self.model.eval()
        self._generate = generate
        self.metadata = dict(
            checkpoint=str(checkpoint),
            weights_sha256=actual,
            prompt=metadata(),
            images_passed_to_model=False,
            reply_source="model_generation",
        )

    def generate(self, request, images):
        text = self.processor.apply_chat_template(
            messages_for(request), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = self.processor(text=[text], images=None, return_tensors="pt", padding=False)["input_ids"][0].tolist()
        if len(ids) > 4096:
            raise ValueError("The full conversation exceeds 4096 input tokens; shorten it before retrying.")
        return self._generate(self.model, self.processor, dict(prefix_ids=ids))


class ToyQ1RBackend(ToyQ1Backend):
    name = "qwen35_q1r_dialogue_continuation"

    def __init__(self, model_dir, checkpoint):
        self.generator = Q1RGenerator(model_dir, checkpoint)
        self.metadata = dict(
            backend=self.name,
            model_inference=True,
            goal_source="q1r_lora_generation",
            reply_source="model_generation",
            natural_language_reply_trained=True,
            deployment_ready=False,
            simulation_only=True,
            inference_physics_frozen=True,
            images_passed_to_model=False,
            generator=self.generator.metadata,
        )
        self.last_trace = None

    def decide_with_images(self, request, images):
        validate_input(request)
        text = request["event"]["text"]
        if request["event"]["kind"] == "user" and any(check(text) for check in (is_cancel, is_pause, is_resume)):
            result = super().decide_with_images(request, images)
            self.last_trace.update(goal_source="host_control", reply_source="host_control")
            return result
        started = time.monotonic()
        self.last_trace = dict(
            backend=self.name,
            goal_source="q1r_lora_generation",
            reply_source="model_generation",
            decision_source="q1r_lora_generation",
            model_called=True,
            images_passed_to_model=False,
            raw_text=None,
            raw_response=None,
            raw_prediction=None,
        )
        try:
            raw, timing = self.generator.generate(request, images)
            self.last_trace.update(raw_text=raw, generation_timing=timing)
            prediction = parse_json_object(raw)
            self.last_trace.update(raw_response=copy.deepcopy(prediction), raw_prediction=copy.deepcopy(prediction))
            response = validate_prediction(prediction, request)
        except Exception as error:
            self.last_trace["rejected"] = dict(type=type(error).__name__, reason=str(error))
            self.last_trace["reply_source"] = "host_failure_notice"
            response = (
                _feedback(request)
                if request["event"]["kind"] == "feedback"
                else _response("这次输出未通过检查，我没有启动新动作，请换一种说法。", "decline")
            )
        return self._finish(response, request, started)
