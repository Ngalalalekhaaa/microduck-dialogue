"""Direct model-generated dialogue and plans, with strict, auditable acceptance."""

import copy
import time

from .setup_skill_head import digest
from .toy_inputs import validate_input
from .toy_protocol import validate_output
from .toy_selector_backend import _continuous, _feedback, _response
from .toy_teacher import is_cancel, is_pause
from .toy_vlm_json import parse_json_object


class ToyVLMBackend:
    """Untrained original VLM; generated replies and plans are never teacher-corrected."""

    name = "qwen35_toy_zero_shot"
    training_ready = False
    requires_lockstep = True

    def __init__(self, model_dir, *, prompt_variant="compact"):
        from .toy_vlm_generation import ToyVLMGenerator

        self.generator = ToyVLMGenerator(model_dir, prompt_variant=prompt_variant)
        self.metadata = dict(
            backend=self.name,
            model_inference=True,
            goal_source="vlm_generation",
            reply_source="vlm_generation",
            feedback_reply_source="trusted_execution_template",
            entity_source="RGB-D three-color ball detector",
            deployment_ready=False,
            simulation_only=True,
            inference_physics_frozen=True,
            implementation_sha256=digest(__file__),
            generator=copy.deepcopy(self.generator.metadata),
        )
        self.last_trace = None

    def _finish(self, response, request, started):
        response = validate_output(response, request)
        self.last_trace.update(response=copy.deepcopy(response), seconds=time.monotonic() - started)
        return response

    def decide_with_images(self, request, images):
        validate_input(request)
        started = time.monotonic()
        self.last_trace = dict(
            backend=self.name,
            goal_source="vlm_generation",
            raw_prediction=None,
            raw_response=None,
            raw_text=None,
            model_called=False,
            reply_source="template",
        )
        if request["event"]["kind"] == "feedback":
            self.last_trace["decision_source"] = "trusted_execution_feedback"
            return self._finish(_feedback(request), request, started)
        text = request["event"]["text"]
        if is_cancel(text):
            self.last_trace["decision_source"] = "host_stop"
            return self._finish(_response("收到，正在取消任务并停止运动。", "cancel"), request, started)
        if is_pause(text):
            self.last_trace["decision_source"] = "host_pause"
            if request["execution"]["state"] == "running" and _continuous(request):
                response = _response("收到，正在暂停当前任务并保留进度。", "pause")
            else:
                response = _response("本轮只支持暂停正在执行的持续行走；取消任务请说‘停下’。", "decline")
            return self._finish(response, request, started)
        self.last_trace["decision_source"] = "vlm_generation"
        try:
            self.last_trace["model_called"] = True
            raw, timing = self.generator.generate(request, images)
            self.last_trace.update(raw_text=raw, generation_timing=timing)
            response = parse_json_object(raw)
            self.last_trace["raw_response"] = copy.deepcopy(response)
            self.last_trace["raw_prediction"] = {
                key: copy.deepcopy(response[key]) for key in ("decision", "goals") if key in response
            }
            validate_output(response, request)
            if response["decision"] in ("pause", "resume") and not _continuous(request):
                raise ValueError("Pause/resume is only verified for continuous walking in this experiment.")
            self.last_trace["reply_source"] = "vlm_generation"
        except Exception as exc:
            self.last_trace["rejected"] = dict(type=type(exc).__name__, reason=str(exc))
            response = _response("这次模型输出未通过检查，我没有启动新动作，请换一种说法。", "decline")
        return self._finish(response, request, started)
