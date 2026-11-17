"""Verified Q1 LoRA plans for live simulation; every spoken reply is a host template."""

import copy
import json
import tempfile
import time
from pathlib import Path

from .ablation_prompt import messages_for
from .ablation_prompt import metadata as prompt_metadata
from .setup_skill_head import digest
from .toy_inputs import validate_input
from .toy_protocol import validate_output
from .toy_selector_backend import _feedback, _response
from .toy_teacher import _goal_label, is_cancel, is_pause, is_resume
from .toy_vlm_json import parse_json_object


def plan_reply(plan):
    """Describe the accepted plan without claiming Q1 generated conversational words."""
    decision = plan["decision"]
    if decision in ("replace", "append"):
        verb = "追加" if decision == "append" else "依次执行"
        return f"好的，我会{verb}：" + "，然后".join(_goal_label(goal) for goal in plan["goals"]) + "。"
    return {
        "respond": "这句话被识别为聊天；当前 Q1 只生成任务计划，尚未训练自然语言回答。",
        "clarify": "请再说明想执行的动作、目标和顺序。",
        "decline": "这次没有生成可执行的任务计划，请查看可用动作或换一种说法。",
        "cancel": "收到，正在取消任务并停止运动。",
        "pause": "收到，正在暂停当前任务并保留剩余任务。",
        "resume": "收到，恢复之前暂停的任务。",
    }[decision]


class Q1PlanGenerator:
    """Reuse the audited Q1 reload and greedy decoder, with fresh input-only P6 tokens."""

    def __init__(self, model_dir, checkpoint):
        from .q0_evaluate import generate
        from .q0_model import load_model, weights_digest

        checkpoint = Path(checkpoint).expanduser().resolve()
        identity = json.loads((checkpoint / "identity.json").read_text())
        if identity.get("step") != 100:
            raise ValueError("Use the selected Q1 checkpoint_100 with its saved identity.json.")
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="microduck-q1-online-") as directory:
            self.model, self.processor, loaded = load_model(model_dir, Path(directory))
        self.model.load_adapter(str(checkpoint), adapter_name="online", is_trainable=False)
        self.model.set_adapter("online")
        parameters = dict(self.model.named_parameters())
        restored = {name: parameters[name.replace(".default.", ".online.")] for name in loaded["trainable_names"]}
        actual_weights = weights_digest(restored)
        if actual_weights != identity["weights_sha256"]:
            raise ValueError("Loaded Q1 adapter differs from checkpoint identity; restore the selected checkpoint.")
        self.model.requires_grad_(False)
        self.model.eval()
        self._generate = generate
        self.max_input_tokens = 4096
        self.metadata = dict(
            checkpoint=str(checkpoint),
            selected_step=identity["step"],
            weights_sha256=actual_weights,
            adapter_file_sha256=digest(checkpoint / "adapter_model.safetensors"),
            model_dir=str(Path(model_dir).expanduser().resolve()),
            input_view="P6",
            images_passed_to_model=False,
            prompt=prompt_metadata("P6"),
            trained_max_goals=6,
            current_capabilities_from_request=True,
            max_input_tokens=self.max_input_tokens,
            generation="audited q0_evaluate.generate; greedy; max_new_tokens=768; max_time=90s",
            loader_seconds=time.monotonic() - started,
            checkpoint_identity_verified=True,
            training_ready=False,
        )

    def encode_request(self, request):
        text = self.processor.apply_chat_template(
            messages_for(request, "P6"), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        batch = self.processor(text=[text], images=None, return_tensors="pt", padding=False)
        ids = batch["input_ids"][0].tolist()
        if len(ids) > self.max_input_tokens:
            raise ValueError(
                f"Input has {len(ids)} tokens; limit is {self.max_input_tokens}. Shorten the conversation."
            )
        if int(batch["attention_mask"].sum()) != len(ids):
            raise ValueError("Q1 requires a complete unpadded request; no truncation or repair is allowed.")
        return dict(prefix_ids=ids)

    def generate(self, request, images):
        return self._generate(self.model, self.processor, self.encode_request(request))


class ToyQ1Backend:
    """Real selected Q1 checkpoint; template responses never supply or correct model plans."""

    name = "qwen35_q1_checkpoint100"
    training_ready = False
    requires_lockstep = True

    def __init__(self, model_dir, checkpoint):
        self.generator = Q1PlanGenerator(model_dir, checkpoint)
        self.metadata = dict(
            backend=self.name,
            model_inference=True,
            goal_source="q1_lora_generation",
            reply_source="template",
            natural_language_reply_trained=False,
            deployment_ready=False,
            simulation_only=True,
            inference_physics_frozen=True,
            images_passed_to_model=False,
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
            goal_source="q1_lora_generation",
            reply_source="template",
            raw_text=None,
            raw_response=None,
            raw_prediction=None,
            model_called=False,
            images_passed_to_model=False,
        )
        if request["event"]["kind"] == "feedback":
            self.last_trace["decision_source"] = "trusted_execution_feedback"
            return self._finish(_feedback(request), request, started)
        if is_cancel(request["event"]["text"]):
            self.last_trace["decision_source"] = "host_stop"
            return self._finish(_response(plan_reply(dict(decision="cancel")), "cancel"), request, started)
        if is_pause(request["event"]["text"]):
            self.last_trace["decision_source"] = "host_pause"
            if request["execution"]["state"] == "running":
                response = _response("收到，任务将在安全动作边界暂停，并保留剩余目标。", "pause")
            else:
                response = _response("当前没有正在执行的任务可以暂停。")
            return self._finish(response, request, started)
        if is_resume(request["event"]["text"]):
            self.last_trace["decision_source"] = "host_resume"
            if request["execution"]["state"] == "paused":
                response = _response(plan_reply(dict(decision="resume")), "resume")
            else:
                response = _response("当前没有暂停的任务；请说明新的任务。", "clarify")
            return self._finish(response, request, started)
        self.last_trace["decision_source"] = "q1_lora_generation"
        try:
            self.last_trace["model_called"] = True
            raw, timing = self.generator.generate(request, images)
            self.last_trace.update(raw_text=raw, generation_timing=timing)
            plan = parse_json_object(raw)
            self.last_trace.update(raw_response=copy.deepcopy(plan), raw_prediction=copy.deepcopy(plan))
            if set(plan) != {"decision", "goals"}:
                raise ValueError(
                    "Q1 must output exactly decision and goals; natural-language reply is a host template."
                )
            response = _response("计划校验中。", plan["decision"], plan["goals"])
            validate_output(response, request)
            response["reply"] = plan_reply(plan)
            validate_output(response, request)
        except Exception as exc:
            self.last_trace["rejected"] = dict(type=type(exc).__name__, reason=str(exc))
            response = _response("这次 Q1 输出未通过检查，我没有启动新动作，请换一种说法。", "decline")
        return self._finish(response, request, started)
