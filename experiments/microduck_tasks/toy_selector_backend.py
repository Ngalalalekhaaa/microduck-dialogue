"""Experimental learned goal plans with explicit templates and immutable model traces."""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .toy_inputs import validate_input
from .toy_protocol import VERSION, validate_output
from .toy_teacher import ToyTeacher, _goal_label, is_cancel, is_pause


def _response(reply, decision="respond", goals=()):
    return dict(schema_version=VERSION, reply=reply, decision=decision, goals=list(goals))


def _continuous(request):
    active = request["execution"]["active"]
    return bool(active and active["type"] == "move" and active["limit"]["kind"] == "until_cancelled")


def _feedback(request):
    result = request["execution"]["last_result"]
    if result is None:
        return _response("执行器尚未提供可核验的结果。")
    label = _goal_label(result["goal"])
    if result["status"] == "succeeded" and result["verified"]:
        return _response(f"{label}已经通过执行验收。")
    return _response(f"{label}未完成：{result['reason']}。")


def _reply(prediction, request):
    decision = prediction["decision"]
    if decision in ("replace", "append"):
        verb = "追加" if decision == "append" else "依次执行"
        return f"好的，我会{verb}：" + "，然后".join(_goal_label(g) for g in prediction["goals"]) + "。"
    if decision == "respond":
        # The teacher may supply words for a chat reply, never a motion plan or a corrected decision.
        chat = ToyTeacher().decide(request)
        if chat["decision"] == "respond" and not chat["goals"]:
            return chat["reply"]
        return "这次我没有识别到需要执行的动作，请换一种说法。"
    return {
        "clarify": "请再明确一下想执行的动作、目标和顺序。",
        "decline": "这次未能生成可执行的任务计划，请查看可用动作或简化指令。",
        "cancel": "收到，正在取消任务并停止运动。",
        "pause": "收到，正在暂停当前任务并保留进度。",
        "resume": "收到，恢复之前暂停的任务。",
    }[decision]


class ToySelectorBackend:
    """Frozen Qwen plus a learned plan head; language replies are not model generations."""

    name = "frozen_vlm_toy_selector"
    training_ready = False
    requires_lockstep = True

    def __init__(self, checkpoint, model_dir):
        from .toy_dataset import contract_digest
        from .toy_head_features import ToyFrozenFeatures, feature_config
        from .toy_selector_model import load_selector

        checkpoint = Path(checkpoint).expanduser().resolve()
        if checkpoint.is_dir():
            checkpoint /= "selector.pt"
        config = feature_config()
        self.selector = load_selector(checkpoint, feature_config=config, contract_sha256=contract_digest())
        self.features = ToyFrozenFeatures(model_dir)
        self.metadata = dict(
            backend=self.name,
            model_inference=True,
            goal_source="learned_selector",
            reply_source="template",
            deployment_ready=False,
            simulation_only=True,
            inference_physics_frozen=True,
            checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            feature_config=config,
            selector=copy.deepcopy(self.selector.metadata),
        )
        self.last_trace = None

    def _finish(self, response, request, started):
        response = validate_output(response, request)
        self.last_trace.update(
            response=copy.deepcopy(response), seconds=time.monotonic() - started, reply_source="template"
        )
        return response

    def decide_with_images(self, request, images):
        validate_input(request)
        started = time.monotonic()
        self.last_trace = dict(
            backend=self.name,
            goal_source="learned_selector",
            raw_prediction=None,
            model_called=False,
            feature_config=self.metadata["feature_config"],
            checkpoint_sha256=self.metadata["checkpoint_sha256"],
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
        self.last_trace["decision_source"] = "learned_selector"
        try:
            self.last_trace["model_called"] = True
            vector, timing = self.features.extract(request, images)
            self.last_trace["feature_timing"] = timing
            features = np.asarray(vector, dtype=np.float32)
            if features.shape != (2048,) or not np.isfinite(features).all():
                raise ValueError("Feature extractor must return one finite 2048-dimensional vector.")
            predictions = self.selector.predict(features[None, :])
            if len(predictions) != 1:
                raise ValueError("The plan head must return exactly one prediction for this observation.")
            prediction = predictions[0]
            # Preserve the actual prediction even when strict contract validation rejects it.
            try:
                json.dumps(prediction, ensure_ascii=False, allow_nan=False)
                self.last_trace["raw_prediction"] = copy.deepcopy(prediction)
            except (ValueError, TypeError):
                self.last_trace["raw_prediction_repr"] = repr(prediction)
                raise ValueError("The plan head returned a non-JSON or nonfinite prediction.") from None
            if not isinstance(prediction, dict) or set(prediction) != {"decision", "goals"}:
                raise ValueError("The plan head must return exactly decision and goals.")
            response = validate_output(_response("已收到模型输出。", **prediction), request)
            if prediction["decision"] in ("pause", "resume") and not _continuous(request):
                raise ValueError("Pause/resume is only verified for continuous walking in this experiment.")
            response["reply"] = _reply(prediction, request)
        except Exception as exc:
            self.last_trace["rejected"] = dict(type=type(exc).__name__, reason=str(exc))
            response = _response("这次技能预测未通过检查，我没有启动新动作，请换一种说法。", "decline")
        return self._finish(response, request, started)
