"""A local vision-language model selects each bounded robot skill.

The HTTP server owns model loading. This module sends actual RGB images to a
loopback server, validates its JSON, and never executes or substitutes an action.
Protocol: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
"""

import base64
import io
import ipaddress
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PROMPT_VERSION = "microduck-skills-5"
SKILLS = (
    "stand",
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "kick_left",
    "kick_right",
)
COLORS = ("red", "yellow", "blue")
PROPOSAL_FIELDS = {"type", "skill", "seconds", "target_color", "reply"}
PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["skill", "answer", "clarify", "finish"]},
        "skill": {"enum": [*SKILLS, None]},
        "seconds": {"type": "number", "minimum": 0, "maximum": 1},
        "target_color": {"enum": [*COLORS, None]},
        "reply": {"type": "string", "maxLength": 1000},
    },
    "required": ["type", "skill", "seconds", "target_color", "reply"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = """You control an HD1910 Microduck SIMULATION by selecting ONE next skill.
Use the current named RGB camera images, the user's ORIGINAL instruction, allowed sensor state,
and previous execution feedback. Choose the skill yourself; no rule planner supplies a choice.
front looks forward/down; near is a virtual camera looking down/back at the feet. Both images
belong to this same instant. A missing ball may be outside the views; never invent its location.
State detections_rgbd, when supplied, are auxiliary color/depth measurements, NOT your detections
and NOT simulator object coordinates. With no detections, reason from RGB but acknowledge uncertainty.
Body axes: +X forward, +Y left, metres. yaw and turns are radians, positive to the left.
State sim_time_s is elapsed simulation time since this task began, not the episode clock.
Available skills: stand maintains balance; move_forward/backward translate along +/-X;
move_left/right translate along +/-Y; turn_left/right rotate yaw at about +/-1 rad/s.
Movement commands request about 0.2 m/s, but actual short-pulse displacement is not exact.
For stand and turns choose 0.02..1.00 seconds; translations 0.02..0.60 seconds.
All durations MUST be multiples of 0.02. Short pulses permit re-observation and correction.
kick_left/right execute an existing learned kick and require EXACTLY 0.50 seconds.
Each motion is followed by executor-controlled standing recovery before the next observation;
that recovery is not another model choice. Kicks require a stationary, aligned body.
Calibrated ball centres for kicking: left (X=0.0895,Y=0.0411), right (X=0.0881,Y=-0.0429),
with +/-0.012m tolerance on each axis. An execution guard may reject unsafe or unaligned kicks;
read its feedback and choose your next skill. A rejection does not mean the action happened.
Follow the requested ball color and do not substitute another ball. Do not claim that an
unseen object is absent from the whole world. Answer view questions from current images.
For a single movement request such as '往前走' or '后退一点', choose ONE bounded pulse;
after feedback confirms that pulse completed, return finish instead of repeating forever.
For a view-description question choose answer immediately, without moving or turning.
After a kick, inspect the new images before reporting its outcome; an out-of-view ball alone
does not prove a successful kick. Never claim physical success based only on issuing a command.
Return exactly one JSON object with all five keys: type, skill, seconds, target_color, reply.
type is skill, answer, clarify, or finish. For skill use one allowed skill, its valid duration,
and the task's target_color (red/yellow/blue, or null for unrelated movement).
For other types skill MUST be null and seconds MUST be 0. Use answer for a visual reply,
clarify when the instruction cannot be resolved, finish when you believe the task is complete.
finish is only your report; independent simulation evaluation decides physical success.
Reply in concise Chinese. No markdown, additional keys, action lists, code, or joint commands.

任务示例（仅说明接口语义，不是本次任务）：
用户“往前走一点”，尚未执行：{"type":"skill","skill":"move_forward","seconds":0.32,"target_color":null,"reply":"向前走一小段。"}
同一任务，上次 move_forward 已执行并站稳：\
{"type":"finish","skill":null,"seconds":0,"target_color":null,"reply":"已经向前走了一小段并停稳。"}
用户“向左转一下”，尚未执行：{"type":"skill","skill":"turn_left","seconds":0.6,"target_color":null,"reply":"向左转一下。"}
同一任务，上次 turn_left 已执行并站稳：\
{"type":"finish","skill":null,"seconds":0,"target_color":null,"reply":"已左转并停稳。"}
左转/右转改变朝向，对应 turn_left/turn_right；左移/右移才对应 move_left/move_right，转向后不需再横移。
用户“看看你眼前有什么”：直接 answer，按实际图像描述；不需要走路。
只走一步、退后一点、转一下等请求，收到执行完成反馈后即可结束，不能因为新画面仍有地板而重复运动。
没有要求找球或踢球时，target_color 必须为 null，不能凭空添加颜色目标。
踢球任务可以需要多次动作；一次靠近并不等于踢球完成。依据新观测继续选择，只有实际尝试踢后才报告结果。
目标颜色明确但球不在当前视野时，可以调用 turn_left/turn_right 转身寻找；看不到球时禁止踢球。
例如用户“踢红球”、当前两图没有红球，可以先 turn_left 1.0秒，然后看新图。
clarify 只用于用户目标不明确；“我需要先找到球”是你应执行的子任务，不是需要用户澄清的问题。
"""


@dataclass(frozen=True)
class SkillInput:
    instruction: str
    images: dict
    state: dict
    context: dict


@dataclass(frozen=True)
class SkillProposal:
    type: str
    skill: str | None
    seconds: float
    target_color: str | None
    reply: str

    def to_dict(self):
        return asdict(self)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _loads(raw):
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid_constant)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number.")
    return float(value)


def task_elapsed_s(current, started):
    """Remove episode clock offsets without exposing floating-point tail noise."""
    current = _number(current, "current simulation time")
    started = _number(started, "task start simulation time")
    if current < started:
        raise ValueError("Observation time cannot precede the current task start.")
    return round(current - started, 6)


def parse_proposal(raw: str) -> SkillProposal:
    """Accept one complete strict JSON proposal; never repair or infer a skill."""
    if not isinstance(raw, str) or len(raw) > 16384:
        raise ValueError("Return a JSON string no longer than 16384 characters.")
    value = _loads(raw)
    if not isinstance(value, dict) or set(value) != PROPOSAL_FIELDS:
        raise ValueError("Return exactly type, skill, seconds, target_color and reply.")
    kind, skill, color = value["type"], value["skill"], value["target_color"]
    if not isinstance(kind, str) or kind not in ("skill", "answer", "clarify", "finish"):
        raise ValueError("type must be skill, answer, clarify or finish.")
    if color is not None and (not isinstance(color, str) or color not in COLORS):
        raise ValueError("target_color must be red, yellow, blue or null.")
    if not isinstance(value["reply"], str) or len(value["reply"]) > 1000:
        raise ValueError("reply must be a string no longer than 1000 characters.")
    seconds = _number(value["seconds"], "seconds")
    if kind == "skill":
        if not isinstance(skill, str) or skill not in SKILLS:
            raise ValueError("Choose a registered Microduck skill.")
        if skill.startswith("kick_"):
            if seconds != 0.5:
                raise ValueError("A kick must last exactly 0.50 seconds.")
        else:
            upper = 0.6 if skill.startswith("move_") else 1.0
            if not 0.02 <= seconds <= upper:
                raise ValueError(f"{skill} duration must be 0.02..{upper:.2f} seconds.")
        if not math.isclose(seconds / 0.02, round(seconds / 0.02), rel_tol=0, abs_tol=1e-8):
            raise ValueError("Skill duration must be a multiple of 0.02 seconds.")
    elif skill is not None or seconds != 0:
        raise ValueError("Non-skill proposals require skill=null and seconds=0.")
    return SkillProposal(kind, skill, seconds, color, value["reply"])


def _keys(value, allowed, name, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"{name} accepts only {sorted(allowed)}; required: {sorted(required)}.")


def _text(value, name, limit=1000):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} must be a string of at most {limit} characters.")


def _proposal_dict(value):
    if value is not None:
        parse_proposal(json.dumps(value, allow_nan=False))


def _input_state(state):
    required = {"yaw", "stationary", "tilt_deg", "sim_time_s"}
    _keys(state, required | {"detections_rgbd"}, "state", required)
    for key in ("yaw", "tilt_deg", "sim_time_s"):
        _number(state[key], key)
    if state["tilt_deg"] < 0 or state["sim_time_s"] < 0 or not isinstance(state["stationary"], bool):
        raise ValueError("State requires nonnegative tilt/time and a boolean stationary flag.")
    if "detections_rgbd" in state:
        detections = state["detections_rgbd"]
        if not isinstance(detections, list) or len(detections) > 20:
            raise ValueError("detections_rgbd must be a list of at most 20 visual measurements.")
        for detection in detections:
            _keys(
                detection,
                {"color", "center_body_m", "camera", "pixel_uv", "pixels", "confidence", "fit_residual_m"},
                "RGB-D detection",
                {"color", "center_body_m"},
            )
            if detection["color"] not in COLORS:
                raise ValueError("RGB-D color must be red, yellow or blue.")
            for key, count in (("center_body_m", 3), ("pixel_uv", 2)):
                if key not in detection:
                    continue
                point = detection[key]
                if not isinstance(point, list) or len(point) != count:
                    raise ValueError(f"{key} must contain {count} finite measurements.")
                for coordinate in point:
                    _number(coordinate, key)
            for key in ("pixels", "confidence", "fit_residual_m"):
                if key in detection:
                    _number(detection[key], key)
            if "camera" in detection and detection["camera"] not in ("front", "near"):
                raise ValueError("RGB-D measurements must come from front or near.")
    return _loads(json.dumps(state, allow_nan=False))


def _input_context(context):
    _keys(context, {"previous_action", "previous_feedback", "history", "selected_color", "step", "task_id"}, "context")
    if "previous_action" in context:
        _proposal_dict(context["previous_action"])
    if "previous_feedback" in context:
        _text(context["previous_feedback"], "previous_feedback")
    if (
        "selected_color" in context
        and context["selected_color"] is not None
        and context["selected_color"] not in COLORS
    ):
        raise ValueError("selected_color must be red, yellow, blue or null.")
    if "step" in context and (type(context["step"]) is not int or context["step"] < 0):
        raise ValueError("step must be a nonnegative integer.")
    if "task_id" in context:
        _text(context["task_id"], "task_id", 100)
    if "history" in context:
        history = context["history"]
        if not isinstance(history, list) or len(history) > 6:
            raise ValueError("history must contain at most six action/feedback records.")
        for record in history:
            _keys(record, {"action", "feedback"}, "history record", {"action", "feedback"})
            _proposal_dict(record["action"])
            _text(record["feedback"], "history feedback")
    # Task identifiers encode collection order, not information the policy needs.
    return _loads(json.dumps({key: value for key, value in context.items() if key != "task_id"}, allow_nan=False))


def _local_endpoint(endpoint):
    if not isinstance(endpoint, str):
        raise ValueError("Set endpoint to a loopback HTTP chat-completions URL.")
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme not in ("http", "https")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/chat/completions"
    ):
        raise ValueError("Use http(s)://127.0.0.1:PORT/v1/chat/completions with no credentials or query.")
    host = parsed.hostname
    host = "127.0.0.1" if host == "localhost" else host
    try:
        address = ipaddress.ip_address(host)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Only numeric loopback addresses or localhost are allowed.") from exc
    if not address.is_loopback:
        raise ValueError("Camera images may only be sent to a loopback VLM server.")
    authority = f"[{address}]" if address.version == 6 else str(address)
    if port is not None:
        authority += f":{port}"
    return urllib.parse.urlunsplit((parsed.scheme, authority, parsed.path, "", ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("VLM HTTP redirects are refused; configure the direct loopback endpoint.")


class LocalVLMBackend:
    """Call a local VLM and retain the serving process's model provenance."""

    name = "local_pretrained_vlm"
    is_trained_vlm = True

    def __init__(self, endpoint="http://127.0.0.1:8081/v1/chat/completions", model="microduck-vlm", timeout=45):
        self.endpoint = _local_endpoint(endpoint)
        _text(model, "model", 200)
        if not model.strip():
            raise ValueError("Provide the local server's model alias.")
        self.model = model
        self.timeout = _number(timeout, "timeout")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive seconds.")
        known_pretrained = self.endpoint == "http://127.0.0.1:8081/v1/chat/completions" and model == "microduck-vlm"
        self.name = "local_pretrained_vlm" if known_pretrained else "local_vlm"
        self.metadata = {
            "backend": self.name,
            "model": model,
            "endpoint": self.endpoint,
            "is_trained_vlm": True,
            "robot_finetuned": False if known_pretrained else None,
            "training": "configured pretrained GGUF"
            if known_pretrained
            else "server model provenance not yet reported",
            "prompt_version": PROMPT_VERSION,
            "timeout_s": self.timeout,
        }
        self.last_trace = {}
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _accept_model_metadata(self, value):
        fields = {"model", "robot_finetuned", "adapter", "adapter_sha256", "base", "prompt_version", "generation_mode"}
        if not isinstance(value, dict) or not fields.issubset(value):
            raise ValueError(
                "Server model metadata requires model, training identity, paths, hash and generation mode."
            )
        for key in fields - {"robot_finetuned"}:
            _text(value[key], f"server metadata {key}", 4096)
            if not value[key].strip():
                raise ValueError(f"Server metadata {key} cannot be empty.")
        if not isinstance(value["robot_finetuned"], bool):
            raise ValueError("Server metadata robot_finetuned must be boolean.")
        if value["model"] != self.model:
            raise ValueError("Server metadata model does not match the requested model alias.")
        if not all(Path(value[key]).is_absolute() for key in ("adapter", "base")):
            raise ValueError("Server metadata adapter and base paths must be absolute.")
        checksum = value["adapter_sha256"]
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise ValueError("Server metadata adapter_sha256 must be a lowercase SHA-256 digest.")
        optional = {
            "inference_format",
            "onnx_model",
            "onnx_manifest_sha256",
            "decoder",
            "decoder_quantization",
            "execution_provider",
            "vision_execution_provider",
            "kv_cache_device",
            "runtime",
        }
        for key in optional.intersection(value):
            _text(value[key], f"server metadata {key}", 4096)
        kept = fields | optional.intersection(value)
        identity = _loads(json.dumps({key: value[key] for key in sorted(kept)}, allow_nan=False))
        self.name = "local_finetuned_vlm" if identity["robot_finetuned"] else "local_pretrained_vlm"
        self.metadata.update(
            backend=self.name,
            robot_finetuned=identity["robot_finetuned"],
            training="server-reported robot finetuning"
            if identity["robot_finetuned"]
            else "server-reported pretrained model",
            server_metadata=identity,
        )
        return identity

    def refresh_model_metadata(self):
        """Read optional local service provenance before recording an episode."""
        parsed = urllib.parse.urlsplit(self.endpoint)
        health = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))
        request = urllib.request.Request(health, headers={"Accept": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read(65537)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Local VLM health request failed; start/check {health}: {exc}") from exc
        if len(body) > 65536:
            raise ValueError("Local VLM health response exceeds the 64 KB protocol limit.")
        result = _loads(body.decode("utf-8"))
        value = result.get("microduck_model_metadata", result) if isinstance(result, dict) else result
        return self._accept_model_metadata(value)

    def decide(self, request: SkillInput) -> SkillProposal:
        """Send current RGB and allowed inputs; failures propagate without rule fallback."""
        started = time.monotonic()
        self.last_trace = {
            "raw_response": None,
            "latency_s": 0.0,
            "model": self.model,
            "endpoint": self.endpoint,
            "prompt_version": PROMPT_VERSION,
            "input_state": None,
            "input_context": None,
        }
        try:
            if not isinstance(request, SkillInput):
                raise TypeError("Pass a SkillInput with current RGB images and whitelisted state.")
            _text(request.instruction, "instruction", 4000)
            if not request.instruction.strip():
                raise ValueError("Provide a nonempty instruction.")
            state, context = _input_state(request.state), _input_context(request.context)
            self.last_trace.update(input_state=state, input_context=context)
            if not isinstance(request.images, dict) or not request.images or set(request.images) - {"front", "near"}:
                raise ValueError("Supply current front and/or near RGB images, without external views.")
            content = []
            for camera in ("front", "near"):
                if camera not in request.images:
                    continue
                rgb = request.images[camera]
                if (
                    not isinstance(rgb, np.ndarray)
                    or rgb.dtype != np.uint8
                    or rgb.ndim != 3
                    or rgb.shape[2] != 3
                    or min(rgb.shape[:2]) < 1
                    or max(rgb.shape[:2]) > 4096
                ):
                    raise ValueError("Camera images must be uint8 H×W×3 RGB, with dimensions 1..4096.")
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format="PNG")
                url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                content.extend(
                    [
                        {"type": "text", "text": f"Current camera: {camera}"},
                        {"type": "image_url", "image_url": {"url": url}},
                    ]
                )
            content.append(
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "instruction": request.instruction,
                            "state": state,
                            "context": context,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                }
            )
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
                "max_tokens": 256,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "microduck_skill",
                        "strict": True,
                        "schema": PROPOSAL_SCHEMA,
                    },
                },
                "chat_template_kwargs": {"enable_thinking": False},
            }
            http_request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            with self._opener.open(http_request, timeout=self.timeout) as response:
                body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise ValueError("Local VLM response exceeds the 2 MB protocol limit.")
            result = _loads(body.decode("utf-8"))
            if isinstance(result, dict):
                if "microduck_model_metadata" in result:
                    self.last_trace["server_metadata"] = self._accept_model_metadata(result["microduck_model_metadata"])
                if isinstance(result.get("model"), str):
                    self.last_trace["server_model"] = result["model"]
                if isinstance(result.get("usage"), dict):
                    # Preserve server accounting separately from our requested alias.
                    self.last_trace["usage"] = _loads(json.dumps(result["usage"], allow_nan=False))
            choices = result.get("choices") if isinstance(result, dict) else None
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ValueError("Local VLM must return exactly one chat-completion choice.")
            choice = choices[0]
            message = choice.get("message")
            raw = message.get("content") if isinstance(message, dict) else None
            self.last_trace["raw_response"] = raw
            if choice.get("finish_reason") not in ("stop", None):
                raise ValueError("Local VLM response was truncated or did not finish normally.")
            return parse_proposal(raw)
        except (urllib.error.URLError, OSError) as exc:
            self.last_trace["error"] = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"Local VLM request failed; start/check {self.endpoint}: {exc}") from exc
        except Exception as exc:
            self.last_trace["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.last_trace["latency_s"] = time.monotonic() - started
