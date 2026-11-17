"""The selected VLM chooses every skill; execution guards never choose a substitute."""

import copy
import json
import math
import queue
import threading
import time

from .language import parse_instruction
from .skill_vlm import SkillInput, _input_context, parse_proposal, task_elapsed_s
from .tasks import TaskSession, evaluate_kick

EMPTY_EXECUTION = {"started": False, "completed": False, "start_time_s": None, "end_time_s": None}


def guard_skill(proposal, observation):
    """Accept or reject a proposal using measured state, without selecting motion."""
    if proposal.type != "skill":
        return {"accepted": True, "reason": "no_motion"}
    tilt = observation["tilt_deg"]
    if not math.isfinite(tilt) or tilt > 25:
        return {"accepted": False, "reason": "body_not_upright"}
    if not proposal.skill.startswith("kick_"):
        return {"accepted": True, "reason": "bounded_skill"}
    if not observation["stationary"]:
        return {"accepted": False, "reason": "kick_requires_stationary_body"}
    target = next((d for d in observation["detections"] if d["color"] == proposal.target_color), None)
    if target is None:
        return {"accepted": False, "reason": "kick_target_not_visible_in_rgbd"}
    point = target["center_body_m"]
    goal = (0.0895, 0.0411) if proposal.skill == "kick_left" else (0.0881, -0.0429)
    if len(point) != 3 or not all(math.isfinite(v) for v in point):
        return {"accepted": False, "reason": "invalid_rgbd_position"}
    if any(abs(point[i] - goal[i]) > 0.012 for i in (0, 1)):
        return {"accepted": False, "reason": "kick_target_outside_foot_region"}
    return {"accepted": True, "reason": "visible_aligned_target_and_stationary_body"}


class SkillTaskSession(TaskSession):
    """Independent model loop; no KickPlanner or lexical intent planner is called."""

    trace_field = "model_trace"

    def __init__(
        self,
        runtime,
        recorder,
        backend,
        *,
        max_steps=40,
        timeout=45,
        rgbd_assist=True,
        expected_color=None,
        verbose=True,
    ):
        super().__init__(runtime, recorder, backend, verbose)
        if not 1 <= max_steps <= 120 or not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError("Use 1..120 VLM steps and a finite 0..120 second inference timeout.")
        self.max_steps, self.timeout = max_steps, timeout
        self.rgbd_assist, self.expected_color = rgbd_assist, expected_color
        self._pending = None

    @property
    def controller_source(self):
        metadata = getattr(self.backend, "metadata", {})
        return "finetuned_vlm" if metadata.get("robot_finetuned") is True else "pretrained_vlm"

    def event(
        self,
        observation,
        phase,
        decision=None,
        execution=None,
        done=False,
        status="running",
        reply="",
        *,
        source=None,
        trace=None,
    ):
        self.event_seq += 1
        self.recorder.append_event(
            {
                "event_id": f"event_{self.event_seq:05d}",
                "task_id": f"task_{self.task_seq:03d}",
                "instruction": self.instruction,
                "intent": {"kind": "vlm_skill_control"},
                "context": copy.deepcopy(self.task_context),
                "observation": observation,
                "phase": phase,
                "decision": decision,
                "execution": execution or dict(EMPTY_EXECUTION),
                "done": done,
                "status": status,
                "reply": reply,
                "controller_source": self.controller_source if source is None else source,
                self.trace_field: trace or {},
            }
        )

    def finish(self, observation, status, reply, evaluation=None):
        self.event(observation, "done", done=True, status=status, reply=reply)
        result = {
            "task_id": f"task_{self.task_seq:03d}",
            "instruction": self.instruction,
            "status": status,
            "reply": reply,
            "evaluation": evaluation,
            "physical_task_verified": status == "success",
            "controller_source": self.controller_source,
        }
        self.results.append(result)
        print(reply, flush=True)
        return result

    def _infer(self, request):
        started = time.monotonic()
        self.last_trace = {
            "raw_response": None,
            "latency_s": 0.0,
            "model": getattr(self.backend, "model", getattr(self.backend, "name", "test_backend")),
            "input_state": copy.deepcopy(request.state),
            "input_context": copy.deepcopy(request.context),
            "request_status": "pending",
        }
        # An abandoned HTTP call can finish, but its private result queue has no
        # path to the executor. Never share mutable backend traces concurrently.
        if self._pending is not None and self._pending.is_alive():
            self.last_trace["request_status"] = "previous_request_still_pending"
            raise RuntimeError("Previous inference is still ending; retry after its HTTP timeout.")
        result = queue.Queue(maxsize=1)

        def predict():
            try:
                proposal = self.backend.decide(request)
                result.put((proposal, copy.deepcopy(self.backend.last_trace), None))
            except Exception as exc:
                result.put((None, copy.deepcopy(self.backend.last_trace), exc))

        self._pending = threading.Thread(target=predict, name="microduck-vlm-http", daemon=True)
        self._pending.start()
        deadline = time.monotonic() + self.timeout
        while not self.runtime.cancel.is_set():
            if self.runtime.failure:
                raise RuntimeError(self.runtime.failure)
            if time.monotonic() >= deadline:
                self.last_trace.update(request_status="timeout", latency_s=time.monotonic() - started)
                raise TimeoutError("VLM inference deadline exceeded; its late response will not execute.")
            try:
                proposal, trace, error = result.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.runtime.cancel.is_set():
                break
            if error is not None:
                self.last_trace = trace
                raise error
            return proposal, trace
        self.last_trace.update(request_status="cancelled", latency_s=time.monotonic() - started)
        return None, {}

    def _execute(self, observation, skill, seconds, *, source=None, trace=None):
        source = self.controller_source if source is None else source
        execution = dict(EMPTY_EXECUTION)
        decision = {"skill": skill, "seconds": seconds, "target_color": self.context.get("selected_color")}
        completed = False
        try:
            if not self.runtime.cancel.is_set():
                completed = self.runtime.execute(skill, seconds)
                execution.update(self.runtime.last_execution)
        except Exception:
            execution.update(self.runtime.last_execution)
            raise
        finally:
            self.event(
                observation,
                "recover" if source == "executor" else "execute",
                decision,
                execution,
                source=source,
                trace=trace,
            )
        return completed

    def _state(self, observation):
        state = {key: observation[key] for key in ("yaw", "stationary", "tilt_deg")}
        state["sim_time_s"] = task_elapsed_s(observation["time_s"], self.task_start_time_s)
        if self.rgbd_assist:
            state["detections_rgbd"] = observation["detections"]
        return state

    def run(self, instruction):
        self.runtime.cancel.clear()
        self.task_seq += 1
        self.instruction = instruction
        self.task_context = {
            **self.context,
            "previous_action": None,
            "previous_feedback": "",
            "history": [],
            "step": 0,
            "task_id": f"task_{self.task_seq:03d}",
        }
        self.last_trace = {}
        observation, images = self.observe()
        self.task_start_time_s = observation["time_s"]
        # Emergency stop is an operator command, never delayed behind inference.
        if parse_instruction(instruction).kind == "stop":
            self.runtime.request_stop()
            return self.finish(observation, "cancelled", "已停止，保持站立。")
        contact_start, row_start = len(self.runtime.contacts), len(self.runtime.rows)
        evaluations, rejected = [], 0
        started = time.monotonic()
        try:
            for step in range(self.max_steps):
                if self.runtime.cancel.is_set():
                    break
                if time.monotonic() - started > 600:
                    return self.finish(observation, "timeout", "任务达到 10 分钟上限，已停止。")
                self.task_context["step"] = step
                if self.verbose:
                    print(f"  VLM 第 {step + 1} 次看图决策…", flush=True)
                proposal, trace = self._infer(
                    SkillInput(instruction, images, self._state(observation), _input_context(self.task_context))
                )
                if proposal is None:
                    self.event(observation, "inference_cancelled", trace=self.last_trace)
                    break
                self.last_trace = trace
                # Validate even injected/test backends. No clipping or guessed action.
                proposal = parse_proposal(json.dumps(proposal.to_dict(), allow_nan=False))
                fresh, _ = self.observe()
                guard = guard_skill(proposal, fresh)
                trace.update(
                    model_proposal=proposal.to_dict(), guard_result=guard, guard_observation_frame=fresh["frame_id"]
                )
                self.last_trace = trace
                if self.runtime.cancel.is_set():
                    break
                if not guard["accepted"]:
                    self.event(observation, "guard_rejected", trace=trace)
                    rejected += 1
                    feedback = guard["reason"]
                    if rejected >= 3:
                        return self.finish(fresh, "guard_rejected", "连续三次动作未通过执行检查，已停止。")
                elif proposal.type != "skill":
                    self.event(observation, "model_response", trace=trace, reply=proposal.reply)
                    if proposal.type == "clarify":
                        return self.finish(fresh, "clarification", proposal.reply)
                    evaluation = {
                        "expected_color": self.expected_color,
                        "kicks": evaluations,
                        "source": "independent_test_target_not_model_prediction",
                    }
                    if self.expected_color:
                        # The acceptance case asks for one kick. Recheck the full
                        # trace at completion, including motion after the kick.
                        evaluations = [
                            evaluate_kick(
                                self.expected_color,
                                e["foot"],
                                e["before"],
                                self.runtime.truth(),
                                e["start"],
                                self.runtime.contacts[contact_start:],
                                self.runtime.rows[row_start:],
                            )
                            for e in evaluations
                        ]
                        evaluation["kicks"] = evaluations
                        passed = len(evaluations) == 1 and evaluations[0]["success"]
                        status = "success" if passed else "verification_failed"
                        evaluation["success"] = passed
                    else:
                        status = "answered" if proposal.type == "answer" else "model_reported"
                    reply = proposal.reply
                    if status == "verification_failed":
                        reply = f"模型报告：{reply}；独立仿真验收未通过。"
                    elif status == "success":
                        reply = f"{reply}（独立仿真验收通过。）"
                    elif status == "model_reported":
                        reply = f"{reply}（这是模型报告，尚未独立验收。）"
                    return self.finish(fresh, status, reply, evaluation)
                else:
                    rejected = 0
                    if proposal.target_color:
                        self.context["selected_color"] = proposal.target_color
                    kick = proposal.skill.startswith("kick_")
                    if kick and self.expected_color:
                        before = self.runtime.truth()  # Evaluation only; never sent to VLM.
                    if not self._execute(observation, proposal.skill, proposal.seconds, trace=trace):
                        break
                    kick_start = self.runtime.last_execution["start_time_s"] if kick else None
                    if self.verbose:
                        print(f"  VLM → {proposal.skill} {proposal.seconds:.2f}s", flush=True)
                    settled, _ = self.observe()
                    if proposal.skill != "stand" and not self._execute(
                        settled, "stand", 3.0 if kick else 1.0, source="executor"
                    ):
                        break
                    if kick and self.expected_color:
                        evaluations.append({"foot": proposal.skill[5:], "before": before, "start": kick_start})
                    feedback = (
                        f"executed_and_recovered: 已实际执行 {proposal.skill} {proposal.seconds:.2f} 秒并恢复站立。"
                        "若原指令只要求走/转一下，单次动作已完成，应结束该任务；"
                        "若原指令要求踢球，则根据新画面继续判断，执行过动作不代表踢球成功。"
                    )
                self.task_context["previous_action"] = proposal.to_dict()
                self.task_context["previous_feedback"] = feedback
                if "selected_color" in self.context:
                    self.task_context["selected_color"] = self.context["selected_color"]
                self.task_context["history"] = (
                    self.task_context["history"] + [{"action": proposal.to_dict(), "feedback": feedback}]
                )[-6:]
                observation, images = self.observe()
            else:
                return self.finish(observation, "step_limit", "模型达到决策次数上限，已停止；任务未验证完成。")
        except Exception as exc:
            self.runtime.request_stop()
            self.event(observation, "model_error", trace=self.last_trace, reply=str(exc))
            return self.finish(observation, "error", f"VLM 任务停止：{type(exc).__name__}: {exc}")
        return self.finish(observation, "cancelled", "任务已中止，保持站立；迟到的模型输出不会执行。")
