"""Bounded v2 plans with one asynchronous request and fresh local execution checks."""

import copy
import json
import math
import queue
import threading
import time

from .efficient_planner import EfficientKickPlanner
from .kick_region import choose_foot, reachable
from .language import parse_instruction
from .plan_inputs import feedback, request
from .plan_protocol import VERSION, parse_plan, resolve_target
from .q1r_navigation_guard import NavigationGuard
from .skill_tasks import SkillTaskSession
from .tasks import evaluate_kick


class PlanTaskSession(SkillTaskSession):
    def _request(self, observation, images, committed):
        return request(
            self.instruction,
            observation,
            images,
            f"task_{self.task_seq}",
            self.task_start_time_s,
            self.outcomes,
            committed,
        )

    def _start_request(self, observation, images, committed=()):
        if self._pending is not None and self._pending.is_alive():
            raise RuntimeError("Wait for the previous inference to finish before another request.")
        request = self._request(observation, images, list(committed))
        ticket = {
            "generation": self.generation,
            "started": time.monotonic(),
            "frame_id": observation["frame_id"],
            "queue": queue.Queue(maxsize=1),
        }

        def predict():
            try:
                plan = self.backend.decide(request)
                ticket["queue"].put((plan, copy.deepcopy(self.backend.last_trace), None))
            except Exception as exc:
                ticket["queue"].put((None, copy.deepcopy(self.backend.last_trace), exc))

        self._pending = threading.Thread(target=predict, name="microduck-plan-http", daemon=True)
        self._pending.start()
        return ticket

    def _receive(self, ticket):
        while not self.runtime.cancel.is_set():
            age = time.monotonic() - ticket["started"]
            if self.runtime.failure:
                raise RuntimeError(self.runtime.failure)
            if age > self.timeout:
                raise TimeoutError("Plan inference timed out; no stale motion will execute.")
            try:
                plan, trace, error = ticket["queue"].get(timeout=0.05)
            except queue.Empty:
                continue
            if self.runtime.cancel.is_set():
                return None
            self._pending.join(timeout=0.1)
            self.last_trace = trace
            if error:
                raise error
            if ticket["generation"] != self.generation or age > 6.0:
                return None
            return parse_plan(json.dumps(plan.to_dict(), allow_nan=False))
        return None

    def _step(self, step, target):
        skill, budget = step["skill"], step["seconds"]
        planner = None
        spent = 0.0
        while spent < budget - 1e-8 and not self.runtime.cancel.is_set():
            pause = getattr(self, "pause_boundary", None)
            if pause is not None:
                paused = pause()
                if paused is not None:
                    self.pause_result = paused
                    return "safe_pause_boundary" if paused.get("paused") else paused["reason"]
            observation, _ = self.observe()
            # Rendering and recording can be slow while a lockstep world stays frozen.
            # Compare capture and execution clocks, not renderer wall-clock duration.
            age = self.runtime.snapshot()["time_s"] - observation["time_s"]
            if not math.isfinite(age) or not 0 <= age <= 0.3:
                return "camera_deadline_exceeded"
            if not math.isfinite(observation["tilt_deg"]) or observation["tilt_deg"] > 25:
                return "body_not_upright"
            navigation = getattr(self, "navigation_guard", None)
            if navigation is not None:
                reason = navigation.check(observation)
                if reason:
                    return reason
            entity = resolve_target(target, observation)
            if target is not None and entity is None:
                return "target_lost"
            if entity:
                self.context["selected_color"] = next(
                    (a["value"] for a in entity["attributes"] if a["name"] == "color"), None
                )
            if skill.startswith("kick_"):
                if entity is None or entity["category"] != "ball" or not observation["stationary"]:
                    return "kick_requires_visible_target_and_standing"
                if not reachable(entity["position_body_m"], skill[5:]):
                    return "kick_target_outside_measured_region"
                before = self.runtime.truth() if self.expected_color else None
                if not self._execute(observation, skill, 0.5, trace=self.last_trace):
                    return "cancelled"
                start = self.runtime.last_execution["start_time_s"]
                observation, _ = self.observe()
                if not self._execute(observation, "stand", 3.0, source="executor"):
                    return "cancelled"
                if before is not None:
                    self.kicks.append({"before": before, "foot": skill[5:], "start": start})
                return "completed"
            actual, duration, heading = skill, min(0.2, budget - spent), None
            if skill == "approach_target":
                if entity is None or entity["category"] != "ball":
                    return "unsupported_navigation_target"
                point = entity["position_body_m"]
                if choose_foot(point) and observation["stationary"]:
                    return "target_reachable"
                if planner is None:
                    planner = EfficientKickPlanner(self.context["selected_color"])
                decision = planner.decide(observation)
                if decision is None:
                    return planner.status
                if decision.skill.startswith("kick_"):
                    return "target_reachable"
                duration = min(decision.seconds, budget - spent)
                if getattr(self, "pause_boundary", None) is not None:
                    duration = min(0.2, duration)
                actual = decision.skill
                heading = getattr(decision, "heading", None)
            elif target is not None and skill == "move_forward":
                x, y, _ = entity["position_body_m"]
                if x < 0.15 or abs(math.atan2(y, x)) > 0.28:
                    return "forward_requires_clear_centered_target"
                heading = observation["yaw"] + math.atan2(y, x)
            if self.runtime.cancel.is_set():
                return "cancelled"
            with self.runtime.lock:
                self.runtime.policy.motion.visual_heading = heading
            duration = round(duration / 0.02) * 0.02
            if not self._execute(observation, actual, duration, trace=self.last_trace):
                return "cancelled"
            spent += duration
        return "cancelled" if self.runtime.cancel.is_set() else "completed"

    def run(self, instruction, *, reset_cancel=True):
        """Start a standalone task, or retain a parent queue's cancellation latch."""
        if reset_cancel:
            self.runtime.cancel.clear()
        self.task_seq += 1
        self.generation = getattr(self, "generation", 0) + 1
        self.instruction, self.task_context, self.last_trace = instruction, {}, {}
        self.outcomes, self.kicks = [], []
        intent = parse_instruction(instruction)
        self.navigation_guard = NavigationGuard(intent.color) if intent.kind == "kick" else None
        self.pause_result = None
        single_motion = intent.kind in ("move", "turn")
        motion_completed = False
        observation, images = self.observe()
        self.task_start_time_s = observation["time_s"]
        if parse_instruction(instruction).kind == "stop":
            self.runtime.request_stop()
            return self.finish(observation, "cancelled", "已停止。")
        contact_start = getattr(self, "evaluation_contact_start", len(self.runtime.contacts))
        row_start = getattr(self, "evaluation_row_start", len(self.runtime.rows))
        started = time.monotonic()
        pending = None
        stale = 0
        rejected = 0
        try:
            for _ in range(self.max_steps):
                if self.runtime.cancel.is_set():
                    break
                if time.monotonic() - started > 600:
                    return self.finish(observation, "timeout", "任务超时，已停止。")
                if pending is None:
                    observation, images = self.observe()
                    pending = self._start_request(observation, images)
                plan = self._receive(pending)
                pending = None
                if plan is None:
                    if self.runtime.cancel.is_set():
                        break
                    stale += 1
                    if stale >= 3:
                        return self.finish(observation, "stale_plan", "连续计划过期，请检查模型延迟。")
                    continue
                stale = 0
                self.last_trace.update(plan=plan.to_dict(), protocol=VERSION)
                self.event(observation, "model_plan", trace=self.last_trace)
                if plan.status == "continue" and single_motion and motion_completed:
                    return self.finish(observation, "command_budget_complete", "单次移动已完成，执行器拒绝重复移动。")
                if plan.status != "continue":
                    evaluation = None
                    status = {"answer": "answered", "clarify": "clarification", "finish": "model_reported"}[plan.status]
                    if self.expected_color:
                        checks = [
                            evaluate_kick(
                                self.expected_color,
                                kick["foot"],
                                kick["before"],
                                self.runtime.truth(),
                                kick["start"],
                                self.runtime.contacts[contact_start:],
                                self.runtime.rows[row_start:],
                            )
                            for kick in self.kicks
                        ]
                        passed = len(checks) == 1 and checks[0]["success"]
                        evaluation = {"success": passed, "kicks": checks}
                        status = "success" if passed else "verification_failed"
                    return self.finish(observation, status, plan.reply, evaluation)
                # Kicks require a new post-recovery image before another inference.
                if (
                    getattr(self.backend, "supports_continuation", True)
                    and not single_motion
                    and not any(s["skill"].startswith("kick_") for s in plan.steps)
                ):
                    observation, images = self.observe()
                    pending = self._start_request(observation, images, plan.steps)
                for step in plan.steps:
                    outcome = self._step(step, plan.target)
                    self.outcomes.append(feedback(step, outcome, plan.target))
                    observation, _ = self.observe()
                    self.event(observation, "plan_step_result", trace=feedback(step, outcome, plan.target))
                    if outcome != "completed":
                        self.generation += 1
                        # Drain but never commit a continuation of an interrupted prefix.
                        if pending is not None:
                            self._receive(pending)
                            pending = None
                        if outcome == "safe_pause_boundary":
                            result = self.finish(observation, "paused", "已站稳暂停，保留当前踢球任务。")
                            result["pause_metrics"] = self.pause_result.get("metrics", {})
                            return result
                        if outcome in (
                            "target_not_found_after_full_search",
                            "target_search_timeout",
                            "target_not_approaching",
                            "navigation_timeout",
                            "invalid_navigation_measurement",
                            "pause_stability_unverified",
                        ):
                            if not self.runtime.cancel.is_set():
                                self._execute(observation, "stand", 0.6, source="executor")
                            result = self.finish(observation, outcome, f"任务未完成：{outcome}，已结束追赶。")
                            result["navigation_metrics"] = (
                                self.navigation_guard.metrics if self.navigation_guard else {}
                            )
                            return result
                        rejected = 0 if outcome == "target_reachable" else rejected + 1
                        if rejected >= 3:
                            return self.finish(observation, "guard_rejected", "连续三次计划条件不成立，已停止。")
                        break
                    rejected = 0
                    if single_motion:
                        motion_completed = True
                        break
            else:
                return self.finish(observation, "step_limit", "达到计划次数上限。")
        except Exception as exc:
            self.runtime.request_stop()
            self.event(observation, "model_error", trace=self.last_trace, reply=str(exc))
            return self.finish(observation, "error", f"计划停止：{type(exc).__name__}: {exc}")
        finally:
            self.generation += 1
            with self.runtime.lock:
                self.runtime.policy.motion.visual_heading = None
        return self.finish(observation, "cancelled", "任务已取消，迟到计划不会执行。")
