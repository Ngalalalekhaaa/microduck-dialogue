"""Offline tasks, actual execution logs, and separate simulation acceptance."""

import math

import numpy as np

from .language import COLORS
from .planner import KickPlanner
from .vlm import RuleVLMBackend, VisionLanguageInput


class TaskSession:
    def __init__(self, runtime, recorder, backend=None, verbose=True, *, teacher="legacy"):
        self.runtime, self.recorder = runtime, recorder
        self.backend = backend or RuleVLMBackend()
        self.context = {}
        self.event_seq = self.task_seq = 0
        self.results = []
        self.verbose = verbose
        if teacher not in ("legacy", "efficient"):
            raise ValueError("Select legacy or efficient teacher.")
        self.teacher = teacher

    def observe(self):
        observation, images, depths = self.runtime.observe()
        stored = self.recorder.record_observation(
            observation["frame_id"],
            observation["time_s"],
            images,
            depths,
            observation["detections"],
        )
        stored.update({key: observation[key] for key in ("yaw", "stationary", "tilt_deg")})
        return stored, images

    def event(self, observation, phase, decision=None, execution=None, done=False, status="running", reply=""):
        self.event_seq += 1
        self.recorder.append_event(
            {
                "event_id": f"event_{self.event_seq:05d}",
                "task_id": f"task_{self.task_seq:03d}",
                "instruction": self.instruction,
                "intent": self.intent.to_dict(),
                "context": self.task_context,
                "observation": observation,
                "phase": phase,
                "decision": decision,
                "execution": execution
                or {"started": False, "completed": False, "start_time_s": None, "end_time_s": None},
                "done": done,
                "status": status,
                "reply": reply,
            }
        )

    def perform(self, observation, phase, skill, seconds):
        execution = {"started": False, "completed": False, "start_time_s": None, "end_time_s": None}
        decision = {"skill": skill, "seconds": seconds, "target_color": self.intent.color}
        if self.runtime.cancel.is_set():
            self.event(observation, phase, decision, execution)
            return False
        try:
            completed = self.runtime.execute(skill, seconds)
        finally:
            execution.update(self.runtime.last_execution)
            self.event(observation, phase, decision, execution)
        if self.verbose:
            positions = {d["color"]: [round(v, 3) for v in d["center_body_m"][:2]] for d in observation["detections"]}
            print(f"  {phase}: {skill} {seconds:.2f}s; visible={positions}", flush=True)
        return completed

    def finish(self, observation, status, reply, evaluation=None):
        self.event(observation, "done", done=True, status=status, reply=reply)
        result = {
            "task_id": f"task_{self.task_seq:03d}",
            "instruction": self.instruction,
            "intent": self.intent.to_dict(),
            "status": status,
            "reply": reply,
            "evaluation": evaluation,
        }
        self.results.append(result)
        print(reply, flush=True)
        return result

    def run(self, instruction):
        self.runtime.cancel.clear()
        self.task_seq += 1
        self.instruction = instruction
        observation, images = self.observe()
        self.task_context = {**self.context, "visible_colors": [d["color"] for d in observation["detections"]]}
        self.intent = self.backend.interpret(VisionLanguageInput(instruction, images, observation, self.task_context))
        self.event(observation, "interpret")
        try:
            if self.intent.kind in ("describe", "clarify", "stop"):
                if self.intent.kind == "stop":
                    self.runtime.request_stop()
                return self.finish(
                    observation, "clarification" if self.intent.kind == "clarify" else "success", self.intent.reply
                )
            if self.intent.kind in ("move", "turn"):
                skill = f"{self.intent.kind}_{self.intent.direction}"
                if not self.perform(observation, "direct", skill, self.intent.seconds):
                    return self.finish(observation, "cancelled", "已中止移动，恢复站立。")
                observation, _ = self.observe()
                self.perform(observation, "settle", "stand", 1.0)
                observation, _ = self.observe()
                return self.finish(
                    observation, "cancelled" if self.runtime.cancel.is_set() else "success", "已停止移动并站稳。"
                )
            self.context["selected_color"] = self.intent.color
            return self.kick(observation)
        except (RuntimeError, TimeoutError, ValueError) as exc:
            self.runtime.request_stop()
            return self.finish(observation, "error", f"任务停止：{exc}")

    def kick(self, observation):
        if self.teacher == "efficient":
            from .efficient_planner import EfficientKickPlanner

            planner = EfficientKickPlanner(self.intent.color)
        else:
            planner = KickPlanner(self.intent.color)
        contact_start = len(self.runtime.contacts)
        row_start = len(self.runtime.rows)
        while not self.runtime.cancel.is_set():
            decision = planner.decide(observation)
            if decision is None:
                message = (
                    "转身搜索后仍未找到目标球。"
                    if planner.status == "not_found"
                    else "未能在限制内对准目标，已停止任务。"
                )
                return self.finish(observation, planner.status, message)
            if decision.phase == "kick":
                # Oracle state is read only for independent evaluation, never for a decision.
                before = self.runtime.truth()
                kick_start = self.runtime.snapshot()["time_s"]
            if self.teacher == "efficient":
                with self.runtime.lock:
                    self.runtime.policy.motion.visual_heading = getattr(decision, "heading", None)
            if not self.perform(observation, decision.phase, decision.skill, decision.seconds):
                break
            if decision.phase == "kick":
                observation, _ = self.observe()
                if not self.perform(observation, "recover", "stand", 3.0):
                    break
                observation, _ = self.observe()
                evaluation = evaluate_kick(
                    self.intent.color,
                    planner.foot,
                    before,
                    self.runtime.truth(),
                    kick_start,
                    self.runtime.contacts[contact_start:],
                    self.runtime.rows[row_start:],
                )
                status = "success" if evaluation["success"] else "verification_failed"
                reply = (
                    f"已踢动{COLORS[self.intent.color]}球并恢复站立。"
                    if evaluation["success"]
                    else "踢球动作已执行，但仿真验收未通过；这条轨迹不会作为成功动作训练样本。"
                )
                return self.finish(observation, status, reply, evaluation)
            observation, _ = self.observe()
            if self.teacher == "legacy" and decision.skill != "stand":
                if not self.perform(observation, "settle", "stand", 1.0):
                    break
                observation, _ = self.observe()
        return self.finish(observation, "cancelled", "任务已中止，恢复站立。")


def evaluate_kick(color, foot, before, after, kick_start, contacts, rows):
    """Ground truth acceptance is deliberately outside the visual planner."""
    valid_positions = all(
        np.asarray(position).shape == (3,) and np.isfinite(position).all()
        for positions in (before, after)
        for position in positions.values()
    )
    displacement = float(np.linalg.norm(after[color][:2] - before[color][:2])) if valid_positions else None
    hit = [
        c
        for c in contacts
        if c["color"] == color
        and c["other_color"] is None
        and kick_start <= c["time_s"] <= kick_start + 0.54
        and foot in c["geom"]
        and "foot" in c["geom"]
    ]
    wrong = [c for c in contacts if c["color"] != color and c["other_color"] is None]
    early = [c for c in contacts if c["color"] == color and c["time_s"] < kick_start]
    valid_trace = (
        bool(rows)
        and math.isfinite(kick_start)
        and all(
            np.isfinite([row[0], row[4], row[5]]).all()
            and np.isfinite(row[1]).all()
            and np.asarray(row[2]).ndim == 1
            and len(row[2]) >= 6
            and np.isfinite(row[2]).all()
            and np.asarray(row[7]).shape == (3,)
            and np.isfinite(row[7]).all()
            for row in rows
        )
    )
    tilt = float(max(row[4] for row in rows)) if valid_trace else None
    final_tilt = float(rows[-1][4]) if valid_trace else None
    planar_speed = float(np.linalg.norm(rows[-1][2][:2])) if valid_trace else None
    angular_speed = float(np.linalg.norm(rows[-1][2][3:6])) if valid_trace else None
    final_height = float(rows[-1][7][2]) if valid_trace else None
    stood_throughout = valid_trace and all(row[7][2] >= 0.065 for row in rows)
    recovered = bool(
        valid_trace
        and rows[-1][3] == "stand"
        and final_tilt < 10
        and planar_speed < 0.025
        and angular_speed < 0.3
        and final_height > 0.09
    )
    return {
        "success": bool(
            valid_positions
            and valid_trace
            and hit
            and not wrong
            and not early
            and displacement >= 0.12
            and tilt < 30
            and stood_throughout
            and recovered
        ),
        "target_displacement_m": displacement,
        "target_foot_contact": bool(hit),
        "non_target_robot_contacts": len(wrong),
        "target_contacts_before_kick": len(early),
        "max_tilt_deg": tilt,
        "valid_trace": valid_trace,
        "recovered_standing": recovered,
        "final_tilt_deg": final_tilt,
        "final_planar_speed_m_s": planar_speed,
        "final_angular_speed_rad_s": angular_speed,
        "final_trunk_height_m": final_height,
        "source": "independent_simulation_ground_truth_not_planner_input",
    }
