"""Run one semantic goal; robot truth is reserved for independent acceptance."""

import json
import math

from .toy_catalog import SELF_ACTIONS


class ToyGoalRunner:
    def __init__(self, session):
        self.session = session
        self.runtime = session.runtime
        self._kick_evidence = {}

    def execute(self, goal, task_id, goal_index):
        if self.runtime.cancel.is_set():
            return dict(success=False, reason="cancelled", metrics={})
        from .toy_motion import ToyMotion

        runtime = self.runtime
        if goal["type"] == "perform":
            from .action_acceptance import evaluate_action

            action = goal["action"]
            if action not in SELF_ACTIONS:
                raise ValueError("Self action outside the toy candidates.")
            checks = []
            for _ in range(goal["repeat"]):
                start = len(runtime.rows)
                if not runtime.perform(action):
                    return dict(success=False, reason="cancelled_or_failed", metrics=dict(checks=checks))
                check = evaluate_action(runtime, action, start)
                checks.append(check)
                if not check["success"]:
                    return dict(success=False, reason=check["reason"], metrics=dict(checks=checks))
            return dict(success=True, reason="motion_and_recovery_verified", metrics=dict(checks=checks))
        if goal["type"] == "kick":
            return self._kick(goal, task_id, goal_index)
        if goal["type"] == "move" and goal["limit"]["kind"] == "until_cancelled":
            while not runtime.cancel.is_set():
                if self.session.pause_requested:
                    settled = self.settle_for_pause()
                    if not settled["success"]:
                        return settled
                    if self.session.pause_requested:
                        return dict(
                            success=False, paused=True, reason="safe_pause_boundary", metrics=settled["metrics"]
                        )
                observation, _, _ = runtime.observe()
                if observation["tilt_deg"] > 25:
                    return dict(success=False, reason="body_not_upright", metrics={})
                if not runtime.execute(f"move_{goal['direction']}", 0.2):
                    break
            return dict(success=False, reason="cancelled", metrics={})
        return ToyMotion(runtime, runtime.observe).execute(goal)

    def settle_for_pause(self):
        """Verify three consecutive stable observations after short stand/hold segments."""
        stable, tilt = 0, None
        for segment in range(10):
            if self.runtime.cancel.is_set() or not self.runtime.execute("stand", 0.2):
                return dict(success=False, reason="cancelled", metrics={})
            observation, _, _ = self.runtime.observe()
            tilt = observation["tilt_deg"]
            settled = math.isfinite(tilt) and tilt < 15 and observation["stationary"] is True
            stable = stable + 1 if settled else 0
            if stable >= 3:
                return dict(
                    success=True,
                    reason="pause_stability_verified",
                    metrics=dict(stand_seconds=0.2 * (segment + 1), stable_observations=stable, final_tilt_deg=tilt),
                )
        return dict(success=False, reason="pause_stability_unverified", metrics=dict(final_tilt_deg=tilt))

    def _navigation_pause_boundary(self):
        """A kick's search/walk can pause; its kick pulse and recovery are atomic."""
        if not self.session.pause_requested:
            return None
        settled = self.settle_for_pause()
        if not settled["success"]:
            return settled
        if self.session.pause_requested:
            return dict(success=False, paused=True, reason="safe_pause_boundary", metrics=settled["metrics"])
        return None

    def _kick(self, goal, task_id, goal_index):
        from .collect_plan_teacher import RGBRecorder
        from .plan_teacher import PlanTeacherBackend, PlanTeacherSession

        colors = {a["name"]: a["value"] for a in goal["target"]["attributes"]}
        color = colors.get("color")
        if color not in ("red", "yellow", "blue"):
            return dict(success=False, reason="unsupported_kick_target", metrics={})
        chinese = {"red": "红", "yellow": "黄", "blue": "蓝"}[color]
        folder = self.runtime.output / "skill_traces" / f"{task_id}_{goal_index:02d}_{self.session.attempt_seq:03d}"
        self.session.attempt_seq += 1
        recorder = RGBRecorder(
            folder, folder.name, self.session.seed, {"backend": "rgbd_rule_teacher", "parent_task": task_id}
        )
        controller = PlanTeacherSession(
            self.runtime, recorder, PlanTeacherBackend(), expected_color=color, verbose=False, max_steps=80
        )
        key = (task_id, goal_index)
        starts = self._kick_evidence.setdefault(key, (len(self.runtime.contacts), len(self.runtime.rows)))
        controller.evaluation_contact_start, controller.evaluation_row_start = starts
        controller.pause_boundary = self._navigation_pause_boundary
        result = controller.run(f"踢{chinese}球", reset_cancel=False)
        (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        success = result["status"] == "success" and result.get("evaluation", {}).get("success") is True
        if result["status"] == "paused":
            return dict(
                success=False,
                paused=True,
                reason="safe_pause_boundary",
                metrics={"skill_trace": str(folder.relative_to(self.runtime.output)), **result["pause_metrics"]},
            )
        self._kick_evidence.pop(key, None)
        return dict(
            success=bool(success),
            reason="target_contact_and_displacement_verified" if success else result["status"],
            metrics=dict(
                evaluation=result.get("evaluation"),
                navigation=result.get("navigation_metrics"),
                skill_trace=str(folder.relative_to(self.runtime.output)),
            ),
        )
