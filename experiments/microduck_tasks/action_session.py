"""One action at a time, with explicit source labels and physical task outcomes."""

import json
from types import SimpleNamespace

from .action_catalog import ALIASES, EXPERIMENTAL, LABELS, OBJECT_TASKS, print_catalog
from .action_objects import object_task
from .language import parse_instruction


class ActionSession:
    def __init__(self, runtime, *, experimental=False, profile="toy"):
        if profile not in ("toy", "research"):
            raise ValueError("Choose the toy or research action profile.")
        self.runtime = runtime
        self.experimental = experimental
        self.profile = profile
        self.instruction = "动作演示；输入 help 查看命令"
        self.backend = SimpleNamespace(metadata={"backend": "action_playground", "hardware": False})
        self.results = []

    def _event(self, instruction, skill, phase, **fields):
        with self.runtime.lock:
            time_s = float(self.runtime.data.time)
        record = dict(
            time_s=time_s,
            instruction=instruction,
            skill=skill,
            phase=phase,
            controller_source="action_playground",
            **fields,
        )
        with (self.runtime.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _kick(self, instruction, color):
        from .collect_plan_teacher import RGBRecorder
        from .plan_teacher import PlanTeacherBackend, PlanTeacherSession

        name = f"kick_{len(self.results):03d}"
        output = self.runtime.output / "teacher_tasks" / name
        recorder = RGBRecorder(output, name, 0, {"backend": "rgbd_rule_teacher", "scene": "action_playground"})
        session = PlanTeacherSession(self.runtime, recorder, PlanTeacherBackend(), expected_color=color)
        result = session.run(instruction)
        (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result["status"] == "success", result["status"]

    def run(self, instruction):
        if instruction in ("help", "帮助", "动作列表"):
            if self.profile == "toy":
                from .toy_catalog import print_catalog as print_toy_catalog

                print_toy_catalog()
            else:
                print_catalog()
            return
        if parse_instruction(instruction).kind == "stop":
            self.runtime.request_stop()
            print("收到，已请求停止当前任务。", flush=True)
            return
        skill = ALIASES.get(instruction, instruction)
        intent = parse_instruction(instruction)
        colored_kick = intent.kind == "kick" and intent.color is not None and skill not in LABELS
        if self.profile == "toy":
            from .toy_catalog import CANDIDATES

            if skill not in CANDIDATES and not colored_kick:
                reply = "这个命令还不能在当前动作播放器中执行。输入 help 查看自身动作和踢球命令。"
                print(reply, flush=True)
                return {"success": False, "outcome": "outside_toy_candidates", "reply": reply}
        if skill not in LABELS and not colored_kick:
            print("没有这个动作；输入 help 查看中文命令或动作 ID。", flush=True)
            return {"success": False, "outcome": "unknown_action"}
        if skill in EXPERIMENTAL and not self.experimental:
            print(f"{EXPERIMENTAL[skill]}。需要演示时用 --experimental 启动。", flush=True)
            return {"success": False, "outcome": "experimental_disabled"}
        self.instruction = instruction
        self.runtime.cancel.clear()
        print(f"好，我开始执行：{instruction if colored_kick else LABELS[skill]}。", flush=True)
        self._event(instruction, skill, "start")
        with self.runtime.lock:
            start_index = len(self.runtime.rows)
        success, reason, error = False, "not_started", None
        evaluation = None
        try:
            if colored_kick:
                success, reason = self._kick(instruction, intent.color)
            elif skill in OBJECT_TASKS:
                success, reason = object_task(self.runtime, skill)
            else:
                success = self.runtime.perform(skill)
                if success:
                    from .action_acceptance import evaluate_action

                    evaluation = evaluate_action(self.runtime, skill, start_index)
                    success, reason = evaluation["success"], evaluation["reason"]
                else:
                    reason = "cancelled_or_failed"
            if self.runtime.cancel.is_set():
                success, reason = False, "cancelled"
        except Exception as exc:
            reason, error = "execution_failed", f"{type(exc).__name__}: {exc}"
        result = dict(
            skill=skill,
            instruction=instruction,
            success=bool(success),
            outcome=reason,
            error=error,
            evaluation=evaluation,
        )
        self.results.append(result)
        self._event(
            instruction, skill, "result", **{k: v for k, v in result.items() if k not in ("skill", "instruction")}
        )
        print("这次动作已通过验收。" if success else "这次任务没有完成，已结束本次尝试。", flush=True)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
