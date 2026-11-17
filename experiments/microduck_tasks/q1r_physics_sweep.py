"""Headless executor exploration with explicit gold semantics, never Q1 predictions."""

import argparse
import copy
import json
import threading
import time
from pathlib import Path

from .bootstrap import configure


def target(color):
    return dict(category="ball", attributes=[dict(name="color", value=color)], entity_id=None)


def perform(action):
    return dict(type="perform", action=action, repeat=1)


def kick(color):
    return dict(type="kick", target=target(color))


def move(direction, seconds=1.0):
    return dict(type="move", direction=direction, limit=dict(kind="duration", seconds=seconds))


def cases():
    from .action_catalog import LABELS
    from .toy_catalog import SELF_ACTIONS

    result = [dict(name=f"action_{a}", text=f"请{LABELS[a]}一次", goals=[perform(a)]) for a in SELF_ACTIONS]
    result += [
        dict(name=f"move_{d}", text=f"实际测试{d}方向走1秒", goals=[move(d)])
        for d in ("forward", "backward", "left", "right")
    ]
    result += [
        dict(name=f"turn_{d}_{a}", text=f"向{d}转{a}度", goals=[dict(type="turn", direction=d, angle_deg=a)])
        for d, a in (("left", 90), ("right", 180), ("left", 360))
    ]
    result += [
        dict(
            name="sit_stand_nod",
            text="坐下来，站起来，再点头",
            goals=[dict(type="posture", pose="sitting"), dict(type="posture", pose="standing"), perform("nod")],
        )
    ]
    result += [
        dict(
            name=f"navigate_{c}",
            text=f"到{c}球旁边停下",
            goals=[dict(type="navigate", target=target(c), standoff_m=0.22)],
        )
        for c in ("red", "yellow", "blue")
    ]
    result += [dict(name=f"kick_default_{c}", text=f"踢{c}球", goals=[kick(c)]) for c in ("red", "yellow", "blue")]
    result += [
        dict(name=f"kick_random_{seed}_{color}", text=f"随机场景踢{color}球", goals=[kick(color)], seed=seed)
        for seed, color in ((8101, "red"), (8102, "yellow"), (8103, "blue"))
    ]
    result += [
        dict(
            name="reproduce_roll_blue_yellow",
            text="做前滚翻，然后踢蓝球，再踢黄球",
            goals=[perform("roll"), kick("blue"), kick("yellow")],
        ),
        dict(
            name="pause_search_resume", text="踢蓝球再点头", goals=[kick("blue"), perform("nod")], hook="search_pause"
        ),
        dict(
            name="pause_approach_resume",
            text="踢红球再点头",
            goals=[kick("red"), perform("nod")],
            hook="approach_pause",
        ),
        dict(name="pause_kick_resume", text="踢红球再点头", goals=[kick("red"), perform("nod")], hook="kick_pause"),
        dict(name="stop_search", text="去踢蓝球", goals=[kick("blue")], hook="search_stop", expected="cancelled"),
        dict(
            name="chat_walk_pause_resume_stop",
            text="一直向前走",
            goals=[dict(type="move", direction="forward", limit=dict(kind="until_cancelled"))],
            hook="chat_walk",
            expected="cancelled",
        ),
        dict(
            name="missing_ball_bounded_search",
            text="去找红球踢一下",
            goals=[kick("red")],
            absent=True,
            expected="target_not_found_after_full_search",
        ),
        dict(
            name="faster_ball_bounded_pursuit",
            text="追红球踢一下",
            goals=[kick("red")],
            moving_target=True,
            expected="target_not_approaching",
        ),
    ]
    return result


class GoldBackend:
    name = "explicit_gold_executor_probe"
    training_ready = False
    metadata = dict(model_called=False, scope="executor_only", ground_truth_semantics=True)

    def __init__(self, case):
        self.case = case
        self.last_trace = {}

    def decide(self, request):
        from .toy_protocol import VERSION
        from .toy_teacher import ToyTeacher

        self.last_trace = dict(model_called=False, decision_source="explicit_gold_test_plan")
        if request["event"]["kind"] == "feedback":
            return ToyTeacher().decide(request)
        text = request["event"]["text"]
        if text == self.case["text"]:
            return dict(
                schema_version=VERSION,
                decision="replace",
                reply="收到，执行本次明确给定的测试计划。",
                goals=copy.deepcopy(self.case["goals"]),
            )
        return dict(schema_version=VERSION, decision="respond", reply="我在继续执行任务，听到你的话了。", goals=[])


def run_case(case, output):
    import mujoco
    import numpy as np

    from .scene import sample_layout
    from .toy_dispatch import ToyInputDispatcher
    from .toy_motion import ToyRuntime
    from .toy_session import ToySession

    folder = output / case["name"]
    folder.mkdir()
    runtime = ToyRuntime(folder, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
    runtime.reset(sample_layout(case["seed"]) if "seed" in case else None)
    if case.get("absent") or case.get("moving_target"):
        with runtime.lock:
            q, v = runtime.ball_q["red"], runtime.ball_v["red"]
            runtime.data.qpos[q : q + 3] = [20, 0, 0.035] if case.get("absent") else [0.45, 0, 0.035]
            if case.get("moving_target"):
                runtime.model.geom_condim[runtime.model.geom("ball_red_geom").id] = 3
                runtime.data.qvel[v : v + 6] = [0.35, 0, 0, 0, 10, 0]
            mujoco.mj_forward(runtime.model, runtime.data)
    session = ToySession(runtime, backend=GoldBackend(case), split_group="q1r_executor_exploration")
    errors, controls, tickets = [], [], []
    dispatch = ToyInputDispatcher(session, threading.Event(), threading.Event(), errors)
    worker = threading.Thread(target=dispatch.dispatch, daemon=True)
    worker.start()
    counts, original_tick = {}, runtime._tick
    fired = set()

    def control(text, tag):
        fired.add(tag)
        controls.append(dict(text=text, time_s=runtime.snapshot()["time_s"], skill=runtime.rows[-1][3]))
        tickets.append(dispatch.submit(text))

    def tick(skill):
        original_tick(skill)
        counts[skill] = counts.get(skill, 0) + 1
        hook = case.get("hook")
        if hook == "search_pause" and skill == "turn_left" and counts[skill] == 3:
            control("暂停", "pause")
        elif hook == "approach_pause" and skill == "move_forward" and counts[skill] == 3:
            control("暂停", "pause")
        elif hook == "kick_pause" and skill.startswith("kick_") and counts[skill] == 3:
            control("暂停", "pause")
        elif hook == "search_stop" and skill == "turn_left" and counts[skill] == 3:
            control("停", "stop")
        elif hook == "chat_walk" and skill == "move_forward":
            if counts[skill] == 3:
                control("你好呀，辛苦啦，你还在走吗？", "chat")
            if counts[skill] == 20:
                control("暂停", "pause")
            if "resumed" in fired and counts[skill] >= 23 and "stop" not in fired:
                control("停", "stop")

    runtime._tick = tick
    started = time.monotonic()
    summary = dict(case=case, semantics_source="explicit_gold_not_q1", unexpected=[], controls=controls)
    try:
        session.run(case["text"], wait=False)
        session.wait(300)
        for done, _box in tickets:
            if not done.wait(20):
                raise TimeoutError("Control dispatcher did not finish.")
        if session.execution["state"] == "paused":
            summary["pause"] = dict(
                execution=copy.deepcopy(session.execution),
                time_s=runtime.snapshot()["time_s"],
                tilt_deg=runtime.snapshot()["tilt_deg"],
                stationary=runtime.snapshot()["stationary"],
                counts=copy.deepcopy(counts),
            )
            if not summary["pause"]["stationary"] or summary["pause"]["tilt_deg"] >= 15:
                summary["unexpected"].append("pause_was_not_stable")
            fired.add("resumed")
            session.run("继续", wait=False)
            session.wait(300)
            for done, _box in tickets:
                if not done.wait(20):
                    raise TimeoutError("Control dispatcher did not finish after resume.")
        summary.update(
            results=copy.deepcopy(session.results),
            execution=copy.deepcopy(session.execution),
            errors=errors,
            counts=counts,
            wall_s=time.monotonic() - started,
            sim_time_s=runtime.snapshot()["time_s"],
            final_tilt_deg=runtime.snapshot()["tilt_deg"],
            final_ball_positions={k: v.tolist() for k, v in runtime.truth().items()},
            max_target_excess=float(np.max(runtime.target_excess)),
        )
        completed = [r for r in session.results if r["status"] == "succeeded"]
        expected = case.get("expected", "succeeded")
        if expected == "succeeded" and len(completed) != len(case["goals"]):
            summary["unexpected"].append("not_all_expected_goals_succeeded")
        elif expected == "cancelled" and not any(r["status"] == "cancelled" for r in session.results):
            summary["unexpected"].append("stop_did_not_record_cancelled_goal")
        elif expected not in ("succeeded", "cancelled") and not any(r["reason"] == expected for r in session.results):
            summary["unexpected"].append("failure_reason_differed_from_expected")
        if case.get("hook", "").endswith("pause") and "pause" not in summary:
            summary["unexpected"].append("pause_boundary_not_reached")
        if errors:
            summary["unexpected"].append("dispatcher_error")
        kick_rows = [r for r in runtime.rows if r[3].startswith("kick_")]
        summary["actual_kick_ticks"] = len(kick_rows)
        summary["goal_statuses"] = [r["status"] for r in session.results]
        summary["goal_reasons"] = [r["reason"] for r in session.results]
    except Exception as exc:
        summary["unexpected"].append(f"{type(exc).__name__}: {exc}")
    finally:
        session.close()
        dispatch.close()
        worker.join(5)
        runtime.save_trace()
        runtime.close()
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configure()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [case for case in cases() if not args.only or case["name"] in args.only]
    summaries = []
    for case in selected:
        existing = args.output / case["name"] / "summary.json"
        if args.resume and existing.exists():
            summary = json.loads(existing.read_text())
        else:
            print(f"BEGIN {case['name']}", flush=True)
            summary = run_case(case, args.output)
        summaries.append(summary)
        (args.output / "sweep.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                dict(
                    case=case["name"],
                    unexpected=summary["unexpected"],
                    reasons=summary.get("goal_reasons"),
                    wall_s=summary.get("wall_s"),
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
