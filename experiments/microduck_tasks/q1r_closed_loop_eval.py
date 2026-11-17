"""Predeclared headless Q1R language-to-physics evaluation; no repaired predictions."""

import argparse
import copy
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from .bootstrap import REPO, configure


def perform(action):
    return dict(type="perform", action=action, repeat=1)


def ball_goal(kind, color, **extra):
    return dict(
        type=kind, target=dict(category="ball", attributes=[dict(name="color", value=color)], entity_id=None), **extra
    )


def turn(text, decision="respond", goals=()):
    return dict(text=text, expected=dict(decision=decision, goals=list(goals), reply="评价参考，不输入模型。"))


def cases():
    return [
        dict(
            id="01_greeting_capabilities",
            turns=[turn("嘿，小鸭子，晚上好呀。"), turn("你现在能表演些什么？给我说几个就行。")],
        ),
        dict(
            id="02_backward_colloquial_3s",
            turns=[
                turn(
                    "往后挪3s吧。",
                    "replace",
                    [dict(type="move", direction="backward", limit=dict(kind="duration", seconds=3))],
                )
            ],
        ),
        dict(
            id="03_nod_left_leg",
            turns=[turn("点个头示意一下，再抬起左脚给我看看。", "replace", [perform("nod"), perform("flamingo_left")])],
        ),
        dict(
            id="04_red_standoff_no_kick",
            turns=[turn("走到红色球边上停好，别踢它。", "replace", [ball_goal("navigate", "red", standoff_m=0.25)])],
        ),
        dict(id="05_kick_red", turns=[turn("帮我把那个红球踢出去一下。", "replace", [ball_goal("kick", "red")])]),
        dict(
            id="06_roll_blue_yellow",
            turns=[
                turn(
                    "先向前翻一圈，再踢蓝球，最后再踢黄球。",
                    "replace",
                    [perform("roll"), ball_goal("kick", "blue"), ball_goal("kick", "yellow")],
                )
            ],
        ),
        dict(id="07_unsupported_purple_block", turns=[turn("踢一下那个紫色方块吧。", "clarify")]),
        dict(id="08_ability_not_authorization", turns=[turn("你会前滚翻吗？我只是问问。")]),
        dict(
            id="09_old_target_correction",
            turns=[
                turn("先把紫色方块踢走。", "clarify"),
                turn("算了，改成踢蓝球。", "replace", [ball_goal("kick", "blue")]),
            ],
        ),
        dict(id="10_steps_not_seconds", turns=[turn("往前迈10步，数着走。", "clarify")]),
        dict(
            id="11_look_then_open_beak",
            turns=[turn("左右看看，接着把嘴张开。", "replace", [perform("look_around"), perform("open_beak")])],
        ),
        dict(
            id="12_walk_chat_pause_resume_stop",
            turns=[
                turn(
                    "一直往前走，等我叫停。",
                    "replace",
                    [dict(type="move", direction="forward", limit=dict(kind="until_cancelled"))],
                )
            ],
            concurrent=True,
        ),
    ]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def run_case(case, backend, output):
    from .q1r_evaluate import assess
    from .toy_dispatch import ToyInputDispatcher
    from .toy_motion import ToyRuntime
    from .toy_session import ToySession

    folder = output / case["id"]
    folder.mkdir()
    runtime = ToyRuntime(folder, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
    runtime.reset()
    session = ToySession(runtime, backend=backend, split_group="q1r_post_training_closed_loop_new_utterances")
    shutdown, errors = threading.Event(), []
    dispatch = ToyInputDispatcher(session, shutdown, threading.Event(), errors)
    thread = threading.Thread(target=dispatch.dispatch, daemon=True)
    thread.start()
    original_tick = runtime._tick
    pulses, tickets, controls = {}, [], []
    flags = set()
    started = time.monotonic()

    def control(text):
        controls.append(dict(text=text, time_s=runtime.snapshot()["time_s"], skill=runtime.rows[-1][3]))
        tickets.append(dispatch.submit(text))

    def tick(skill):
        original_tick(skill)
        pulses[skill] = pulses.get(skill, 0) + 1
        if runtime.data.time > 82 or time.monotonic() - started > 150:
            if "budget" not in flags:
                flags.add("budget")
                control("停")
            return
        if not case.get("concurrent") or skill != "move_forward":
            return
        if pulses[skill] == 3:
            flags.add("chat")
            control("你走你的，我想听你说句鼓励的话。")
        if pulses[skill] == 20:
            flags.add("pause")
            control("暂停")
        if "resumed" in flags and pulses[skill] >= 23 and "stop" not in flags:
            flags.add("stop")
            control("停")

    def drain():
        for done, _ in tickets:
            if not done.wait(180):
                raise TimeoutError("Queued dialogue did not finish.")

    runtime._tick = tick
    result = dict(case=case, turns=[], controls=controls, unexpected=[])
    try:
        for prescribed in case["turns"]:
            before_rows, before_events = len(runtime.rows), len(session.recorder.events)
            response = session.run(prescribed["text"], wait=False)
            session.wait(180)
            drain()
            events = session.recorder.events[before_events:]
            event = next(
                e for e in events if e["event"] == "dialogue" and e["request"]["event"]["text"] == prescribed["text"]
            )
            trace = event.get("model_trace", {})
            raw = trace.get("raw_text")
            if raw is None:
                raw = json.dumps(trace.get("raw_prediction"), ensure_ascii=False)
            score = assess(raw, dict(expected=prescribed["expected"], request=event["request"]))
            result["turns"].append(
                dict(
                    text=prescribed["text"],
                    response=response,
                    score=score,
                    model_called=trace.get("model_called"),
                    reply_source=trace.get("reply_source"),
                    raw_text=raw,
                    trace=trace,
                    result_events=[e for e in events if e["event"] == "goal_result"],
                    physics_ticks=len(runtime.rows) - before_rows,
                )
            )
            if not score["semantic_plan_exact"]:
                result["unexpected"].append(
                    dict(
                        layer="language",
                        text=prescribed["text"],
                        reason="first_prediction_differs_from_predeclared_semantics",
                        score=score,
                    )
                )
        if case.get("concurrent"):
            result["before_resume"] = copy.deepcopy(session.execution)
            if session.execution["state"] == "paused":
                snap = runtime.snapshot()
                result["pause_verified"] = dict(
                    stationary=snap["stationary"], tilt_deg=snap["tilt_deg"], time_s=snap["time_s"]
                )
                flags.add("resumed")
                result["resume_response"] = session.run("继续", wait=False)
                session.wait(180)
                drain()
            else:
                result["unexpected"].append(
                    dict(layer="control", reason="continuous_walk_did_not_reach_expected_pause", flags=sorted(flags))
                )
        result["execution"] = copy.deepcopy(session.execution)
        result["goal_results"] = copy.deepcopy(session.results)
        result["model_dialogues"] = [copy.deepcopy(e) for e in session.recorder.events if e["event"] == "dialogue"]
        for row in session.results:
            if row["status"] == "failed":
                result["unexpected"].append(
                    dict(layer="physics_or_executor", reason=row["reason"], goal=row["goal"], metrics=row["metrics"])
                )
        result.update(
            wall_s=time.monotonic() - started,
            sim_time_s=runtime.snapshot()["time_s"],
            skill_ticks=pulses,
            local_control_flags=sorted(flags),
            errors=errors,
        )
        if "budget" in flags:
            result["unexpected"].append(
                dict(layer="harness", reason="predeclared_80_sim_second_or_150_wall_second_guard")
            )
        if errors:
            result["unexpected"].append(dict(layer="dialogue_runtime", errors=errors))
    except Exception as exc:
        result["unexpected"].append(dict(layer="evaluation_exception", reason=f"{type(exc).__name__}: {exc}"))
    finally:
        session.close()
        dispatch.close()
        thread.join(5)
        runtime.save_trace()
        runtime.close()
        write(folder / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    declared = cases()
    write(args.output / "predeclared_cases.json", declared)
    probes = counterfactuals()
    write(args.output / "counterfactual_predeclared.json", probes)
    source = Path(__file__)
    (args.output / source.name).write_bytes(source.read_bytes())
    sys.path.insert(0, str(REPO / ".cache/microduck_head/qlora_deps"))
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
    configure()
    from .q1r_backend import ToyQ1RBackend

    checkpoint = REPO / "experiments/microduck_tasks/outputs/q1r_20261116/train/checkpoint_000513"
    metadata = dict(
        checkpoint=str(checkpoint),
        script_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model_goals_repaired=False,
        new_training=False,
        renderer_opened=False,
        raw_camera_images_passed_to_vlm=False,
        physics_perception="simulated RGB-D and body IMU; oracle state only for independent acceptance",
        host_controls="pause/resume/stop use existing host control; marked separately from learned responses",
        predefined_case_count=len(declared),
    )
    write(args.output / "metadata.json", metadata)
    started = time.monotonic()
    backend = ToyQ1RBackend(REPO / ".cache/microduck_head/model", checkpoint)
    metadata.update(backend=backend.metadata, load_seconds=time.monotonic() - started)
    write(args.output / "metadata.json", metadata)
    results = []
    for case in declared:
        print(f"BEGIN {case['id']}", flush=True)
        result = run_case(case, backend, args.output)
        results.append(result)
        write(
            args.output / "summary.json",
            dict(metadata=metadata, completed_cases=len(results), cases=results, wall_s=time.monotonic() - started),
        )
        print(
            json.dumps(
                dict(
                    case=case["id"],
                    unexpected=result["unexpected"],
                    replies=[r["response"]["reply"] for r in result["turns"]],
                    wall_s=result.get("wall_s"),
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
    from .q1r_evaluate import assess

    probe_results = []
    for probe in probes:
        response = backend.decide_with_images(probe["request"], {})
        trace = copy.deepcopy(backend.last_trace)
        probe_results.append(dict(**probe, response=response, trace=trace, score=assess(trace["raw_text"], probe)))
        write(args.output / "counterfactual_results.json", probe_results)
        print(
            json.dumps(dict(probe=probe["id"], response=response, rejected=trace.get("rejected")), ensure_ascii=False),
            flush=True,
        )


def counterfactuals():
    from .toy_fixtures import observed_ball, request

    base = request("请踢红球", planned=True)
    base["observation"]["entities"] = [observed_ball("red")]
    base["observation"]["entities"][0]["position_body_m"][2] = -0.08
    probes = []
    for name in (
        "normal",
        "sitting",
        "fallen",
        "motion_gate_closed",
        "stale_1s",
        "two_red_balls",
        "kick_unavailable",
        "not_visible_but_searchable",
    ):
        req = copy.deepcopy(base)
        decision, goals = "replace", [ball_goal("kick", "red")]
        if name == "sitting":
            req["observation"]["posture"] = "sitting"
            goals.insert(0, dict(type="posture", pose="standing"))
        elif name == "fallen":
            req["observation"]["posture"] = "fallen"
            decision, goals = "decline", []
        elif name == "motion_gate_closed":
            req["observation"]["environment"]["motion_allowed"] = False
            decision, goals = "decline", []
        elif name == "stale_1s":
            req["observation"]["age_s"] = 1.0
            decision, goals = "decline", []
        elif name == "two_red_balls":
            other = observed_ball("red", 1)
            other["position_body_m"][2] = -0.08
            req["observation"]["entities"].append(other)
            decision, goals = "clarify", []
        elif name == "kick_unavailable":
            req["capabilities"]["goal_types"].remove("kick")
            decision, goals = "clarify", []
        elif name == "not_visible_but_searchable":
            req["observation"]["entities"] = []
        probes.append(
            dict(
                id=name,
                request=req,
                expected=dict(decision=decision, goals=goals, reply="状态对照参考"),
                physical_executed=False,
                source="predeclared_synthetic_counterfactual",
            )
        )
    return probes


if __name__ == "__main__":
    main()
