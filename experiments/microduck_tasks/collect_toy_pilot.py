"""Small actual-physics pilot covering dialogue, queues, and interruption."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def wait_sim(session, seconds):
    start, deadline = session.runtime.snapshot()["time_s"], time.monotonic() + 30
    while session.runtime.snapshot()["time_s"] < start + seconds:
        if session.worker_error is not None:
            raise RuntimeError("Pilot worker failed.") from session.worker_error
        if time.monotonic() > deadline:
            raise TimeoutError("Pilot action did not advance simulation time.")
        time.sleep(0.002)


def collect(output):
    from .action_runtime import LAYOUT
    from .scene import sample_layout
    from .toy_dataset import export, read_episode
    from .toy_motion import ToyRuntime
    from .toy_session import ToySession

    output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("multi_blue_red_turn", 201, False, ["你好，你叫什么", "你现在能看到什么", "先踢蓝球，再踢红球，再原地转一圈"]),
        ("multi_yellow_blue_turn", 202, True, ["你都会做什么", "先踢黄球，再踢蓝球，再向右转一圈"]),
        ("navigate_kick_turn", 203, True, ["先走到黄球旁边，再踢蓝球，再原地转一圈"]),
        ("posture_roll", 204, True, ["先点头，再坐下，再站起来", "做一下前滚翻", "先张嘴，再闭嘴"]),
        ("self_balance", 205, True, ["先抬左脚，再抬右脚", "先环顾四周，再左右摇摆"]),
        (
            "self_dance",
            206,
            True,
            ["做一下月球漫步", "跳个舞", "蹲下再站起来", "做一下坐站", "做一下俯身复原", "做一下害羞舞步"],
        ),
        ("finite_moves", 207, True, ["向前走1秒", "向后走1秒", "向左走1秒", "向右走1秒", "向右转90度"]),
        ("continuous_control", 208, True, []),
        ("append_and_cancel", 209, True, []),
        ("grounded_dialogue", 210, True, ["你好", "你现在能看到什么", "踢紫球", "飞起来", "谢谢", "踢红球"]),
        ("missing_target", 211, False, ["先踢蓝球，再踢红球，再原地转一圈"]),
    ]
    summaries = []
    for name, seed, random_layout, commands in cases:
        folder = output / "episodes" / name
        folder.mkdir(parents=True)
        runtime = ToyRuntime(folder, toy_xy=(1.5, 1.5), block_xy=(1.5, -1.5))
        layout = sample_layout(seed) if random_layout else dict(LAYOUT)
        if name == "missing_target":
            layout["blue"] = [10.0, 10.0]
        runtime.reset(layout)
        session = ToySession(runtime, seed=seed, split_group=name)
        session.recorder.metadata["layout"] = layout
        session.recorder.save_metadata()
        checks = {}
        error = None
        try:
            print(f"\nEpisode: {name}", flush=True)
            for command in commands:
                print(f"用户 > {command}", flush=True)
                session.run(command)
            if name == "continuous_control":
                session.run("一直往前走不要停", wait=False)
                wait_sim(session, 0.6)
                revision = session.execution["revision"]
                response = session.run("你现在能看到什么", wait=False)
                checks["visual_chat_preserves_task"] = (
                    response["decision"] == "respond"
                    and session.execution["revision"] == revision
                    and session.execution["state"] == "running"
                )
                session.run("暂停", wait=False)
                session.wait()
                checks["paused"] = session.execution["state"] == "paused"
                session.run("继续", wait=False)
                wait_sim(session, 0.4)
                session.run("停下", wait=False)
                checks["cancel_clears_queue"] = (
                    session.execution["active"] is None
                    and not session.execution["pending"]
                    and session.execution["state"] == "idle"
                )
            elif name == "append_and_cancel":
                session.run("向前走3秒", wait=False)
                response = session.run("再点头", wait=False)
                checks["append_accepted"] = response["decision"] == "append"
                session.wait()
                checks["append_order"] = [row["goal"]["type"] for row in session.results] == ["move", "perform"]
                session.run("先向后走5秒，再踢红球，再转一圈", wait=False)
                wait_sim(session, 0.4)
                task_id = session.task_id
                session.run("停下", wait=False)
                starts = [e for e in session.recorder.events if e["event"] == "goal_start" and e["task_id"] == task_id]
                checks["cancel_suppresses_tail"] = len(starts) == 1 and not session.execution["pending"]
            session.run("你现在能看到什么", wait=False)
            if name == "missing_target":
                checks["missing_target_reported_failed"] = (
                    len(session.results) == 1
                    and session.results[0]["status"] == "failed"
                    and session.results[0]["reason"] in ("step_limit", "target_not_found")
                    and runtime.failure is None
                )
                checks["tail_not_executed"] = not session.execution["pending"] and session.execution["active"] is None
            if name not in ("continuous_control", "append_and_cancel", "grounded_dialogue", "missing_target"):
                minimum = dict(
                    multi_blue_red_turn=3,
                    multi_yellow_blue_turn=3,
                    navigate_kick_turn=3,
                    posture_roll=6,
                    self_balance=4,
                    self_dance=7,
                    finite_moves=5,
                )
                checks["intended_actions_executed"] = len(session.results) >= minimum[name]
            if checks and not all(checks.values()):
                raise AssertionError(checks)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(error, flush=True)
        finally:
            session.close()
            runtime.save_trace()
            session.recorder.metadata.update(physics=runtime.physics, policy_sha256=runtime.policy.hashes)
            session.recorder.save_metadata()
            runtime.close()
        _, rows = read_episode(folder)
        summary = dict(
            episode_id=name,
            seed=seed,
            layout=layout,
            dialogue_rows=len(rows),
            checks=checks,
            error=error,
            expected_failure=name == "missing_target",
            goals=Counter(row["status"] for row in session.results),
            results=session.results,
            metadata=session.recorder.metadata,
        )
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        summaries.append(summary)
    dataset = export(output / "episodes", output / "dataset")
    report = dict(
        episodes=len(summaries),
        dataset=dataset,
        checks_passed=all(s["error"] is None and all(s["checks"].values()) for s in summaries),
        goals=Counter(row["status"] for s in summaries for row in s["results"]),
        errors={s["episode_id"]: s["error"] for s in summaries if s["error"]},
        failed_goals=[
            row for s in summaries for row in s["results"] if row["status"] == "failed" and not s["expected_failure"]
        ],
        expected_failures=[s["episode_id"] for s in summaries if s["expected_failure"]],
        summaries=[{k: s[k] for k in ("episode_id", "dialogue_rows", "checks", "goals", "error")} for s in summaries],
        performed_actions=sorted(
            {
                row["goal"]["action"]
                for s in summaries
                for row in s["results"]
                if row["success"] and row["goal"]["type"] == "perform"
            }
        ),
    )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    from .bootstrap import configure

    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = collect(args.output)
    if not report["checks_passed"] or report["failed_goals"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
