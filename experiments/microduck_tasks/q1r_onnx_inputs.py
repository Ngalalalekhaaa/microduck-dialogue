"""Freeze actual interactive requests for paired backend timing, without inference."""

import argparse
import hashlib
import json
from pathlib import Path

from .toy_fixtures import kick, perform
from .toy_inputs import validate_input
from .toy_protocol import VERSION, validate_output


def checksum(value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def read_events(path):
    return {row["event_id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}


def build(output):
    root = Path(__file__).parent / "outputs"
    desktop = root / "toy_desktop_20261117_103749/episode_001/events.jsonl"
    closed = root / "q1r_20261116/evaluation_20261117/closed_loop"
    selections = [
        (desktop, "event_000001", "desktop_001_greeting", "greeting", "respond", [], "你好，我在。"),
        (
            desktop,
            "event_000003",
            "desktop_003_capabilities",
            "long_capability_answer",
            "respond",
            [],
            "我可以执行当前开放的自身动作、移动、转身、靠近和踢红黄蓝球，也可以聊天。",
        ),
        (
            desktop,
            "event_000009",
            "desktop_009_dance_pair",
            "dance_composition",
            "replace",
            [perform("caught_dancing"), perform("moonwalk")],
            "我会先做害羞舞，再做太空步。",
        ),
        (
            desktop,
            "event_000016",
            "desktop_016_ambiguous_kick",
            "ambiguous_target",
            "clarify",
            [],
            "请说明要踢红球、黄球还是蓝球。",
        ),
        (
            desktop,
            "event_000017",
            "desktop_017_red_then_yellow",
            "elliptical_kick_sequence",
            "replace",
            [kick("red"), kick("yellow")],
            "收到，我会先踢红球，再踢黄球。",
        ),
        (
            desktop,
            "event_000025",
            "desktop_025_referent",
            "history_referent",
            "replace",
            [kick("yellow")],
            "你指的是刚才走近的黄球，我会踢黄球。",
        ),
        (
            desktop,
            "event_000026",
            "desktop_026_block",
            "unsupported_object",
            "clarify",
            [],
            "当前不支持走近或踢方块，可以改成红、黄或蓝球。你想选哪个？",
        ),
        (
            desktop,
            "event_000027",
            "desktop_027_green_capability",
            "unsupported_capability_question",
            "respond",
            [],
            "目前不能可靠识别或踢绿球，支持的是红球、黄球和蓝球。",
        ),
        (
            closed / "03_nod_left_leg/events.jsonl",
            "event_000007",
            "feedback_left_leg_wrong_reply",
            "feedback_historical_wrong_reply",
            "respond",
            [],
            "抬左脚已完成，并且执行验收通过。",
        ),
        (
            closed / "05_kick_red/events.jsonl",
            "event_000004",
            "feedback_red_kick_correct_reply",
            "feedback_historical_correct_reply",
            "respond",
            [],
            "踢红球已完成，并通过接触和位移验收。",
        ),
    ]
    notes = {
        "desktop_017_red_then_yellow": "前文踢球正在补颜色，两项均应踢球；冻结输出把黄色误改为靠近。",
        "desktop_025_referent": "前文踢一下等待目标；已验收走近黄球，当前观察新鲜。冻结输出误称观察过期。",
        "desktop_027_green_capability": "这是能力询问，不能执行动作，也不能声称支持能力表之外的绿球。",
        "feedback_left_leg_wrong_reply": "已验收的是抬左脚，冻结回复却说踢红球；参考输出不是正确标签。",
        "feedback_red_kick_correct_reply": "last_result明确踢红球成功并验收，冻结回复正确。",
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "benchmark_inputs.jsonl"
    manifest_path = output / "benchmark_inputs_manifest.json"
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("Benchmark inputs are immutable; use a fresh output directory.")
    records = []
    for path, event_id, identifier, group, decision, goals, reply in selections:
        event = read_events(path)[event_id]
        request = validate_input(event["request"])
        trace = event["model_trace"]
        expected = dict(decision=decision, goals=goals, reply=reply)
        validate_output(dict(schema_version=VERSION, **expected), request)
        if not trace["model_called"] or trace["reply_source"] != "model_generation":
            raise ValueError("Benchmark references must be actual Q1R model generations.")
        records.append(
            dict(
                id=identifier,
                group=group,
                request=request,
                request_sha256=checksum(request),
                expected=expected,
                expected_provenance="agent_semantic_review_of_actual_request_not_reference_copy",
                human_reviewed=False,
                expected_reply_exact_match_required=False,
                semantic_note=notes.get(identifier, "按本轮明确请求与实际能力标注。"),
                source=dict(
                    path=str(path.resolve()),
                    event_id=event_id,
                    file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    event_sha256=checksum(event),
                    historical_simulation_time_s=event["time_s"],
                ),
                frozen_reference=dict(
                    raw_text=trace["raw_text"],
                    response=event["response"],
                    timing=trace["generation_timing"],
                    backend=trace["backend"],
                    historical_turn_seconds=trace["seconds"],
                    fair_benchmark=False,
                    timing_role="historical_reference_only",
                ),
                media_passed_to_model=False,
                physical_replay_required=False,
            )
        )
    destination.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
    manifest = dict(
        version="q1r-onnx-actual-inputs-1",
        rows=len(records),
        user_requests=8,
        executor_feedback=2,
        ids=[r["id"] for r in records],
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        timing_policy="Historical desktop/closed-loop timings are context only, never the paired speed baseline.",
        comparison_policy="Use identical requests; distinguish timing, output consistency and correctness.",
        limitations=[
            "No blind benchmark read.",
            "References intentionally retain existing model errors.",
            "Expected replies are agent-authored semantic guides, not exact-string targets or human labels.",
        ],
        input_extraction_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/microduck_tasks/outputs/q1r_onnx_20261117"))
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
