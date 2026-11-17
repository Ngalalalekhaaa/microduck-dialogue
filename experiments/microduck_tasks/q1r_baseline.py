"""Run the selected old Q1 once per independent probe, preserving every raw output."""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .q1r_suite import cases, extra_cases, write_suite
from .toy_protocol import validate_output
from .toy_teacher import _goal_label


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def canonical(goals):
    result = copy.deepcopy(goals)
    for goal in result:
        target = goal.get("target")
        if target and target.get("attributes"):
            target["attributes"].sort(key=lambda row: (row["name"], row["value"]))
            target["entity_id"] = None
    return result


def describe(plan):
    decisions = {
        "replace": "执行新计划",
        "append": "追加任务",
        "respond": "回答但不改任务",
        "clarify": "澄清后再决定",
        "decline": "拒绝当前运动",
        "cancel": "停止并清空",
        "pause": "安全暂停并保留任务",
        "resume": "恢复未完成任务",
    }
    goals = " → ".join(_goal_label(goal) for goal in plan["goals"])
    return decisions[plan["decision"]] + ("：" + goals if goals else "")


def changed_policy_case(row):
    return row["group"] == "unsupported" or (
        row["group"] == "pinyin_and_noisy_text" and row.get("expected", {}).get("decision") == "clarify"
    )


def assess(row, response, trace):
    expected = row["expected"]
    actual = {key: response[key] for key in ("decision", "goals")}
    host_response_plan_exact = actual["decision"] == expected["decision"] and canonical(actual["goals"]) == canonical(
        expected["goals"]
    )
    causes = []
    protocol = dict(accepted=True, reason=None)
    try:
        validate_output(response, row["request"])
    except (TypeError, ValueError) as exc:
        protocol = dict(accepted=False, reason=str(exc))
    rejected = trace.get("rejected")
    source = trace.get("decision_source", "unknown")
    raw_plan_exact = None
    if trace.get("model_called"):
        raw_plan = trace.get("raw_prediction")
        raw_plan_exact = (
            isinstance(raw_plan, dict)
            and set(raw_plan) == {"decision", "goals"}
            and not rejected
            and raw_plan["decision"] == expected["decision"]
            and canonical(raw_plan["goals"]) == canonical(expected["goals"])
        )
    plan_exact = bool(raw_plan_exact) if trace.get("model_called") else host_response_plan_exact
    if rejected:
        reason = rejected["reason"]
        if "Input has" in reason:
            kind = "input_context_limit"
            explanation = "当前长队列或历史超出旧Q1的4096输入token限制，模型尚未生成；属于宿主输入预算限制。"
        elif any(term in reason for term in ("unseen target", "Multiple objects", "bound entity")):
            kind = "target_grounding_rejected"
            explanation = "模型生成了当前观察或搜索能力无法唯一支持的目标，宿主校验拒绝，因此没有启动动作。"
        elif any(term in reason for term in ("capacity", "many goals", "unbounded", "until_cancelled")):
            kind = "queue_contract_rejected"
            explanation = "模型计划超过队列容量或把动作排在持续任务之后，宿主校验拒绝。"
        elif any(term in reason for term in ("posture", "standing", "motion gate", "observation", "edge guard")):
            kind = "state_contract_rejected"
            explanation = "模型没有正确使用当前姿态、观察新鲜度或运动安全条件，宿主校验拒绝。"
        else:
            kind = "protocol_rejected"
            explanation = "原始生成未通过结构化协议或运行时上下文校验，宿主返回拒绝；原始输出未被修复。"
        causes.append(dict(category=kind, explanation=explanation, evidence=reason))
    if not plan_exact and not (rejected and host_response_plan_exact):
        if changed_policy_case(row) and actual["decision"] == "decline" and not rejected:
            category = "policy_contract_update"
            explanation = (
                "旧P6提示要求库外请求decline，本轮用户改为clarify；这一差异是行为契约更新，不能单独归因为旧模型没学会。"
            )
        elif actual["decision"] != expected["decision"]:
            category = "host_control_mismatch" if source.startswith("host_") else "intent_or_state_misclassification"
            explanation = (
                f"语义预期为“{describe(expected)}”，实际被判断为“{describe(actual)}”；"
                f"当前执行状态是{row['request']['execution']['state']}，"
                f"姿态是{row['request']['observation']['posture']}。仅根据输出能定位意图或状态判断错误，"
                "不能据此断言模型内部的具体原因。"
            )
        elif len(actual["goals"]) != len(expected["goals"]):
            redundant_stand = (
                row["request"]["observation"]["posture"] == "standing"
                and actual["goals"][:1] == [dict(type="posture", pose="standing")]
                and canonical(actual["goals"][1:]) == canonical(expected["goals"])
            )
            category = "redundant_standing_transition" if redundant_stand else "omitted_or_extra_goals"
            explanation = (
                "已站立时额外安排了一次站起，属于冗余目标，不是左右动作颠倒。"
                if redundant_stand
                else f"期望{len(expected['goals'])}个目标，实际{len(actual['goals'])}个；有动作遗漏或额外生成。"
            ) + f"预期“{describe(expected)}”；实际“{describe(actual)}”。"
        else:
            category = "action_target_parameter_mismatch"
            explanation = (
                "目标数量相同但动作、对象、方向、参数、顺序或重复次数不一致。"
                f"预期“{describe(expected)}”；实际“{describe(actual)}”。"
            )
        causes.append(dict(category=category, explanation=explanation, evidence=dict(expected=expected, actual=actual)))
    reply_gap = (
        expected["decision"] == "respond" and trace.get("model_called") and trace.get("reply_source") == "template"
    )
    if reply_gap:
        causes.append(
            dict(
                category="reply_capability_gap",
                explanation="旧Q1只训练decision/goals，聊天由固定模板替代，未回答具体问题；属于尚未实现的语言回复能力。",
                evidence=response["reply"],
            )
        )
    return dict(
        plan_exact=plan_exact,
        raw_plan_exact=raw_plan_exact,
        host_response_plan_exact=host_response_plan_exact,
        expected_decision=expected["decision"],
        reply_capability_gap=bool(reply_gap),
        semantic_reply_scored=False,
        reply_exact_match_used=False,
        protocol=protocol,
        model_called=bool(trace.get("model_called")),
        model_inference_completed=bool(trace.get("generation_timing")),
        decision_source=source,
        unexpected=bool(causes),
        causes=causes,
    )


def summary(records, total, elapsed):
    counts = Counter()
    grouped = defaultdict(lambda: dict(total=0, plan_exact=0, unexpected=0, reply_capability_gap=0))
    for record in records:
        assessment = record["assessment"]
        stats = grouped[record["group"]]
        stats["total"] += 1
        for field in ("plan_exact", "unexpected", "reply_capability_gap"):
            stats[field] += int(assessment[field])
        counts.update(cause["category"] for cause in assessment["causes"])
    plan_ok = sum(record["assessment"]["plan_exact"] for record in records)
    stable = [record for record in records if not changed_policy_case(record)]
    supported = [record for record in records if record["assessment"].get("expected_decision") in ("replace", "append")]
    return dict(
        completed=len(records),
        total=total,
        plan_exact=plan_ok,
        plan_exact_rate=plan_ok / len(records) if records else None,
        model_calls=sum(record["assessment"]["model_called"] for record in records),
        model_plan_exact=sum(
            record["assessment"]["plan_exact"] for record in records if record["assessment"]["model_called"]
        ),
        host_response_plan_exact=sum(record["assessment"].get("host_response_plan_exact", False) for record in records),
        host_control_cases=sum(not record["assessment"]["model_called"] for record in records),
        unexpected_cases=sum(record["assessment"]["unexpected"] for record in records),
        reply_capability_gaps=sum(record["assessment"]["reply_capability_gap"] for record in records),
        categories=dict(counts),
        unchanged_policy_cases=len(stable),
        unchanged_policy_exact=sum(record["assessment"]["plan_exact"] for record in stable),
        supported_action_cases=len(supported),
        supported_action_exact=sum(record["assessment"]["plan_exact"] for record in supported),
        groups=dict(grouped),
        seconds=elapsed,
        physical_executed=False,
        old_blind_access=False,
        training_allowed=False,
        scoring_note="计划语义精确比较；有效实体绑定ID可等价。回复不做字符串精确分数；旧模板缺失单列。",
    )


def review(output):
    """Apply only explicitly logged gold corrections without rerunning any prediction."""
    output = Path(output)
    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    suite = [json.loads(line) for line in (output / "suite.jsonl").read_text().splitlines()]
    live_gold = {row["id"]: row for row in cases() + extra_cases()}
    gold = {row["id"]: live_gold[row["id"]] for row in suite}
    correction_path = output / "gold_corrections.json"
    corrections = json.loads(correction_path.read_text()) if correction_path.exists() else {}
    lines = ["Q1 语义压力测试逐次异常说明", "实际物理执行：否；这是语言规划和宿主协议检查。", ""]
    provenance_path = output / "provenance_correction.json"
    provenance_correction = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if provenance_correction:
        lines += ["来源纠正：测试预期由编码代理独立构造，未经人工审核；原始文件错误的human标签保留审计。", ""]
    for record in records:
        row = gold[record["id"]]
        if record["source"] != row["source"]:
            if not provenance_correction:
                raise ValueError("A provenance change needs an explicit correction record.")
            record["original_provenance"] = record["source"]
            record["source"] = row["source"]
        record["human_reviewed"] = False
        if row["request"] != record["request"]:
            raise ValueError("A review must never replace the actual model input.")
        if row["expected"] != record["expected"]:
            correction = corrections[record["id"]]
            if correction["corrected_expected_goals"] != row["expected"]["goals"]:
                raise ValueError("An unlogged gold change cannot be silently rescored.")
            record["original_expected"] = record["expected"]
            record["gold_correction"] = correction
            record["expected"] = row["expected"]
        record["original_assessment"] = record["assessment"]
        record["assessment"] = assess(row, record["response"], record["model_trace"])
        if not record["assessment"]["unexpected"]:
            continue
        lines += [f"{record['id']} [{record['group']}] {record['request']['event']['text']}"]
        lines += ["预期：" + json.dumps(record["expected"], ensure_ascii=False)]
        lines += ["原始输出：" + str(record["raw"])]
        lines += ["宿主返回：" + json.dumps(record["response"], ensure_ascii=False)]
        lines += [f"原因 [{item['category']}]：{item['explanation']}" for item in record["assessment"]["causes"]]
        lines.append("")
    for name, items in (("reviewed_predictions.jsonl", records), ("reviewed_suite.jsonl", list(gold.values()))):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items))
    (output / "unexpected_review.txt").write_text("\n".join(lines))
    status = json.loads((output / "status.json").read_text())
    report = summary(records, len(gold), status["seconds"])
    report.update(
        gold_corrections=corrections,
        original_predictions_preserved=True,
        review_state=status["state"],
        provenance_correction=provenance_correction,
        human_reviewed=False,
    )
    save(output / "reviewed_summary.json", report)
    return report


def combine_runs(root):
    """Join the two disjoint runs by id for matched model-call comparisons after training."""
    root = Path(root)
    records, reports, identities = [], [], []
    for name in ("baseline", "baseline_extra"):
        directory = root / name
        if json.loads((directory / "status.json").read_text())["state"] != "complete":
            raise ValueError("Both baseline runs must finish before publishing the combined result.")
        reports.append(review(directory))
        identities.append(json.loads((directory / "model_identity.json").read_text()))
        for line in (directory / "reviewed_predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["source_run"] = name
            row["eligible_model_comparison"] = row["assessment"]["model_called"]
            records.append(row)
    if identities[0]["generator"]["weights_sha256"] != identities[1]["generator"]["weights_sha256"]:
        raise ValueError("The old model weights must be identical across baseline runs.")
    suite = [json.loads(line) for line in (root / "suite.jsonl").read_text().splitlines()]
    expected_ids = {row["id"] for row in suite}
    if len(records) != len(expected_ids) or {row["id"] for row in records} != expected_ids:
        raise ValueError("The combined baseline must contain every development id exactly once.")
    (root / "baseline_reviewed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records)
    )
    report = summary(records, len(suite), sum(item["seconds"] for item in reports))
    report.update(
        model_plan_exact_rate=report["model_plan_exact"] / report["model_calls"],
        original_runs_preserved=True,
        source_runs=["baseline", "baseline_extra"],
        human_reviewed=False,
        source="agent_authored_synthetic_semantic",
        gold_corrections=reports[0]["gold_corrections"],
        provenance_correction=reports[0]["provenance_correction"],
        comparison_rule="Compare only ids with model_called=true in both runs; host controls are not model ability.",
        score_rule="Validator rejection cannot count as correct raw model planning even if host fallback matches gold.",
    )
    save(root / "baseline_summary.json", report)
    (root / "baseline_unexpected.txt").write_text(
        "\n\n".join((root / name / "unexpected_review.txt").read_text() for name in ("baseline", "baseline_extra"))
    )
    return report


def run(args):
    root = Path(__file__).resolve().parents[2]
    overlay = root / ".cache/microduck_head/qlora_deps"
    if str(overlay) not in sys.path:
        sys.path.insert(0, str(overlay))
    from .toy_q1_backend import ToyQ1Backend

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.jsonl").exists():
        raise ValueError("Refusing to overwrite or retry prior baseline; select a fresh output directory.")
    metadata = write_suite(output, extra=args.extra)
    rows = [json.loads(line) for line in (output / "suite.jsonl").read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    started = time.monotonic()
    save(
        output / "status.json", dict(state="loading_q1", pid=os.getpid(), completed=0, total=len(rows), suite=metadata)
    )
    backend = ToyQ1Backend(args.model_dir, args.checkpoint)
    save(output / "model_identity.json", backend.metadata)
    save(
        output / "execution_manifest.json",
        dict(
            pid=os.getpid(),
            checkpoint=args.checkpoint,
            physical_executed=False,
            retry=False,
            generation="greedy",
            fixture_source="agent_authored_synthetic_semantic",
            suite_sha256=metadata["sha256"],
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
    records = []
    with (output / "predictions.jsonl").open("w") as predictions, (output / "unexpected.jsonl").open("w") as unexpected:
        for row in rows:
            save(
                output / "status.json",
                dict(
                    state="running",
                    pid=os.getpid(),
                    current_id=row["id"],
                    current_request=row["request"]["event"]["text"],
                    **summary(records, len(rows), time.monotonic() - started),
                ),
            )
            response = backend.decide_with_images(copy.deepcopy(row["request"]), None)
            trace = copy.deepcopy(backend.last_trace)
            record = dict(
                **row,
                response=response,
                raw=trace.get("raw_text"),
                model_trace=trace,
                assessment=assess(row, response, trace),
            )
            predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            predictions.flush()
            if record["assessment"]["unexpected"]:
                unexpected.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                unexpected.flush()
            records.append(record)
            if len(records) % 10 == 0:
                print(
                    json.dumps(
                        dict(
                            completed=len(records),
                            total=len(rows),
                            last=row["id"],
                            plan_exact=sum(r["assessment"]["plan_exact"] for r in records),
                        ),
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    report = summary(records, len(rows), time.monotonic() - started)
    save(output / "summary.json", report)
    save(output / "status.json", dict(state="complete", pid=os.getpid(), **report))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/microduck_tasks/outputs/q1r_20261116/baseline")
    parser.add_argument("--model-dir", default=".cache/microduck_head/model")
    parser.add_argument(
        "--checkpoint", default="experiments/microduck_tasks/outputs/staged_plan_20261116/q1/train1200/checkpoint_100"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--extra", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--combine-root")
    options = parser.parse_args()
    if options.combine_root:
        print(json.dumps(combine_runs(options.combine_root), ensure_ascii=False, indent=2))
    elif options.review_only:
        print(json.dumps(review(options.output), ensure_ascii=False, indent=2))
    else:
        run(options)
