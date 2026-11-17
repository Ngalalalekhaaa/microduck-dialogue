"""Paired development reports, family summaries, and predeclared finalist selection."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .ablation_prompt import PARENTS
from .ablation_score import fraction, summarize
from .run_ablation import save


def load_rows(directory):
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json")) if p.stem not in ("progress", "report")]


def details(rows):
    common_actions = [r for r in rows if r["domain"] == "common" and r["expected"]["goals"]]
    full_actions = [r for r in rows if r["domain"] == "full_only"]
    model_route = [r for r in rows if not r["host_control"]]
    control_route = [r for r in rows if r["host_control"]]
    confusion = Counter(
        (r["expected"]["decision"], (r["score"]["parsed"] or {}).get("decision", "invalid_json")) for r in rows
    )
    return dict(
        common_action_exact=fraction(common_actions, "accepted_exact"),
        full_only_action_exact=fraction(full_actions, "accepted_exact"),
        model_route_exact=fraction(model_route, "accepted_exact"),
        host_route_model_exact=fraction(control_route, "accepted_exact"),
        host_route_count=len(control_route),
        all_rows_use_raw_model=True,
        host_routing_not_executed=True,
        families={
            family: fraction([r for r in rows if r["family_id"] == family], "accepted_exact")
            for family in sorted({r["family_id"] for r in rows})
        },
        decision_confusion=[dict(expected=a, predicted=b, count=n) for (a, b), n in sorted(confusion.items())],
        action_refused_ids=[
            r["id"]
            for r in rows
            if r["expected"]["goals"]
            and (r["score"]["parsed"] or {}).get("decision") in ("respond", "clarify", "decline")
        ],
        full_only_cases=[
            dict(id=r["id"], expected=r["expected"], actual=r["score"]["parsed"], score=r["score"])
            for r in full_actions
        ],
    )


def paired(parent, child):
    previous = {r["id"]: r for r in parent}
    current = {r["id"]: r for r in child}
    ids = sorted(previous.keys() & current.keys())
    if any(previous[i]["request"] != current[i]["request"] for i in ids):
        raise ValueError("A paired comparison must use exactly the same input facts.")
    fixed = [i for i in ids if current[i]["score"]["accepted_exact"] and not previous[i]["score"]["accepted_exact"]]
    broken = [i for i in ids if previous[i]["score"]["accepted_exact"] and not current[i]["score"]["accepted_exact"]]
    return dict(
        pairs=len(ids),
        improved_ids=fixed,
        regressed_ids=broken,
        exact_rate_change=(len(fixed) - len(broken)) / len(ids) if ids else None,
        small_correlated_development_sample=True,
    )


def rank_key(item):
    """Prefer fewer raw false starts, then complete action plans and legality."""
    name, report = item
    metrics = report["summary"]["metrics"]
    return (
        -metrics["nonaction_attempted_start"]["correct"],
        metrics["action_plan_exact"]["rate"] or 0,
        metrics["context_valid"]["rate"] or 0,
        metrics["accepted_exact"]["rate"] or 0,
        name,
    )


def report(root):
    configs = json.loads((root / "config.json").read_text())
    rows = {v: load_rows(root / v) for v in PARENTS if (root / v).exists()}
    reports = {
        v: dict(summary=summarize(items, expected_count=len(configs["case_ids"])), details=details(items))
        for v, items in rows.items()
    }
    pairs = {v: paired(rows[parent], rows[v]) for v, parent in PARENTS.items() if v in rows and parent in rows}
    complete = all(v in rows and len(rows[v]) == 96 for v in PARENTS)
    ranked = sorted(reports.items(), key=rank_key, reverse=True) if complete else []
    result = dict(
        complete_screen=complete,
        variants=reports,
        paired=pairs,
        finalist_selection=(
            "raw false starts ascending, action exact descending, context legality descending, "
            "total exact descending; ties by name"
        ),
        finalists=[name for name, _ in ranked[:2]],
        blind_test_used=False,
        training_started=False,
        physical_executions=0,
    )
    save(root / "comparison.json", result)
    lines = [
        "# P 阶段离线开发评估",
        "",
        "首次原始生成；没有重试、修复或教师兜底。以下分数不表示物理执行成功。",
        "",
        "| 方案 | 完成 | 协议/状态合法 | 动作完整计划 | 3–6步 | 非动作误启动尝试 | P50秒 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, value in reports.items():
        summary = value["summary"]
        metrics = summary["metrics"]
        counts = [
            f"{metrics[k]['correct']}/{metrics[k]['total']}"
            for k in ("context_valid", "action_plan_exact", "three_six_exact", "nonaction_attempted_start")
        ]
        latency = summary["costs"]["p50_seconds"]
        latency_text = f"{latency:.2f}" if latency is not None else "未运行"
        lines.append(
            f"| {variant} | {summary['completed']}/{summary['expected']} | "
            + " | ".join(counts)
            + f" | {latency_text} |"
        )
    lines += [
        "",
        "P1对P0、P2对P1、P3对P2、P4/P5/P6分别对P3；"
        "逐条改善/退化、共同域/完整域、宿主路由和决策混淆见 comparison.json。",
        "",
        "96条是筛查，不用于宣布通过原型门槛。240条也属于合成语义开发集；视觉真值、自然回复质量、异步与真实机器人尚未验收。",
    ]
    (root / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = report(args.root)
    print(json.dumps(dict(complete_screen=result["complete_screen"], finalists=result["finalists"])))


if __name__ == "__main__":
    main()
