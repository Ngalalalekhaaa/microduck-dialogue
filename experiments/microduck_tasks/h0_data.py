"""Build grouped common-domain training data and seal a separate blind split."""

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from .ablation_contract import checksum, common_domain, dumps, validate_context, wrap_plan
from .h0_language import action_request, inventory, phrase, random_plans
from .h0_scenarios import boundary_case, context, queue_case, state_case
from .run_ablation import save
from .toy_dataset import digest
from .toy_fixtures import observed_ball, request
from .toy_inputs import model_payload, validate_input

VERSION = "microduck-h0-data-1"


def base_state():
    state = request("待填当前事件", planned=True)
    state["capabilities"]["max_goals"] = 6
    state["observation"]["entities"] = [observed_ball(color, i) for i, color in enumerate(("red", "yellow", "blue"))]
    return state


def make_row(
    state, text, decision, goals, split, group, index, template, *, context_name="explicit_state", alternatives=()
):
    state["event"]["text"] = text
    expected = dict(decision=decision, goals=copy.deepcopy(goals))
    validate_input(state)
    validate_context(wrap_plan(expected), state)
    if not common_domain(goals):
        raise ValueError("H0 generated labels must round-trip through the unchanged small-head vocabulary.")
    signature = checksum(goals)
    family = f"{template}:{signature[:12]}"
    return dict(
        id=f"{split}_{group}_{index:04}",
        split=split,
        group=group,
        request=state,
        expected=expected,
        target=wrap_plan(expected),
        family_id=family,
        split_group=family,
        episode_id=None,
        template_family=template,
        source_kind="synthetic_semantic",
        human_reviewed=False,
        physical_verified=False,
        requires_pixels=False,
        images={},
        context=context_name,
        domain="common",
        goal_count=len(goals),
        plan_signature=signature,
        acceptable_decisions=[decision, *alternatives],
    )


def generate(split):
    if split not in ("train", "blind"):
        raise ValueError("Only construct the new training and sealed blind splits.")
    rows, atoms = [], inventory()
    single = 300 if split == "train" else 80
    for i in range(single):
        state = base_state()
        name = context(state, i)
        goals = [copy.deepcopy(atoms[i % len(atoms)])]
        text, template = action_request(goals, split, i)
        rows.append(make_row(state, text, "replace", goals, split, "single", i, template, context_name=name))
    counts = {length: 100 for length in range(2, 7)} if split == "train" else {2: 40, 3: 40, 4: 27, 5: 27, 6: 26}
    for length, count in counts.items():
        for i, goals in enumerate(random_plans(count, length, 4100 + length + (1000 if split == "blind" else 0))):
            state = base_state()
            name = context(state, i)
            text, template = action_request(goals, split, i)
            group = f"multi{length}" if split == "train" else "two_three" if length <= 3 else "four_six"
            rows.append(
                make_row(state, text, "replace", goals, split, group, length * 100 + i, template, context_name=name)
            )
    for i in range(120 if split == "train" else 80):
        state = base_state()
        goal = copy.deepcopy(atoms[(i // 12 * 7 + i % 3) % len(atoms)])
        text, decision, goals, mode = queue_case(state, goal, split, i)
        if split == "train":
            text += ("", "这是现在的队列操作。", "请处理这次任务变更。")[(i // 12) % 3]
        else:
            text += ("", "按此更新这轮任务。", "我指的是现在这次执行。")[(i // 12) % 3]
        rows.append(
            make_row(
                state,
                text,
                decision,
                goals,
                split,
                "interaction" if split == "train" else "queue",
                i,
                f"{split}:queue:{mode}:{i // 36}",
            )
        )
    for i in range(60 if split == "train" else 80):
        state = base_state()
        text, decision, goals, mode = state_case(state, split, i)
        # Resume/completed and permission/freshness counterfactual pairs share a family.
        pair = 3 if mode in (3, 4, 11) else 5 if mode in (5, 6, 10) else mode
        row = make_row(
            state,
            text,
            decision,
            goals,
            split,
            "interaction" if split == "train" else "state",
            200 + i,
            f"{split}:state:{pair}:{i // 12}",
        )
        row["family_id"] = row["split_group"] = f"{split}:state_pair:{pair}:{i // 12}"
        rows.append(row)
    if split == "train":
        for i in range(120):
            state = base_state()
            name = context(state, i if i < 90 else (i - 90) % 10 + (i - 90) // 10)
            word = phrase(atoms[i % len(atoms)], split)
            texts = (
                f"我只想了解能力，不是下命令：你支持{word}吗？",
                f"关于{word}这项能力，能向我介绍一下吗？不必实际表演。",
            )
            smalltalk = (
                "早上好呀，小鸭。",
                "告诉我你的名字吧。",
                "今天跟你聊天很开心。",
                "谢谢你陪着我。",
                "我们聊聊天，不用表演动作。",
                "我刚刚看了一本有趣的书。",
                "今天的天气真不错啊。",
                "你好，我是你的新朋友。",
                "我想听你介绍一下自己。",
                "你在这里陪我就很好。",
            )
            text = smalltalk[(i - 90) % len(smalltalk)] if i >= 90 else texts[i // len(atoms)]
            template = (
                f"train:chat:{(i - 90) % 10}" if i >= 90 else f"train:capability:{i // len(atoms)}:{i % len(atoms)}"
            )
            rows.append(make_row(state, text, "respond", [], split, "chat", i, template, context_name=name))
    for i in range(100 if split == "train" else 80):
        state = base_state()
        goal = atoms[(i // 10 * 3) % len(atoms)]
        text, decision, goals, mode = boundary_case(state, goal, split, i)
        # Distinct surface formulations, capped at three per scenario source.
        if split == "train":
            text += ("", "请不要另作解释或替换动作。", "请按这一轮的原意理解。")[(i // 10) % 3]
        else:
            text += ("", "这是我完整的意思。", "就按本条要求判断。")[(i // 10) % 3]
        context(state, i // 10)
        rows.append(make_row(state, text, decision, goals, split, "boundary", i, f"{split}:boundary:{mode}:{i // 30}"))
    return rows


def validation_rows(suite):
    original = json.loads((suite / "dev240.json").read_text())
    if any(row["request"]["capabilities"]["max_goals"] != 6 for row in original):
        raise ValueError("Legacy H0 requires an explicit max_goals=6 validation suite; use its frozen data.")
    return [
        dict(
            copy.deepcopy(r),
            split="val",
            target=wrap_plan(r["expected"]),
            split_group=r["family_id"],
            template_family=f"original_dev:{r['group']}",
            episode_id=None,
            plan_signature=checksum(r["expected"]["goals"]),
        )
        for r in original
    ]


def audit(splits):
    owners = {name: {} for name in ("id", "family_id", "template_family", "input", "utterance")}
    result = {}
    for split, rows in splits.items():
        seen_inputs = set()
        for row in rows:
            validate_context(wrap_plan(row["expected"]), row["request"])
            key = checksum(model_payload(row["request"]))
            if key in seen_inputs:
                raise ValueError(f"Duplicate model input inside {split}: {row['id']}")
            seen_inputs.add(key)
            facts = {**row, "input": key, "utterance": "".join(row["request"]["event"]["text"].split())}
            for field, mapping in owners.items():
                previous = mapping.setdefault(facts[field], split)
                if previous != split:
                    raise ValueError(f"Cross-split {field}: {row['id']}")
            if split != "val" and not common_domain(row["expected"]["goals"]):
                raise ValueError("Generated label is outside the common domain.")
        contexts = defaultdict(Counter)
        for row in rows:
            for goal in row["expected"]["goals"]:
                contexts[goal.get("action", goal["type"])][row["context"]] += 1
        result[split] = dict(
            rows=len(rows),
            groups=dict(Counter(r["group"] for r in rows)),
            decisions=dict(Counter(r["expected"]["decision"] for r in rows)),
            unique_action_plans=len({r["plan_signature"] for r in rows if r["expected"]["goals"]}),
            family_count=len({r["family_id"] for r in rows}),
            action_contexts=dict(contexts),
        )
    result["cross_split_exact_input_or_utterance_overlap"] = 0
    result["human_reviewed"] = False
    result["limitations"] = [
        "Synthetic authored templates, not independent real users.",
        "Family IDs represent authored provenance, not proof against all semantic similarity.",
        "Existing dev240 is a reused development set, not a blind test.",
    ]
    return result


def freeze(suite, output):
    splits = dict(train=generate("train"), val=validation_rows(suite), blind=generate("blind"))
    if [len(splits[k]) for k in splits] != [1200, 240, 480]:
        raise ValueError("Expected exactly 1200 train, 240 development and 480 sealed blind decisions.")
    checks = audit(splits)
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        path = output / ("sealed/blind480.jsonl" if split == "blind" else f"{split}.jsonl")
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(dumps(row) + "\n" for row in rows))
    manifest = dict(
        version=VERSION,
        input_view="P6",
        training_splits=["train", "val"],
        files_sha256={s: digest(output / f"{s}.jsonl") for s in ("train", "val")},
        sealed_blind_sha256=digest(output / "sealed/blind480.jsonl"),
        blind_rows=480,
        development_suite_sha256=checksum(json.loads((suite / "dev240.json").read_text())),
        human_reviewed=False,
        blind_predictions_made=False,
        audit=checks,
        blind_release_policy=(
            "Freeze route, checkpoint, seed rule, input renderer, decoding and scoring before first blind inference."
        ),
    )
    save(output / "manifest.json", manifest)
    (output / "sealed/blind480.jsonl").chmod(0o444)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(dumps(freeze(args.suite, args.output)), flush=True)


if __name__ == "__main__":
    main()
