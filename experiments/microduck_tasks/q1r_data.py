"""Generate and freeze diverse synthetic semantic Q1R train/development data.

No real image, physical trajectory, motor success, or human-review claim is made.
Development wording families never appear as training wording families. The
separate root-authored baseline suite is not read by this data generator.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .q1r_data_actions import generate_actions
from .q1r_data_boundaries import generate_boundaries
from .q1r_data_colloquial import generate_colloquial
from .q1r_data_context import generate_context
from .q1r_data_core import SOURCE, Builder, digest, dumps
from .q1r_data_dialogue import generate_dialogue
from .toy_inputs import validate_input
from .toy_protocol import VERSION, validate_output


def build():
    builder = Builder()
    for generate in (generate_actions, generate_dialogue, generate_context, generate_boundaries, generate_colloquial):
        generate(builder)
    return builder


def audit(rows):
    families, utterances, inputs = {}, {}, set()
    reports = {}
    for split in ("train", "val"):
        selected = [r for r in rows if r["split"] == split]
        for row in selected:
            if set(row) != {
                "id",
                "split",
                "group",
                "family_id",
                "request",
                "expected",
                "rationale",
                "source",
                "physical_verified",
                "human_reviewed",
            }:
                raise ValueError("Unexpected raw row fields.")
            validate_input(row["request"])
            validate_output(dict(schema_version=VERSION, **row["expected"]), row["request"])
            if set(row["expected"]) != {"decision", "goals", "reply"}:
                raise ValueError("Answer must supervise complete decision/goals/reply.")
            if row["source"] != SOURCE or row["physical_verified"] or row["human_reviewed"]:
                raise ValueError("Synthetic source provenance must remain explicit.")
            family = row["family_id"]
            if family in families and families[family] != split:
                raise ValueError("Language family crossed train/development split.")
            families[family] = split
            text = row["request"]["event"]["text"]
            if text in utterances and utterances[text] != split:
                raise ValueError(f"Exact target utterance crossed splits: {text}")
            utterances[text] = split
            key = digest(row["request"])
            if key in inputs:
                raise ValueError("Duplicate complete request.")
            inputs.add(key)
            if row["request"]["event"]["kind"] == "feedback" and row["expected"]["decision"] != "respond":
                raise ValueError("Executor feedback must not authorize motion.")
            if row["group"].startswith("chat_") and row["expected"]["decision"] != "respond":
                raise ValueError("Chat must leave the action queue alone.")
            execution = row["request"]["execution"]
            result = execution["last_result"]
            if execution["state"] == "failed" and (result is None or result["status"] != "failed"):
                raise ValueError("Synthetic failed execution must carry its matching failure result.")
            if result is not None and result["reason"] == "safe_pause_boundary":
                if execution["state"] != "paused" or execution["active"] != result["goal"]:
                    raise ValueError("Synthetic pause result must retain the paused active goal.")
        goals = [g for r in selected for g in r["expected"]["goals"]]
        reports[split] = dict(
            count=len(selected),
            families=len({r["family_id"] for r in selected}),
            distinct_target_utterances=len({r["request"]["event"]["text"] for r in selected}),
            groups=dict(sorted(Counter(r["group"] for r in selected).items())),
            decisions=dict(sorted(Counter(r["expected"]["decision"] for r in selected).items())),
            history_lengths=dict(sorted(Counter(len(r["request"]["history"]) for r in selected).items())),
            goal_lengths=dict(sorted(Counter(len(r["expected"]["goals"]) for r in selected).items())),
            self_actions=dict(sorted(Counter(g["action"] for g in goals if g["type"] == "perform").items())),
            goal_types=dict(sorted(Counter(g["type"] for g in goals).items())),
            maximum_reply_characters=max(len(r["expected"]["reply"]) for r in selected),
        )
    return dict(
        schema_validation_errors=0,
        duplicate_inputs=0,
        train_val_family_overlap=0,
        train_val_exact_utterance_overlap=0,
        train_val_input_overlap=0,
        result_state_inconsistencies=0,
        splits=reports,
        semantic_checks=[
            "Every full input and schema-wrapped complete output validated against current executable contract.",
            "Whole wording families stay in one split; skill concepts are intentionally shared.",
            "Images and physical validation are not claimed; observations and results are fictional.",
            "Chat and feedback authorize no goals; success language is grounded in explicit verified executor status.",
            "Cap 15 includes existing active/pending goals; unsupported goals clarify instead of partial execution.",
            "Known sitting adds standing before movement; unknown/fallen posture does not assume recovery.",
            "Numbers, colours and sampled context variants are not counted as independent language families.",
            "Independent baseline cases were not read or copied; only error categories informed new phrases.",
        ],
        human_reviewed=False,
        physical_verified=False,
        limitations=[
            "Synthetic labels can contain semantic mistakes despite validation and semantic tests.",
            "Validation measures held-out wording and contexts, not physical or open-domain chat success.",
            "Context/parameter expansion means row count is not independent language diversity.",
        ],
    )


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("raw_train.jsonl", "raw_val.jsonl", "manifest.json", "audit.json"):
        if (output / name).exists():
            raise FileExistsError(f"Frozen data already exists: {output / name}; use a new version directory.")
    builder = build()
    report = audit(builder.rows)
    files = {}
    for split in ("train", "val"):
        rows = [r for r in builder.rows if r["split"] == split]
        builder.rng.shuffle(rows)
        path = output / f"raw_{split}.jsonl"
        path.write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")
        files[split] = dict(path=path.name, count=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    sources = sorted(Path(__file__).parent.glob("q1r_data*.py")) + [Path(__file__).with_name("q1r_language.py")]
    manifest = dict(
        version="q1r-synthetic-semantic-rich-1",
        source=SOURCE,
        files=files,
        seed=1611,
        schema_version=VERSION,
        answer_fields=["decision", "goals", "reply"],
        maximum_goals=15,
        human_reviewed=False,
        physical_verified=False,
        raw_pixels=False,
        blind_suite_access=False,
        duplicate_generation_attempts_skipped=builder.duplicate_skips,
        implementation_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        split_policy="Whole hand-authored wording families stay in one split, without parameter-random splitting.",
        scope="Synthetic semantic supervision; no real camera, runtime or physical acceptance evidence claimed.",
        user_authorized_synthetic_training=True,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/microduck_tasks/outputs/q1r_20261116/data"))
    args = parser.parse_args()
    manifest, report = generate(args.output)
    print(dumps({"files": manifest["files"], "audit": report}))
