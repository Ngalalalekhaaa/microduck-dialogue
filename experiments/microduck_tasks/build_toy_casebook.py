"""Export reviewable draft examples, never a train/val/test SFT dataset."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .toy_catalog import manifest
from .toy_context_examples import examples as context_examples
from .toy_examples import examples
from .toy_inputs import model_payload
from .toy_protocol import VERSION, validate_output
from .toy_schema import SYSTEM, response_schema


def check_case(row):
    if row["source"] != "hand_authored_design_fixture" or row["training_ready"] is not False:
        raise ValueError("Design fixtures cannot be relabeled as training-ready recordings.")
    if row["physical_verified"] is not False or row["media"]:
        raise ValueError("These design fixtures have no physical evidence or captured images.")
    validate_output(row["target"], row["input"])
    model_payload(row["input"])


def require_training_ready(row):
    # A separate, versioned live executor/exporter must exist before promotion.
    if row["source"] == "hand_authored_design_fixture" or "draft" in row["input"]["schema_version"]:
        raise ValueError("Draft design fixtures are not SFT data; collect matched online executor traces first.")
    raise ValueError("The v3 online trace exporter has not been implemented yet.")


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def build(output):
    rows = examples() + context_examples()
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate design case id.")
    for row in rows:
        check_case(row)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "design_cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "review.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            ["case_id", "capability_profile", "utterance", "reply", "decision", "goals", "history", "execution"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["case_id"],
                    row["capability_profile"],
                    row["input"]["event"]["text"],
                    row["target"]["reply"],
                    row["target"]["decision"],
                    *[
                        json.dumps(v, ensure_ascii=False)
                        for v in (row["target"]["goals"], row["input"]["history"], row["input"]["execution"])
                    ],
                ]
            )
    _json(output / "response.schema.json", response_schema())
    _json(output / "candidates.json", manifest())
    (output / "system_prompt.txt").write_text(SYSTEM, encoding="utf-8")
    source = Path(__file__).parent
    files = sorted(
        [
            *source.glob("toy_*.py"),
            source / "build_toy_casebook.py",
            *[source / name for name in ("action_catalog.py", "action_session.py", "action_playground.py")],
        ]
    )
    report = {
        "schema_version": VERSION,
        "training_ready": False,
        "physical_verified": False,
        "cases": len(rows),
        "families": len({row["family_id"] for row in rows}),
        "decisions": dict(Counter(row["target"]["decision"] for row in rows)),
        "profiles": dict(Counter(row["capability_profile"] for row in rows)),
        "validation": "strict structure and capability checks only; not language-model or robot accuracy",
        "split": "design_only; no training/evaluation split has been assigned",
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())},
    }
    _json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New review directory; existing directories are refused."
    )
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
