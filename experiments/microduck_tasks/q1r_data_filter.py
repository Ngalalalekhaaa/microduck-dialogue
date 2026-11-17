"""Exclude development user utterances without using development answer labels."""

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from .q1r_data import audit
from .q1r_data_core import digest, dumps


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def freeze_filtered(source, suite, output):
    source, suite, output = Path(source), Path(suite), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads((source / "manifest.json").read_text())
    train, val = read_rows(source / "raw_train.jsonl"), read_rows(source / "raw_val.jsonl")
    # Only extract the current input text. No expected answer, rationale or response
    # from the independent development suite is used to author or modify labels.
    dev_texts = {
        row["request"]["event"]["text"] for row in read_rows(suite) if row["request"]["event"]["kind"] == "user"
    }
    normalized = {normalize(text) for text in dev_texts}
    kept, excluded = [], []
    for row in train:
        text = row["request"]["event"]["text"]
        if row["request"]["event"]["kind"] == "user" and normalize(text) in normalized:
            excluded.append(
                dict(
                    id=row["id"],
                    family_id=row["family_id"],
                    group=row["group"],
                    current_text=text,
                    reason="exact" if text in dev_texts else "normalized",
                )
            )
        else:
            kept.append(row)
    before, after = audit(train + val), audit(kept + val)
    missing = set(before["splits"]["train"]["groups"]) - set(after["splits"]["train"]["groups"])
    if missing:
        raise ValueError(f"Filtering removed whole supervision groups: {sorted(missing)}")
    checks = dict(
        source_manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        development_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
        development_answer_labels_used=False,
        policy="NFKC, casefold, remove whitespace and Unicode punctuation; compare current user event.text only",
        development_distinct_user_texts=len(dev_texts),
        excluded_rows=len(excluded),
        exact_match_rows=sum(row["reason"] == "exact" for row in excluded),
        normalized_only_rows=sum(row["reason"] == "normalized" for row in excluded),
        excluded_distinct_current_texts=len({row["current_text"] for row in excluded}),
        before=before["splits"]["train"],
        after=after["splits"]["train"],
        remaining_normalized_current_user_text_overlap=0,
        validation_changed=False,
        limitation="Development exclusion is not blind testing; paraphrase and history overlap are not ruled out.",
    )
    (output / "raw_train.jsonl").write_text("".join(dumps(row) + "\n" for row in kept), encoding="utf-8")
    (output / "raw_val.jsonl").write_bytes((source / "raw_val.jsonl").read_bytes())
    (output / "excluded_development_texts.jsonl").write_text(
        "".join(dumps(row) + "\n" for row in excluded), encoding="utf-8"
    )
    manifest = dict(source_manifest)
    manifest["version"] = "q1r-synthetic-semantic-rich-final-1"
    manifest["development_text_exclusion"] = checks
    manifest["implementation_sha256"] = dict(source_manifest["implementation_sha256"])
    manifest["implementation_sha256"][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["files"] = {
        split: dict(
            path=f"raw_{split}.jsonl",
            count=len(rows),
            sha256=hashlib.sha256((output / f"raw_{split}.jsonl").read_bytes()).hexdigest(),
        )
        for split, rows in (("train", kept), ("val", val))
    }
    after["development_text_exclusion"] = checks
    after["semantic_checks"][-1] = "Development input texts used only for exclusion; no development gold labels used."
    for name, document in (("manifest.json", manifest), ("audit.json", after)):
        (output / name).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dict(
        files=manifest["files"],
        exclusion={k: v for k, v in checks.items() if k not in ("before", "after")},
        before_count=len(train),
        after_count=len(kept),
        families=after["splits"]["train"]["families"],
        groups=len(after["splits"]["train"]["groups"]),
        manifest_sha256=digest(manifest),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(dumps(freeze_filtered(args.source, args.suite, args.output)))
