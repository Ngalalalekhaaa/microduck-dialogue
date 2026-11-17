"""Report real collection coverage, repetitions, and scene/family split isolation."""

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .toy_dataset import digest, dumps
from .toy_train_probe import read_records


def _normalized(value):
    return "".join(
        char.casefold()
        for char in unicodedata.normalize("NFKC", value)
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def _numeric_identity(value):
    """Compare finite JSON numbers by value without mutating samples or booleans."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Plan identity requires finite numbers.")
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {key: _numeric_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numeric_identity(item) for item in value]
    return value


def _label(goal):
    goal = _numeric_identity(goal)
    kind = goal["type"]
    if kind == "perform":
        return f"perform:{goal['action']}:x{goal['repeat']}"
    if kind in ("kick", "navigate"):
        color = next((item["value"] for item in goal["target"]["attributes"] if item["name"] == "color"), "unspecified")
        return f"{kind}:{color}"
    if kind == "turn":
        return f"turn:{goal['direction']}:{goal['angle_deg']}"
    if kind == "move":
        return f"move:{goal['direction']}:{goal['limit']['kind']}"
    return f"posture:{goal['pose']}"


def _coverage(rows):
    utterances, normalized, templates = Counter(), Counter(), Counter()
    decisions, goal_types, actions, combinations, lengths, feedback = (Counter() for _ in range(6))
    plans = set()
    for row in rows:
        event, target = row["request"]["event"], row["target"]
        decisions[target["decision"]] += 1
        lengths[len(target["goals"])] += 1
        if event["kind"] == "user":
            utterances[event["text"]] += 1
            normalized[_normalized(event["text"])] += 1
            templates[re.sub(r"\d+(?:\.\d+)?|[红蓝黄]", "<value>", _normalized(event["text"]))] += 1
        else:
            feedback[row["request"]["execution"]["last_result"]["status"]] += 1
        if target["goals"]:
            plans.add(dumps(_numeric_identity(target["goals"])))
            combinations[" -> ".join(_label(goal) for goal in target["goals"])] += 1
        for goal in target["goals"]:
            goal_types[goal["type"]] += 1
            if goal["type"] == "perform":
                actions[goal["action"]] += 1
    return {
        "rows": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "families": len({row["split_group"] for row in rows}),
        "scene_seeds": len({row["seed"] for row in rows}),
        "user_rows": sum(utterances.values()),
        "feedback_rows": sum(feedback.values()),
        "unique_user_utterances": len(utterances),
        "unique_normalized_user_utterances": len(normalized),
        "unique_color_number_templates": len(templates),
        "repeated_normalized_user_rows": sum(normalized.values()) - len(normalized),
        "most_repeated_utterances": dict(utterances.most_common(12)),
        "unique_exact_plans": len(plans),
        "unique_goal_sequences": len(combinations),
        "decisions": dict(decisions),
        "goal_types": dict(goal_types),
        "self_actions": dict(actions),
        "goal_sequences": dict(combinations),
        "goals_per_response": dict(lengths),
        "feedback_status": dict(feedback),
    }


def _overlaps(rows_by_split, field):
    seen = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            seen[str(row[field])].add(split)
    return {key: sorted(splits) for key, splits in seen.items() if len(splits) > 1}


def _difference_hash(path):
    with Image.open(path) as image:
        pixels = image.convert("L").resize((9, 8)).tobytes()
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def _media_coverage(root, rows_by_split):
    counts, by_split, perceptual = Counter(), defaultdict(set), defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            for camera, relative in row["images"].items():
                sha = row["image_sha256"][camera]
                if sha not in counts:
                    perceptual[_difference_hash(root / relative)].add(sha)
                counts[sha] += 1
                by_split[sha].add(split)
    same_bucket = sum(len(values) * (len(values) - 1) // 2 for values in perceptual.values())
    adjacent = 0
    for value, hashes in perceptual.items():
        for bit in range(64):
            neighbor = value ^ (1 << bit)
            if neighbor > value:
                adjacent += len(hashes) * len(perceptual.get(neighbor, ()))
    return {
        "camera_frame_references": sum(counts.values()),
        "unique_png_sha256": len(counts),
        "repeated_exact_frame_references": sum(counts.values()) - len(counts),
        "image_hashes_shared_across_splits": sum(len(splits) > 1 for splits in by_split.values()),
        "distinct_image_pairs_with_dhash_distance_at_most_1": same_bucket + adjacent,
        "near_duplicate_method": "64-bit grayscale difference hash; compare distinct PNG hashes, Hamming distance <= 1",
        "interpretation": "Stationary views can repeat legitimately; coarse visual similarity does not prove leakage.",
    }


def _source_coverage(source_root, included):
    statuses, kinds, by_action, source_versions, validations = (Counter() for _ in range(5))
    found = set()
    for path in sorted(Path(source_root).rglob("metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("episode_id") not in included or not (path.parent / "events.jsonl").exists():
            continue
        episode = metadata["episode_id"]
        if episode in found:
            raise ValueError(f"Repeated source episode id {episode}; use the accepted episodes directory.")
        found.add(episode)
        kinds[metadata.get("collection_kind", "unspecified")] += 1
        validations[str(metadata.get("collection_validated", "legacy_unspecified"))] += 1
        source_versions[hashlib.sha256(dumps(metadata.get("source_sha256", {})).encode()).hexdigest()] += 1
        for line in (path.parent / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "goal_result":
                statuses[event["status"]] += 1
                by_action[f"{_label(event['goal'])}:{event['status']}"] += 1
    if found != included:
        raise ValueError("The source directory does not contain every exported episode.")
    return {
        "episodes": len(found),
        "collection_kinds": dict(kinds),
        "collection_validation": dict(validations),
        "goal_results": dict(statuses),
        "goal_results_by_action": dict(by_action),
        "source_code_versions": len(source_versions),
    }


def audit(dataset_root, *, source_root=None):
    """Validate exported rows and report coverage without claiming trained quality."""
    root = Path(dataset_root)
    manifest = json.loads((root / "manifest.json").read_text())
    rows_by_split = {}
    for split, expected in manifest["files_sha256"].items():
        path = root / f"{split}.jsonl"
        if digest(path) != expected:
            raise ValueError("Split checksum changed after dataset export.")
        rows_by_split[split] = read_records(path) if path.stat().st_size else []
    rows = [row for group in rows_by_split.values() for row in group]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated training-row ids in the exported dataset.")
    isolation = {field: _overlaps(rows_by_split, field) for field in ("episode_id", "seed", "split_group")}
    if any(isolation.values()):
        raise ValueError("Split leakage: an episode, scene seed, or paraphrase family appears in multiple splits.")
    report = {
        "dataset_manifest_sha256": digest(root / "manifest.json"),
        "training_ready": manifest["training_ready"],
        "deployment_ready": False,
        "split_rule": manifest["split_rule"],
        "split_isolation_passed": True,
        "splits": {split: _coverage(group) for split, group in rows_by_split.items()},
        "overall": _coverage(rows),
        "media": _media_coverage(root, rows_by_split),
        "limits": [
            "Counts describe rule-teacher coverage, not VLM performance or execution success rates.",
            "Raw utterance counts include color/number substitutions; template counts expose this inflation.",
            "Distinct seeded scenes can still produce identical views; RGB hashes are reported, not hidden.",
            "Shared words and generic greetings across splits are expected; family and seed isolation are enforced.",
        ],
    }
    if source_root is not None:
        report["source"] = _source_coverage(source_root, {row["episode_id"] for row in rows})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new audit report path.")
    report = audit(args.dataset, source_root=args.source)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(dumps({"audit": str(args.output), "rows": report["overall"]["rows"], "passed": True}))


if __name__ == "__main__":
    main()
