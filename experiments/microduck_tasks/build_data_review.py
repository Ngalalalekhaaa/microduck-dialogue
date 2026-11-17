"""Build a local, offline data browser without modifying training examples."""

import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "outputs/training_dataset_final_20261114"
OUTPUT = ROOT / "outputs/data_quality_review_20261115"


def read(path):
    return json.loads(path.read_text())


def url(path):
    return Path(os.path.relpath(path, OUTPUT)).as_posix()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = read(DATASET / "manifest.json")
    source = {(x["episode_id"], x["task_id"]): x for x in manifest["accepted_tasks"]}
    predictions = {}
    for name, path in (
        ("QLoRA", ROOT / "outputs/qlora_20261114/predictions.jsonl"),
        ("ONNX INT8", ROOT / "outputs/onnx_test_20261115/evaluation8_compact/predictions.jsonl"),
    ):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            prediction = row.get("prediction")
            target = row["target"]
            matched = isinstance(prediction, dict) and all(
                prediction.get(k) == target[k] for k in ("type", "skill", "seconds", "target_color")
            )
            predictions.setdefault(row["id"], {})[name] = {"prediction": prediction, "matched": matched}
    highlights = read(OUTPUT / "audit_highlights.json") if (OUTPUT / "audit_highlights.json").exists() else {}
    highlight_rows = highlights if isinstance(highlights, list) else highlights.get("highlights", [])
    highlighted = {r["sample_id"]: r for r in highlight_rows if "sample_id" in r}
    summaries = {}
    records = []
    for split in ("train", "val", "test"):
        for index, line in enumerate((DATASET / f"{split}.jsonl").read_text().splitlines()):
            row = json.loads(line)
            row.pop("messages")
            entry = source[(row["episode_id"], row["task_id"])]
            directory = Path(entry["path"])
            if str(directory) not in summaries:
                summaries[str(directory)] = read(directory / "summary.json")
            result = next(
                (r for r in summaries[str(directory)].get("results", []) if r["task_id"] == row["task_id"]), {}
            )
            row["review"] = {
                "split": split,
                "index": index,
                "line": index + 1,
                "included": True,
                "phase": row["source"]["rule_phase_audit_only"],
                "task_kind": row["source"]["rule_task_kind_audit_only"],
                "images": {k: url(DATASET / v) for k, v in row["images"].items()},
                "source_summary": url(directory / "summary.json"),
                "source_events": url(directory / "events.jsonl"),
                "task_result": result,
                "models": predictions.get(row["sample_id"], {}),
                "highlight": highlighted.get(row["sample_id"]),
            }
            records.append(row)
    failed_tasks = []
    for entry in manifest["excluded"]:
        if not entry["reason"].startswith("task_not_successful:"):
            continue
        directory = Path(entry["path"])
        summary = read(directory / "summary.json")
        result = next((r for r in summary.get("results", []) if r["task_id"] == entry["task_id"]), {})
        failed_tasks.append({**entry, "result": result})
        for index, line in enumerate((directory / "events.jsonl").read_text().splitlines()):
            event = json.loads(line)
            if event["task_id"] != entry["task_id"]:
                continue
            obs = event["observation"]
            decision = event.get("decision") or {}
            records.append(
                {
                    "sample_id": f"{entry['episode_id']}:{event['event_id']}",
                    "episode_id": entry["episode_id"],
                    "task_id": entry["task_id"],
                    "scene_seed": entry["scene_seed"],
                    "instruction": event["instruction"],
                    "state": {
                        **{k: obs[k] for k in ("yaw", "stationary", "tilt_deg")},
                        "sim_time_s": obs["time_s"],
                        "detections_rgbd": obs["detections"],
                    },
                    "context": event.get("context", {}),
                    "target": {
                        "type": "skill" if decision else "record",
                        "skill": decision.get("skill"),
                        "seconds": decision.get("seconds", 0),
                        "target_color": event["intent"].get("color"),
                        "reply": event.get("reply", ""),
                    },
                    "source": {"event_id": event["event_id"], "execution": event["execution"]},
                    "review": {
                        "split": "excluded",
                        "index": index,
                        "line": index + 1,
                        "included": False,
                        "phase": event["phase"],
                        "task_kind": "kick",
                        "exclusion": entry["reason"],
                        "images": {k: url(directory / v) for k, v in obs["images"].items()},
                        "source_summary": url(directory / "summary.json"),
                        "source_events": url(directory / "events.jsonl"),
                        "task_result": result,
                        "models": {},
                        "raw_event": event,
                    },
                }
            )
    audit = read(OUTPUT / "audit_dataset.json") if (OUTPUT / "audit_dataset.json").exists() else {}
    payload = {
        "created": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "dataset": str(DATASET),
        "counts": manifest["counts"],
        "excluded_counts": manifest["excluded_counts"],
        "records": records,
        "failed_tasks": failed_tasks,
        "audit": audit,
        "highlights": highlight_rows,
        "split_rows": dict(Counter(r["review"]["split"] for r in records)),
    }
    (OUTPUT / "review_data.js").write_text(
        "window.REVIEW_DATA=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
    )
    for name in ("index.html", "review.js"):
        shutil.copyfile(ROOT / "review_assets" / name, OUTPUT / name)
    build = {
        "rows": len(records),
        "split_rows": payload["split_rows"],
        "failed_tasks": len(failed_tasks),
        "highlight_count": len(highlight_rows),
        "source_dataset_sha256": manifest["jsonl_sha256"],
        "scope": "Read-only browser. Excluded raw decisions are not training labels. Manual notes stay in browser.",
    }
    (OUTPUT / "browser_build.json").write_text(json.dumps(build, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**build, "open": str(OUTPUT / "index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
