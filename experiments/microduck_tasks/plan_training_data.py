"""Export successful audited teacher traces to v2; never label model predictions.

Seeds remain in their collection split. Future observations and oracle acceptance
stay out of student inputs; actual future actions provide bounded chunk labels.
"""

import argparse
import copy
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from .dataset import _asset, _demonstration_source, _json, _load_episodes, split_for_seed
from .plan_inputs import BOUNDARY_CONTRACT, LEGACY_CONTRACT, contract_digest
from .plan_protocol import CATALOG, SYSTEM, VERSION, entities, parse_plan, validate_input
from .plan_teacher_data import task_rows as semantic_task_rows
from .training_data import _completed


def chunk(actions, index, target):
    steps, total = [], 0.0
    while index < len(actions) and len(steps) < 4:
        event = actions[index]
        decision = event["decision"]
        skill, seconds = decision["skill"], decision["seconds"]
        if target is None and skill.startswith("kick_"):
            break
        if target is not None and target["id"] not in {e["id"] for e in entities(event["observation"])}:
            break
        navigation = event["intent"]["kind"] == "kick" and event["phase"] in {
            "face",
            "approach",
            "align",
            "avoid",
            "settle",
        }
        if navigation:
            if target is None:
                break
            skill, seconds = "approach_target", 0.0
            while index < len(actions):
                future = actions[index]
                if future["phase"] not in {"face", "approach", "align", "avoid", "settle"}:
                    break
                duration = future["decision"]["seconds"]
                if seconds + duration > 2.000001 or total + seconds + duration > 3.000001:
                    break
                seconds += duration
                index += 1
            if seconds == 0:
                break
        else:
            index += 1
        if total + seconds > 3.000001:
            break
        steps.append({"skill": skill, "seconds": round(seconds, 6)})
        total += seconds
        if skill.startswith("kick_"):
            break
    return steps


def task_rows(events, result):
    final = events[-1]
    if sum(e["done"] for e in events) != 1:
        raise ValueError("requires_one_terminal_event")
    if not final["done"] or final["status"] not in ("success", "clarification"):
        raise ValueError("task_not_successful")
    if final["intent"]["kind"] == "stop":
        raise ValueError("operator_stop_not_model_label")
    if final["intent"]["kind"] == "kick" and not (result or {}).get("evaluation", {}).get("success"):
        raise ValueError("kick_not_independently_verified")
    executed = [e for e in events if e["decision"] is not None]
    if any(not _completed(e) for e in executed):
        raise ValueError("incomplete_action_or_recovery")
    if any(e["observation"]["time_s"] > e["execution"]["start_time_s"] + 1e-6 for e in executed):
        raise ValueError("future_observation")
    actions = [
        e
        for e in events
        if e["decision"] is not None
        and e["phase"] != "recover"
        and not (final["intent"]["kind"] in ("move", "turn") and e["phase"] == "settle")
    ]
    if any(not _completed(e) for e in actions):
        raise ValueError("incomplete_action")
    if any(
        abs(e["execution"]["end_time_s"] - e["execution"]["start_time_s"] - e["decision"]["seconds"]) > 0.0201
        for e in actions
    ):
        raise ValueError("action_duration_mismatch")
    history = []
    for index, event in enumerate([*actions, final]):
        obs = event["observation"]
        color = final["intent"].get("color")
        target = next(
            ({k: e[k] for k in ("id", "category", "attributes")} for e in entities(obs) if e["id"] == f"ball:{color}"),
            None,
        )
        steps = chunk(actions, index, target) if event is not final else []
        status = (
            "continue"
            if steps
            else "clarify"
            if final["status"] == "clarification"
            else ("answer" if final["intent"]["kind"] == "describe" else "finish")
        )
        if not steps and event is not final:
            continue
        label = parse_plan(
            _json(
                {
                    "schema_version": VERSION,
                    "status": status,
                    "target": target,
                    "steps": steps,
                    "reply": final["reply"] if event is final else "",
                }
            )
        ).to_dict()
        state = {key: obs[key] for key in ("yaw", "stationary", "tilt_deg")}
        state.update(
            entities=entities(obs),
            observation_id=obs["frame_id"],
            task_elapsed_s=round(obs["time_s"] - events[0]["observation"]["time_s"], 6),
            available_skills=CATALOG,
            schema_version=VERSION,
        )
        row = {
            "format_version": "microduck-plan-sft-2",
            "instruction": event["instruction"],
            "state": state,
            "context": {"task_id": event["task_id"], "committed_steps": [], "recent_outcomes": history[-6:]},
            "target": label,
            "system_prompt": SYSTEM,
        }
        validate_input(row["state"], row["context"])
        yield event, row
        if event is not final:
            actual = {key: event["decision"][key] for key in ("skill", "seconds")}
            continuation = chunk(actions, index + 1, target)
            if continuation and not actual["skill"].startswith("kick_"):
                ahead = copy.deepcopy(row)
                ahead["variant"] = "continuation"
                ahead["context"]["committed_steps"] = [actual]
                ahead["target"]["steps"] = continuation
                parse_plan(_json(ahead["target"]))
                yield event, ahead
            history = [*history, {"step": actual, "outcome": "completed", "target": target}]


def export(input_root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new v2 dataset output directory.")
    samples = {split: [] for split in ("train", "val", "test")}
    copies, excluded, assignments, contracts = {}, [], {}, set()
    for root, metadata, events in _load_episodes(Path(input_root)):
        if not _demonstration_source(metadata, events):
            excluded.append({"episode": metadata["episode_id"], "reason": "not_a_rule_demonstration"})
            continue
        meta = metadata.get("metadata", {})
        semantic = meta.get("teacher") == "efficient_plan" and meta.get("input_contract") == BOUNDARY_CONTRACT
        if semantic and meta.get("input_contract_sha256") != contract_digest():
            raise ValueError("The collected semantic executor changed; use its pinned source to export these traces.")
        if meta.get("teacher") != "efficient" and not semantic:
            excluded.append({"episode": metadata["episode_id"], "reason": "requires_efficient_teacher"})
            continue
        contracts.add(BOUNDARY_CONTRACT if semantic else LEGACY_CONTRACT)
        if len(contracts) != 1:
            raise ValueError("Do not mix primitive relabels with semantic boundary recordings.")
        seed = metadata["seed"]
        split = metadata.get("metadata", {}).get("training_split", split_for_seed(seed))
        if split not in samples or seed in assignments and assignments[seed] != split:
            raise ValueError("A scene seed cannot cross dataset splits.")
        assignments[seed] = split
        summary = json.loads((root / "summary.json").read_text())
        results = {r["task_id"]: r for r in summary["results"]}
        tasks = defaultdict(list)
        for event in events:
            tasks[event["task_id"]].append(event)
        for task, task_events in tasks.items():
            try:
                rows = list((semantic_task_rows if semantic else task_rows)(task_events, results.get(task)))
            except (ValueError, TypeError, KeyError) as exc:
                excluded.append({"episode": metadata["episode_id"], "task": task, "reason": str(exc)})
                continue
            for event, row in rows:
                row.update(
                    id=f"{metadata['episode_id']}:{event['event_id']}:{row.get('variant', 'current')}",
                    scene_seed=seed,
                    episode_id=metadata["episode_id"],
                    task_id=task,
                    images={},
                )
                for camera, source in event["observation"]["images"].items():
                    dest = f"images/{metadata['episode_id']}/{Path(source).name}"
                    row["images"][camera] = dest
                    copies[dest] = _asset(root, source)
                samples[split].append(row)
    output.mkdir(parents=True)
    hashes = {}
    for relative, source in copies.items():
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[relative] = hashlib.sha256(dest.read_bytes()).hexdigest()
    file_hashes = {}
    for split, rows in samples.items():
        encoded = "".join(_json(row) + "\n" for row in rows).encode()
        (output / f"{split}.jsonl").write_bytes(encoded)
        file_hashes[split] = hashlib.sha256(encoded).hexdigest()
    manifest = {
        "format_version": "microduck-plan-sft-2",
        "input_contract": next(iter(contracts), None),
        "input_contract_sha256": contract_digest() if contracts == {BOUNDARY_CONTRACT} else None,
        "prompt_version": VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "counts": {split: len(rows) for split, rows in samples.items()},
        "seed_splits": assignments,
        "excluded": excluded,
        "jsonl_sha256": file_hashes,
        "image_sha256": hashes,
        "training_started": False,
        "scope": "simulation teacher, not model success",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.input, args.output)
    print(json.dumps({"counts": result["counts"], "excluded": len(result["excluded"])}, indent=2))


if __name__ == "__main__":
    main()
