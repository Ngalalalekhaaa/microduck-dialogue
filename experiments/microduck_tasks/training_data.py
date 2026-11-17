"""Export rule demonstrations for the exact deployed, per-step skill interface.

Oracle acceptance filters whole tasks, never inputs or model claims. Stored
intent/phase fields are audit facts only. Recovery is an executor operation;
the planner's separate 0.6 second alignment stand remains a learned choice.
"""

import argparse
import copy
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .dataset import _asset, _demonstration_source, _json, _load_episodes, split_for_seed
from .skill_tasks import guard_skill
from .skill_vlm import (
    PROMPT_VERSION,
    SKILLS,
    SYSTEM_PROMPT,
    _input_context,
    _input_state,
    parse_proposal,
    task_elapsed_s,
)

FORMAT_VERSION = "microduck-skill-sft-1"
SPLITS = ("train", "val", "test")
SKILL_REPLIES = {
    "stand": "保持站立，等待身体稳定。",
    "move_forward": "向前走一小段。",
    "move_backward": "向后退一小段。",
    "move_left": "向左移动一小段。",
    "move_right": "向右移动一小段。",
    "turn_left": "向左转一小段。",
    "turn_right": "向右转一小段。",
    "kick_left": "执行左脚踢球动作。",
    "kick_right": "执行右脚踢球动作。",
}


def execution_feedback(proposal):
    """Match SkillTaskSession feedback, including its stand-only convention."""
    return (
        f"executed_and_recovered: 已实际执行 {proposal['skill']} {proposal['seconds']:.2f} 秒并恢复站立。"
        "若原指令只要求走/转一下，单次动作已完成，应结束该任务；"
        "若原指令要求踢球，则根据新画面继续判断，执行过动作不代表踢球成功。"
    )


def _completed(event):
    execution = event["execution"]
    return execution["started"] and execution["completed"] and execution["end_time_s"] > execution["start_time_s"]


def _is_recovery(event, preceding):
    if preceding is None or preceding["decision"]["skill"] == "stand":
        return False
    kick = preceding["decision"]["skill"].startswith("kick_")
    decision = event["decision"]
    return (
        decision["skill"] == "stand"
        and decision["seconds"] == (3.0 if kick else 1.0)
        and (event["phase"] == ("recover" if kick else "settle") or event.get("controller_source") == "executor")
    )


def _actions(events):
    """Pair each model motion with its completed, fixed executor recovery."""
    actions, recoveries, pending = [], [], None
    for event in events:
        if event["decision"] is None:
            continue
        if not _completed(event):
            raise ValueError("incomplete_action")
        execution = event["execution"]
        elapsed = execution["end_time_s"] - execution["start_time_s"]
        if not math.isclose(elapsed, event["decision"]["seconds"], rel_tol=0, abs_tol=0.0201):
            raise ValueError("executed_duration_differs_from_proposal")
        if event["observation"]["time_s"] > execution["start_time_s"] + 1e-6:
            raise ValueError("action_observation_is_from_the_future")
        if _is_recovery(event, pending):
            if event["execution"]["start_time_s"] < pending["execution"]["end_time_s"] - 1e-6:
                raise ValueError("recovery_precedes_action_completion")
            recoveries.append(event)
            pending = None
            continue
        if pending is not None:
            raise ValueError("missing_completed_executor_recovery")
        if event["phase"] == "recover" or event.get("controller_source") == "executor":
            raise ValueError("unpaired_executor_recovery")
        actions.append(event)
        if event["decision"]["skill"] != "stand":
            pending = event
    if pending is not None:
        raise ValueError("missing_completed_executor_recovery")
    return actions, recoveries


def _target(kind, skill=None, seconds=0, color=None, reply=""):
    return parse_proposal(
        _json({"type": kind, "skill": skill, "seconds": seconds, "target_color": color, "reply": reply})
    ).to_dict()


def _task_rows(events, metadata, result, rgbd_assist):
    finals = [event for event in events if event["done"]]
    if len(finals) != 1 or finals[0] is not events[-1]:
        raise ValueError("missing_or_ambiguous_terminal_event")
    final = finals[0]
    kind = final["intent"]["kind"]
    if kind == "stop":
        raise ValueError("operator_stop_bypasses_model")
    if kind not in ("kick", "move", "turn", "describe", "clarify"):
        raise ValueError("unsupported_rule_task")
    required_status = "clarification" if kind == "clarify" else "success"
    if final["status"] != required_status:
        raise ValueError(f"task_not_successful:{final['status']}")
    evaluation = result.get("evaluation") if result else None
    if kind == "kick" and not (
        result
        and result.get("status") == "success"
        and isinstance(evaluation, dict)
        and evaluation.get("success") is True
    ):
        raise ValueError("kick_missing_independent_success_evaluation")
    if any(event["instruction"] != final["instruction"] for event in events):
        raise ValueError("instruction_changed_within_task")
    if any(event["intent"].get("kind") != kind for event in events):
        raise ValueError("intent_changed_within_task")
    actions, recoveries = _actions(events)
    if kind in ("describe", "clarify") and actions:
        raise ValueError("non_motion_task_executed_motion")
    if kind in ("move", "turn") and len(actions) != 1:
        raise ValueError("single_motion_requires_one_executed_action")
    if kind == "kick" and sum(event["decision"]["skill"].startswith("kick_") for event in actions) != 1:
        raise ValueError("kick_requires_one_executed_kick")
    color = final["intent"].get("color") if kind == "kick" else None
    context = {"previous_action": None, "previous_feedback": "", "history": [], "step": 0, "task_id": final["task_id"]}
    prior_color = events[0].get("context", {}).get("selected_color")
    if prior_color is not None:
        context["selected_color"] = prior_color
    task_started = events[0]["observation"]["time_s"]
    rows = []
    for event in actions:
        decision = event["decision"]
        skill = decision["skill"]
        proposal = _target("skill", skill, decision["seconds"], color, SKILL_REPLIES.get(skill, ""))
        guard = guard_skill(parse_proposal(_json(proposal)), event["observation"])
        if not guard["accepted"]:
            raise ValueError(f"deployed_guard_rejected:{guard['reason']}")
        rows.append(_row(event, metadata, context, proposal, rgbd_assist, task_started))
        feedback = execution_feedback(proposal)
        context.update(previous_action=proposal, previous_feedback=feedback, step=context["step"] + 1)
        if proposal["target_color"] is not None:
            context["selected_color"] = proposal["target_color"]
        context["history"] = (context["history"] + [{"action": proposal, "feedback": feedback}])[-6:]
    if kind == "kick":
        reply = "已执行踢球动作并恢复站立；是否踢中目标需要独立验收。"
    else:
        reply = final["reply"]
    terminal_type = {"describe": "answer", "clarify": "clarify"}.get(kind, "finish")
    rows.append(
        _row(final, metadata, context, _target(terminal_type, color=color, reply=reply), rgbd_assist, task_started)
    )
    return rows, recoveries


def _row(event, metadata, context, target, rgbd_assist, task_started):
    observation = event["observation"]
    if set(observation["images"]) != {"front", "near"}:
        raise ValueError("training_requires_front_and_near_rgb")
    state = {key: observation[key] for key in ("yaw", "stationary", "tilt_deg")}
    state["sim_time_s"] = task_elapsed_s(observation["time_s"], task_started)
    if rgbd_assist:
        state["detections_rgbd"] = observation["detections"]
    state, context = _input_state(state), _input_context(context)
    images = {
        camera: (Path("images") / metadata["episode_id"] / Path(relative).name).as_posix()
        for camera, relative in observation["images"].items()
    }
    data = {"instruction": event["instruction"], "state": state, "context": context}
    content = []
    for camera in ("front", "near"):
        content.extend(
            [{"type": "text", "text": f"Current camera: {camera}"}, {"type": "image", "image": images[camera]}]
        )
    content.append({"type": "text", "text": json.dumps(data, ensure_ascii=False, allow_nan=False)})
    return {
        "format_version": FORMAT_VERSION,
        "sample_id": f"{metadata['episode_id']}:{event['event_id']}",
        "episode_id": metadata["episode_id"],
        "task_id": event["task_id"],
        "scene_seed": metadata["seed"],
        "images": images,
        **data,
        "target": copy.deepcopy(target),
        "label": copy.deepcopy(target),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
            {"role": "assistant", "content": _json(target)},
        ],
        "source": {
            "event_id": event["event_id"],
            "frame_id": observation["frame_id"],
            "rule_phase_audit_only": event["phase"],
            "rule_task_kind_audit_only": event["intent"]["kind"],
            "execution": copy.deepcopy(event["execution"]),
        },
    }


def _stats(rows):
    text_lengths = [
        len(SYSTEM_PROMPT) + len(row["messages"][1]["content"][-1]["text"]) + len(row["messages"][-1]["content"])
        for row in rows
    ]
    return {
        "samples": len(rows),
        "scene_seeds": sorted({row["scene_seed"] for row in rows}),
        "episodes": len({row["episode_id"] for row in rows}),
        "tasks": len({(row["episode_id"], row["task_id"]) for row in rows}),
        "types": dict(Counter(row["target"]["type"] for row in rows)),
        "skills": dict(Counter(row["target"]["skill"] for row in rows if row["target"]["skill"])),
        "colors": dict(Counter(row["target"]["target_color"] or "none" for row in rows)),
        "rule_phases_audit_only": dict(Counter(row["source"]["rule_phase_audit_only"] for row in rows)),
        "text_characters_not_tokens": {
            f"p{percentile}": int(np.percentile(text_lengths, percentile)) if text_lengths else 0
            for percentile in (0, 50, 90, 95, 99, 100)
        },
        "missing_skills": [skill for skill in SKILLS if not any(row["target"]["skill"] == skill for row in rows)],
    }


def _recording_roots(root):
    """Snapshot a collection's finalized cases, excluding active/retry folders."""
    summary_path = root / "summary.json"
    document = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if document.get("format_version") != "microduck-rule-training-collection-1":
        return [root], None
    roots = []
    for case in document.get("cases", []):
        directory = Path(case["output"]).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("A collection case path escapes its recording root.")
        if (directory / "episode.json").is_file():
            roots.append(directory)
    return roots, {
        "root": str(root.resolve()),
        "planned_cases": document.get("planned_total"),
        "completed_cases": len(document.get("cases", [])),
        "cases_with_recordings": len(roots),
        "plan_sha256": document.get("plan_sha256"),
        "updated_at": document.get("updated_at"),
    }


def export_training_dataset(input_roots, output_root, *, seed_splits=None, rgbd_assist=True):
    """Build a new portable dataset; no inference, training or source mutation."""
    if isinstance(input_roots, (str, Path)):
        input_roots = [input_roots]
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"Choose a new training dataset directory: {output_root}")
    if seed_splits is not None and any(
        type(seed) is not int or split not in SPLITS for seed, split in seed_splits.items()
    ):
        raise ValueError("seed_splits maps integer scene seeds to train, val or test.")
    episodes, identifiers, collection_snapshots = [], set(), []
    for root in input_roots:
        roots, snapshot = _recording_roots(Path(root))
        if snapshot:
            collection_snapshots.append(snapshot)
        for recording_root in roots:
            for episode in _load_episodes(recording_root):
                identifier = episode[1]["episode_id"]
                if identifier in identifiers:
                    raise ValueError(f"Duplicate episode_id across source roots: {identifier}")
                identifiers.add(identifier)
                episodes.append(episode)
    if not episodes:
        raise ValueError("Provide at least one recording root.")
    samples, copies, excluded, accepted, ignored = {split: [] for split in SPLITS}, {}, [], [], Counter()
    assignments = {}
    for root, metadata, events in episodes:
        seed = metadata["seed"]
        if seed_splits is not None and seed not in seed_splits:
            raise ValueError(f"Explicit seed split is missing scene seed {seed}.")
        declared_split = metadata.get("metadata", {}).get("training_split")
        split = seed_splits[seed] if seed_splits is not None else declared_split or split_for_seed(seed)
        if split not in SPLITS or declared_split is not None and declared_split != split:
            raise ValueError(f"Explicit seed split disagrees with collection metadata for seed {seed}.")
        if seed in assignments and assignments[seed] != split:
            raise ValueError(f"Scene seed {seed} has conflicting split assignments.")
        assignments[seed] = split
        source = {"episode_id": metadata["episode_id"], "scene_seed": seed, "split": split, "path": str(root.resolve())}
        if not _demonstration_source(metadata, events):
            excluded.append({**source, "reason": "model_prediction_not_demonstration"})
            continue
        collection = metadata.get("metadata", {})
        if collection.get("collection_kind") == "direct_dialogue" and collection.get("collection_verified") is not True:
            excluded.append({**source, "reason": "direct_collection_not_independently_verified"})
            continue
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        results = {result["task_id"]: result for result in summary.get("results", [])}
        tasks = defaultdict(list)
        for event in events:
            if "task_id" not in event:
                raise ValueError(f"Every training event needs an explicit task_id: {root}")
            tasks[event["task_id"]].append(event)
        for task_id, task_events in tasks.items():
            try:
                rows, recoveries = _task_rows(task_events, metadata, results.get(task_id), rgbd_assist)
            except (ValueError, KeyError, TypeError) as error:
                excluded.append({**source, "task_id": task_id, "reason": str(error), "raw_events": len(task_events)})
                continue
            samples[split].extend(rows)
            ignored.update(executor_recovery=len(recoveries))
            accepted.append({**source, "task_id": task_id, "samples": len(rows)})
            by_id = {event["event_id"]: event for event in task_events}
            for row in rows:
                event = by_id[row["source"]["event_id"]]
                for camera, destination in row["images"].items():
                    original = _asset(root, event["observation"]["images"][camera])
                    if destination in copies and original != copies[destination]:
                        raise ValueError("Distinct frames collide at one exported image path.")
                    copies[destination] = original
    manifest = {
        "format_version": FORMAT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "training_started_by_exporter": False,
        "export_only": True,
        "note_training_status_in": "training run directory/run_status.json",
        "rgbd_assist": rgbd_assist,
        "scope": "Simulation rule imitation; held-out seed evaluation required before claiming generalization.",
        "split_unit": "scene_seed; all tasks, colors and paraphrases of a seed stay together",
        "seed_splits": {str(seed): split for seed, split in assignments.items()},
        "counts": {split: _stats(rows) for split, rows in samples.items()},
        "accepted_tasks": accepted,
        "collection_snapshots": collection_snapshots,
        "excluded": excluded,
        "excluded_counts": dict(Counter(item["reason"] for item in excluded)),
        "ignored_event_counts": dict(ignored),
        "image_files": len(copies),
        "image_bytes": sum(path.stat().st_size for path in copies.values()),
    }
    output_root.mkdir(parents=True)
    for relative, original in copies.items():
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    manifest["jsonl_sha256"] = {}
    for split, rows in samples.items():
        encoded = "".join(_json(row) + "\n" for row in rows).encode()
        (output_root / f"{split}.jsonl").write_bytes(encoded)
        manifest["jsonl_sha256"][split] = hashlib.sha256(encoded).hexdigest()
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def parse_seed_list(value):
    """Read comma-separated integers and inclusive ranges, e.g. 0,1000-1079."""
    result = set()
    for item in value.split(","):
        if not item.strip():
            continue
        parts = item.strip().split("-")
        if len(parts) == 1:
            result.add(int(parts[0]))
        elif len(parts) == 2 and all(part.isdigit() for part in parts):
            start, end = map(int, parts)
            if end < start or end - start > 100000:
                raise ValueError("Use ascending seed ranges no larger than 100000.")
            result.update(range(start, end + 1))
        else:
            raise ValueError("Use comma-separated nonnegative seeds or ascending ranges.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Recording root; repeat for multiple roots.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rgb-only", action="store_true")
    for split in SPLITS:
        parser.add_argument(f"--{split}-seeds", help="Comma-separated seeds or inclusive ranges.")
    args = parser.parse_args()
    provided = [getattr(args, f"{split}_seeds") is not None for split in SPLITS]
    seed_splits = None
    if any(provided):
        if not all(provided):
            parser.error("Specify train, val and test seed lists together; an empty list is allowed.")
        seed_splits = {}
        for split in SPLITS:
            for seed in parse_seed_list(getattr(args, f"{split}_seeds")):
                if seed in seed_splits:
                    parser.error(f"Scene seed {seed} occurs in multiple splits.")
                seed_splits[seed] = split
    manifest = export_training_dataset(args.input, args.output, seed_splits=seed_splits, rgbd_assist=not args.rgb_only)
    print(
        json.dumps(
            {"output": args.output, "counts": manifest["counts"], "excluded": manifest["excluded_counts"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
