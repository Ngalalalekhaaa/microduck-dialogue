"""Export actual dialogue traces, preserving causal inputs and episode grouping."""

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from .toy_inputs import validate_input
from .toy_protocol import VERSION, goal, text, validate_output
from .toy_schema import SYSTEM

TRACE_VERSION = "microduck-toy-trace-1"
FORMAT_VERSION = "microduck-toy-dialogue-sft-1"
SOURCE = "toy_rule_teacher"


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_digest():
    root = Path(__file__).parent
    names = ("toy_inputs.py", "toy_protocol.py", "toy_schema.py", "toy_catalog.py")
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in names)).hexdigest()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Use a nonempty path-safe string for episode/event identifiers.")
    return value


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Trace time must be finite and nonnegative.")
    return value


def _asset(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Camera paths must be relative to their episode.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Camera asset is missing or escapes its episode.")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("Require actual RGB PNG camera frames.")
        image.verify()
    return path


def _result(event, starts):
    goal(event["goal"])
    text(event["reason"], "executor result reason", 200)
    status = event.get("status")
    if status not in ("succeeded", "failed", "cancelled", "blocked"):
        raise ValueError("Every goal result needs an explicit executor status.")
    if type(event.get("success")) is not bool or type(event.get("verified")) is not bool:
        raise ValueError("Results require boolean success and verified fields.")
    if event["success"] != (status == "succeeded") or (event["success"] and not event["verified"]):
        raise ValueError("Only independently verified success may be labeled succeeded.")
    key = (event["task_id"], event["goal_index"])
    if event["success"] and starts.get(key) != event["goal"]:
        raise ValueError("Successful goals must match an earlier goal_start.")
    return {name: event[name] for name in ("goal", "status", "reason", "verified")}


def read_episode(root):
    """Validate provenance, public history, images and executor-result causality."""
    root = Path(root).resolve()
    metadata = json.loads((root / "metadata.json").read_text())
    if (
        metadata.get("format_version") != TRACE_VERSION
        or metadata.get("schema_version") != VERSION
        or metadata.get("recording_source") != "runtime"
        or metadata.get("controller_source") != SOURCE
        or metadata.get("training_ready") is not True
        or metadata.get("contract_sha256") != contract_digest()
    ):
        raise ValueError("Require current runtime traces; design fixtures and relabeled demos cannot train.")
    if "collection_validated" in metadata and metadata["collection_validated"] is not True:
        raise ValueError("Collection validation failed or is incomplete; quarantine this episode before export.")
    episode = _identifier(metadata["episode_id"])
    if type(metadata.get("seed")) is not int or not isinstance(metadata.get("split_group"), str):
        raise ValueError("Episode needs its scene seed and nonempty paraphrase-family split_group.")
    if not metadata["split_group"].strip():
        raise ValueError("split_group must be nonempty.")
    history, results, starts, identifiers, rows = [], {}, {}, set(), []
    last_time = -1
    trace = root / "events.jsonl"
    for line in trace.read_text().splitlines():
        event = json.loads(line)
        identifier = _identifier(event["event_id"])
        if identifier in identifiers:
            raise ValueError("Duplicate event id inside one episode.")
        identifiers.add(identifier)
        timestamp = _time(event["time_s"])
        if timestamp < last_time - 1e-6:
            raise ValueError("Trace events must be written in causal time order.")
        last_time = timestamp
        kind = event["event"]
        if kind == "goal_start":
            starts[event["task_id"], event["goal_index"]] = copy.deepcopy(event["goal"])
        elif kind == "goal_result":
            results[identifier] = _result(event, starts)
        elif kind == "dialogue":
            request = validate_input(event["request"])
            response = validate_output(event["response"], request)
            if event.get("controller_source") != SOURCE:
                raise ValueError("Only explicitly identified rule-teacher labels are accepted here.")
            if event["revision"] != request["execution"]["revision"]:
                raise ValueError("Dialogue revision must match the input snapshot.")
            if request["history"] != history[-12:]:
                raise ValueError("Input history differs from previously recorded public dialogue.")
            last_result = request["execution"]["last_result"]
            if last_result is not None and last_result not in results.values():
                raise ValueError("Input execution history contains an unrecorded or future result.")
            observed = _time(event["observation_time_s"])
            if observed > timestamp + 1e-6:
                raise ValueError("Future camera images cannot supervise an earlier decision.")
            if abs(request["observation"]["age_s"] - (timestamp - observed)) > 0.05:
                raise ValueError("Recorded observation age contradicts capture and decision timestamps.")
            if set(event["media"]) != {"front", "near"}:
                raise ValueError("Each dialogue input needs both recorded camera views.")
            images = {camera: _asset(root, relative) for camera, relative in event["media"].items()}
            hashes = {camera: digest(path) for camera, path in images.items()}
            if event.get("media_sha256", hashes) != hashes:
                raise ValueError("Camera data changed since recording.")
            if request["event"]["kind"] == "feedback":
                actual = results.get(event.get("result_ref"))
                if actual is None or request["execution"]["last_result"] != actual:
                    raise ValueError("Feedback must reference the exact preceding executor result.")
                if response["decision"] != "respond":
                    raise ValueError("Executor feedback cannot secretly authorize another user task.")
            else:
                history.append({"role": "user", "text": request["event"]["text"]})
            history.append({"role": "assistant", "text": response["reply"]})
            rows.append(
                {
                    "format_version": FORMAT_VERSION,
                    "id": f"{episode}_{identifier}",
                    "episode_id": episode,
                    "seed": metadata["seed"],
                    "split_group": metadata["split_group"],
                    "controller_source": SOURCE,
                    "training_ready": True,
                    "supervision_scope": "dialogue_and_semantic_planning_not_motor_success",
                    "contract_sha256": contract_digest(),
                    "system_prompt": SYSTEM,
                    "request": copy.deepcopy(request),
                    "target": response,
                    "images": images,
                    "image_sha256": hashes,
                    "provenance": {
                        "event_id": identifier,
                        "turn_id": event["turn_id"],
                        "observation_id": event["observation_id"],
                        "observation_time_s": observed,
                        "time_s": timestamp,
                        "trace_sha256": digest(trace),
                    },
                }
            )
    if not rows:
        raise ValueError("Episode has no actual dialogue samples.")
    return metadata, rows


def _components(episodes):
    """Join shared scene seeds and paraphrase families into indivisible groups."""
    components = []
    for metadata, _ in episodes:
        tags = {f"seed:{metadata['seed']}", f"family:{metadata['split_group']}"}
        members = {metadata["episode_id"]}
        remaining = []
        for other_tags, other_members in components:
            if tags & other_tags:
                tags |= other_tags
                members |= other_members
            else:
                remaining.append((other_tags, other_members))
        components = remaining + [(tags, members)]
    components.sort(key=lambda item: hashlib.sha256(dumps(sorted(item[0])).encode()).hexdigest())
    return components


def _split_episodes(episodes, *, three_way=False):
    """Assign whole connected groups; a shared scene or family never leaks."""
    components = _components(episodes)
    if three_way:
        if len(components) < 3:
            raise ValueError("Three-way export needs at least three independent scene/family groups.")
        names = ("train", "val", "test")
        desired = dict(zip(names, (0.8 * len(episodes), 0.1 * len(episodes), 0.1 * len(episodes)), strict=True))
        assigned, sizes = {}, Counter()
        # Largest groups first, with stable hash order resolving equal sizes.
        ordered = sorted(components, key=lambda item: -len(item[1]))
        for index, (_, members) in enumerate(ordered):
            empty = [name for name in names if sizes[name] == 0]
            candidates = empty if len(ordered) - index == len(empty) else names
            split = max(candidates, key=lambda name: desired[name] - sizes[name])
            assigned.update(dict.fromkeys(members, split))
            sizes[split] += len(members)
        return assigned
    heldout = components[-1][1] if len(components) > 1 else set()
    return {meta["episode_id"]: "val" if meta["episode_id"] in heldout else "train" for meta, _ in episodes}


def export(input_root, output, *, three_way=False):
    input_root, output = Path(input_root), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing datasets are never overwritten.")
    paths = sorted(input_root.rglob("metadata.json"))
    paths = [p for p in paths if (p.parent / "events.jsonl").exists()]
    if not paths:
        raise ValueError("No runtime dialogue episodes found; design casebooks are not training datasets.")
    episodes = [read_episode(path.parent) for path in paths]
    ids = [meta["episode_id"] for meta, _ in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode ids across recordings.")
    splits = _split_episodes(episodes, three_way=three_way)
    output.mkdir(parents=True)
    counts, decisions, lengths, result_status = Counter(), Counter(), Counter(), Counter()
    names = ("train", "val", "test") if three_way else ("train", "val")
    handles = {split: (output / f"{split}.jsonl").open("w") for split in names}
    try:
        for meta, rows in episodes:
            split = splits[meta["episode_id"]]
            for row in rows:
                for camera, source in row["images"].items():
                    relative = f"media/{row['episode_id']}/{row['id']}_{camera}.png"
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    row["images"][camera] = relative
                handles[split].write(dumps(row) + "\n")
                counts[split] += 1
                decisions[row["target"]["decision"]] += 1
                lengths[len(row["target"]["goals"])] += 1
                if row["request"]["event"]["kind"] == "feedback":
                    result_status[row["request"]["execution"]["last_result"]["status"]] += 1
    finally:
        for handle in handles.values():
            handle.close()
    report = {
        "format_version": FORMAT_VERSION,
        "contract_sha256": contract_digest(),
        "source": SOURCE,
        "training_ready": True,
        "deployment_ready": False,
        "episodes": len(episodes),
        "rows": dict(counts),
        "decisions": dict(decisions),
        "goals_per_response": dict(lengths),
        "feedback_status": dict(result_status),
        "episode_splits": splits,
        "split_rule": (
            "whole episodes; union all shared scene seeds and paraphrase families; "
            + (
                "deterministic largest-group-first, targeting 80/10/10 episode counts"
                if three_way
                else "last group held out"
            )
        ),
        "split_targets": {"train": 0.8, "val": 0.1, "test": 0.1} if three_way else None,
        "independent_split_groups": len(_components(episodes)),
        "split_episode_counts": dict(Counter(splits.values())),
        "limits": [
            "Rule-teacher dialogue and semantic-plan supervision, not an evaluation of a trained VLM.",
            "A proposed plan does not assert successful execution; failures and cancellations remain labeled.",
            "RGB-D entities currently support the visual replies; open-world visual narration is not covered.",
            "A grouped holdout alone does not establish unseen-language or unseen-task generalization.",
        ],
        "files_sha256": {split: digest(output / f"{split}.jsonl") for split in handles},
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--three-way", action="store_true", help="Split independent groups into approximately 80/10/10."
    )
    args = parser.parse_args()
    print(json.dumps(export(args.input, args.output, three_way=args.three_way), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
