"""Record execution traces and export verified-source rule demonstrations.

This is a versioned interchange format, not a trained VLM or a trainer-specific
dataset. Depth and detector outputs supervise the rule teacher, while exported
student messages contain RGB, language and task context only. RGB-only distance
estimation must be evaluated separately before replacing the RGB-D teacher.
"""

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

FORMAT_VERSION = "microduck-vlm-1"
_RULE_BACKENDS = {"rule", "rgbd_rule_teacher"}
_MODEL_SOURCES = {"pretrained_vlm", "finetuned_vlm", "frozen_vlm_trained_skill_head"}
_CONTROLLER_SOURCES = _MODEL_SOURCES | {"executor", "rule"}
_TOKEN = re.compile(r"[A-Za-z0-9_-]+\Z")
_EVENT_FIELDS = {
    "event_id",
    "instruction",
    "intent",
    "observation",
    "phase",
    "decision",
    "execution",
    "done",
    "status",
    "reply",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _token(value, name):
    value = str(value)
    if not _TOKEN.fullmatch(value):
        raise ValueError(f"Use only letters, digits, underscores and hyphens for {name}: {value!r}")
    return value


def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number of seconds.")


def _asset(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Image and depth paths must be relative to the episode directory.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes its episode directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Episode asset is missing: {path}")
    return path


def _validate_event(event, root):
    missing = _EVENT_FIELDS - event.keys()
    if missing:
        raise ValueError(f"Event is missing required fields: {sorted(missing)}")
    _token(event["event_id"], "event_id")
    if "task_id" in event:
        _token(event["task_id"], "task_id")
    if "context" in event and not isinstance(event["context"], dict):
        raise ValueError("Conversation context must be an object.")
    if "controller_source" in event:
        source = event["controller_source"]
        if not isinstance(source, str) or source not in _CONTROLLER_SOURCES:
            raise ValueError(f"Controller source must be one of: {', '.join(sorted(_CONTROLLER_SOURCES))}.")
    for name in ("model_trace", "teacher_trace"):
        trace = event.get(name)
        if trace is not None:
            if not isinstance(trace, dict):
                raise ValueError(f"{name} must be an object or null.")
            if trace.get("latency_s") is not None:
                _time(trace["latency_s"], f"{name}.latency_s")
    for key in ("instruction", "phase", "status", "reply"):
        if not isinstance(event[key], str):
            raise ValueError(f"Event {key} must be a string.")
    if not isinstance(event["done"], bool) or not isinstance(event["intent"], dict):
        raise ValueError("Event done must be boolean and intent must be an object.")
    observation = event["observation"]
    if not isinstance(observation, dict) or not {"frame_id", "time_s", "images", "depths", "detections"}.issubset(
        observation
    ):
        raise ValueError("Observation requires frame_id, time_s, images, depths and detections.")
    _token(observation["frame_id"], "frame_id")
    _time(observation["time_s"], "observation.time_s")
    if not isinstance(observation["images"], dict) or not observation["images"]:
        raise ValueError("Observation must contain at least one RGB image.")
    if not isinstance(observation["depths"], dict) or not isinstance(observation["detections"], list):
        raise ValueError("Observation depths must be an object and detections must be a list.")
    for group in ("images", "depths"):
        for camera, relative in observation[group].items():
            _token(camera, "camera")
            path = _asset(root, relative)
            if group == "images":
                with Image.open(path) as image:
                    image.verify()
    execution = event["execution"]
    if not isinstance(execution, dict) or not {"started", "completed", "start_time_s", "end_time_s"}.issubset(
        execution
    ):
        raise ValueError("Execution requires explicit started/completed flags and start/end timestamps.")
    if not all(isinstance(execution[key], bool) for key in ("started", "completed")):
        raise ValueError("Execution started/completed must be boolean.")
    start, end = execution["start_time_s"], execution["end_time_s"]
    if not execution["started"]:
        if execution["completed"] or start is not None or end is not None:
            raise ValueError("An unstarted action cannot be completed or have execution timestamps.")
    else:
        _time(start, "execution.start_time_s")
        if end is not None:
            _time(end, "execution.end_time_s")
            if end < start:
                raise ValueError("Execution end precedes its start.")
        if execution["completed"] and end is None:
            raise ValueError("A completed action requires an end timestamp.")
    decision = event["decision"]
    if decision is not None:
        if not isinstance(decision, dict) or not isinstance(decision.get("skill"), str) or not decision["skill"]:
            raise ValueError("Decision must be null or an object containing a nonempty skill.")
        _time(decision.get("seconds"), "decision.seconds")
    if execution["started"] and decision is None:
        raise ValueError("A started action requires the decision that was executed.")
    _json(event)


class Recorder:
    """Own a new episode directory; never infer that a proposed action executed."""

    def __init__(self, root, episode_id, seed, metadata):
        self.root = Path(root)
        episode_id = _token(episode_id, "episode_id")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Use an integer scene seed so identical scenes share a dataset split.")
        payload = {"format_version": FORMAT_VERSION, "episode_id": episode_id, "seed": seed, "metadata": metadata}
        encoded = _json(payload)
        self.root.mkdir(parents=True, exist_ok=True)
        if any((self.root / name).exists() for name in ("episode.json", "events.jsonl", "frames")):
            raise FileExistsError(f"Choose a new episode directory; recording already exists in {self.root}")
        (self.root / "frames").mkdir()
        (self.root / "episode.json").write_text(encoded + "\n", encoding="utf-8")
        (self.root / "events.jsonl").touch(exist_ok=False)
        self._events = set()
        self._frames = set()

    def record_observation(self, frame_id, time_s, images, depths, perception):
        """Save a synchronized RGB-D observation and return relative asset paths."""
        frame_id = _token(frame_id, "frame_id")
        _time(time_s, "time_s")
        if frame_id in self._frames:
            raise ValueError(f"Frame {frame_id} is already recorded; reuse its returned observation instead.")
        if not images or set(depths) - set(images) or not isinstance(perception, list):
            raise ValueError("Provide RGB images, depth cameras matching RGB, and a perception list.")
        _json(perception)
        for camera, rgb in images.items():
            _token(camera, "camera")
            names = [f"frames/{frame_id}_{camera}.png"]
            if camera in depths:
                names.append(f"frames/{frame_id}_{camera}_depth.npz")
            if any((self.root / name).exists() for name in names):
                raise FileExistsError("Frame and camera names collide with an existing recording asset.")
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("RGB images must be uint8 arrays with shape (height, width, 3).")
            if min(rgb.shape[:2]) < 1:
                raise ValueError("RGB images cannot be empty.")
            if camera in depths:
                depth = np.asarray(depths[camera])
                if depth.shape != rgb.shape[:2] or not np.issubdtype(depth.dtype, np.floating):
                    raise ValueError("Depth must be a floating array matching its RGB image dimensions.")
                if not np.isfinite(depth).all() or (depth < 0).any():
                    raise ValueError("Depth must contain finite nonnegative metre values.")
        observation = {"frame_id": frame_id, "time_s": time_s, "images": {}, "depths": {}, "detections": perception}
        for camera, rgb in images.items():
            relative = f"frames/{frame_id}_{camera}.png"
            Image.fromarray(rgb).save(self.root / relative)
            observation["images"][camera] = relative
            if camera in depths:
                relative = f"frames/{frame_id}_{camera}_depth.npz"
                np.savez_compressed(self.root / relative, depth_m=depths[camera])
                observation["depths"][camera] = relative
        self._frames.add(frame_id)
        return observation

    def append_event(self, event):
        """Persist caller-provided execution facts, including unfinished attempts."""
        _validate_event(event, self.root)
        if event["event_id"] in self._events:
            raise ValueError(f"Duplicate event_id: {event['event_id']}")
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json(event) + "\n")
        self._events.add(event["event_id"])


def split_for_seed(seed):
    """Group all episodes from the same seed, including language paraphrases."""
    bucket = int(hashlib.sha256(f"{FORMAT_VERSION}:scene:{seed}".encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def _load_episodes(input_root):
    episodes, identifiers = [], set()
    for metadata_path in sorted(input_root.rglob("episode.json")):
        root = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported recording format: {metadata_path}")
        episode_id = _token(metadata["episode_id"], "episode_id")
        if episode_id in identifiers:
            raise ValueError(f"Duplicate episode_id across recordings: {episode_id}")
        identifiers.add(episode_id)
        seed = metadata["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Episode seed must be an integer: {metadata_path}")
        events, event_ids = [], set()
        for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            _validate_event(event, root)
            if event["event_id"] in event_ids:
                raise ValueError(f"Duplicate event_id in {root}: {event['event_id']}")
            event_ids.add(event["event_id"])
            events.append(event)
        episodes.append((root, metadata, events))
    if not episodes:
        raise ValueError(f"No episode.json recordings found below {input_root}")
    return episodes


def _backend(metadata):
    """Legacy recordings without an explicit source came from the rule teacher."""
    return metadata.get("metadata", {}).get("backend", "rule")


def _demonstration_source(metadata, events):
    """Task success alone does not certify a model's predictions as labels."""
    return (
        _backend(metadata) in _RULE_BACKENDS
        and not metadata.get("metadata", {}).get("is_trained_vlm", False)
        and not metadata.get("metadata", {}).get("head_trained", False)
        and not any(event.get("controller_source") in _MODEL_SOURCES or event.get("model_trace") for event in events)
    )


def _sample(kind, event, metadata, images, target, previous_action):
    context = {"instruction": event["instruction"], "context": event.get("context", {})}
    if kind != "intent":
        context.update(intent=event["intent"], previous_action=previous_action)
    return {
        "format_version": FORMAT_VERSION,
        "sample_id": f"{metadata['episode_id']}:{event['event_id']}:{kind}",
        "episode_id": metadata["episode_id"],
        "task_id": event.get("task_id", "task"),
        "scene_seed": metadata["seed"],
        "kind": kind,
        "source_event": {
            "event_id": event["event_id"],
            "frame_id": event["observation"]["frame_id"],
            "time_s": event["observation"]["time_s"],
            "phase": event["phase"],
            "execution": event["execution"],
            "done": event["done"],
            "status": event["status"],
        },
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Microduck simulation supervision: {kind}. Return one JSON object. "
                            "Use the supplied RGB views and text; do not assume unseen objects are visible."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": path, "camera": camera} for camera, path in sorted(images.items())],
                    {"type": "text", "text": _json(context)},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": _json(target)}]},
        ],
        "teacher": {
            "backend": _backend(metadata),
            "uses_depth": bool(event["observation"]["depths"]),
            "detections": event["observation"]["detections"],
        },
    }


def export_vlm_dataset(input_root, output_root):
    """Export validated, self-contained RGB samples; retain failures in raw logs.

    Only rule episodes produce supervision. Pretrained VLM proposals and replies
    remain unlabelled raw traces even when a task succeeds. Rule intent labels
    are exported once per task. Motor imitation includes only fully executed
    actions from tasks ending in explicit success, never terminal suggestions or
    partial runs. A decision's phase stays in audit metadata, not student input.
    All images and samples from the same scene seed stay in the same split.
    """
    input_root, output_root = Path(input_root), Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"Choose a new dataset output directory: {output_root}")
    episodes = _load_episodes(input_root)
    samples = {name: [] for name in ("train", "val", "test")}
    manifest = {
        "format_version": FORMAT_VERSION,
        "teacher": "Rule demonstrations only; source backends are recorded per sample and episode",
        "student_inputs": (
            "RGB images, language, intent, conversation context and previous action; "
            "no current phase, depth or oracle coordinates"
        ),
        "limitation": (
            "Rule teacher uses depth. RGB-only observability and sim-to-real transfer are not established. "
            "Model predictions are not correct demonstration labels, including predictions in successful tasks."
        ),
        "training_started": False,
        "split_policy": "SHA-256 scene seed, 80/10/10 buckets; same seed always shares one split",
        "episodes": [],
        "excluded": [],
        "source_backends": {},
    }
    output_root.mkdir(parents=True)
    for root, metadata, events in episodes:
        split = split_for_seed(metadata["seed"])
        backend = _backend(metadata)
        source = manifest["source_backends"].setdefault(
            backend, {"episodes": 0, "events": 0, "exported_samples": 0, "excluded_episodes": 0}
        )
        source["episodes"] += 1
        source["events"] += len(events)
        eligible = _demonstration_source(metadata, events)
        finals = {event.get("task_id", "task"): event for event in events if event["done"]}
        successful = {task for task, final in finals.items() if final["status"] == "success"}
        seen_tasks, previous = set(), {}
        manifest["episodes"].append(
            {
                "episode_id": metadata["episode_id"],
                "seed": metadata["seed"],
                "split": split,
                "source": str(root.resolve()),
                "metadata": metadata.get("metadata", {}),
                "backend": backend,
                "supervision_eligible": eligible,
            }
        )
        if not eligible:
            source["excluded_episodes"] += 1
            manifest["excluded"].append(
                {
                    "episode_id": metadata["episode_id"],
                    "backend": backend,
                    "reason": "model_prediction_not_demonstration",
                    "events": len(events),
                }
            )
            continue
        for event in events:
            task = event.get("task_id", "task")
            prior = previous.get(task)
            execution = event["execution"]
            executed = execution["started"] and execution["completed"]
            targets = []
            if task not in seen_tasks:
                targets.append(("intent", event["intent"]))
                seen_tasks.add(task)
            if executed and task in successful and execution["end_time_s"] > execution["start_time_s"]:
                targets.append(("action", {"action": event["decision"]}))
            if event["reply"]:
                targets.append(("reply", {"reply": event["reply"], "status": event["status"]}))
            if targets:
                images = {}
                for camera, relative in event["observation"]["images"].items():
                    destination = Path("images") / metadata["episode_id"] / Path(relative).name
                    (output_root / destination).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(_asset(root, relative), output_root / destination)
                    images[camera] = destination.as_posix()
                for kind, target in targets:
                    samples[split].append(_sample(kind, event, metadata, images, target, prior))
                    source["exported_samples"] += 1
            if executed:
                previous[task] = event["decision"]
    for split, rows in samples.items():
        (output_root / f"{split}.jsonl").write_text("".join(_json(row) + "\n" for row in rows), encoding="utf-8")
    manifest["counts"] = {
        split: {
            "total": len(rows),
            **{kind: sum(row["kind"] == kind for row in rows) for kind in ("intent", "action", "reply")},
        }
        for split, rows in samples.items()
    }
    (output_root / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return manifest
