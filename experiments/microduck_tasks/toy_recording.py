"""Append-only live dialogue events and synchronized, lossless RGB observations."""

import copy
import hashlib
import json
import threading
from pathlib import Path

from PIL import Image

from .toy_protocol import VERSION


def source_hashes():
    root = Path(__file__).parent
    names = [*root.glob("toy_*.py"), root / "runtime.py", root / "action_runtime.py", root / "action_policy.py"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(names)}


class ToyRecorder:
    def __init__(
        self, output, *, episode_id, seed, split_group, controller_source="toy_rule_teacher", training_ready=True
    ):
        from .toy_dataset import contract_digest

        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "events.jsonl"
        if self.path.exists():
            raise FileExistsError("Use a new episode directory; recorded dialogue cannot be overwritten.")
        self.lock = threading.RLock()
        self.seq = 0
        self.events = []
        self.metadata = dict(
            format_version="microduck-toy-trace-1",
            schema_version=VERSION,
            episode_id=episode_id,
            seed=seed,
            split_group=split_group,
            controller_source=controller_source,
            recording_source="runtime",
            training_ready=training_ready,
            backend=controller_source,
            hardware=False,
            source_sha256=source_hashes(),
            contract_sha256=contract_digest(),
            visual_teacher="RGB-D three-color ball detections, not unrestricted scene captioning",
        )
        self.save_metadata()

    def save_metadata(self):
        (self.output / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2))

    def append(self, event, **fields):
        with self.lock:
            self.seq += 1
            row = copy.deepcopy(
                dict(
                    event=event,
                    event_id=f"event_{self.seq:06d}",
                    controller_source=self.metadata["controller_source"],
                    **fields,
                )
            )
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self.events.append(row)
            return row

    def images(self, observation, images):
        media, hashes = {}, {}
        with self.lock:
            folder = self.output / "observations"
            folder.mkdir(exist_ok=True)
            for camera in ("front", "near"):
                path = folder / f"{observation['frame_id']}_{camera}.png"
                if path.exists():
                    raise FileExistsError("Observation ids must identify one captured frame.")
                Image.fromarray(images[camera]).save(path)
                media[camera] = str(path.relative_to(self.output))
                hashes[camera] = hashlib.sha256(path.read_bytes()).hexdigest()
        return media, hashes
