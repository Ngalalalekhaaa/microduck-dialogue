"""Paired-seed teacher evaluation; records failures and path quality without resampling."""

import argparse
import json
from pathlib import Path

import numpy as np

from .bootstrap import configure


class MemoryRecorder:
    def __init__(self):
        self.events = []

    def record_observation(self, frame_id, time_s, images, depths, detections):
        return {"frame_id": frame_id, "time_s": time_s, "detections": detections}

    def append_event(self, event):
        self.events.append(event)


def quality(rows, events):
    if not rows:
        return {}
    xy = np.array([row[7][:2] for row in rows])
    yaw = np.unwrap([row[5] for row in rows])
    motions = [e for e in events if e["decision"] is not None]
    return {
        "path_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        "elapsed_s": rows[-1][0] - rows[0][0],
        "decisions": len(motions),
        "align_decisions": sum(e["phase"] == "align" for e in motions),
        "yaw_travel_rad": float(np.abs(np.diff(yaw)).sum()),
        "stand_s": sum(e["decision"]["seconds"] for e in motions if e["decision"]["skill"] == "stand"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", choices=("legacy", "efficient"), default="efficient")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--colors", nargs="+", default=["red", "yellow", "blue"])
    parser.add_argument("--nominal-motion", action="store_true")
    args = parser.parse_args()
    configure()
    from .runtime import Runtime
    from .scene import sample_layout
    from .tasks import TaskSession

    args.output.mkdir(parents=True, exist_ok=False)
    runtime = Runtime(args.output)
    runtime.policy.motion.enabled = not args.nominal_motion
    results = []
    try:
        for seed in args.seeds:
            for color in args.colors:
                runtime.reset(sample_layout(seed))
                recorder = MemoryRecorder()
                session = TaskSession(runtime, recorder, verbose=False, teacher=args.teacher)
                start = len(runtime.rows)
                result = session.run(f"踢{dict(red='红', yellow='黄', blue='蓝')[color]}球")
                result.update(seed=seed, color=color, quality=quality(runtime.rows[start:], recorder.events))
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                (args.output / "results.json").write_text(
                    json.dumps(
                        {"teacher": args.teacher, "nominal_motion": args.nominal_motion, "results": results},
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
