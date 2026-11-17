"""Collect exact semantic decision boundaries with the deployed executor."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .bootstrap import configure
from .dataset import Recorder
from .evaluate_skill_head import direct_acceptance
from .plan_inputs import BOUNDARY_CONTRACT, contract_digest
from .plan_teacher import PlanTeacherBackend, PlanTeacherSession
from .train_vlm import write_json


class RGBRecorder(Recorder):
    """Retain synchronized RGB and derived entities; raw depth is not a model input."""

    def record_observation(self, frame_id, time_s, images, depths, perception):
        return super().record_observation(frame_id, time_s, images, {}, perception)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, default=[3300, 3301, 3302])
    parser.add_argument(
        "--resume", action="store_true", help="Continue recorded, completed cases without overwriting them."
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != 3:
        raise ValueError("Use distinct train/validation/test scene seeds.")
    configure()
    from .runtime import Runtime
    from .scene import sample_layout

    args.output.mkdir(parents=True, exist_ok=args.resume)
    source = Path(__file__).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
    cases = [(f"踢{c}球", color, None) for c, color in (("红", "red"), ("黄", "yellow"), ("蓝", "blue"))]
    cases += [
        (text, None, skill)
        for text, skill in (
            ("往前走一点", "move_forward"),
            ("后退一点", "move_backward"),
            ("向左移一点", "move_left"),
            ("向右移一点", "move_right"),
            ("左转一下", "turn_left"),
            ("右转一下", "turn_right"),
            ("看看眼前有什么", "answer"),
        )
    ]
    results = json.loads((args.output / "collection.json").read_text()) if args.resume else []
    completed = {r["episode"] for r in results}
    for result in results:
        saved = json.loads((args.output / result["episode"] / "summary.json").read_text())
        if saved["metadata"].get("input_contract_sha256") != contract_digest():
            raise ValueError("Do not resume a collection with a changed executor contract.")
    for split, seed in zip(("train", "val", "test"), args.seeds, strict=True):
        for index, (instruction, color, direct) in enumerate(cases):
            episode = f"semantic_{seed}_{index:02d}"
            if episode in completed:
                continue
            used = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
            if used > 384 * 1024**2 or shutil.disk_usage(args.output).free < 2 * 1024**3:
                raise RuntimeError("Keep collection below 384 MiB and at least 2 GiB free.")
            output = args.output / episode
            output.mkdir()
            runtime = Runtime(output)
            metadata = dict(
                backend="rgbd_rule_teacher",
                teacher="efficient_plan",
                is_trained_vlm=False,
                input_contract=BOUNDARY_CONTRACT,
                input_contract_sha256=contract_digest(),
                training_split=split,
                source_sha256=hashes,
                hardware=False,
                evaluation_mode="lockstep",
                physics=runtime.physics,
                policy_sha256=runtime.policy.hashes,
                raw_depth_saved=False,
            )
            recorder = RGBRecorder(output, episode, seed, metadata)
            result = None
            try:
                runtime.reset(sample_layout(seed))
                session = PlanTeacherSession(
                    runtime, recorder, PlanTeacherBackend(), expected_color=color, verbose=False
                )
                row_start = max(0, len(runtime.rows) - 1)
                result = session.run(instruction)
                if direct and direct != "answer":
                    result["evaluation"] = direct_acceptance(runtime.rows[row_start:], direct)
                result["source"] = "rule_teacher_not_learned_model"
            finally:
                runtime.close()
                runtime.save_trace()
                write_json(output / "contacts.json", runtime.contacts)
                write_json(output / "summary.json", dict(metadata=metadata, results=[result] if result else []))
            results.append(dict(episode=episode, seed=seed, split=split, **result))
            write_json(args.output / "collection.json", results)
            print(
                json.dumps(dict(episode=episode, status=result["status"], evaluation=result["evaluation"])), flush=True
            )


if __name__ == "__main__":
    main()
