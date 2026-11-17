"""Closed-loop held-out simulation evaluation of a fixed frozen-VLM skill head."""

import argparse
import cProfile
import io
import json
import math
import pstats
import time
from collections import Counter
from pathlib import Path

from .bootstrap import configure


def direct_acceptance(rows, skill):
    """Independent physical direction check, using the collection's existing thresholds."""
    expected = {
        "move_forward": (0, 1, 0.003),
        "move_backward": (0, -1, 0.003),
        "move_left": (1, 1, 0.003),
        "move_right": (1, -1, 0.003),
        "turn_left": (2, 1, 0.05),
        "turn_right": (2, -1, 0.05),
    }
    first, last = rows[0], rows[-1]
    dx, dy = last[7][:2] - first[7][:2]
    yaw = first[5]
    delta = [
        float(math.cos(yaw) * dx + math.sin(yaw) * dy),
        float(-math.sin(yaw) * dx + math.cos(yaw) * dy),
        float((last[5] - yaw + math.pi) % (2 * math.pi) - math.pi),
    ]
    axis, sign, threshold = expected[skill]
    return dict(
        success=sign * delta[axis] >= threshold,
        body_displacement_xy_m=delta[:2],
        yaw_delta_rad=delta[2],
        signed_progress=sign * delta[axis],
        minimum_progress=threshold,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3200, 3201, 3202])
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--skip-direct", action="store_true")
    args = parser.parse_args()
    configure()
    from .evaluate_teacher import MemoryRecorder, quality
    from .head_backend import SkillHeadBackend, SkillHeadTaskSession
    from .plan_protocol import CATALOG, VERSION, entities
    from .runtime import Runtime
    from .scene import sample_layout
    from .setup_skill_head import digest
    from .train_vlm import write_json

    args.output.mkdir(parents=True, exist_ok=False)
    backend = SkillHeadBackend(args.checkpoint, args.model)
    backend.metadata["evaluation_mode"] = "lockstep_simulation_paused_during_inference"
    runtime = Runtime(args.output)
    backend.metadata.update(
        motion_calibration_sha256=digest(Path(__file__).with_name("motion_calibration.json")),
        policy_sha256=runtime.policy.hashes,
        physics=runtime.physics,
        scene_seeds=args.seeds,
        max_steps=args.max_steps,
        source_sha256={path.name: digest(path) for path in Path(__file__).parent.glob("*.py")},
    )
    results = []
    cases = [
        (seed, f"踢{dict(red='红', yellow='黄', blue='蓝')[color]}球", color, None)
        for seed in args.seeds
        for color in ("red", "yellow", "blue")
    ]
    if not args.skip_direct:
        cases += [
            (args.seeds[0], text, None, skill)
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
    try:
        # Warm up kernels on a separate scene, not one of the scored test cases.
        runtime.reset(sample_layout(3199))
        obs, images, _ = runtime.observe()
        state = {k: obs[k] for k in ("yaw", "stationary", "tilt_deg")}
        state.update(
            entities=entities(obs),
            observation_id=obs["frame_id"],
            task_elapsed_s=0.0,
            available_skills=CATALOG,
            schema_version=VERSION,
        )
        profiler = cProfile.Profile()
        profiler.enable()
        backend.features.extract(
            "看看眼前有什么",
            state,
            dict(task_id="warmup", committed_steps=[], recent_outcomes=[]),
            images,
            vision=backend.vision,
        )
        profiler.disable()
        profile_text = io.StringIO()
        pstats.Stats(profiler, stream=profile_text).sort_stats("cumulative").print_stats(30)
        (args.output / "warmup_profile.txt").write_text(profile_text.getvalue())
        for index, (seed, instruction, color, expected_skill) in enumerate(cases):
            runtime.reset(sample_layout(seed))
            recorder = MemoryRecorder()
            session = SkillHeadTaskSession(
                runtime, recorder, backend, max_steps=args.max_steps, expected_color=color, verbose=False
            )
            row_start = len(runtime.rows)
            started = time.monotonic()
            result = session.run(instruction)
            predictions = [
                e["model_trace"]["raw_prediction"]
                for e in recorder.events
                if e["phase"] in ("model_plan", "model_error") and "raw_prediction" in e["model_trace"]
            ]
            outcomes = [e["model_trace"]["outcome"] for e in recorder.events if e["phase"] == "plan_step_result"]
            actions = [e["decision"]["skill"] for e in recorder.events if e["phase"] == "execute"]
            physical = None
            if expected_skill and expected_skill != "answer":
                physical = direct_acceptance(runtime.rows[max(0, row_start - 1) :], expected_skill)
            if expected_skill:
                correct = (
                    (not actions and result["status"] == "answered")
                    if expected_skill == "answer"
                    else (
                        bool(actions)
                        and all(s == expected_skill for s in actions)
                        and result["status"] in ("model_reported", "command_budget_complete")
                        and physical["success"]
                    )
                )
            else:
                correct = result["physical_task_verified"]
            result.update(
                seed=seed,
                color=color,
                expected_skill=expected_skill,
                accepted=correct,
                direct_physical_check=physical,
                wall_seconds=time.monotonic() - started,
                quality=quality(runtime.rows[row_start:], recorder.events),
                raw_predictions=predictions,
                outcomes=dict(Counter(outcomes)),
                executed_skills=dict(Counter(actions)),
            )
            results.append(result)
            write_json(args.output / f"case_{index:02d}_events.json", recorder.events)
            write_json(args.output / "results.json", dict(backend=backend.metadata, results=results, complete=False))
            print(json.dumps(result, ensure_ascii=False), flush=True)
        write_json(args.output / "results.json", dict(backend=backend.metadata, results=results, complete=True))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
