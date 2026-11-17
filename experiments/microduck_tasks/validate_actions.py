"""Bounded local validation; all failures remain in the generated report."""

import argparse
import hashlib
import json
from pathlib import Path

from .bootstrap import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure()
    from .action_runtime import ActionRuntime
    from .action_session import ActionSession

    cases = []
    for repeat in range(2):
        for skill in (
            "sit_stand",
            "ground_pick",
            "roll",
            "flamingo_left",
            "flamingo_right",
            "happy_hop",
            "moonwalk",
            "look_around",
            "nod",
            "body_sway",
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
            "turn_left",
            "turn_right",
        ):
            cases.append((f"{skill}_{repeat}", skill, {}, repeat))
    for i, xy in enumerate(((0.065, 0), (0.063, 0), (0.067, 0), (0.065, 0.002), (0.065, -0.002))):
        cases.append((f"pick_place_{i}", "pick_place", {"toy_xy": xy, "support_height": 0.014}, 0))
    cases.append(("push_block", "push_block", {"block_xy": (0.14, 0)}, 0))
    results = []
    for name, skill, options, repeat in cases:
        output = args.output / name
        output.mkdir()
        runtime = ActionRuntime(output, **options)
        session = ActionSession(runtime, profile="research")
        result = None
        try:
            runtime.reset()
            # Second trial starts after extra settling and a head gesture, not
            # the identical cold-start policy history from trial zero.
            if repeat:
                runtime.perform("look_around")
            result = session.run(skill)
        except Exception as exc:
            result = dict(skill=skill, success=False, outcome="exception", error=str(exc))
        finally:
            runtime.close()
            runtime.save_trace()
        result.update(
            case=name,
            options=options,
            max_raw_excess_rad=runtime.policy.max_raw_excess,
            raw_clipped_ticks=runtime.policy.raw_clip_ticks,
            action_ticks=runtime.policy.action_ticks,
        )
        results.append(result)
        (output / "summary.json").write_text(
            json.dumps(dict(metadata={"backend": "action_playground"}, results=[result]), ensure_ascii=False, indent=2)
        )
        sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("action*.py")}
        (args.output / "summary.json").write_text(
            json.dumps(dict(source_sha256=sources, results=results), ensure_ascii=False, indent=2)
        )


if __name__ == "__main__":
    main()
