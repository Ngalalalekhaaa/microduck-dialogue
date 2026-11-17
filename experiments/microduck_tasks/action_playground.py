"""Show pinned Microduck actions and physical object interactions locally."""

import argparse
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from .bootstrap import configure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--action", action="append", help="Repeat for a continuous action sequence; accepts Chinese.")
    parser.add_argument("--showcase", action="store_true", help="Standing gestures, sit/stand, balance, hop and roll.")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--profile", choices=("toy", "research"), default="toy")
    parser.add_argument("--preset", choices=("auto", "open", "pickup", "push"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true", help="Render a replay from the actual saved physics states.")
    args = parser.parse_args(argv)
    from .action_catalog import ALIASES, OBJECT_TASKS, SHOWCASE, print_catalog
    from .action_models import download

    if args.profile == "toy":
        from .toy_catalog import SHOWCASE, print_catalog

    if args.list:
        print_catalog()
        return 0
    if args.download:
        download()
        print("Pinned action weights are ready.")
        return 0
    configure()
    from .action_runtime import ActionRuntime
    from .action_session import ActionSession
    from .demo import _stop_task, interactive
    from .live_view import LiveView

    actions = list(SHOWCASE) if args.showcase else (args.action or [])
    if not actions and not args.interactive:
        parser.error("Choose --interactive, --showcase, --action NAME, --list or --download.")
    keys = [ALIASES.get(s, s) for s in actions]
    preset = args.preset
    if preset == "auto":
        preset = (
            "push"
            if "push_block" in keys
            else "pickup"
            if args.profile == "research" and (args.interactive or any(k in OBJECT_TASKS for k in keys))
            else "open"
        )
    output = (
        args.output
        or Path(__file__).parent / "outputs" / f"actions_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    options = {"toy_xy": (0.065, 0), "support_height": 0.014} if preset == "pickup" else {}
    if preset == "push":
        options["block_xy"] = (0.14, 0)
    runtime = ActionRuntime(output, **options)
    session = ActionSession(runtime, experimental=args.experimental, profile=args.profile)
    metadata = dict(
        backend="action_playground",
        hardware=False,
        profile=args.profile,
        preset=preset,
        data_contract="microduck-action-demo-1",
        training_ready=False,
        mouth="independent torque-limited simulated hinge",
        physics=runtime.physics,
        policy_sha256=runtime.policy.hashes,
    )
    window = worker = None
    errors = []
    sequence_failed = []

    def run_sequence():
        try:
            for action in actions:
                result = session.run(action)
                if result and not result["success"]:
                    sequence_failed.append(True)
                    break
        except BaseException as exc:
            errors.append(exc)

    try:
        runtime.reset()
        if args.interactive:
            print_catalog()
            interactive(session, viewer=not args.no_viewer)
        elif args.no_viewer:
            run_sequence()
        else:
            window = LiveView(session)
            runtime.start_realtime()
            worker = threading.Thread(target=run_sequence, name="microduck-action-sequence")
            worker.start()
            while worker.is_alive():
                if not window.pump():
                    runtime.request_stop()
                    break
                time.sleep(0.01)
            while not window.closed and not runtime.failure and window.pump():
                time.sleep(0.02)
        if errors:
            raise errors[0]
    finally:
        _stop_task(runtime, worker)
        if window:
            window.close()
        runtime.close()
        runtime.save_trace()
        (output / "summary.json").write_text(
            json.dumps(
                dict(
                    metadata=metadata,
                    results=session.results,
                    failure=runtime.failure,
                    max_raw_target_excess_rad=runtime.policy.max_raw_excess,
                    raw_clipped_ticks=runtime.policy.raw_clip_ticks,
                    new_policy_ticks=runtime.policy.action_ticks,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"Saved: {output}")
    if args.video:
        from .recording import render_video

        print(render_video(output))
    return 1 if errors or sequence_failed or runtime.failure or any(not r["success"] for r in session.results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
