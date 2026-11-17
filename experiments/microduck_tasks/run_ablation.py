"""Freeze development data or replay P candidates without executing any robot."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from .ablation_cases import VERSION, build_cases
from .ablation_contract import checksum, dumps
from .ablation_prompt import PARENTS, metadata
from .ablation_score import score, summarize
from .evaluate_toy_vlm_language import source_snapshot
from .toy_teacher import is_cancel, is_pause


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def freeze(source, output):
    observation, images, provenance = source_snapshot(source)
    cases = build_cases(observation)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "dev240.json", cases)
    save(output / "screen96_ids.json", [row["id"] for row in cases if row["screen"]])
    manifest = dict(
        version=VERSION,
        cases_sha256=checksum(cases),
        source=provenance,
        images=images,
        count=len(cases),
        screen_count=sum(r["screen"] for r in cases),
        source_kind="synthetic_semantic_with_recorded_image_pair",
        blind_test=False,
        physical_validation=False,
        human_reviewed=False,
        limitations=[
            "Synthetic instructions and states; not a random sample of real users.",
            "A recorded image pair is shared; altered observation fields are listed per case.",
            "No pixel-required tasks; visual grounding is deferred to V/E.",
            "Family members all belong to development; none are blind-test or training data.",
        ],
    )
    save(output / "manifest.json", manifest)
    return manifest


def evaluate(args):
    manifest = json.loads((args.suite / "manifest.json").read_text())
    cases = json.loads((args.suite / "dev240.json").read_text())
    if checksum(cases) != manifest["cases_sha256"]:
        raise ValueError("Development cases changed after freezing; make a new suite.")
    for camera, path in manifest["images"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest["source"]["image_sha256"][camera]:
            raise ValueError("Source image changed after freezing.")
    cases = [r for r in cases if args.size == 240 or r["screen"]]
    if args.limit:
        cases = cases[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    run = dict(
        suite_sha256=manifest["cases_sha256"],
        variants=args.variants,
        size=args.size,
        case_ids=[r["id"] for r in cases],
        prompts={v: metadata(v) for v in args.variants},
        raw_model_only=True,
        host_controls_scored_separately=True,
        repairs=0,
        retries=0,
        teacher_fallback=False,
        physical_executions=0,
    )
    if args.grammar_max_whitespace is not None:
        run["grammar_max_whitespace"] = args.grammar_max_whitespace
        run["experiment_label"] = "P5b_bounded_whitespace"
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run:
        raise ValueError("This output already contains a different experiment; use a new directory.")
    save(config_path, run)
    from .ablation_generation import AblationGenerator

    generator = AblationGenerator(args.model, grammar_max_whitespace=args.grammar_max_whitespace)
    save(args.output / "model.json", generator.metadata)
    for variant in args.variants:
        directory = args.output / variant
        directory.mkdir(exist_ok=True)
        rows = []
        for case in cases:
            destination = directory / f"{case['id']}.json"
            if destination.exists():
                rows.append(json.loads(destination.read_text()))
                continue
            raw, timing, rendered, error = "", {}, None, None
            started = time.monotonic()
            try:
                raw, timing, rendered = generator.generate_variant(case["request"], manifest["images"], variant)
            except Exception as exc:
                error = dict(type=type(exc).__name__, reason=str(exc))
                timing = dict(seconds=time.monotonic() - started)
            text = case["request"]["event"]["text"]
            control = "host_cancel" if is_cancel(text) else "host_pause" if is_pause(text) else None
            row = dict(
                **case,
                raw_text=raw,
                timing=timing,
                generation_error=error,
                rendered_input=rendered,
                rendered_sha256=checksum(rendered),
                host_control=control,
                score=score(raw, case, plan_only=variant not in ("P0", "P1")),
            )
            save(destination, row)
            rows.append(row)
            report = summarize(rows, expected_count=len(cases))
            save(directory / "progress.json", report)
            save(args.output / "status.json", dict(variant=variant, case=case["id"], **report))
            print(
                dumps(
                    dict(
                        variant=variant,
                        case=case["id"],
                        completed=len(rows),
                        exact=row["score"]["accepted_exact"],
                        valid=row["score"]["context_valid"],
                    )
                ),
                flush=True,
            )
            if error:
                raise RuntimeError(f"Generation failed; preserved original error at {destination}: {error}")
        save(directory / "report.json", summarize(rows, expected_count=len(cases)))
    save(args.output / "complete.json", dict(complete=True, variants=args.variants, count=len(cases)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-from", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=list(PARENTS), default=["P0"])
    parser.add_argument("--size", type=int, choices=(96, 240), default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--grammar-max-whitespace", type=int)
    parser.add_argument("--model", type=Path, default=Path(".cache/microduck_head/model"))
    args = parser.parse_args()
    if args.grammar_max_whitespace is not None and (args.grammar_max_whitespace < 1 or args.variants != ["P5"]):
        parser.error("A bounded-whitespace correction requires only P5 and a positive bound.")
    if args.freeze_from:
        print(dumps(freeze(args.freeze_from, args.output)), flush=True)
    elif args.suite:
        evaluate(args)
    else:
        parser.error("Use --freeze-from for new data or --suite for replay.")


if __name__ == "__main__":
    main()
