"""Collect a small, fixed-split v2 pipeline sample with a bounded storage budget."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=3100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = []
    for offset, split in enumerate(("train", "val", "test")):
        seed = args.seed_start + offset
        for color in ("red", "yellow", "blue"):
            plan.append({"seed": seed, "split": split, "color": color})
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for case in plan:
        used = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
        if used > 256 * 1024**2 or shutil.disk_usage(args.output).free < 2 * 1024**3:
            raise RuntimeError("Sample collection stopped: keep 2 GiB free and the sample under 256 MiB.")
        output = args.output / f"seed_{case['seed']}_{case['color']}"
        chinese = {"red": "红", "yellow": "黄", "blue": "蓝"}[case["color"]]
        command = [
            sys.executable,
            "-m",
            "experiments.microduck_tasks.demo",
            "--output",
            str(output),
            "--teacher",
            "efficient",
            "--seed",
            str(case["seed"]),
            "--command",
            "看看眼前有什么",
            "--command",
            f"踢{chinese}球",
        ]
        # Each color starts from the same seeded world; direct motions follow the kick.
        if case["color"] == "red":
            for instruction in ("往前走", "后退一点", "左移", "右移", "左转一下", "右转一下"):
                command += ["--command", instruction]
        with (args.output / f"{output.name}.log").open("w") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        for filename in ("episode.json", "summary.json"):
            path = output / filename
            if path.exists():
                value = json.loads(path.read_text())
                value["metadata"]["training_split"] = case["split"]
                path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        result = {**case, "process_exit_code": proc.returncode, "output": str(output)}
        results.append(result)
        (args.output / "collection.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
