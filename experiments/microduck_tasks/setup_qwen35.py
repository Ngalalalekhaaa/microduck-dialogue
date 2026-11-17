"""Pinned Qwen3.5-2B trial service; preserves existing Qwen3-VL weights and services.

Uses Unsloth's GGUF conversion of Qwen's model and the already installed llama.cpp.
"""

import argparse
import json
import subprocess
from pathlib import Path

from . import setup_vlm

CACHE = setup_vlm.REPO / ".cache/microduck_qwen35"
REVISION = "f6d5376be1edb4d416d56da11e5397a961aca8ae"
REPO = "unsloth/Qwen3.5-2B-GGUF"
FILES = (
    ("Qwen3.5-2B-Q4_K_M.gguf", 1280835840, "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"),
    ("mmproj-F16.gguf", 668227264, "7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    command = setup_vlm.loader_command()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.download:
        setup_vlm.CACHE = CACHE
        for name, size, sha in FILES:
            setup_vlm.download(
                (name, size, sha, f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true")
            )
    for name, size, sha in FILES:
        path = CACHE / name
        if not path.is_file() or path.stat().st_size != size or setup_vlm.digest(path) != sha:
            raise ValueError(f"Run --download to obtain the pinned file: {path}")
    command += [
        "--model",
        str(CACHE / FILES[0][0]),
        "--mmproj",
        str(CACHE / FILES[1][0]),
        "--alias",
        "microduck-qwen35",
        "--host",
        "127.0.0.1",
        "--port",
        "8084",
        "--ctx-size",
        "8192",
        "--n-gpu-layers",
        "99",
        "--parallel",
        "1",
        "--threads",
        "4",
        "--jinja",
        "--no-warmup",
    ]
    (CACHE / "deployment.json").write_text(
        json.dumps(
            {
                "repository": REPO,
                "revision": REVISION,
                "base_model": "Qwen/Qwen3.5-2B",
                "files": FILES,
                "command": command,
                "robot_finetuned": False,
            },
            indent=2,
        )
        + "\n"
    )
    if args.serve:
        subprocess.run(command, check=True, cwd=Path(command[3]).parent)


if __name__ == "__main__":
    main()
