"""Isolated Qwen3.5 frozen-feature runtime, sharing only the existing Torch install."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

MODEL_REPO = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
WEIGHTS_SHA256 = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".cache/microduck_head"))
    parser.add_argument("--shared-python", default="/home/luckysir/microduck/microduck_rl/.venv/bin/python")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 13 * 1024**3:
        raise RuntimeError("Free 13 GiB before setup to leave space for the robot workspace.")
    python = root / "venv/bin/python"
    if not python.exists():
        subprocess.run([args.shared_python, "-m", "venv", str(root / "venv")], check=True)
    script = "import sysconfig;print(sysconfig.get_path('purelib'))"
    shared = subprocess.check_output([args.shared_python, "-c", script], text=True).strip()
    local = subprocess.check_output([str(python), "-c", script], text=True).strip()
    (Path(local) / "microduck_shared_packages.pth").write_text(shared + "\n")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "transformers==5.4.0",
            "accelerate==1.12.0",
            "bitsandbytes==0.49.2",
            "fla-core==0.4.1",
            "einops==0.8.2",
        ],
        check=True,
    )
    if args.download:
        code = (
            "from huggingface_hub import snapshot_download;"
            f"snapshot_download({MODEL_REPO!r},revision={MODEL_REVISION!r},local_dir={str(root / 'model')!r},"
            "allow_patterns=['*.json','*.txt','*.jinja','*.safetensors','README.md'])"
        )
        subprocess.run(
            [str(python), "-c", code],
            check=True,
            env=dict(os.environ, HF_HUB_DISABLE_XET="1", HF_HUB_DOWNLOAD_TIMEOUT="120"),
        )
        if digest(root / "model" / WEIGHTS) != WEIGHTS_SHA256:
            raise RuntimeError("Qwen3.5 weights failed SHA256 verification.")
    manifest = dict(
        model_repo=MODEL_REPO,
        revision=MODEL_REVISION,
        weights_sha256=WEIGHTS_SHA256,
        python=str(python),
        shared_packages=shared,
        original_environment_modified=False,
    )
    (root / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
