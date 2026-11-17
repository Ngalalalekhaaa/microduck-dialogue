"""Build an isolated training overlay without modifying the existing robot environment."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from experiments.microduck_tasks.train_vlm import MIN_FREE_BYTES, MODEL_REPO, MODEL_REVISION, MODEL_SHA256, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".cache/microduck_train"))
    parser.add_argument(
        "--shared-python", type=Path, default=Path("/home/luckysir/microduck/microduck_rl/.venv/bin/python")
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    required = MIN_FREE_BYTES + (5 * 1024**3 if not (root / "model/model.safetensors").exists() else 512 * 1024**2)
    if shutil.disk_usage(root).free < required:
        raise RuntimeError("Free additional storage before setup; model/environment must leave at least 8 GiB free.")
    environment = root / "venv"
    python = environment / "bin/python"
    if not python.exists():
        subprocess.run([str(args.shared_python), "-m", "venv", str(environment)], check=True)
    shared = subprocess.check_output(
        [str(args.shared_python), "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"], text=True
    ).strip()
    local = subprocess.check_output(
        [str(python), "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"], text=True
    ).strip()
    (Path(local) / "microduck_shared_packages.pth").write_text(shared + "\n")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            str(Path(__file__).with_name("training_requirements.txt")),
        ],
        check=True,
    )
    if args.download:
        script = (
            "from huggingface_hub import snapshot_download;"
            f"snapshot_download({MODEL_REPO!r},revision={MODEL_REVISION!r},local_dir={str(root / 'model')!r},"
            "allow_patterns=['*.json','*.txt','*.safetensors','README.md'])"
        )
        environment_vars = dict(os.environ, HF_HUB_DISABLE_XET="1", HF_HUB_DOWNLOAD_TIMEOUT="120")
        subprocess.run([str(python), "-c", script], env=environment_vars, check=True)
        if digest(root / "model/model.safetensors") != MODEL_SHA256:
            raise RuntimeError("Downloaded base weights failed SHA256 verification.")
    manifest = {
        "shared_python": str(args.shared_python),
        "shared_site_packages_read_only": shared,
        "training_python": str(python),
        "model_repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "weights_sha256": MODEL_SHA256,
        "original_environment_modified": False,
    }
    (root / "training_environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
