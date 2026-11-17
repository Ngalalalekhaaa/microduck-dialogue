"""Use the existing local model/environment; never install into the training venv."""

import importlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MICRODUCK = Path(os.environ.get("MICRODUCK_ROOT", "/home/luckysir/microduck")).expanduser().resolve()
ROBOT_ASSETS = MICRODUCK / "microduck_rl_hd1910/src/mjlab_microduck/robot/microduck"
MODEL_PATH = MICRODUCK / "deploy_hd1910/models/hd1910_walk_1849.onnx"


def configure():
    for path in (REPO, MICRODUCK / "scripts", MICRODUCK / "deploy_hd1910"):
        sys.path.insert(0, str(path))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    importlib.import_module("strands_robots")
    for path in (ROBOT_ASSETS / "scene.xml", MODEL_PATH, MICRODUCK / "scripts/render_hd1910_controller.py"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; set MICRODUCK_ROOT to the local microduck workspace")
