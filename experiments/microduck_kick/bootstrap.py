"""Resolve the existing local simulation dependencies without installing anything."""

import importlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MICRODUCK = Path(os.environ.get("MICRODUCK_ROOT", "/home/luckysir/microduck")).expanduser().resolve()
ASSETS = MICRODUCK / "microduck_rl_hd1910/src/mjlab_microduck/robot/microduck"
STAND_MODEL = MICRODUCK / "deploy_hd1910/models/hd1910_walk_1849.onnx"
KICK_MODELS = REPO / ".cache/microduck_kick_research_20261114"
KICK_HASHES = {
    "left": "d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a",
    "right": "147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af",
}


def configure():
    for path in (REPO, MICRODUCK / "scripts", MICRODUCK / "deploy_hd1910"):
        sys.path.insert(0, str(path))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    importlib.import_module("strands_robots")
    for path in (ASSETS / "scene_ball.xml", STAND_MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; set MICRODUCK_ROOT to the local workspace.")
