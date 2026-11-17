"""Start the desktop with isolated ORT and CUDA libraries before native imports."""

import os
import sys

from .bootstrap import REPO


def runtime_environment(current):
    cache = REPO / ".cache/microduck_head"
    overlay = cache / "onnx_deps"
    libraries = (cache / "onnx_cuda/nvidia/cu13/lib", cache / "onnx_cudnn/nvidia/cudnn/lib")
    required = (
        overlay / "onnxruntime/__init__.py",
        overlay / "onnxruntime_genai/__init__.py",
        libraries[0] / "libcudart.so.13",
        libraries[1] / "libcudnn.so.9",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; restore the isolated ONNX dependencies listed in "
                "experiments/microduck_tasks/PROJECT_BRIEF_ZH.md."
            )
    environment = dict(current)
    for key, first in (("PYTHONPATH", [overlay]), ("LD_LIBRARY_PATH", libraries)):
        paths = [str(path) for path in first]
        paths.extend(path for path in current.get(key, "").split(os.pathsep) if path)
        environment[key] = os.pathsep.join(dict.fromkeys(paths))
    return environment


def ensure_desktop_environment(argv):
    """Re-exec only when needed: the ELF loader reads LD_LIBRARY_PATH at startup."""
    environment = runtime_environment(os.environ)
    if any(environment[key] != os.environ.get(key) for key in ("PYTHONPATH", "LD_LIBRARY_PATH")):
        command = [sys.executable, "-m", "experiments.microduck_tasks.toy_desktop", *argv]
        os.execve(sys.executable, command, environment)
    overlay = str(REPO / ".cache/microduck_head/onnx_deps")
    if overlay in sys.path:
        sys.path.remove(overlay)
    sys.path.insert(0, overlay)
