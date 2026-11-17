#!/usr/bin/env bash
set -euo pipefail
demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$demo_dir/../.." && pwd)"
microduck_root="${MICRODUCK_ROOT:-/home/luckysir/microduck}"
microduck_python="${MICRODUCK_PYTHON:-$microduck_root/microduck_rl/.venv/bin/python}"
if [[ ! -x "$microduck_python" ]]; then
    echo "Python not found: $microduck_python. Set MICRODUCK_PYTHON to the existing MuJoCo/ONNX/BAM environment." >&2
    exit 1
fi
export MICRODUCK_ROOT="$microduck_root"
export MUJOCO_GL="${MUJOCO_GL:-egl}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd -- "$repo_dir"
exec env -u PYTHONPATH "$microduck_python" -m experiments.microduck_tasks.demo "$@"
