#!/usr/bin/env bash
set -euo pipefail
demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
microduck_root="${MICRODUCK_ROOT:-/home/luckysir/microduck}"
microduck_python="${MICRODUCK_PYTHON:-$microduck_root/microduck_rl/.venv/bin/python}"
export MICRODUCK_ROOT="$microduck_root"
export MUJOCO_GL="${MUJOCO_GL:-egl}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset PYTHONPATH
exec "$microduck_python" "$demo_dir/demo.py" "$@"
