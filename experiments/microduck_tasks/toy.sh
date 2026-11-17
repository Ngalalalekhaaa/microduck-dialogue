#!/usr/bin/env bash
set -euo pipefail
toy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$toy_dir/../.." && pwd)"
microduck_root="${MICRODUCK_ROOT:-/home/luckysir/microduck}"
model_backend=false
previous_arg=""
for toy_arg in "$@"; do
  if [[ "$toy_arg" == "--backend=selector" || "$toy_arg" == "--backend=vlm" ||
        ( "$previous_arg" == "--backend" && ( "$toy_arg" == "selector" || "$toy_arg" == "vlm" ) ) ]]; then
    model_backend=true
  fi
  previous_arg="$toy_arg"
done
if "$model_backend"; then
  microduck_python="${MICRODUCK_PYTHON:-$repo_dir/.cache/microduck_head/venv/bin/python}"
else
  microduck_python="${MICRODUCK_PYTHON:-$microduck_root/microduck_rl/.venv/bin/python}"
fi
export MICRODUCK_ROOT="$microduck_root"
export MUJOCO_GL="${MUJOCO_GL:-egl}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd -- "$repo_dir"
exec env -u PYTHONPATH "$microduck_python" -m experiments.microduck_tasks.toy_demo "$@"
