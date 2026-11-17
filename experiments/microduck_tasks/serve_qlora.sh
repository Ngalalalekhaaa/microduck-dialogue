#!/usr/bin/env bash
# Serve only local frozen QLoRA weights for the HD1910 simulation client.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$task_dir/../.." && pwd)"
python_bin="$project_root/.cache/microduck_train/venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing training environment: $python_bin. Restore the existing microduck_train venv first." >&2
    exit 1
fi
cd -- "$project_root"
exec "$python_bin" -m experiments.microduck_tasks.serve_adapter "$@"
