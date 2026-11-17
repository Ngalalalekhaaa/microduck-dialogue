#!/usr/bin/env bash
# Local Qwen service; no Python packages or global libraries are changed.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$task_dir/setup_vlm.py" --serve "$@"
