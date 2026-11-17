#!/usr/bin/env bash
# Benchmark the experimental CUDA export against frozen real Q1R requests.
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$task_root"
task_cache="$task_root/.cache/microduck_head"
task_output="$task_root/experiments/microduck_tasks/outputs/q1r_onnx_20261117"
for task_dir in onnx_deps onnx_cuda/nvidia/cu13/lib onnx_cudnn/nvidia/cudnn/lib; do
    if [[ ! -d "$task_cache/$task_dir" ]]; then
        echo "Missing $task_cache/$task_dir; install the isolated dependencies recorded in PROJECT_BRIEF_ZH.md." >&2
        exit 1
    fi
done
export PYTHONPATH="$task_cache/onnx_deps${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$task_cache/onnx_cuda/nvidia/cu13/lib:$task_cache/onnx_cudnn/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ $# -eq 0 ]]; then
    set -- --model "$task_output/onnx_int8_fp16_head" \
        --inputs "$task_output/encoded_inputs.jsonl" \
        --output "$task_output/retest_$(date +%Y%m%d_%H%M%S).jsonl" --repeats 2
fi
exec "$task_cache/venv/bin/python" -u -m experiments.microduck_tasks.q1r_onnx_benchmark "$@"
