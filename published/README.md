# Published experiment records

These are selected copies of the local experiment outputs, preserved without rewriting the JSON or data rows.
`manifest.json` records the source path, byte size and SHA-256 of each copied file.
Absolute paths in historical metadata describe the original machine; they are provenance, not portable launch paths.

- `q1r_20261116/data_final/`: 8,196 synthetic training examples and 343 held-out synthetic validation examples,
  their audit and generation manifest. These are not raw user conversations.
- `q1r_20261116/train/`: actual configuration, parameter inventory, initialization, update history,
  validation curve and final selection results. No model or optimizer tensors are included.
- `q1r_20261116/selected_model.json`: metadata for the selected checkpoint 513, not the checkpoint weights.
- `q1r_20261116/post_training/comparison.json`: the matched 310-case development comparison.
- `q1r_20261116/evaluation_20261117/`: assessment, data coverage review and closed-loop summary.
- `rgb_tof_20261116/FOOT_REACH_SUMMARY.json`: the independent sensor-controller development regression.

The 480 sealed blind-test examples, intermediate checkpoints, raw desktop conversations, recordings,
model caches and large simulation trajectories are excluded. Paths to other original outputs in historical
reports will only resolve in the author's full local workspace. Development and validation scores do not
establish general physical success rates or readiness for real hardware.
