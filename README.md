# MicroDuck Dialogue

Chinese dialogue, task planning, and skill execution for a MicroDuck robot in MuJoCo simulation.
This research repository combines a fine-tuned Qwen3.5-2B planner with existing walking, posture, and kicking controllers.
The desktop interface uses the native MuJoCo viewer and a Chinese chat window.

[中文项目介绍](PROJECT_BRIEF_ZH.md) · [Detailed experiment notes](experiments/microduck_tasks/PROJECT_BRIEF_ZH.md) · [Upstream README](README_UPSTREAM.md)

## How it works

A request such as “先点头，再去踢红球” (“nod, then kick the red ball”) follows this path:

```text
Chinese instruction + structured observations + state + history
    -> Q1R: decision + goals + reply
    -> plan validation and task queue
    -> existing motion skills in MuJoCo
    -> measured execution result and a generated reply
```

The planner supports 13 self-motion skills, movement, turns, and ball approach/kicking tasks.
The queue allows at most 15 unfinished goals. Unsupported tasks should trigger clarification.
Ordinary pause, resume, and stop commands have a host control path; pausing waits for a stable execution boundary.
The current Q1R model receives **text and structured observations, not raw camera images**.
The default executor uses simulated RGB-D and IMU observations. A separate RGB/ToF branch is described below.

## Current model and training

Q1R continues the earlier Q1 planning adapter and adds natural-language replies to structured decisions and goals.
The current desktop backend is **PyTorch**, using `checkpoint_000513`. High-level ONNX export remains an experiment.
Existing low-level ONNX motion policies are separate from that planner backend choice.

| Item | Q1R configuration |
| --- | --- |
| Base model | Qwen3.5-2B |
| Method | QLoRA, 4-bit NF4 base weights |
| Trainable parameters | 8,409,600 language-layer LoRA parameters |
| LoRA | rank 8, alpha 16, dropout 0.05 |
| Data | 8,196 synthetic training examples; 343 validation examples; 42 scenario categories |
| Optimizer / peak learning rate | AdamW / `5e-5` |
| Batch / gradient accumulation | 1 / 16 |
| Training / selected checkpoint | 2 epochs, 1,026 updates / step 513 |

Base weights, the vision encoder, embeddings, output head, and motion controllers were not updated.
The first-epoch checkpoint was selected using a validation rule that prioritizes fewer incorrect action starts.
See [training configuration](published/q1r_20261116/train/training_config.json), [parameter audit](published/q1r_20261116/train/model_preflight.json), and [data audit](published/q1r_20261116/data_final/audit.json).

## Results and limits

| Evaluation | Observed result | Scope |
| --- | --- | --- |
| Complete plan correctness | Q1: 167/310 (53.9%); Q1R: 268/310 (86.5%) | Same 310 development cases; not a blind test |
| Q1R controlling MuJoCo | 11/11 finite goals passed, including 4 kicks | Seven finite-action sessions; a small sample |
| Separate RGB + 8×8 ToF + IMU/encoders branch | 16/16 v6 cases passed | Development regression; not integrated into the default chat entry |

Evidence: [Q1/Q1R comparison](published/q1r_20261116/post_training/comparison.json), [closed-loop summary](published/q1r_20261116/evaluation_20261117/closed_loop/summary.json), [ToF summary](published/rgb_tof_20261116/FOOT_REACH_SUMMARY.json).

State constraints, queue boundaries, and execution feedback remain unreliable in some cases.
For example, the model has reported the next queued action as already completed.
Validation catches some invalid plans but cannot catch every semantically incorrect plan.
These results do not establish real-hardware performance, general conversation quality, or a universal kicking success rate.

## Running the desktop experiment

**This is a source/data/report release, not a complete downloadable model bundle.**
Model weights, training state, runtime caches, recordings, and personal chat sessions are excluded.
A fresh clone cannot run Q1R until the following local prerequisites are prepared:

1. Linux graphical desktop, NVIDIA CUDA/PyTorch environment, MuJoCo, the project dependencies, and PyQt5 for the interpreter selected by `--qt-python` (default `/usr/bin/python3`).
2. Robot assets and motion resources from [microduck-project](https://github.com/Ngalalalekhaaa/microduck-project); set `MICRODUCK_ROOT` to that checkout. Matching kicking weights must also be prepared in the cache location defined by [bootstrap.py](experiments/microduck_tasks/bootstrap.py).
3. The pinned Qwen base model and compatible inference dependencies. [setup_skill_head.py](experiments/microduck_tasks/setup_skill_head.py) accepts `--shared-python /path/to/cuda/python --download`; QLoRA additionally requires the PEFT version checked by [q0_model.py](experiments/microduck_tasks/q0_model.py).
4. A validated Q1R adapter checkpoint, including `identity.json`, with its matching `results.json` in the checkpoint's parent directory. Adapter weights are not included in this release.

After preparing these resources, run from the repository root with your actual paths:

```bash
export MICRODUCK_ROOT=/path/to/microduck-project
/path/to/prepared/python -m experiments.microduck_tasks.toy_desktop \
  --backend q1r \
  --model-dir /path/to/Qwen3.5-2B \
  --checkpoint /path/to/q1r/train/checkpoint_000513
```

Historical notes contain paths from the original workstation; replace them with your own paths.
Published metadata records the original run and is not a replacement for the missing weights.

## Repository map and attribution

| Path | Contents |
| --- | --- |
| [experiments/microduck_tasks/](experiments/microduck_tasks/) | Planning, data generation, training, evaluation, native UI, and simulated execution |
| [published/](published/) | Selected synthetic data, configurations, audits, and evaluation summaries |
| [RGB_TOF_SIM_ZH.md](experiments/microduck_tasks/RGB_TOF_SIM_ZH.md) | Independent sensor-fusion and kicking experiment |
| [reference/](reference/) | Original experiment plan and companion kit; planned steps are not all completed |
| [strands_robots/](strands_robots/) | Reused Strands Robots library source |

This independent repository includes source from [Strands Robots](https://github.com/strands-labs/robots), with its [Apache-2.0 license](LICENSE), [NOTICE](NOTICE), and [upstream README](README_UPSTREAM.md) preserved.
It does not claim authorship of the upstream library. [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) records the source revision.
Robot resources and model weights retain their respective upstream licenses and are obtained separately.

The [publication check record](published/PUBLICATION_CHECKS.json) lists the test scope and an unresolved type-check finding in unchanged upstream code.
GitHub Actions are disabled for this initial snapshot; inherited upstream workflows have not been configured for this independent repository.
