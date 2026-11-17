"""Pipeline orchestration never starts a second extractor or ignores failed stages."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import run_toy_selector_pipeline as pipeline


@pytest.fixture
def paths(tmp_path):
    values = {key: tmp_path / key for key in ("features", "dataset", "output", "run_dir", "model")}
    for key in ("features", "dataset", "model"):
        values[key].mkdir()
    (values["dataset"] / "manifest.json").write_text("{}")
    (values["features"] / "manifest.json").write_text('{"complete":true}')
    return values


def _successful_subprocesses(monkeypatch, paths):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command[1] == "-m"
        assert kwargs["check"] is False and kwargs["timeout"] > 0
        assert kwargs["stderr"] == subprocess.STDOUT
        if command[2].endswith("train_toy_selector"):
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            output = paths["output"]
            output.mkdir()
            (output / "selector.pt").write_text("mock checkpoint")
            (output / "results.json").write_text(
                json.dumps(
                    {
                        "training_seconds": 1,
                        "evaluation": {
                            "test": {
                                "decision_user": {"accuracy": 0.75, "macro_f1_present": 0.6},
                                "action_plan_exact": {"accuracy": 0.25, "support": 4},
                            }
                        },
                    }
                )
            )
        else:
            assert command[2].endswith("evaluate_toy_selector")
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            output = paths["run_dir"] / "evaluation"
            output.mkdir()
            (output / "report.json").write_text('{"summary":{"semantic_pass":2,"cases":4}}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    return commands


def test_completed_features_run_training_then_evaluation_without_pid(paths, monkeypatch):
    commands = _successful_subprocesses(monkeypatch, paths)
    result = pipeline.run_pipeline(**paths)
    assert result["complete"] and result["stage"] == "complete"
    assert len(commands) == 2
    assert not any("extract" in command[2] for command in commands)
    assert result["summary"]["test_user_decision"] == {"accuracy": 0.75, "macro_f1_present": 0.6}
    assert result["summary"]["test_action_plan_exact"]["accuracy"] == 0.25
    assert result["summary"]["test_false_start_user"] is None
    assert result["summary"]["simulation_evaluation"] == {"semantic_pass": 2, "cases": 4}
    assert (paths["run_dir"] / "summary.json").is_file()
    assert all((paths["run_dir"] / log).is_file() for log in ("feature_wait.log", "train.log", "evaluate.log"))


def test_waits_for_existing_worker_then_uses_final_manifest(paths, monkeypatch):
    commands = _successful_subprocesses(monkeypatch, paths)
    manifest = paths["features"] / "manifest.json"
    manifest.write_text('{"complete":false}')
    monkeypatch.setattr(pipeline, "_process_identity", lambda pid: f"{pid}:1")
    pauses = []

    def finish(seconds):
        pauses.append(seconds)
        manifest.write_text('{"complete":true}')

    monkeypatch.setattr(pipeline.time, "sleep", finish)
    result = pipeline.run_pipeline(**paths, feature_pid=123)
    assert result["complete"] and pauses == [10] and len(commands) == 2
    waited = json.loads((paths["run_dir"] / "feature_wait.log").read_text())
    assert waited["feature_process_identity"] == "123:1"


@pytest.mark.parametrize("identities", [(None, None), ("123:1", None), ("123:1", "123:2")])
def test_dead_zombie_or_reused_pid_fails_without_starting_training(paths, monkeypatch, identities):
    (paths["features"] / "manifest.json").write_text('{"complete":false}')
    answers = iter(identities)
    monkeypatch.setattr(pipeline, "_process_identity", lambda pid: next(answers))
    with pytest.raises(RuntimeError, match="before completion"):
        pipeline.run_pipeline(**paths, feature_pid=123)
    status = json.loads((paths["run_dir"] / "status.json").read_text())
    assert status["stage"] == "failed" and status["failed_stage"] == "waiting_features"
    assert not paths["output"].exists() and not (paths["run_dir"] / "train.log").exists()


@pytest.mark.parametrize("failure", ["returncode", "timeout"])
def test_training_subprocess_failure_never_runs_evaluation(paths, monkeypatch, failure):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        pipeline.run_pipeline(**paths)
    status = json.loads((paths["run_dir"] / "status.json").read_text())
    assert status["stage"] == "failed" and status["failed_stage"] == "training"
    assert len(calls) == 1 and not (paths["run_dir"] / "evaluation").exists()


def test_existing_checkpoint_directory_is_preserved(paths):
    paths["output"].mkdir()
    (paths["output"] / "selector.pt").write_text("existing")
    with pytest.raises(FileExistsError, match="Training output already exists"):
        pipeline.run_pipeline(**paths)
    assert (paths["output"] / "selector.pt").read_text() == "existing"
    assert json.loads((paths["run_dir"] / "status.json").read_text())["stage"] == "failed"


def test_linux_process_identity_rejects_zombie_and_tracks_start_time(monkeypatch):
    monkeypatch.setattr(pipeline.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(pipeline.sys, "platform", "linux")
    fields = ["S"] + ["0"] * 18 + ["998877"]
    monkeypatch.setattr(Path, "read_text", lambda path: "123 (name with ) parentheses) " + " ".join(fields))
    assert pipeline._process_identity(123) == "123:998877"
    fields[0] = "Z"
    assert pipeline._process_identity(123) is None


def test_worker_wait_is_bounded(paths, monkeypatch):
    (paths["features"] / "manifest.json").write_text('{"complete":false}')
    clock = [0.0]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(pipeline, "_process_identity", lambda pid: "123:1")
    with pytest.raises(TimeoutError, match="time limit"):
        pipeline.run_pipeline(**paths, feature_pid=123, timeout_seconds=20)
    assert clock[0] == 20
    status = json.loads((paths["run_dir"] / "status.json").read_text())
    assert status["failed_stage"] == "waiting_features" and not status["complete"]
