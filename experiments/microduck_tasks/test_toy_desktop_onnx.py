"""Desktop routing and native-library setup must select the actual ONNX backend."""

import os
import sys

import pytest

from . import onnx_bootstrap, toy_desktop


@pytest.fixture
def runtime_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(onnx_bootstrap, "REPO", tmp_path)
    cache = tmp_path / ".cache/microduck_head"
    for relative in (
        "onnx_deps/onnxruntime/__init__.py",
        "onnx_deps/onnxruntime_genai/__init__.py",
        "onnx_cuda/nvidia/cu13/lib/libcudart.so.13",
        "onnx_cudnn/nvidia/cudnn/lib/libcudnn.so.9",
    ):
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return cache


def test_explicit_onnx_routes_without_loading_training_checkpoint(monkeypatch):
    from . import q1r_onnx_backend

    args = toy_desktop.arguments(["--backend", "onnx"])
    assert args.backend == "onnx" and args.checkpoint is None
    calls, instance = [], object()

    def factory(*, model_dir):
        calls.append(model_dir)
        return instance

    monkeypatch.setattr(q1r_onnx_backend, "ToyQ1ROnnxBackend", factory)
    assert toy_desktop.make_backend(args) is instance
    assert calls == [args.onnx_model]


def test_default_routes_to_selected_pytorch_q1r(monkeypatch):
    from . import q1r_backend

    args = toy_desktop.arguments([])
    assert args.backend == "q1r" and args.checkpoint.name == "checkpoint_000513"
    calls, instance = [], object()

    def factory(*, model_dir, checkpoint):
        calls.append((model_dir, checkpoint))
        return instance

    monkeypatch.setattr(q1r_backend, "ToyQ1RBackend", factory)
    assert toy_desktop.make_backend(args) is instance
    assert calls == [(args.model_dir, args.checkpoint)]


def test_onnx_rejects_an_adapter_argument_instead_of_ignoring_it():
    with pytest.raises(SystemExit):
        toy_desktop.arguments(["--backend", "onnx", "--checkpoint", "different-adapter"])


def test_native_environment_prepends_once_and_preserves_existing_paths(runtime_tree):
    original = {"LD_LIBRARY_PATH": "/existing/libs", "PYTHONPATH": "/existing/python", "DISPLAY": ":0"}
    prepared = onnx_bootstrap.runtime_environment(original)
    assert prepared["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(runtime_tree / "onnx_cuda/nvidia/cu13/lib"),
        str(runtime_tree / "onnx_cudnn/nvidia/cudnn/lib"),
        "/existing/libs",
    ]
    assert prepared["PYTHONPATH"].split(os.pathsep) == [str(runtime_tree / "onnx_deps"), "/existing/python"]
    assert prepared["DISPLAY"] == ":0" and original["PYTHONPATH"] == "/existing/python"
    assert onnx_bootstrap.runtime_environment(prepared) == prepared


def test_reexec_keeps_cli_arguments_before_importing_native_libraries(runtime_tree, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    calls = []

    def replace(executable, command, environment):
        calls.append((executable, command, environment))
        raise SystemExit(0)

    monkeypatch.setattr(os, "execve", replace)
    arguments = ["--backend", "onnx", "--output", "a directory with spaces", "--seed", "12"]
    with pytest.raises(SystemExit):
        onnx_bootstrap.ensure_desktop_environment(arguments)
    executable, command, environment = calls[0]
    assert executable == sys.executable
    assert command == [sys.executable, "-m", "experiments.microduck_tasks.toy_desktop", *arguments]
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(runtime_tree / "onnx_deps")


def test_prepared_process_does_not_reexec(runtime_tree, monkeypatch):
    prepared = onnx_bootstrap.runtime_environment(os.environ)
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        monkeypatch.setenv(key, prepared[key])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "execve", lambda *_: pytest.fail("Environment is already correct"))
    onnx_bootstrap.ensure_desktop_environment([])
    assert sys.path[0] == str(runtime_tree / "onnx_deps")


def test_missing_native_dependency_has_actionable_error(runtime_tree):
    (runtime_tree / "onnx_cudnn/nvidia/cudnn/lib/libcudnn.so.9").unlink()
    with pytest.raises(FileNotFoundError, match="PROJECT_BRIEF_ZH.md"):
        onnx_bootstrap.runtime_environment({})
