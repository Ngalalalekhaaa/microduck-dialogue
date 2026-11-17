"""Native viewer isolation, validation and worker lifecycle without a desktop."""

import contextlib
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from . import toy_native_viewer


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    from .bootstrap import configure

    configure()
    mujoco = pytest.importorskip("mujoco")
    native = pytest.importorskip("mujoco.viewer")

    scene = tmp_path / "scene.xml"
    scene.write_text(
        '<mujoco><worldbody><body name="trunk_base" pos="0 0 .2">'
        '<freejoint/><geom size=".035"/></body></worldbody></mujoco>'
    )
    snapshot = dict(qpos=np.array([0, 0, 0.2, 1, 0, 0, 0.0]), qvel=np.zeros(6), time_s=1, display_frame_id=4)
    calls = []
    handle = SimpleNamespace(
        cam=mujoco.MjvCamera(),
        opt=mujoco.MjvOption(),
        lock=contextlib.nullcontext,
        sync=lambda: calls.append("sync"),
        close=lambda: calls.append("close"),
        is_running=lambda: "close" not in calls,
    )

    def launch(model, data, **options):
        calls.append(options)
        handle.model, handle.data = model, data
        return handle

    window = SimpleNamespace(
        window_id=123,
        find=lambda: 123,
        geometry=lambda: (20, 30, 900, 700),
        place=lambda *rect: calls.append(rect),
        close=lambda: calls.append("window_close"),
    )
    monkeypatch.setattr(native, "launch_passive", launch)
    monkeypatch.setattr(toy_native_viewer, "_X11Window", lambda: window)
    instance = toy_native_viewer.NativeViewer(scene, lambda: snapshot)
    yield instance, snapshot, calls
    instance.close()


def test_native_state_is_independent_and_menu_never_writes_source(viewer):
    instance, snapshot, calls = viewer
    assert calls[0] == dict(show_left_ui=False, show_right_ui=False)
    assert instance.model is instance.handle.model
    assert not np.shares_memory(instance.data.qpos, snapshot["qpos"])
    instance.data.qpos[0] = 99  # A display-only perturbation cannot change the source cache.
    assert snapshot["qpos"][0] == 0
    instance.handle.cam.azimuth = 35
    snapshot["qpos"][0], snapshot["time_s"], snapshot["display_frame_id"] = 0.4, 2, 5
    instance.update()
    assert instance.data.qpos[0] == 0.4 and instance.data.time == 2 and instance.frame_id == 5
    assert instance.handle.cam.azimuth == 35 and calls[-1] == "sync"


def test_native_geometry_passes_only_own_window(viewer):
    instance, _, calls = viewer
    assert instance.window_id == 123 and instance.geometry() == (20, 30, 900, 700)
    instance.place(70, 80, 900, 650)
    assert calls[-1] == (70, 80, 900, 650)


def test_default_camera_survives_first_sync_and_later_user_zoom_is_preserved(viewer):
    instance, _, _ = viewer
    camera = instance.handle.cam
    tracking = instance.mujoco.mjtCamera.mjCAMERA_TRACKING
    assert camera.type == tracking and camera.distance == pytest.approx(0.9)
    instance.update()
    assert camera.type == tracking and camera.distance == pytest.approx(0.9)
    camera.distance, camera.azimuth, camera.elevation = 1.4, 70, -35
    camera.type = instance.mujoco.mjtCamera.mjCAMERA_FREE
    for _ in range(3):
        instance.update()
    assert camera.type == instance.mujoco.mjtCamera.mjCAMERA_FREE
    assert (camera.distance, camera.azimuth, camera.elevation) == pytest.approx((1.4, 70, -35))


@pytest.mark.parametrize(
    "field,value", [("qpos", np.full(7, np.nan)), ("qvel", np.full(6, np.inf)), ("time_s", np.nan)]
)
def test_nonfinite_snapshot_rejected_before_native_state_change(viewer, field, value):
    instance, snapshot, _ = viewer
    snapshot[field] = value
    with pytest.raises(ValueError, match="finite"):
        instance.update()
    assert np.isfinite(instance.data.qpos).all() and np.isfinite(instance.data.qvel).all()
    assert instance.data.time == 1


def test_wrong_scene_snapshot_is_rejected(viewer):
    instance, snapshot, _ = viewer
    snapshot["qpos"] = np.zeros(8)
    with pytest.raises(ValueError, match="shape"):
        instance.update()


def test_worker_reads_cache_without_world_or_ui_lock(viewer):
    instance, _, calls = viewer
    synced = threading.Event()
    instance.handle.sync = synced.set
    instance.start()
    thread = instance.worker
    instance.start()
    assert instance.worker is thread and synced.wait(1)
    instance.close()
    assert not thread.is_alive() and not instance.is_running() and "window_close" in calls


def test_worker_error_is_exposed_without_writing_simulation(viewer):
    instance, snapshot, _ = viewer
    snapshot["time_s"] = float("inf")
    instance.start()
    assert instance.shutdown.wait(1)
    assert "finite" in instance.error and not instance.is_running()


def test_close_joins_native_render_owner_before_x11_cleanup(viewer):
    instance, _, calls = viewer
    instance.native_threads = [SimpleNamespace(join=lambda timeout: calls.append(("native_join", timeout)))]
    instance.close()
    assert calls[-3:] == ["close", ("native_join", 3), "window_close"]


@pytest.mark.parametrize("hz", [0, 121, np.nan, np.inf])
def test_invalid_native_sync_rate_is_rejected(viewer, hz):
    with pytest.raises(ValueError, match="rate"):
        toy_native_viewer.NativeViewer("unused.xml", lambda: None, hz=hz)
