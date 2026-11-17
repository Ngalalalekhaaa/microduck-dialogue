"""CPU-only checks for encoder geometry and synchronized sensor caching."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def configured(monkeypatch):
    # No renderer or ONNX session is created by these tests.
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("LIBGL_ALWAYS_SOFTWARE", "1")
    from .bootstrap import configure

    try:
        configure()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return pytest.importorskip("mujoco")


def test_encoder_camera_matches_two_head_poses_independent_of_world_root(configured, tmp_path):
    from microduck_deploy.policy import HOME, JOINT_NAMES

    from .rgb_tof_scene import build_sensor_scene, camera_from_encoders

    mujoco = configured
    path = tmp_path / "sensor_scene.xml"
    build_sensor_scene(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    root = int(model.joint("trunk_base_freejoint").qposadr[0])
    body = model.body("trunk_base").id
    camera = model.camera("face_rgb").id
    indices = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    moved_head = HOME.copy()
    moved_head[JOINT_NAMES.index("head_pitch")] += 0.25
    moved_head[JOINT_NAMES.index("head_yaw")] -= 0.3
    roots = ([0, 0, 0.12, 1, 0, 0, 0], [0.7, -0.4, 0.35, np.cos(0.3), 0, 0, np.sin(0.3)])
    head_rotations = []
    for encoders in (HOME.copy(), moved_head):
        original = encoders.copy()
        predicted_position, predicted_rotation = camera_from_encoders(model, encoders)
        np.testing.assert_array_equal(encoders, original)
        np.testing.assert_allclose(predicted_rotation.T @ predicted_rotation, np.eye(3), atol=1e-10)
        for pose in roots:
            data.qpos[root : root + 7] = pose
            data.qpos[indices] = encoders
            mujoco.mj_forward(model, data)
            body_rotation = data.xmat[body].reshape(3, 3)
            expected_position = body_rotation.T @ (data.cam_xpos[camera] - data.xpos[body])
            expected_rotation = body_rotation.T @ data.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1])
            np.testing.assert_allclose(predicted_position, expected_position, atol=1e-10)
            np.testing.assert_allclose(predicted_rotation, expected_rotation, atol=1e-10)
        head_rotations.append(predicted_rotation)
    assert not np.allclose(head_rotations[0], head_rotations[1])


def _packet(time_s, colour):
    return {
        "time_s": time_s,
        "rgb_time_s": time_s,
        "rgb": np.full((2, 2, 3), colour, dtype=np.uint8),
        "ranges_m": np.full((8, 8), 0.3),
        "valid": np.ones((8, 8), dtype=bool),
        "quality": np.ones((8, 8)),
    }


def test_sensor_cache_returns_the_whole_original_frame_without_rendering(configured):
    from .rgb_tof_sensor import RGBToFSensor

    sensor = object.__new__(RGBToFSensor)
    sensor.last = _packet(1.0, 12)
    # Deliberately no renderer or simulation data: a cache hit must not touch either.
    returned = RGBToFSensor.capture.__wrapped__(sensor, np.ones(3), np.ones(3), 1.05)
    assert returned is sensor.last
    assert returned["rgb_time_s"] == returned["time_s"] == 1.0
    np.testing.assert_array_equal(returned["rgb"], np.full((2, 2, 3), 12))


def test_runtime_reuses_frame_encoder_pose_and_rejects_stale_ranges(configured, monkeypatch):
    pytest.importorskip("onnxruntime")
    from . import rgb_tof_runtime as module
    from .rgb_tof_geometry import RGBToFConfig

    runtime = object.__new__(module.SensorRuntime)
    runtime.lock = threading.RLock()
    runtime.config = RGBToFConfig(width=2, height=2)
    runtime.model = object()
    runtime.qi = np.arange(3)
    runtime.data = SimpleNamespace(time=1.0, qpos=np.array([0.1, 0.2, 0.3]), qvel=np.zeros(3))
    runtime.snapshot = lambda: dict(time_s=runtime.data.time, yaw=0, tilt_deg=0, stationary=True, skill="stand")
    runtime.cameras = SimpleNamespace(packet=_packet(1.0, 12))
    runtime.cameras.capture = lambda qpos, qvel, time_s: runtime.cameras.packet
    runtime.packet = runtime.camera_pose = None
    runtime.frame_seq = 0
    runtime.observation_hook = None
    runtime.head_tracking = True
    runtime.target_color = "red"
    runtime.policy = SimpleNamespace(head_command=np.zeros(4))
    encoder_calls, fused = [], []

    def forward_kinematics(model, encoders):
        encoder_calls.append(encoders.copy())
        return encoders.copy(), np.eye(3)

    def fuse(rgb, ranges, *, config, valid, camera_position_body, camera_rotation_body):
        fused.append((rgb, camera_position_body.copy(), valid.copy()))
        return [], {
            "associations": [{"color": "red", "box_uv": [0, 1, 1, 1], "pixels": 2}],
            "rejected_rgb_candidates": [],
        }

    monkeypatch.setattr(module, "camera_from_encoders", forward_kinematics)
    foot_calls = []

    def foot_kinematics(model, encoders):
        foot_calls.append(encoders.copy())
        return {
            side: SimpleNamespace(position_body_m=encoders.copy(), rotation_body_from_foot=np.eye(3))
            for side in ("left", "right")
        }

    monkeypatch.setattr(module, "feet_from_encoders", foot_kinematics)
    monkeypatch.setattr(module, "fuse_balls", fuse)
    first, first_images, _ = runtime.observe()
    first_head = runtime.policy.head_command.copy()
    runtime.data.time = 1.04
    runtime.data.qpos[:] = [0.4, 0.5, 0.6]
    second, second_images, _ = runtime.observe()
    np.testing.assert_array_equal(runtime.policy.head_command, first_head)
    assert len(encoder_calls) == 1
    assert len(foot_calls) == 1
    assert second["feet"] == first["feet"]
    assert first_images["face_rgb"] is second_images["face_rgb"]
    assert second["sensor_time_s"] == first["sensor_time_s"] == 1.0
    assert second["sensor_age_s"] == pytest.approx(0.04)
    np.testing.assert_array_equal(fused[1][1], [0.1, 0.2, 0.3])

    runtime.data.time = 1.08
    runtime.cameras.packet = _packet(1.08, 24)
    fresh, fresh_images, _ = runtime.observe()
    assert runtime.policy.head_command[1] > first_head[1]
    assert len(encoder_calls) == 2
    assert len(foot_calls) == 2
    assert fresh["feet"]["left"]["position_body_m"] == [0.4, 0.5, 0.6]
    assert fresh["sensor_age_s"] == 0
    assert fresh_images["face_rgb"] is not first_images["face_rgb"]
    np.testing.assert_array_equal(fused[2][1], [0.4, 0.5, 0.6])

    runtime.data.time = 1.3
    runtime.observe()
    assert not fused[-1][2].any()
    assert runtime.cameras.packet["valid"].all()
    assert len(encoder_calls) == 2
    assert len(foot_calls) == 2


def test_zone_return_rejects_isolated_subray_and_accepts_supported_cluster(configured):
    from .rgb_tof_sensor import zone_return

    isolated = np.full(25, np.nan)
    isolated[0] = 0.15
    value, quality = zone_return(isolated)
    assert np.isnan(value)
    assert quality == 0
    isolated[1:4] = [0.5, 0.504, 0.508]
    value, quality = zone_return(isolated)
    assert value == pytest.approx(0.504)
    assert quality == pytest.approx(3 / 25)


def _recovery_runtime(monkeypatch):
    """Exercise real RGB/ToF fusion and recovery without simulation or rendering."""
    pytest.importorskip("onnxruntime")
    from . import rgb_tof_runtime as module
    from .test_rgb_tof_fusion import _sensor_inputs
    from .test_rgb_tof_handoff import _case, _joined_red_foot

    _, arguments = _case()
    config, pose = arguments["config"], arguments["pose"]
    point = np.asarray(arguments["locked"]["center_body_m"])
    camera = pose[1].T @ (point - pose[0])
    rgb, ranges = _sensor_inputs(config, camera, calibrated_response=True)
    runtime = object.__new__(module.SensorRuntime)
    runtime.lock = threading.RLock()
    runtime.config, runtime.model = config, object()
    runtime.qi = np.arange(3)
    runtime.data = SimpleNamespace(time=1.0, qpos=np.zeros(3), qvel=np.zeros(3))
    runtime.snapshot = lambda: dict(time_s=runtime.data.time, yaw=0.0, tilt_deg=0.0, stationary=True, skill="stand")
    runtime.cameras = SimpleNamespace()
    runtime.cameras.capture = lambda qpos, qvel, time_s: runtime.cameras.packet
    runtime.packet = runtime.camera_pose = runtime.last_fused_target = None
    runtime.foot_poses = None
    runtime.frame_seq = 0
    runtime.observation_hook = None
    runtime.head_tracking = False
    runtime.target_color = "red"
    runtime.policy = SimpleNamespace(head_command=np.zeros(4), motion=SimpleNamespace(visual_heading=None))
    monkeypatch.setattr(module, "camera_from_encoders", lambda model, encoders: pose)
    monkeypatch.setattr(
        module,
        "feet_from_encoders",
        lambda model, encoders: {
            foot: SimpleNamespace(
                position_body_m=np.array([0.005, sign * 0.042, -0.117]), rotation_body_from_foot=np.eye(3)
            )
            for foot, sign in (("left", 1), ("right", -1))
        },
    )

    def publish(timestamp, *, joined=False, valid=True):
        runtime.data.time = timestamp
        runtime.cameras.packet = dict(
            time_s=timestamp,
            rgb_time_s=timestamp,
            rgb=_joined_red_foot(rgb) if joined else rgb.copy(),
            ranges_m=ranges.copy(),
            valid=np.full((8, 8), valid, dtype=bool),
            quality=np.ones((8, 8)),
        )

    publish(1.0)
    return runtime, publish


def test_fresh_recovery_updates_prior_but_duplicate_packet_does_not(configured, monkeypatch):
    runtime, publish = _recovery_runtime(monkeypatch)
    original, _, _ = runtime.observe()
    assert len(original["detections"]) == 1
    original_prior = runtime.last_fused_target
    runtime.data.time = 1.04
    runtime.observe()
    assert runtime.last_fused_target is original_prior
    assert runtime.last_fused_target[0] == 1.0

    publish(1.08, joined=True)
    recovered, _, _ = runtime.observe()
    assert len(recovered["detections"]) == 1
    assert recovered["detections"][0]["rgb_geometry"] == "partial_arc"
    assert recovered["detections"][0]["tof_zone_count"] >= 2
    assert runtime.diagnostics["near_ball_recovery"]["prior_age_s"] == pytest.approx(0.08)
    recovered_prior = runtime.last_fused_target
    assert recovered_prior is not original_prior
    assert recovered_prior[0] == 1.08

    runtime.data.time = 1.11
    cached, _, _ = runtime.observe()
    assert len(cached["detections"]) == 1
    assert runtime.last_fused_target is recovered_prior
    assert cached["sensor_time_s"] == 1.08
    assert cached["sensor_age_s"] == pytest.approx(0.03)


def test_stale_cached_rgb_and_tof_cannot_extend_recovery_or_move_robot(configured, monkeypatch):
    from .rgb_tof_run import SensorServo

    runtime, publish = _recovery_runtime(monkeypatch)
    runtime.observe()
    publish(1.08, joined=True)
    recovered, _, _ = runtime.observe()
    prior = runtime.last_fused_target
    original_valid = runtime.cameras.packet["valid"].copy()
    runtime.data.time = 1.201
    stale, _, _ = runtime.observe()
    assert stale["detections"] == []
    assert runtime.diagnostics["near_ball_recovery"]["reason"] == "no_current_tof"
    assert runtime.last_fused_target is prior
    np.testing.assert_array_equal(runtime.cameras.packet["valid"], original_valid)

    actions = []

    def execute(skill, seconds):
        actions.append((skill, seconds))
        runtime.data.time += seconds
        return True

    runtime.execute = execute
    servo = SensorServo(runtime, kick_strategy="foot_relative")
    servo.last_target = recovered["detections"][0]["center_body_m"]
    with pytest.raises(RuntimeError, match="stationary reacquisition"):
        servo.observe()
    assert actions == [("stand", 0.2)] * 9
    assert runtime.last_fused_target is prior
    assert runtime.last_fused_target[0] == 1.08


def test_frames_without_current_tof_do_not_keep_prior_alive(configured, monkeypatch):
    runtime, publish = _recovery_runtime(monkeypatch)
    runtime.observe()
    prior = runtime.last_fused_target
    for timestamp in (1.2, 1.4, 1.6):
        publish(timestamp, joined=True, valid=False)
        observation, _, _ = runtime.observe()
        assert observation["detections"] == []
        assert runtime.diagnostics["near_ball_recovery"]["reason"] == "no_current_tof"
        assert runtime.last_fused_target is prior
    publish(1.81, joined=True)
    expired, _, _ = runtime.observe()
    assert expired["detections"] == []
    assert runtime.diagnostics["near_ball_recovery"]["reason"] == "prior_expired_or_invalid_age"
    assert runtime.last_fused_target is prior


def test_fresh_complete_silhouette_reacquires_after_track_expiry(configured, monkeypatch):
    runtime, publish = _recovery_runtime(monkeypatch)
    runtime.observe()
    publish(2.0)
    observation, _, _ = runtime.observe()
    assert len(observation["detections"]) == 1
    assert "near_ball_recovery" not in runtime.diagnostics
    assert runtime.last_fused_target[0] == 2.0


def test_reset_forgets_prior_and_cached_encoder_poses(configured, monkeypatch):
    from .runtime import Runtime

    runtime, _ = _recovery_runtime(monkeypatch)
    runtime.observe()
    runtime.estimator = SimpleNamespace(reset=lambda: None)
    monkeypatch.setattr(Runtime, "reset", lambda self, layout: None)
    runtime.reset({})
    assert runtime.last_fused_target is None
    assert runtime.packet is None
    assert runtime.camera_pose is None
    assert runtime.foot_poses is None
    assert runtime.cameras.last is None


@pytest.mark.parametrize("invalid", ["future", "packet_nan", "packet_inf", "clock_nan", "rgb_nan", "rgb_mismatch"])
def test_invalid_sensor_times_cannot_refresh_prior_or_drive_head(configured, monkeypatch, invalid):
    runtime, publish = _recovery_runtime(monkeypatch)
    runtime.observe()
    prior = runtime.last_fused_target
    publish(1.08)
    runtime.head_tracking = True
    packet = runtime.cameras.packet
    if invalid == "future":
        runtime.data.time = 1.07
    elif invalid == "packet_nan":
        packet["time_s"] = packet["rgb_time_s"] = np.nan
    elif invalid == "packet_inf":
        packet["time_s"] = packet["rgb_time_s"] = np.inf
    elif invalid == "clock_nan":
        runtime.data.time = np.nan
    elif invalid == "rgb_nan":
        packet["rgb_time_s"] = np.nan
    else:
        packet["rgb_time_s"] = 1.07
    head = runtime.policy.head_command.copy()
    observation, _, _ = runtime.observe()
    assert observation["detections"] == []
    assert runtime.last_fused_target is prior
    np.testing.assert_array_equal(runtime.policy.head_command, head)
    assert packet["valid"].all()
