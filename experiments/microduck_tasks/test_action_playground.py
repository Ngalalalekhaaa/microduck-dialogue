"""Physical grasp evidence, policy-boundary checks, and registry isolation."""

import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def configured():
    from .bootstrap import configure

    try:
        configure()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    for module in ("mujoco", "onnxruntime", "bam"):
        pytest.importorskip(module)


@pytest.mark.parametrize("bad", ["no_contact", "floor_support", "far_from_beak", "too_short", "too_low"])
def test_hold_gate_rejects_false_grasps(configured, bad):
    from .action_objects import held

    row = dict(
        toy_xyz=[0, 0, 0.2], pocket_xyz=[0, 0, 0.2], grip_force_n=0.1, toy_contacts=["grip_upper_pad", "grip_lower_pad"]
    )
    rows = [copy.deepcopy(row) for _ in range(50)]
    runtime = SimpleNamespace(lock=threading.RLock(), metrics=rows)
    assert held(runtime)
    if bad == "no_contact":
        rows[25]["grip_force_n"] = 0
    elif bad == "floor_support":
        rows[25]["toy_contacts"].append("floor")
    elif bad == "far_from_beak":
        rows[25]["toy_xyz"][0] = 1
    elif bad == "too_short":
        rows.pop()
    else:
        rows[25]["toy_xyz"][2] = 0.015
    assert not held(runtime)


def test_scene_keeps_policy_joint_count_and_has_no_object_weld(configured, tmp_path):
    from .action_runtime import ActionRuntime

    runtime = ActionRuntime(tmp_path, toy_xy=(0.065, 0), support_height=0.014)
    try:
        model = runtime.model
        assert runtime.controller.q_target.shape == (14,)
        assert model.nu == 15
        assert model.jnt_type[model.joint("grasp_toy_free").id] == 0  # Free joint, no kinematic attachment.
        toy = model.body("grasp_toy").id
        assert toy not in model.eq_obj1id and toy not in model.eq_obj2id
        assert np.allclose(model.actuator_forcerange[runtime.mouth_act], [-0.15, 0.15])
        assert model.actuator_gear[runtime.mouth_act, 0] == 1
        assert not np.any(runtime.data.xfrc_applied) and not np.any(runtime.data.qfrc_applied)
    finally:
        runtime.close()


def test_pick_carry_place_uses_physics_and_lands_inside_mat(configured, tmp_path):
    from .action_objects import object_task
    from .action_runtime import ActionRuntime

    runtime = ActionRuntime(tmp_path, toy_xy=(0.065, 0), support_height=0.014)
    try:
        runtime.reset()
        success, reason = object_task(runtime, "pick_place")
        assert success and reason == "released_on_floor"
        rows = runtime.metrics
        assert max(m["toy_xyz"][2] for m in rows) > 0.18
        final = rows[-1]
        assert abs(final["toy_xyz"][0] - 0.32) < 0.088 and abs(final["toy_xyz"][1]) < 0.088
        assert final["toy_xyz"][2] < 0.017 and final["grip_force_n"] == 0
        assert not np.any(runtime.data.xfrc_applied) and not np.any(runtime.data.qfrc_applied)
        assert max(m["grip_penetration_m"] for m in rows) < 0.003
        assert np.max(np.abs(runtime.data.qvel)) < 1
    finally:
        runtime.close()


def test_new_models_refuse_nonfinite_outputs_before_clipping(configured):
    from microduck_deploy.policy import HOME, JOINT_NAMES

    from strands_robots import JointState, Observation, Pose

    from .action_policy import ActionPolicy

    policy = ActionPolicy()
    policy.start("look_around", 0)
    policy.limits = np.tile([-1, 1], (14, 1))
    policy.sessions["walk"] = SimpleNamespace(run=lambda *_: [np.full((1, 14), np.nan)])
    obs = Observation(
        0,
        JointState(JOINT_NAMES, HOME, np.zeros(14), np.zeros(14)),
        Pose([0, 0, 0.12], [1, 0, 0, 0], angular_velocity=np.zeros(3)),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        policy.act(obs)


def test_special_posture_allowance_never_leaks_to_navigation(configured):
    from .action_runtime import ActionRuntime

    runtime = ActionRuntime.__new__(ActionRuntime)
    runtime.qa = 0
    runtime.data = SimpleNamespace(qpos=np.array([0, 0, 0.08]), time=1)
    runtime.policy = SimpleNamespace(start_time=0)
    assert not runtime._posture_failed("roll", 170)
    assert runtime._posture_failed("move_forward", 170)
    runtime.data.time = 4
    assert runtime._posture_failed("roll", 170)


def test_timeline_policy_state_advances_and_restarts(configured):
    from microduck_deploy.policy import HOME, JOINT_NAMES

    from strands_robots import JointState, Observation, Pose

    from .action_policy import ActionPolicy

    policy = ActionPolicy()
    policy.start("caught_dancing", 0)
    inputs_seen = []

    def run(names, inputs):
        inputs_seen.append(inputs["h_in"].copy())
        return [np.zeros((1, 14), np.float32), inputs["h_in"] + 1, inputs["c_in"] + 1]

    policy.sessions["caught_dancing"] = SimpleNamespace(run=run)
    obs = Observation(
        0,
        JointState(JOINT_NAMES, HOME, np.zeros(14), np.zeros(14)),
        Pose([0, 0, 0.12], [1, 0, 0, 0], angular_velocity=np.zeros(3)),
    )
    policy.act(obs)
    policy.act(obs)
    policy.start("caught_dancing", 1)
    policy.act(obs)
    assert [float(x.item()) for x in inputs_seen] == [0, 1, 0]


def test_playground_recording_has_its_own_truthful_controller_label():
    from .recording import _controller_labels

    labels = _controller_labels({"backend": "action_playground"}, {})
    assert "Scripted" in labels[1]
    assert "VLM" not in labels[0] and "teacher" not in labels[0]
