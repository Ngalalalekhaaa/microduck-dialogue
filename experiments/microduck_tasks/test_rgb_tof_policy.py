"""Exercise head command slots with a stub ONNX session and no simulator."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def policy():
    from .bootstrap import configure

    try:
        configure()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    pytest.importorskip("onnxruntime")
    from .motion import HeadingHold
    from .rgb_tof_policy import SensorPolicy

    class Session:
        def run(self, outputs, inputs):
            self.inputs = inputs["obs"].copy()
            return [np.full((1, 14), 0.01, dtype=np.float32)]

    instance = object.__new__(SensorPolicy)
    instance.motion = HeadingHold(enabled=False)
    instance.sessions = {name: Session() for name in ("walk", "kick_left", "kick_right")}
    instance.reset()
    return instance


def test_walk_head_commands_and_kick_zero_head_contract(policy):
    from microduck_deploy.policy import HOME, JOINT_NAMES

    # Only sensor-supported fields exist. Access to position or body velocity fails.
    obs = SimpleNamespace(
        pose=SimpleNamespace(orientation=np.array([1, 0, 0, 0]), angular_velocity=np.zeros(3)),
        joints=SimpleNamespace(names=JOINT_NAMES, positions=HOME.copy(), velocities=np.zeros(14)),
    )
    policy.head_command[1] = 0.4
    policy.skill = "move_forward"
    targets = policy.act(obs)
    vector = policy.sessions["walk"].inputs[0]
    assert vector.shape == (61,)
    assert vector[3:6] == pytest.approx([0, 0, -1])
    assert vector[48:51] == pytest.approx([0.2, 0, 0])
    assert vector[51:55] == pytest.approx([0, 0.4, 0, 0])
    assert [targets[name] for name in JOINT_NAMES] == pytest.approx(HOME + 0.01)
    for foot in ("left", "right"):
        policy.skill = f"kick_{foot}"
        policy.act(obs)
        assert policy.sessions[policy.skill].inputs[0, 51:61] == pytest.approx(np.zeros(10))
    policy.reset()
    assert policy.head_command == pytest.approx(np.zeros(4))
