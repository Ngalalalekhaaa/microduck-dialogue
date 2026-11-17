"""A provisional face sensor bracket; dimensions are not real-robot calibration."""

import math

import mujoco
import numpy as np
from microduck_deploy.policy import HOME, JOINT_NAMES

from .scene import build_scene


def build_sensor_scene(path, *, fovy_deg=100.0, pitch_deg=55.0):
    build_scene(path)
    spec = mujoco.MjSpec.from_file(str(path))
    model = spec.compile()
    reference = mujoco.MjData(model)
    root = model.joint("trunk_base_freejoint").qposadr[0]
    reference.qpos[root : root + 7] = [0, 0, 0, 1, 0, 0, 0]
    for name, value in zip(JOINT_NAMES, HOME, strict=True):
        reference.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, reference)
    head = model.body("jaw_soft").id
    rotation = reference.xmat[head].reshape(3, 3)
    position_body = np.array([0.14, 0.0, 0.16])
    pitch = math.radians(pitch_deg)
    right = np.array([0.0, -1.0, 0.0])
    up = np.array([math.sin(pitch), 0.0, math.cos(pitch)])
    spec.body("jaw_soft").add_camera(
        name="face_rgb",
        pos=rotation.T @ (position_body - reference.xpos[head]),
        xyaxes=np.concatenate((rotation.T @ right, rotation.T @ up)),
        fovy=fovy_deg,
    )
    spec.compile()
    path.write_text(spec.to_xml())


def camera_from_encoders(model, joint_positions):
    """Forward kinematics with fixed identity base; world pose and ball states are absent."""
    data = mujoco.MjData(model)
    root = model.joint("trunk_base_freejoint").qposadr[0]
    data.qpos[root : root + 7] = [0, 0, 0, 1, 0, 0, 0]
    for name, value in zip(JOINT_NAMES, joint_positions, strict=True):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    camera = model.camera("face_rgb").id
    rotation = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    return data.cam_xpos[camera].copy(), rotation
