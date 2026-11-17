"""Encoder-only foot-site kinematics, independent of live simulation state.

Foot frames are the model's ``left_foot``/``right_foot`` sites attached to the
ankle bodies, not the ankle joint origins or collision-mesh centroids. At HOME
both x axes point forward, both y axes point toward the robot's left, and both
z axes point upward, with opposite approximately five-degree foot rolls.
"""

from dataclasses import dataclass

import numpy as np

FOOT_SITES = {"left": "left_foot", "right": "right_foot"}


def _finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


@dataclass(frozen=True)
class FootPose:
    """Foot-site origin in trunk metres and rotation mapping foot axes to trunk."""

    foot: str
    position_body_m: np.ndarray
    rotation_body_from_foot: np.ndarray

    def __post_init__(self):
        if self.foot not in FOOT_SITES:
            raise ValueError("Choose foot 'left' or 'right'")
        position = _finite(self.position_body_m, (3,), "Foot position")
        rotation = _finite(self.rotation_body_from_foot, (3, 3), "Foot rotation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0) or not np.isclose(
            np.linalg.det(rotation), 1, atol=1e-8, rtol=0
        ):
            raise ValueError("Foot rotation must be a right-handed orthonormal matrix")
        position.setflags(write=False)
        rotation.setflags(write=False)
        object.__setattr__(self, "position_body_m", position)
        object.__setattr__(self, "rotation_body_from_foot", rotation)

    @property
    def site_name(self):
        return FOOT_SITES[self.foot]

    def to_foot(self, point_body_m):
        """Transform a sensor-estimated body-frame ball centre into foot axes."""
        point = _finite(point_body_m, (3,), "Body-frame point")
        return self.rotation_body_from_foot.T @ (point - self.position_body_m)

    def to_body(self, point_foot_m):
        point = _finite(point_foot_m, (3,), "Foot-frame point")
        return self.position_body_m + self.rotation_body_from_foot @ point


def feet_from_encoders(model, joint_positions):
    """Return left/right FootPose from exactly 14 absolute joint angles in rad.

    Ordering is ``microduck_deploy.policy.JOINT_NAMES``. A private MjData has its
    floating root fixed at identity and receives only these encoder readings.
    No live MjData, body/root truth, ball position, velocity or contact is read;
    the model is used only as the calibrated rigid-link kinematic description.
    Finite observed angles are not clipped to nominal actuator limits.
    """
    import mujoco
    from microduck_deploy.policy import JOINT_NAMES

    joints = _finite(joint_positions, (14,), "Joint encoders")
    if len(JOINT_NAMES) != 14:
        raise ValueError("Foot kinematics requires the inspected 14-joint encoder contract")
    try:
        root = model.joint("trunk_base_freejoint")
        trunk = model.body("trunk_base").id
        joint_ids = [model.joint(name).id for name in JOINT_NAMES]
        sites = {foot: model.site(site_name).id for foot, site_name in FOOT_SITES.items()}
        ankles = {foot: model.body(f"ankle_{foot}").id for foot in FOOT_SITES}
    except KeyError as error:
        raise ValueError("Restore the calibrated trunk, 14 joints and left/right foot sites") from error
    if int(root.type[0]) != int(mujoco.mjtJoint.mjJNT_FREE) or int(root.bodyid[0]) != trunk:
        raise ValueError("The inspected floating root must belong to trunk_base")
    if any(int(model.site_bodyid[sites[foot]]) != ankles[foot] for foot in FOOT_SITES):
        raise ValueError("Each foot site must belong to its corresponding ankle body")
    if any(int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_HINGE) for joint in joint_ids):
        raise ValueError("The 14 calibrated encoder joints must all be hinge joints")
    data = mujoco.MjData(model)
    root_address = int(root.qposadr[0])
    data.qpos[root_address : root_address + 7] = [0, 0, 0, 1, 0, 0, 0]
    data.qpos[model.jnt_qposadr[joint_ids]] = joints
    mujoco.mj_kinematics(model, data)
    return {
        foot: FootPose(foot, data.site_xpos[site], data.site_xmat[site].reshape(3, 3)) for foot, site in sites.items()
    }
