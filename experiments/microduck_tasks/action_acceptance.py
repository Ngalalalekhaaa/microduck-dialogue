"""Simulation-only outcome checks; these facts are never policy observations."""

import numpy as np
from microduck_deploy.policy import JOINT_NAMES


def _longest(mask):
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best * 0.02


def evaluate_action(runtime, skill, start_index):
    rows = runtime.rows[start_index:]
    metrics = runtime.metrics[start_index:]
    active = [m for m in metrics if m["skill"] == skill]
    if not rows or not active:
        return {"success": False, "reason": "no_executed_action"}
    recovered = all(m["tilt_deg"] < 15 and m["trunk_z"] > 0.09 for m in metrics[-50:])
    delta = rows[-1][7][:2] - rows[0][7][:2]
    yaw = rows[0][5]
    forward, lateral = np.dot(delta, [np.cos(yaw), np.sin(yaw)]), np.dot(delta, [-np.sin(yaw), np.cos(yaw)])
    success = recovered
    if skill in ("sit_stand", "official_sit"):
        success &= min(m["trunk_z"] for m in active) < 0.075
    elif skill in ("ground_pick", "official_pick"):
        success &= min(m["pocket_xyz"][2] for m in active) < 0.06
    elif skill in ("roll", "official_roll"):
        success &= max(m["tilt_deg"] for m in active) > 150
    elif skill.startswith("flamingo_"):
        foot = 0 if skill.endswith("left") else 1
        success &= _longest(m["foot_z"][foot] > 0.04 and m["feet_on_floor"][1 - foot] for m in active) >= 0.5
    elif skill in ("happy_hop", "hop"):
        success &= (
            _longest(not any(m["feet_on_floor"]) and min(m["foot_z"]) > 0.015 and m["tilt_deg"] < 30 for m in active)
            >= 0.04
        )
    elif skill in ("move_forward", "run", "official_walk", "velstand"):
        success &= forward > 0.1
    elif skill in ("move_backward", "moonwalk"):
        success &= forward < -0.1
    elif skill == "move_left":
        success &= lateral > 0.1
    elif skill == "move_right":
        success &= lateral < -0.1
    elif skill in ("turn_left", "turn_right"):
        change = np.unwrap([r[5] for r in rows])[-1] - rows[0][5]
        success &= change > 0.3 if skill == "turn_left" else change < -0.3
    elif skill in ("look_around", "nod"):
        joint = "head_yaw" if skill == "look_around" else "head_pitch"
        index = runtime.qi[JOINT_NAMES.index(joint)]
        values = [r[1][index] for r in rows if r[3] == skill]
        success &= np.ptp(values) > (0.3 if skill == "look_around" else 0.15)
    elif skill == "body_sway":
        success &= max(m["tilt_deg"] for m in active) > 3
    elif skill == "open_beak":
        success &= metrics[-1]["mouth_rad"] > 0.45
    elif skill == "close_beak":
        success &= metrics[-1]["mouth_rad"] < 0.1
    elif skill == "pirouette":
        success &= abs(np.unwrap([r[5] for r in rows])[-1] - rows[0][5]) > np.pi
    elif skill in ("groove", "caught_dancing"):
        joint = "left_knee" if skill == "groove" else "head_yaw"
        index = runtime.qi[JOINT_NAMES.index(joint)]
        values = [r[1][index] for r in rows if r[3] == skill]
        success &= np.ptp(values) > 0.15
    elif skill == "bow":
        success &= active[0]["pocket_xyz"][2] - min(m["pocket_xyz"][2] for m in active) > 0.02
    elif skill == "headstand":
        success = False  # This isolated fold stage does not implement the full contact-based routine.
    elif skill.startswith("kick_"):
        # Empty kicks are motion previews. Target kick tasks use evaluate_kick separately.
        success &= any(r[3] == skill for r in rows)
    return dict(
        success=bool(success),
        recovered=bool(recovered),
        forward_m=float(forward),
        lateral_m=float(lateral),
        min_trunk_z_m=min(m["trunk_z"] for m in active),
        max_tilt_deg=max(m["tilt_deg"] for m in active),
        reason="verified_motion" if success else "motion_semantics_or_recovery_not_verified",
    )
