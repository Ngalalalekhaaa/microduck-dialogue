"""Docked manipulation demonstrations with independent physical verification."""

import numpy as np

from .action_scene import MOUTH_OPEN, TOY_RADIUS

GRIP_ANGLE = 0.3


def held(runtime, seconds=1.0):
    """Require continuous beak support, clearance and proximity for a full window."""
    with runtime.lock:
        rows = list(runtime.metrics[-round(seconds / 0.02) :])
    if len(rows) < round(seconds / 0.02):
        return False
    return all(
        row["toy_xyz"][2] - TOY_RADIUS > 0.08
        and np.linalg.norm(np.subtract(row["toy_xyz"], row["pocket_xyz"])) < 0.035
        and row["grip_force_n"] > 0
        and row["toy_contacts"]
        and all(name.startswith("grip_") for name in row["toy_contacts"])
        for row in rows
    )


def pulse(runtime, skill, seconds):
    remaining = round(seconds / 0.02)
    while remaining:
        count = min(remaining, 150)
        if not runtime.execute(skill, count * 0.02):
            return False
        remaining -= count
    return True


def pickup(runtime):
    """Fixed docking pose: learned crouch + bounded jaw close, then measured hold."""
    with runtime.lock:
        runtime.mouth_target = MOUTH_OPEN
        runtime.policy.start("ground_pick", float(runtime.data.time))
    if not pulse(runtime, "ground_pick", 1.2):
        return False
    with runtime.lock:
        runtime.mouth_target = GRIP_ANGLE
    return pulse(runtime, "ground_pick", 2.8) and pulse(runtime, "stand", 2.0) and held(runtime)


def carry(runtime):
    if not held(runtime):
        return False
    return pulse(runtime, "move_forward", 1.6) and pulse(runtime, "stand", 2.0) and held(runtime)


def release(runtime, *, low=False):
    if not held(runtime):
        return False
    if low:
        with runtime.lock:
            runtime.policy.start("ground_pick", float(runtime.data.time))
        if not pulse(runtime, "ground_pick", 1.2):
            return False
    with runtime.lock:
        runtime.mouth_target = MOUTH_OPEN
    if low and not pulse(runtime, "ground_pick", 2.8):
        return False
    if not pulse(runtime, "stand", 2.0):
        return False
    row = runtime.metrics[-1]
    return row["toy_xyz"][2] < TOY_RADIUS + 0.005 and row["grip_force_n"] == 0


def throw(runtime):
    if not held(runtime):
        return False
    start = np.array(runtime.metrics[-1]["toy_xyz"])
    with runtime.lock:
        runtime.mouth_target = np.deg2rad(20)
        runtime.policy.start("beak_throw", float(runtime.data.time))
    if not pulse(runtime, "beak_throw", 0.72):
        return False
    with runtime.lock:
        runtime.mouth_target = MOUTH_OPEN
    if not pulse(runtime, "beak_throw", 1.68) or not pulse(runtime, "stand", 3.0):
        return False
    row = runtime.metrics[-1]
    return row["grip_force_n"] == 0 and row["toy_xyz"][0] - start[0] > 0.15 and row["toy_xyz"][2] < 0.02


def object_task(runtime, name):
    if name == "push_block":
        start = np.array(runtime.metrics[-1]["block_xyz"])
        first = len(runtime.metrics)
        completed = pulse(runtime, "move_forward", 1.6) and pulse(runtime, "stand", 2.0)
        end = np.array(runtime.metrics[-1]["block_xyz"])
        contacted = any(m["block_contact_n"] > 0 for m in runtime.metrics[first:])
        success = completed and contacted and end[0] - start[0] > 0.05
        return success, "block_pushed" if success else "push_not_physically_verified"
    if name in ("pick_up", "pick_place", "pick_throw"):
        if not pickup(runtime):
            return False, "pickup_not_physically_verified"
        if name == "pick_up":
            return True, "held_for_one_second"
    if name in ("carry", "pick_place"):
        if not carry(runtime):
            return False, "carry_not_physically_verified"
        if name == "carry":
            return True, "carried_and_held"
    if name in ("release", "place", "pick_place"):
        success = release(runtime, low=name != "release")
        if name == "pick_place" and success:
            x, y, _ = runtime.metrics[-1]["toy_xyz"]
            success = abs(x - 0.32) < 0.088 and abs(y) < 0.088
        return success, "released_on_floor" if success else "release_not_physically_verified"
    if name in ("throw", "pick_throw"):
        success = throw(runtime)
        return success, "released_and_thrown" if success else "throw_not_physically_verified"
    raise ValueError(f"Unknown object task {name}")
