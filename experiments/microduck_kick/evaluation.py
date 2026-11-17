"""Explicit acceptance criteria for a fixed-ball kick and its standing recovery."""

import numpy as np


def assess(
    arrays,
    contacts,
    *,
    foot,
    kick_seconds,
    warmup_seconds,
    recovery_seconds,
    lag_seconds=0.02,
    failure=None,
    control=False,
):
    t = arrays["time"]
    start_index = int(np.argmin(abs(t - warmup_seconds)))
    started = t[-1] >= warmup_seconds - 1e-8
    start_ball = arrays["ball_xyz"][start_index]
    pre = t <= warmup_seconds + 1e-8
    drift = float(np.max(np.linalg.norm(arrays["ball_xyz"][pre, :2] - arrays["ball_xyz"][0, :2], axis=1)))
    pre_contacts = [c for c in contacts if c["time_s"] < warmup_seconds - 1e-8]
    strike_end = warmup_seconds + kick_seconds + lag_seconds
    strike_contacts = [
        c
        for c in contacts
        if warmup_seconds - 1e-8 <= c["time_s"] <= strike_end + 1e-8 and c["geom"] == f"{foot}_foot_collision"
    ]
    later_contacts = [c for c in contacts if c["time_s"] > strike_end + 1e-8]
    other_contacts = [
        c for c in contacts if c["time_s"] >= warmup_seconds - 1e-8 and c["geom"] != f"{foot}_foot_collision"
    ]
    # Scene heading is initially +X. Keep this direction fixed rather than rewarding turning.
    forward = float(arrays["ball_xyz"][-1, 0] - start_ball[0])
    lateral = float(arrays["ball_xyz"][-1, 1] - start_ball[1])
    strike_index = int(np.flatnonzero(t <= strike_end + 1e-8)[-1])
    strike_forward = float(arrays["ball_xyz"][strike_index, 0] - start_ball[0])
    moved_during_strike = bool(strike_forward >= 0.03)
    maximum_drift = float(np.max(np.linalg.norm(arrays["ball_xyz"][:, :2] - start_ball[:2], axis=1)))
    end = warmup_seconds + kick_seconds + recovery_seconds
    final = t >= end - 0.5 - 1e-8
    full = bool(t[-1] >= end - 1e-8 and np.count_nonzero(final) >= 25)
    speeds = np.linalg.norm(arrays["trunk_velocity"][:, :2], axis=1)
    stable = bool(
        full
        and np.max(arrays["tilt_deg"][final]) < 12
        and np.min(arrays["trunk_xyz"][final, 2]) > 0.09
        and np.max(speeds[final]) < 0.05
        and np.max(np.linalg.norm(arrays["gyro"][final], axis=1)) < 0.4
    )
    clean = bool(started and drift <= 0.01 and not pre_contacts)
    hit = bool(strike_contacts) and not control
    attribution_clean = not later_contacts and not other_contacts
    success = bool(
        not control
        and clean
        and failure is None
        and hit
        and moved_during_strike
        and attribution_clean
        and forward >= 0.10
        and stable
    )
    if failure:
        status = failure
    elif not clean:
        status = "setup_contaminated"
    elif control:
        status = "control_stable" if stable and maximum_drift < 0.01 and not contacts else "control_failed"
    elif success:
        status = "kick_success"
    elif not hit:
        status = "missed_ball"
    elif not attribution_clean:
        status = "other_or_late_robot_contact"
    elif not moved_during_strike:
        status = "no_forward_launch_during_kick"
    elif forward < 0.10:
        status = "insufficient_forward_displacement"
    else:
        status = "unstable_recovery"
    return {
        "status": status,
        "success": success,
        "control": control,
        "setup_clean": clean,
        "pre_kick_ball_drift_m": drift,
        "pre_kick_robot_contacts": len(pre_contacts),
        "ball_at_kick_start_world_m": start_ball.tolist(),
        "ball_final_world_m": arrays["ball_xyz"][-1].tolist(),
        "ball_forward_m": forward,
        "ball_lateral_m": lateral,
        "designated_foot_hit": hit,
        "ball_forward_at_strike_end_m": strike_forward,
        "maximum_ball_xy_drift_m": maximum_drift,
        "strike_window_end_s": strike_end,
        "post_strike_robot_contacts": len(later_contacts),
        "non_designated_foot_contacts": len(other_contacts),
        "attribution_clean": attribution_clean,
        "strike_contact_samples": len(strike_contacts),
        "first_strike_contact_s": strike_contacts[0]["time_s"] if strike_contacts else None,
        "strike_normal_impulse_ns": float(sum(c["normal_impulse_ns"] for c in strike_contacts)),
        "recovered_stable": stable,
        "completed_seconds": float(t[-1]),
        "max_tilt_deg": float(np.max(arrays["tilt_deg"])),
        "min_trunk_height_m": float(np.min(arrays["trunk_xyz"][:, 2])),
        "final_tilt_deg": float(arrays["tilt_deg"][-1]),
        "final_trunk_height_m": float(arrays["trunk_xyz"][-1, 2]),
        "final_xy_speed_m_s": float(speeds[-1]),
    }
