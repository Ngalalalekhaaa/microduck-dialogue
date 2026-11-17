"""Body attitude and settled-state estimates using only IMU and encoders.

The gyro is body-frame rad/s and accelerometer is body-frame specific force
(an upright stationary sensor reports +g on z). Yaw starts at zero and drifts:
gravity does not provide an absolute heading. ``stationary`` means the sensor
window looks settled; constant-speed sliding cannot be ruled out by an IMU.
This module never reads a simulator body transform, root velocity or contact.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

GRAVITY = 9.80665


def _vector(value, name, width=None):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not result.size or (width is not None and result.size != width):
        raise ValueError(f"{name} must be a vector of length {width or 'at least 1'}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array(
        (
            w * v - x * i - y * j - z * k,
            w * i + x * v + y * k - z * j,
            w * j - x * k + y * v + z * i,
            w * k + x * j - y * i + z * v,
        )
    )


def _gravity(q):
    w, x, y, z = q
    return np.array((2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1))


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _orientation(gravity, yaw):
    roll = math.atan2(-gravity[1], -gravity[2])
    pitch = math.atan2(gravity[0], math.hypot(gravity[1], gravity[2]))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


@dataclass(frozen=True)
class BodyEstimate:
    timestamp: float
    projected_gravity: np.ndarray
    orientation_wxyz: np.ndarray
    angular_velocity: np.ndarray
    joint_velocities: np.ndarray
    yaw: float
    tilt_deg: float
    stationary: bool
    confidence: float
    accel_correction_used: bool

    def is_fresh(self, now, max_age_s=0.1):
        if not math.isfinite(now) or not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Freshness needs a finite time and positive finite maximum age")
        return -1e-9 <= now - self.timestamp <= max_age_s


class BodyEstimator:
    """Complementary attitude filter and a trailing settled-state window.

    Acceleration corrections reject magnitudes far from 1 g; sustained linear
    acceleration near 1 g can still bias tilt. ``confidence`` is a deterministic
    quality flag (1/.5/.25), not a calibrated probability or a safety guarantee.
    Initialize while standing. Runtime samples must have strictly increasing
    timestamps; a missing interval raises rather than silently bridging it.
    """

    def __init__(
        self,
        *,
        complementary_tau_s=0.5,
        accel_rejection_fraction=0.2,
        max_sample_gap_s=0.1,
        settle_window_s=0.3,
        gyro_limit_rad_s=0.3,
        accel_std_limit_m_s2=0.8,
        joint_speed_limit_rad_s=0.5,
    ):
        values = (
            complementary_tau_s,
            max_sample_gap_s,
            settle_window_s,
            gyro_limit_rad_s,
            accel_std_limit_m_s2,
            joint_speed_limit_rad_s,
        )
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Estimator time constants and thresholds must be positive and finite")
        if not math.isfinite(accel_rejection_fraction) or not 0 < accel_rejection_fraction < 1:
            raise ValueError("Acceleration rejection fraction must be between 0 and 1")
        self.tau = complementary_tau_s
        self.accel_rejection = accel_rejection_fraction
        self.max_gap = max_sample_gap_s
        self.window = settle_window_s
        self.gyro_limit = gyro_limit_rad_s
        self.accel_std_limit = accel_std_limit_m_s2
        self.joint_speed_limit = joint_speed_limit_rad_s
        self.reset()

    def reset(self):
        self.state = None
        self._joints = None
        self._history = deque()

    def update(self, time_s, gyro_body, accel_body, joint_positions):
        if not math.isfinite(time_s):
            raise ValueError("IMU timestamp must be finite")
        gyro = _vector(gyro_body, "gyro_body", 3)
        accel = _vector(accel_body, "accel_body", 3)
        joints = _vector(joint_positions, "joint_positions")
        norm = float(np.linalg.norm(accel))
        accepted = abs(norm - GRAVITY) <= GRAVITY * self.accel_rejection
        if self.state is None:
            if not accepted:
                raise ValueError("Initialize IMU while standing with acceleration near 1 g")
            gravity = -accel / norm
            orientation = _orientation(gravity, 0.0)
            velocities = np.zeros_like(joints)
        else:
            dt = time_s - self.state.timestamp
            if dt <= 0 or dt > self.max_gap + 1e-9:
                raise ValueError(f"Invalid IMU sample interval {dt:.6f} s; restart while standing")
            if joints.shape != self._joints.shape:
                raise ValueError("Joint encoder count changed; reset the estimator")
            velocities = (joints - self._joints) / dt
            rate = float(np.linalg.norm(gyro))
            angle = rate * dt
            delta = np.r_[math.cos(angle / 2), gyro * (math.sin(angle / 2) / rate if rate else 0)]
            orientation = _multiply(self.state.orientation_wxyz, delta)
            orientation /= np.linalg.norm(orientation)
            gravity = _gravity(orientation)
            if accepted:
                weight = -math.expm1(-dt / self.tau)
                gravity = (1 - weight) * gravity - weight * accel / norm
                gravity /= np.linalg.norm(gravity)
                orientation = _orientation(gravity, _yaw(orientation))
        self._history.append((float(time_s), gyro, accel, velocities))
        cutoff = time_s - self.window
        while len(self._history) > 1 and self._history[1][0] <= cutoff + 1e-9:
            self._history.popleft()
        full = time_s - self._history[0][0] >= self.window - 1e-9
        gyros = np.array([row[1] for row in self._history])
        accels = np.array([row[2] for row in self._history])
        speeds = np.array([row[3] for row in self._history])
        stationary = bool(
            full
            and accepted
            and np.linalg.norm(gyros, axis=1).max() < self.gyro_limit
            and np.linalg.norm(accels.std(axis=0)) < self.accel_std_limit
            and np.sqrt(np.mean(speeds**2, axis=1)).max() < self.joint_speed_limit
        )
        self.state = BodyEstimate(
            timestamp=float(time_s),
            projected_gravity=gravity.copy(),
            orientation_wxyz=orientation.copy(),
            angular_velocity=gyro.copy(),
            joint_velocities=velocities.copy(),
            yaw=_yaw(orientation),
            tilt_deg=math.degrees(math.acos(float(np.clip(-gravity[2], -1, 1)))),
            stationary=stationary,
            confidence=(1.0 if full else 0.5) if accepted else 0.25,
            accel_correction_used=accepted,
        )
        self._joints = joints
        return self.state
