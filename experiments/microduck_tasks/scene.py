"""Reproducible physical balls, with explicit front and near-field RGB-D cameras."""

import math

import mujoco
import numpy as np

from .bootstrap import ASSETS

COLORS = {"red": [1, 0.01, 0.015, 1], "yellow": [1, 0.90, 0.01, 1], "blue": [0.01, 0.08, 1, 1]}
# Six-dimensional contacts activate rolling resistance. This is a simulation
# surface assumption, not a measured coefficient for the physical robot's floor.
BALL_FRICTION = [0.5, 0.005, 0.001]
CAMERAS = {
    "front": {"position": [0.14, 0, 0.16], "pitch_deg": 45, "fovy_deg": 85},
    "near": {"position": [0.17, 0, 0.12], "pitch_deg": 110, "fovy_deg": 90},
}


def sample_layout(seed, radius_min=0.3, radius_max=0.6, min_separation=0.20):
    if (
        not np.isfinite([radius_min, radius_max, min_separation]).all()
        or not 0.2 <= radius_min <= radius_max <= 2
        or min_separation < 0.08
    ):
        raise ValueError("Use 0.2 <= radius_min <= radius_max <= 2 and separation >=0.08 m.")
    rng = np.random.default_rng(seed)
    layout = {}
    for color in COLORS:
        for _ in range(1000):
            radius = rng.uniform(radius_min, radius_max)
            angle = rng.uniform(-np.pi, np.pi)
            point = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            if all(np.linalg.norm(point - previous) >= min_separation for previous in layout.values()):
                layout[color] = point.tolist()
                break
        else:
            raise ValueError("Could not sample separated balls; widen the radius range or reduce separation.")
    return layout


def build_scene(path):
    spec = mujoco.MjSpec.from_file(str(ASSETS / "scene_ball.xml"))
    spec.meshdir = str(ASSETS / "assets")
    spec.body("ball").name = "ball_red"
    spec.joint("ball_free").name = "ball_red_free"
    red = spec.geom("ball_geom")
    red.name, red.rgba = "ball_red_geom", COLORS["red"]
    red.condim, red.friction = 6, BALL_FRICTION
    for color in ("yellow", "blue"):
        body = spec.worldbody.add_body(
            name=f"ball_{color}",
            pos=[0.4, 0, 0.035],
            mass=0.015,
            ipos=[0, 0, 0],
            inertia=[1.225e-5] * 3,
            explicitinertial=True,
        )
        body.add_freejoint(name=f"ball_{color}_free")
        body.add_geom(
            name=f"ball_{color}_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.035, 0, 0],
            rgba=COLORS[color],
            friction=BALL_FRICTION,
            condim=6,
        )
    # A neutral floor keeps the blue ball distinct without privileged segmentation.
    texture = spec.texture("groundplane")
    texture.rgb1, texture.rgb2, texture.markrgb = [0.34, 0.34, 0.34], [0.22, 0.22, 0.22], [0.6, 0.6, 0.6]
    for name, calibration in CAMERAS.items():
        pitch = math.radians(calibration["pitch_deg"])
        spec.body("trunk_base").add_camera(
            name=name,
            pos=calibration["position"],
            xyaxes=[0, -1, 0, math.sin(pitch), 0, math.cos(pitch)],
            fovy=calibration["fovy_deg"],
        )
    spec.compile()
    path.write_text(spec.to_xml())
