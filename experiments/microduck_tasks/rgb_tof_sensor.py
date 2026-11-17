"""Simulation sensor boundary: RGB plus one scalar radial return per ToF zone.

World geometry is used ONLY inside this sensor generator. No object identities,
winning subray indices, dense depth or body/world poses leave ``capture``.
The return is the nearest range cluster supported by >=2 of 25 zone subrays.
This is a geometric ToF approximation, not an ST optical/firmware simulation.
"""

import mujoco
import numpy as np

from strands_robots._gl import gl_thread

from .rgb_tof_geometry import zone_unit_rays


def zone_return(distances, *, minimum=0.04, maximum=2.0, cluster_width=0.012):
    values = np.sort(np.asarray(distances)[np.isfinite(distances) & (np.asarray(distances) >= minimum)])
    values = values[values <= maximum]
    for value in values:
        cluster = values[(values >= value) & (values <= value + cluster_width)]
        if len(cluster) >= 2:
            return float(np.median(cluster)), len(cluster) / len(distances)
    return float("nan"), 0.0


class RGBToFSensor:
    @gl_thread
    def __init__(self, model, config, *, seed=17, noise_m=0.002, dropout=0.0):
        self.model, self.config = model, config
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, width=config.width, height=config.height)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        self.camera = model.camera("face_rgb").id
        self.rng = np.random.default_rng(seed)
        self.noise_m, self.dropout = noise_m, dropout
        self.last = None
        self.rays = np.stack(
            [zone_unit_rays(config, row, column, samples=5) for row in range(8) for column in range(8)]
        )

    @gl_thread
    def capture(self, qpos, qvel, time_s):
        if self.last is not None and time_s - self.last["time_s"] < 1 / 15 - 1e-9:
            return self.last
        self.data.qpos[:], self.data.qvel[:], self.data.time = qpos, qvel, time_s
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="face_rgb", scene_option=self.option)
        rgb = self.renderer.render().copy()
        camera_rotation = self.data.cam_xmat[self.camera].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        translation = np.asarray(self.config.tof_translation_camera_m)
        rotation = np.asarray(self.config.tof_rotation_camera).reshape(3, 3)
        origin = self.data.cam_xpos[self.camera] + camera_rotation @ translation
        rays = (self.rays.reshape(-1, 3) @ rotation.T @ camera_rotation.T).copy()
        distances = np.full(len(rays), -1.0)
        # IDs are required by MuJoCo but are never examined or exported.
        ids = np.full(len(rays), -1, np.int32)
        mujoco.mj_multiRay(
            self.model,
            self.data,
            origin,
            rays.reshape(-1),
            self.option.geomgroup,
            True,
            -1,
            ids,
            distances,
            None,
            len(rays),
            2.0,
        )
        ranges, quality = zip(*(zone_return(row) for row in distances.reshape(64, -1)), strict=True)
        ranges, quality = np.array(ranges).reshape(8, 8), np.array(quality).reshape(8, 8)
        ranges += self.rng.normal(0, self.noise_m, (8, 8))
        valid = np.isfinite(ranges) & (ranges >= 0.04) & (ranges <= 2.0)
        valid &= self.rng.random((8, 8)) >= self.dropout
        ranges[~valid] = np.nan
        self.last = dict(time_s=time_s, ranges_m=ranges, valid=valid, quality=quality, rgb=rgb, rgb_time_s=time_s)
        return self.last

    @gl_thread
    def close(self):
        self.renderer.close()
