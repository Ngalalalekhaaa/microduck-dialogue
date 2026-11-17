"""Replay saved joint states into a two-camera video without moving the live data."""

import os
import select
import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from strands_robots._gl import gl_thread


class _ReplayRenderer:
    @gl_thread
    def __init__(self, runtime, result):
        self.model = runtime.model
        self.body, self.marker = runtime.body, runtime.marker
        self.data = mujoco.MjData(self.model)
        self.original_position = self.model.geom_pos[self.marker].copy()
        self.original_colour = self.model.geom_rgba[self.marker].copy()
        self.original_size = (self.model.vis.global_.offwidth, self.model.vis.global_.offheight)
        self.renderer = None
        try:
            self.model.geom_pos[self.marker] = [*result["target_world_xy_m"], 0.003]
            self.model.geom_rgba[self.marker] = (
                [0.15, 0.15, 0.15, 1] if result.get("target_hidden", False) else [1, 0.015, 0.015, 1]
            )
            self.model.vis.global_.offwidth = max(640, self.original_size[0])
            self.model.vis.global_.offheight = max(480, self.original_size[1])
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)
            self.option = mujoco.MjvOption()
            self.option.geomgroup[3:] = 0
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.azimuth = 135
            self.camera.elevation = -25
            self.camera.distance = 0.75
        except BaseException:
            self.close()
            raise

    @gl_thread
    def frame(self, qpos, qvel, time_s):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = float(time_s)
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.data.xpos[self.body] + [0.08, 0, 0.015]
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        side = self.renderer.render().copy()
        self.renderer.update_scene(self.data, camera="navigation", scene_option=self.option)
        first_person = self.renderer.render().copy()
        return side, first_person

    @gl_thread
    def close(self):
        try:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
        finally:
            self.model.geom_pos[self.marker] = self.original_position
            self.model.geom_rgba[self.marker] = self.original_colour
            self.model.vis.global_.offwidth, self.model.vis.global_.offheight = self.original_size


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _compose(side, first_person, time_s, skill, tilt, status, font):
    frame = Image.new("RGB", (1280, 576), (18, 24, 32))
    frame.paste(Image.fromarray(side), (0, 48))
    frame.paste(Image.fromarray(first_person), (640, 48))
    draw = ImageDraw.Draw(frame)
    draw.text((16, 12), "Microduck | side view", font=font, fill="white")
    draw.text((656, 12), f"Navigation camera | outcome: {status}", font=font, fill="white")
    draw.text((16, 540), f"t={time_s:5.2f}s | {skill} | tilt={tilt:.1f} deg", font=font, fill="white")
    draw.text((656, 540), "RGB-D + rule planner; no VLM / no hardware", font=font, fill=(157, 221, 245))
    return frame


def _write_frame(process, frame):
    """Use a bounded writable-pipe wait so a stuck encoder cannot hang a replay."""
    descriptor = process.stdin.fileno()
    remaining = memoryview(frame.tobytes())
    while remaining:
        if process.poll() is not None:
            raise RuntimeError(f"Video encoder exited early with status {process.returncode}.")
        _, writable, _ = select.select([], [descriptor], [], 30)
        if not writable:
            raise TimeoutError("Video encoder stopped accepting frames for 30 seconds.")
        try:
            count = os.write(descriptor, remaining[: 1024 * 1024])
        except BlockingIOError:
            continue
        if count == 0:
            raise RuntimeError("Video encoder closed its input before the replay finished.")
        remaining = remaining[count:]


def render_video(runtime, result: dict, output: Path) -> None:
    """Create a 10 fps real-time MP4, contact sheet, and first/last onboard frames.

    Only saved 50 Hz trajectories are replayed. ``runtime.data`` is never read or
    modified; target coordinates restore the recorded scene for visualization.
    All rendering and temporary model changes run on the Strands GL thread.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new video output path; this one already exists: {output}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Install ffmpeg and rerun with --video to encode the replay.")
    with np.load(Path(result["output"]) / "trajectory.npz", allow_pickle=False) as archive:
        time_s, qpos, qvel = (archive[name].copy() for name in ("time", "qpos", "qvel"))
        skills, tilts = (archive[name].copy() for name in ("skill", "tilt_deg"))
    count = len(time_s)
    if count == 0 or qpos.shape != (count, runtime.model.nq) or qvel.shape != (count, runtime.model.nv):
        raise ValueError("Replay requires a nonempty trajectory with qpos/qvel matching this MuJoCo model.")
    if time_s.shape != (count,) or skills.shape != (count,) or tilts.shape != (count,):
        raise ValueError("Replay time, skill, and tilt arrays must have one value for every recorded state.")
    if any(not np.isfinite(values).all() for values in (time_s, qpos, qvel, tilts)):
        raise ValueError("Replay trajectory contains nonfinite states; inspect the simulation result first.")
    if count > 1 and not np.allclose(np.diff(time_s), 0.02, atol=1e-5, rtol=0):
        raise ValueError("Replay expects a 50 Hz trajectory to preserve real-time playback at 10 fps.")
    indices = list(range(0, count, 5))
    contact_indices = set(np.linspace(0, len(indices) - 1, min(8, len(indices)), dtype=int).tolist())
    contact_frames = []
    font = _font(20)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = _ReplayRenderer(runtime, result)
    process = None
    completed = False
    try:
        with tempfile.TemporaryFile() as error_log:
            process = subprocess.Popen(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    "1280x576",
                    "-framerate",
                    "10",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=error_log,
                bufsize=0,
            )
            os.set_blocking(process.stdin.fileno(), False)
            for frame_index, index in enumerate(indices):
                side, first_person = renderer.frame(qpos[index], qvel[index], time_s[index])
                if frame_index == 0:
                    Image.fromarray(first_person).save(output.parent / "first_person_start.png")
                frame = _compose(
                    side,
                    first_person,
                    time_s[index],
                    str(skills[index]),
                    float(tilts[index]),
                    str(result["status"]),
                    font,
                )
                _write_frame(process, frame)
                if frame_index in contact_indices:
                    contact_frames.append(frame.resize((640, 288), Image.Resampling.LANCZOS))
            process.stdin.close()
            returncode = process.wait(timeout=60)
            if returncode:
                error_log.seek(0)
                error = error_log.read(8192).decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg failed with status {returncode}: {error}")
        _, first_person = renderer.frame(qpos[-1], qvel[-1], time_s[-1])
        Image.fromarray(first_person).save(output.parent / "first_person_end.png")
        contact = Image.new("RGB", (1280, 288 * ((len(contact_frames) + 1) // 2)), (18, 24, 32))
        for index, frame in enumerate(contact_frames):
            contact.paste(frame, ((index % 2) * 640, (index // 2) * 288))
        contact.save(output.with_suffix(".jpg"), quality=92)
        completed = True
    finally:
        try:
            if process is not None:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
        finally:
            renderer.close()
            if not completed:
                output.unlink(missing_ok=True)
