"""Replay recorded kick states from two external cameras without resimulating."""

import argparse
import json
import os
import select
import shutil
import subprocess
import tempfile
from pathlib import Path

if __name__ == "__main__":
    from bootstrap import configure

    configure()

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from strands_robots._gl import gl_thread  # noqa: E402


class _ReplayRenderer:
    @gl_thread
    def __init__(self, scene_path: Path, robot_start: np.ndarray, ball_xyz: np.ndarray):
        import mujoco

        self.mujoco = mujoco
        self.renderer = None
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.model.vis.global_.offwidth = max(640, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(480, self.model.vis.global_.offheight)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        bounds = np.vstack((ball_xyz, robot_start + [-0.15, -0.15, -0.15], robot_start + [0.20, 0.15, 0.25]))
        lower, upper = bounds.min(axis=0), bounds.max(axis=0)
        overview_center = 0.5 * (lower + upper)
        radius = 0.5 * float(np.linalg.norm(upper - lower)) + 0.04
        half_fov = np.deg2rad(float(self.model.vis.global_.fovy) * 0.5)
        overview_distance = max(0.8, radius / np.sin(half_fov) * 1.10)
        self.cameras = []
        for azimuth, elevation, lookat, camera_distance in (
            (90, -18, [robot_start[0] + 0.10, robot_start[1], 0.12], 0.70),
            (
                140,
                -43,
                overview_center,
                overview_distance,
            ),
        ):
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth, camera.elevation = azimuth, elevation
            camera.lookat[:] = lookat
            camera.distance = camera_distance
            self.cameras.append(camera)
        try:
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        except BaseException:
            self.close()
            raise

    @gl_thread
    def frame(self, qpos: np.ndarray, qvel: np.ndarray, time_s: float):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = float(time_s)
        self.mujoco.mj_forward(self.model, self.data)
        frames = []
        for camera in self.cameras:
            self.renderer.update_scene(self.data, camera=camera, scene_option=self.option)
            frames.append(self.renderer.render().copy())
        return frames

    @gl_thread
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _compose(views, time_s, skill, tilt, ball_forward, result, font):
    frame = Image.new("RGB", (1280, 576), (18, 24, 32))
    for index, view in enumerate(views):
        frame.paste(Image.fromarray(view), (index * 640, 48))
    draw = ImageDraw.Draw(frame)
    foot = str(result["foot"])
    variant = str(result.get("variant", "HD1910"))
    draw.text((12, 6), f"HD1910 simulation | official {foot} kick | {variant}", font=font, fill="white")
    draw.text((12, 28), "Side view (fixed) | saved physics states", font=font, fill=(157, 221, 245))
    draw.text((652, 6), "Oblique view (fixed) | no autonomous visual approach", font=font, fill="white")
    draw.text((652, 28), f"Outcome: {result['status']} | success={result['success']}", font=font, fill=(157, 221, 245))
    draw.text((12, 536), f"t={time_s:5.2f}s | phase: {skill} | tilt={tilt:.1f} deg", font=font, fill="white")
    draw.text(
        (652, 536),
        f"Ball dx={ball_forward:+.3f} m | commanded kick={float(result['kick_seconds']):.2f}s",
        font=font,
        fill="white",
    )
    return frame


def _write_frame(process, frame):
    descriptor = process.stdin.fileno()
    remaining = memoryview(frame.tobytes())
    while remaining:
        if process.poll() is not None:
            raise RuntimeError(f"Video encoder exited early with status {process.returncode}.")
        _, writable, _ = select.select([], [descriptor], [], 30)
        if not writable:
            raise TimeoutError("Video encoder stopped accepting frames for 30 seconds.")
        try:
            written = os.write(descriptor, remaining[: 1024 * 1024])
        except BlockingIOError:
            continue
        if not written:
            raise RuntimeError("Video encoder closed its input before the replay finished.")
        remaining = remaining[written:]


def render_case(scene_path: Path, case_dir: Path, output: Path) -> None:
    """Write a 25 fps MP4 and JPG contact sheet from a saved 50 Hz trajectory.

    The ball free joint is replayed from qpos, never repositioned or simulated.
    The close camera stays near the kick; the fixed overview covers the complete
    recorded ball trajectory. Failed trials retain their recorded outcome label.
    """
    scene_path, case_dir, output = Path(scene_path), Path(case_dir), Path(output)
    if output.exists() or output.with_suffix(".jpg").exists():
        raise FileExistsError(f"Choose a new replay output path: {output}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Install ffmpeg before encoding the kick replay.")
    result = json.loads((case_dir / "result.json").read_text())
    with np.load(case_dir / "trajectory.npz", allow_pickle=False) as archive:
        times, qpos, qvel, skills, tilts, balls = (
            archive[name].copy() for name in ("time", "qpos", "qvel", "skill", "tilt_deg", "ball_xyz")
        )
    count = len(times)
    if not count or times.shape != (count,) or skills.shape != (count,) or tilts.shape != (count,):
        raise ValueError("Replay requires a nonempty trajectory with one time, skill, and tilt per state.")
    if qpos.ndim != 2 or qvel.ndim != 2 or len(qpos) != count or len(qvel) != count or balls.shape != (count, 3):
        raise ValueError("Replay qpos, qvel, and ball_xyz must have one row for each recorded state.")
    if any(not np.isfinite(values).all() for values in (times, qpos, qvel, tilts, balls)):
        raise ValueError("Replay contains nonfinite states; inspect the simulation result before rendering.")
    if count > 1 and not np.allclose(np.diff(times), 0.02, atol=1e-5, rtol=0):
        raise ValueError("Replay requires 50 Hz states for real-time playback at 25 fps.")
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = _ReplayRenderer(scene_path, qpos[0, :3], balls)
    process = None
    completed = False
    try:
        if qpos.shape[1] != renderer.model.nq or qvel.shape[1] != renderer.model.nv:
            raise ValueError("Recorded qpos/qvel do not match the scene; use the scene from this experiment.")
        indices = list(range(0, count, 2))
        samples = set(np.linspace(0, len(indices) - 1, min(8, len(indices)), dtype=int).tolist())
        kick_indices = np.flatnonzero(np.char.startswith(skills.astype(str), "kick"))
        if len(kick_indices):
            kick_start, kick_end = float(times[kick_indices[0]]) - 0.02, float(times[kick_indices[-1]])
            strike = result.get("first_strike_contact_s")
            strike = float(strike) if strike is not None else kick_start + 0.10
            desired_times = (
                times[0],
                kick_start - 0.12,
                strike,
                strike + 0.12,
                kick_end,
                kick_end + 0.4,
                (kick_end + times[-1]) * 0.5,
                times[-1],
            )
            samples = {int(np.argmin(np.abs(times[indices] - desired))) for desired in desired_times}
        contact_frames = []
        font = _font(18)
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
                    "25",
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
                views = renderer.frame(qpos[index], qvel[index], times[index])
                frame = _compose(
                    views,
                    float(times[index]),
                    str(skills[index]),
                    float(tilts[index]),
                    float(balls[index, 0] - balls[0, 0]),
                    result,
                    font,
                )
                _write_frame(process, frame)
                if frame_index in samples:
                    contact_frames.append(frame.resize((640, 288), Image.Resampling.LANCZOS))
            process.stdin.close()
            returncode = process.wait(timeout=60)
            if returncode:
                error_log.seek(0)
                error = error_log.read(8192).decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg failed with status {returncode}: {error}")
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
                output.with_suffix(".jpg").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render_case(args.scene, args.case, args.out)


if __name__ == "__main__":
    main()
