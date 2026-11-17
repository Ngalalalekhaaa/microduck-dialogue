"""Render saved physics states with external, front and near camera views."""

import argparse
import json
import os
import select
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from strands_robots._gl import gl_thread


class _ReplayRenderer:
    @gl_thread
    def __init__(self, scene_path):
        import mujoco

        self.mujoco = mujoco
        self.side_renderer = self.camera_renderer = None
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.body = self.model.body("trunk_base").id
        for name in ("front", "near"):
            self.model.camera(name)
        self.model.vis.global_.offwidth = max(960, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(720, self.model.vis.global_.offheight)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth, self.camera.elevation, self.camera.distance = 135, -35, 1.35
        try:
            self.side_renderer = mujoco.Renderer(self.model, width=960, height=720)
            self.camera_renderer = mujoco.Renderer(self.model, width=320, height=240)
        except BaseException:
            self.close()
            raise

    @gl_thread
    def frame(self, qpos, qvel, time_s):
        self.data.qpos[:], self.data.qvel[:], self.data.time = qpos, qvel, float(time_s)
        self.mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.data.xpos[self.body] + [0.08, 0, 0.015]
        self.side_renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        views = [self.side_renderer.render().copy()]
        for name in ("front", "near"):
            self.camera_renderer.update_scene(self.data, camera=name, scene_option=self.option)
            views.append(self.camera_renderer.render().copy())
        return views

    @gl_thread
    def close(self):
        for name in ("side_renderer", "camera_renderer"):
            renderer = getattr(self, name, None)
            if renderer is not None:
                renderer.close()
                setattr(self, name, None)


@lru_cache(maxsize=4)
def _font(size):
    for name in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _events(path):
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        time_s = event.get("observation", {}).get("time_s", event.get("time_s"))
        if isinstance(time_s, (int, float)) and np.isfinite(time_s):
            events.append((float(time_s), event))
    return sorted(events, key=lambda item: item[0])


def _controller_labels(metadata, event):
    metadata = metadata or {}
    model = metadata.get("vlm") or metadata
    identity = (event.get("model_trace") or {}).get("server_metadata") or model.get("server_metadata") or {}
    source = event.get("controller_source")
    backend = metadata.get("backend", model.get("backend"))
    if backend == "action_playground" or source == "action_playground":
        return "Pinned action policies", "Scripted skill selection", "Physics demo / simulation only"
    if (
        identity.get("head_trained") is True
        or model.get("head_trained") is True
        or metadata.get("head_trained") is True
        or backend == "qwen35-frozen-skill-head"
        or source == "frozen_vlm_trained_skill_head"
    ):
        return (
            "Frozen VLM + trained skill head",
            "Backbone frozen / skill head trained",
            "Local inference / simulation only",
        )
    if (
        identity.get("robot_finetuned") is True
        or model.get("robot_finetuned") is True
        or metadata.get("robot_finetuned") is True
        or backend == "local_finetuned_vlm"
        or source == "finetuned_vlm"
    ):
        adapter = identity.get("adapter")
        detail = f"Adapter: {Path(adapter).name}" if isinstance(adapter, str) and adapter else "Local VLM"
        title = "ONNX QLoRA VLM" if identity.get("inference_format") == "onnx" else "QLoRA VLM"
        return title, detail, "Local inference / simulation only"
    if backend == "local_pretrained_vlm" or source == "pretrained_vlm":
        return "Local pretrained VLM", "", "Local inference / simulation only"
    if backend == "local_vlm" or model.get("is_trained_vlm") is True:
        return "Local VLM", "", "Local inference / simulation only"
    return "RGB-D rule teacher", "", "Rule control / simulation only"


def _replay_metadata(run_dir):
    for name in ("summary.json", "episode.json"):
        path = run_dir / name
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8")).get("metadata")
            if isinstance(metadata, dict):
                return metadata
    return {}


def _compose(views, time_s, skill, tilt, event, font, *, live=False, model_metadata=None):
    canvas = Image.new("RGB", (1280, 816), (18, 24, 32))
    for view, offset in zip(views, ((0, 48), (960, 48), (960, 336)), strict=True):
        canvas.paste(Image.fromarray(view), offset)
    draw = ImageDraw.Draw(canvas)
    source = "LIVE" if live else "recorded physics"
    draw.text((12, 9), f"HD1910 simulation | external camera follows robot | {source}", font=font, fill="white")
    draw.text((972, 9), "Front RGB camera", font=font, fill="white")
    draw.text((972, 300), "Near RGB camera", font=font, fill="white")
    draw.text((972, 594), f"Time: {time_s:.2f} s", font=font, fill="white")
    draw.text((972, 626), f"Action: {skill}", font=font, fill="white")
    draw.text((972, 658), f"Body tilt: {tilt:.1f} deg", font=font, fill="white")
    title, detail, subtitle = _controller_labels(model_metadata, event)
    draw.text((972, 704), title, font=font, fill=(157, 221, 245))
    while draw.textlength(detail, font=_font(14)) > 296 and len(detail) > 12:
        detail = detail[:-4] + "..."
    draw.text((972, 732), detail, font=_font(14), fill=(157, 221, 245))
    draw.text((972, 752), subtitle, font=_font(14), fill=(157, 221, 245))
    instruction = str(event.get("instruction", "Saved trajectory replay"))
    label = f"Instruction: {instruction}"
    while draw.textlength(label, font=font) > 1240 and len(label) > 20:
        label = label[:-4] + "..."
    draw.text((12, 778), label, font=font, fill="white")
    return canvas


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


def render_video(run_dir, output_path=None):
    """Write a 25 fps MP4 and JPG contact sheet; return the MP4 Path.

    Reads ``scene.xml`` and the 50 Hz ``trajectory.npz`` in ``run_dir``. Optional
    ``events.jsonl`` supplies instruction captions. Ball and robot coordinates
    come exclusively from saved qpos; playback never advances physics. Rendering
    uses a private MuJoCo model entirely on the Strands GL thread.
    """
    run_dir = Path(run_dir)
    output = Path(output_path) if output_path is not None else run_dir / "demo.mp4"
    if output.exists() or output.with_suffix(".jpg").exists():
        raise FileExistsError(f"Choose a new replay output path: {output}")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Use an .mp4 output path for the recorded video.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Install ffmpeg before rendering the simulation replay.")
    with np.load(run_dir / "trajectory.npz", allow_pickle=False) as archive:
        times, qpos, qvel, skills, tilts = (
            archive[name].copy() for name in ("time", "qpos", "qvel", "skill", "tilt_deg")
        )
    count = len(times)
    if not count or times.shape != (count,) or skills.shape != (count,) or tilts.shape != (count,):
        raise ValueError("Replay requires nonempty time, skill and tilt arrays with one value per state.")
    if qpos.ndim != 2 or qvel.ndim != 2 or len(qpos) != count or len(qvel) != count:
        raise ValueError("Replay requires one qpos and qvel row per recorded state.")
    if any(not np.isfinite(values).all() for values in (times, qpos, qvel, tilts)):
        raise ValueError("Replay contains nonfinite states; inspect the simulation result before rendering.")
    if count > 1 and not np.allclose(np.diff(times), 0.02, atol=1e-5, rtol=0):
        raise ValueError("Replay requires 50 Hz recorded states for real-time playback at 25 fps.")
    events = _events(run_dir / "events.jsonl")
    model_metadata = _replay_metadata(run_dir)
    indices = list(range(0, count, 2))
    samples = set(np.linspace(0, len(indices) - 1, min(8, len(indices)), dtype=int).tolist())
    kicks = np.flatnonzero(np.char.startswith(skills.astype(str), "kick"))
    if len(kicks):
        # Preserve contact moments even when the search phase lasts a long time.
        samples.update(int(np.argmin(abs(np.asarray(indices) - index))) for index in (kicks[0], kicks[len(kicks) // 2]))
    renderer = _ReplayRenderer(run_dir / "scene.xml")
    if model_metadata.get("backend") == "action_playground":
        renderer.camera.distance = 0.75
    process, completed = None, False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if qpos.shape[1] != renderer.model.nq or qvel.shape[1] != renderer.model.nv:
            raise ValueError("Recorded qpos/qvel do not match scene.xml; use the scene from this run.")
        contact_frames, font = [], _font(20)
        event_index, event = 0, {}
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
                    "1280x816",
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
            for number, index in enumerate(indices):
                while event_index < len(events) and events[event_index][0] <= times[index] + 1e-8:
                    event = events[event_index][1]
                    event_index += 1
                views = renderer.frame(qpos[index], qvel[index], times[index])
                frame = _compose(
                    views,
                    float(times[index]),
                    str(skills[index]),
                    float(tilts[index]),
                    event,
                    font,
                    model_metadata=model_metadata,
                )
                _write_frame(process, frame)
                if number in samples:
                    contact_frames.append(frame.resize((640, 408), Image.Resampling.LANCZOS))
            process.stdin.close()
            code = process.wait(timeout=60)
            if code:
                error_log.seek(0)
                detail = error_log.read(8192).decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg failed with status {code}: {detail}")
        contact = Image.new("RGB", (1280, 408 * ((len(contact_frames) + 1) // 2)), (18, 24, 32))
        for index, frame in enumerate(contact_frames):
            contact.paste(frame, ((index % 2) * 640, (index // 2) * 408))
        contact.save(output.with_suffix(".jpg"), quality=92)
        completed = True
        return output
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(render_video(args.run_dir, args.output))


if __name__ == "__main__":
    main()
