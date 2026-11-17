"""Render recorded simulation states for review; never run a policy or controller."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _indices(times, fps, maximum_seconds):
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Trajectory times must be finite and nondecreasing.")
    duration = float(times[-1] - times[0])
    count = min(max(1, math.ceil(duration * fps) + 1), int(maximum_seconds * fps))
    targets = np.linspace(times[0], times[-1], count)
    right = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(times[left] - targets) < abs(times[right] - targets), left, right)


def _draw_watermark(rgb, timestamp, skill, font):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 48), fill=(20, 20, 20))
    draw.text((10, 5), "OFFLINE SIMULATION REPLAY / EVALUATION ONLY", font=font, fill="white")
    draw.text((10, 26), f"recorded time {timestamp:6.2f}s   skill: {skill}", font=font, fill=(255, 225, 120))
    return image


def _render(run, filename, fps, maximum_seconds, azimuth, elevation, distance):
    import mujoco
    from OpenGL import GL

    scene_path, trajectory_path = run / "scene.xml", run / "trajectory.npz"
    output = run / filename
    preview_path, metadata_path = output.with_name(output.stem + "_preview.png"), output.with_suffix(".json")
    for path in (output, preview_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Preserve existing replay {path}; choose a different --filename.")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        times, qpos = archive["time"].copy(), archive["qpos"].copy()
        qvel, skills = archive["qvel"].copy(), archive["skill"].copy()
    indices = _indices(times, fps, maximum_seconds)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if qpos.shape != (len(times), model.nq) or qvel.shape != (len(times), model.nv):
        raise ValueError("Trajectory state dimensions do not match the saved scene.")
    if skills.shape != times.shape or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("Replay needs one skill per finite recorded state.")
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.time = qpos[0], qvel[0], float(times[0])
    mujoco.mj_forward(model, data)
    trunk = model.body("trunk_base").id
    red = model.body("ball_red").id
    lookat = (data.xpos[trunk] + data.xpos[red]) / 2
    lookat[2] = 0.11
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(480, model.vis.global_.offheight)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    kick_indices = np.flatnonzero(np.char.startswith(skills.astype(str), "kick_"))
    selected_time = times[kick_indices[len(kick_indices) // 2]] if len(kick_indices) else times[len(times) // 2]
    preview_frame = int(np.argmin(abs(times[indices] - selected_time)))
    frames = []
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        gl_renderer = GL.glGetString(GL.GL_RENDERER).decode("utf-8", errors="replace")
        if not any(name in gl_renderer.lower() for name in ("llvmpipe", "softpipe", "software", "swrast")):
            raise RuntimeError(f"Replay requires a CPU renderer; EGL selected {gl_renderer!r}.")
        for frame_index, index in enumerate(indices):
            data.qpos[:], data.qvel[:], data.time = qpos[index], qvel[index], float(times[index])
            # mj_forward refreshes geometry; no mj_step or controller is invoked.
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            overlay = _draw_watermark(renderer.render().copy(), times[index], str(skills[index]), font)
            if frame_index == preview_frame:
                overlay.save(preview_path)
            frames.append(overlay.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
        optimize=False,
    )
    metadata = {
        "source": "offline_simulation_state_replay_evaluation_only",
        "policy_or_controller_invoked": False,
        "physics_stepped": False,
        "render_backend": "Mesa software EGL",
        "gl_renderer": gl_renderer,
        "gif": output.name,
        "preview": preview_path.name,
        "source_trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        "source_scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "source_states": len(times),
        "source_duration_s": float(times[-1] - times[0]),
        "frame_count": len(frames),
        "fps": fps,
        "playback_duration_s": len(frames) / fps,
        "sampled_full_trajectory": True,
        "sampled_times_s": times[indices].tolist(),
        "preview_time_s": float(times[indices[preview_frame]]),
        "camera": {"lookat": lookat.tolist(), "distance": distance, "azimuth": azimuth, "elevation": elevation},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return {key: value for key, value in metadata.items() if key != "sampled_times_s"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--filename", default="overview.gif")
    parser.add_argument("--fps", type=int, choices=(8, 10), default=8)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--azimuth", type=float, default=110.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--distance", type=float, default=1.1)
    args = parser.parse_args()
    if Path(args.filename).name != args.filename or Path(args.filename).suffix.lower() != ".gif":
        parser.error("--filename must be a .gif filename inside the run directory.")
    if not math.isfinite(args.max_seconds) or not 1 <= args.max_seconds <= 60:
        parser.error("--max-seconds must be finite and between 1 and 60.")
    if not all(math.isfinite(value) for value in (args.azimuth, args.elevation, args.distance)) or args.distance <= 0:
        parser.error("Camera angles and positive distance must be finite.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["LP_NUM_THREADS"] = "2"
    from strands_robots._gl import on_gl_thread

    result = on_gl_thread(
        _render,
        args.run.resolve(),
        args.filename,
        args.fps,
        args.max_seconds,
        args.azimuth,
        args.elevation,
        args.distance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
