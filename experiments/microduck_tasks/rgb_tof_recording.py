"""Reviewable RGB/ToF artifacts from sensor packets, without simulator access."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _label(draw, xy, text, font, colour="white"):
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    draw.rectangle((left - 1, top - 1, right + 1, bottom + 1), fill=(18, 18, 18))
    draw.text(xy, text, fill=colour, font=font)


def _overlay(rgb, observation, packet, diagnostics, font):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    associations = diagnostics.get("associations", [])
    supported = {(zone["row"], zone["column"]) for item in associations for zone in item.get("supporting_zones", [])}
    for zone in diagnostics.get("projections", []):
        points = np.asarray(zone.get("polygon_uv", []), dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            continue
        polygon = [tuple(point) for point in points]
        accepted = (zone["row"], zone["column"]) in supported
        colour = (40, 255, 90) if accepted else (125, 125, 125)
        draw.line([*polygon, polygon[0]], fill=colour, width=2 if accepted else 1)
    colours = {"red": (255, 100, 100), "yellow": (255, 240, 50), "blue": (90, 170, 255)}
    candidates = associations + diagnostics.get("rejected_rgb_candidates", [])
    for item in candidates:
        box = np.asarray(item.get("box_uv", []), dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        colour = colours.get(item.get("color"), "white")
        draw.rectangle(tuple(box), outline=colour, width=2)
        distance = item.get("fused_center_range_camera_m")
        text = item.get("color", "ball")
        if distance is not None:
            text += f" cam r={distance:.3f}m"
        else:
            text += " rejected"
        xy = (max(0, min(int(box[0]), image.width - 130)), max(28, int(box[1]) - 13))
        _label(draw, xy, text, font, colour)
    time_s = float(packet["time_s"])
    age_s = float(observation["time_s"]) - time_s
    tilt = observation.get("tilt_deg")
    tilt_text = "?" if tilt is None else f"{tilt:.1f}"
    _label(draw, (3, 2), f"sensor {time_s:.2f}s  age {age_s:.3f}s", font)
    _label(draw, (3, 15), f"IMU tilt {tilt_text}deg  settled={observation.get('stationary', '?')}", font)
    _label(draw, (3, image.height - 14), "grey: ToF zones   green: accepted support", font)
    return image


class Recorder:
    """Store each sensor timestamp once; GIF is a bounded 5-fps review summary.

    ``save`` closes no external resources and can be called for interim review.
    Raw frames and JSONL are written on each hook call, even before ``save``.
    GIF sampling starts at 5 Hz of simulation time and decimates if necessary;
    playback is a 5-fps summary, not a real-time recording.
    """

    def __init__(self, output, config):
        self.output, self.config = Path(output), config
        self.output.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output / "sensor_frames"
        self.log_path = self.output / "observations.jsonl"
        self.config_path = self.output / "sensor_recording_config.json"
        if any(path.exists() for path in (self.frames_dir, self.log_path, self.config_path)):
            raise FileExistsError(f"Use a fresh sensor recording directory: {self.output}")
        self.frames_dir.mkdir()
        self.log_path.touch(exist_ok=False)
        self.config_path.write_text(json.dumps(_jsonable(asdict(config)), indent=2, allow_nan=False) + "\n")
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 10)
        except OSError:
            self.font = ImageFont.load_default()
        self.seen = set()
        self.frames = 0
        self.first_overlay = self.last_overlay = None
        self.first_time_s = self.last_time_s = None
        self.gif_frames = []
        self.gif_interval_s = 0.2

    def __call__(self, observation, packet, diagnostics):
        time_s = float(packet["time_s"])
        if not math.isfinite(time_s):
            raise ValueError("Record a finite sensor timestamp.")
        if time_s in self.seen:
            return
        rgb = np.asarray(packet["rgb"])
        ranges = np.asarray(packet["ranges_m"])
        valid, quality = np.asarray(packet["valid"]), np.asarray(packet["quality"])
        shape = (self.config.tof_rows, self.config.tof_columns)
        if rgb.dtype != np.uint8 or rgb.shape != (self.config.height, self.config.width, 3):
            raise ValueError("RGB packet dimensions must match the recording calibration.")
        if any(value.shape != shape for value in (ranges, valid, quality)):
            raise ValueError("Record one range, validity and quality value per ToF zone.")
        relative = Path("sensor_frames") / f"frame_{self.frames:06d}.npz"
        np.savez_compressed(
            self.output / relative,
            rgb=rgb,
            ranges_m=ranges,
            valid=valid,
            quality=quality,
            time_s=time_s,
            rgb_time_s=float(packet.get("rgb_time_s", time_s)),
            observation_time_s=float(observation["time_s"]),
        )
        record = {
            "index": self.frames,
            "sensor_time_s": time_s,
            "rgb_time_s": float(packet.get("rgb_time_s", time_s)),
            "npz": relative.as_posix(),
            "observation": observation,
            "diagnostics": diagnostics,
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
        overlay = _overlay(rgb, observation, packet, diagnostics, self.font)
        if self.first_overlay is None:
            self.first_overlay, self.first_time_s = overlay, time_s
            overlay.save(self.output / "first_overlay.png")
        self.last_overlay, self.last_time_s = overlay, time_s
        if not self.gif_frames or time_s - self.gif_frames[-1][0] >= self.gif_interval_s - 1e-9:
            if len(self.gif_frames) >= 180:
                self.gif_frames = self.gif_frames[::2]
                self.gif_interval_s *= 2
            self.gif_frames.append((time_s, overlay.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)))
        self.frames += 1
        self.seen.add(time_s)

    def save(self):
        gif_frames = list(self.gif_frames)
        if self.last_overlay is not None:
            self.last_overlay.save(self.output / "last_overlay.png")
            if gif_frames[-1][0] != self.last_time_s:
                gif_frames.append(
                    (self.last_time_s, self.last_overlay.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
                )
            images = [frame for _, frame in gif_frames]
            images[0].save(
                self.output / "sensor_trace.gif",
                save_all=True,
                append_images=images[1:],
                duration=200,
                loop=0,
                disposal=2,
                optimize=False,
            )
        summary = {
            "sensor_frames": self.frames,
            "first_time_s": self.first_time_s,
            "last_time_s": self.last_time_s,
            "raw_frames": "sensor_frames/frame_*.npz",
            "observations": self.log_path.name,
            "calibration": self.config_path.name,
            "first_overlay": "first_overlay.png" if self.frames else None,
            "last_overlay": "last_overlay.png" if self.frames else None,
            "gif": "sensor_trace.gif" if self.frames else None,
            "gif_frames": len(gif_frames),
            "gif_fps": 5,
            "gif_mode": "sampled_summary_not_realtime",
            "gif_sampling_interval_s": self.gif_interval_s,
            "json_nonfinite_values": "null; raw sensor values preserved in npz",
            "source": "caller_sensor_packets_and_fusion_diagnostics_only",
        }
        (self.output / "recording_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary
