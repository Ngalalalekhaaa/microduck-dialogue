"""Image + language backend boundary, with a local rule teacher implementation.

This module does not contain a trained VLM. A future local/API VLM implements the
same protocol, sees the current images and instruction, and returns a validated
intent. Rule observations must be computed from those images, not oracle state.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .language import Intent, describe_view, parse_instruction


@dataclass(frozen=True)
class VisionLanguageInput:
    instruction: str
    images: dict[str, np.ndarray]
    observation: dict
    context: dict


@runtime_checkable
class VLMBackend(Protocol):
    def interpret(self, request: VisionLanguageInput) -> Intent:
        """Interpret the current instruction and visual observations."""
        ...


class RuleVLMBackend:
    """Offline teacher: lexical intent grammar and geometric RGB-D detections."""

    name = "rgbd_rule_teacher"
    is_trained_vlm = False

    def interpret(self, request: VisionLanguageInput) -> Intent:
        if not isinstance(request, VisionLanguageInput):
            raise TypeError("Pass a VisionLanguageInput containing current camera images.")
        if not isinstance(request.images, dict) or not request.images:
            return Intent("clarify", reply="当前没有相机图像，请先恢复相机观测。")
        for name, image in request.images.items():
            if (
                not isinstance(name, str)
                or not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[2] != 3
                or image.shape[0] == 0
                or image.shape[1] == 0
                or image.dtype != np.uint8
            ):
                return Intent("clarify", reply="相机图像格式无效，需要 H×W×3 的 uint8 RGB 图像。")
        if not isinstance(request.observation, dict) or not isinstance(request.context, dict):
            raise TypeError("observation and context must be dictionaries.")
        intent = parse_instruction(request.instruction, request.context)
        if intent.kind == "describe":
            detections = request.observation.get("detections")
            if not isinstance(detections, list):
                return Intent("clarify", reply="当前还没有视觉检测结果，请先完成本帧检测。")
            return Intent("describe", reply=describe_view(detections))
        return intent
