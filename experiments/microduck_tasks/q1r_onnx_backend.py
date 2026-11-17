"""Native-chat Q1R ONNX backend sharing the existing task and host-control checks."""

import copy

from .q1r_backend import ToyQ1RBackend
from .q1r_onnx_runtime import Q1ROnnxGenerator


class ToyQ1ROnnxBackend(ToyQ1RBackend):
    name = "qwen35_q1r_onnx_int8_dialogue"

    def __init__(self, model_dir, tokenizer_dir=None):
        self.generator = Q1ROnnxGenerator(model_dir, tokenizer_dir)
        self.metadata = dict(
            backend=self.name,
            model_inference=True,
            goal_source="q1r_onnx_int8_generation",
            reply_source="model_generation",
            natural_language_reply_trained=True,
            deployment_ready=False,
            simulation_only=True,
            inference_physics_frozen=True,
            images_passed_to_model=False,
            generator=self.generator.metadata,
        )
        self.last_trace = None

    def decide_with_images(self, request, images):
        response = super().decide_with_images(request, images)
        if self.last_trace.get("model_called"):
            self.last_trace.update(
                goal_source="q1r_onnx_int8_generation",
                decision_source="q1r_onnx_int8_generation",
                onnx_generation=copy.deepcopy(self.generator.last_generation),
            )
        return response
