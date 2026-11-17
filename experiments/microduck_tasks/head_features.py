"""Frozen, input-only Qwen3.5 features shared by dataset extraction and inference."""

import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .plan_protocol import CATALOG, validate_input
from .setup_skill_head import MODEL_REPO, MODEL_REVISION, WEIGHTS, WEIGHTS_SHA256, digest

FEATURE_VERSION = "qwen35-skill-input-1"
SYSTEM = """You observe a mobile robot through two RGB cameras, body sensors and past execution outcomes.
Understand the user's instruction and the current situation to choose the NEXT skill.
Available skills: stand, move_forward, move_backward, move_left, move_right, turn_left, turn_right,
approach_target, kick_left, kick_right. approach_target visually follows a visible target and stops
within reach; it never kicks. Also possible: answer a view question, clarify an ambiguous request,
or finish a completed task. A single move or turn request should execute once then finish.
The body-frame coordinates are metres: x forward, y left, z up. History is already completed.
Use the current images and observations; do not assume a hidden target is visible.
"""


def _rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def payload(instruction, state, context):
    validate_input(state, context)
    if context["committed_steps"]:
        raise ValueError("This first head is trained only at decision boundaries; wait for the committed steps.")
    # Deliberately omit task/frame ids, elapsed counters, and all training labels.
    sensors = {key: state[key] for key in ("yaw", "stationary", "tilt_deg", "entities")}
    return _rounded(dict(instruction=instruction, state=sensors, recent_outcomes=context["recent_outcomes"]))


def messages(instruction, state, context, vision=True):
    content = []
    if vision:
        for camera in ("front", "near"):
            content.extend([dict(type="text", text=f"Current camera: {camera}"), dict(type="image")])
    content.append(dict(type="text", text=json.dumps(payload(instruction, state, context), ensure_ascii=False)))
    return [dict(role="system", content=SYSTEM), dict(role="user", content=content)]


def label(row):
    plan = row["target"]
    return plan["steps"][0]["skill"] if plan["status"] == "continue" else plan["status"]


def feature_config(vision=True):
    return dict(
        version=FEATURE_VERSION,
        implementation_sha256=digest(__file__),
        model_repo=MODEL_REPO,
        revision=MODEL_REVISION,
        weights_sha256=WEIGHTS_SHA256,
        prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
        vision=vision,
        pooling="last_valid_input_token_before_any_assistant_answer",
        dtype="bfloat16",
        quantization="NF4-double",
        skip_quantization=["visual", "lm_head"],
        min_pixels=65536,
        max_pixels=98304,
        max_input_tokens=4096,
        transformers="5.4.0",
        bitsandbytes="0.49.2",
        gated_delta_backend="fla-core==0.4.1",
        convolution_backend="torch",
        right_padding_multiple=1024,
        use_cache=False,
        labels_include=list(CATALOG) + ["answer", "clarify", "finish"],
    )


class FrozenFeatures:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

        self.torch = torch
        if version("fla-core") != "0.4.1":
            raise ValueError("Use fla-core==0.4.1 to preserve the feature extraction backend.")
        torch.set_num_threads(4)
        started = time.monotonic()
        if digest(Path(model_dir) / WEIGHTS) != WEIGHTS_SHA256:
            raise ValueError("Use the pinned and verified Qwen3.5 base model.")
        self.processor = AutoProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
            min_pixels=65536,
            max_pixels=98304,
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["visual", "lm_head"],
            ),
        ).eval()
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if hasattr(module, "chunk_gated_delta_rule"):
                if not module.chunk_gated_delta_rule.__module__.startswith("fla."):
                    raise RuntimeError("FLA gated delta kernels are unavailable; check the isolated environment.")
                module.causal_conv1d_fn = None
        self.model.config.use_cache = False
        self.load_seconds = time.monotonic() - started

    def encode(self, instruction, state, context, images, *, vision=True):
        import torch
        from PIL import Image

        chat = messages(instruction, state, context, vision)
        text = self.processor.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pil = []
        if vision:
            for name in ("front", "near"):
                image = images[name]
                if isinstance(image, (str, Path)):
                    with Image.open(image) as source:
                        pil.append(source.convert("RGB"))
                elif isinstance(image, Image.Image):
                    pil.append(image.convert("RGB"))
                else:
                    pil.append(Image.fromarray(np.asarray(image)).convert("RGB"))
        batch = self.processor(text=[text], images=pil or None, return_tensors="pt", padding=False)
        length = batch["input_ids"].shape[1]
        if length > 4096:
            raise ValueError(f"Input length {length} exceeds 4096; truncation is disabled.")
        # Fixed buckets avoid recompiling Triton kernels for every token length.
        # Causal right padding cannot affect the valid input token we pool below.
        extra = ((length + 1023) // 1024) * 1024 - length
        for key, value in (
            ("input_ids", self.processor.tokenizer.pad_token_id),
            ("attention_mask", 0),
            ("mm_token_type_ids", 0),
            ("token_type_ids", 0),
        ):
            if key in batch:
                batch[key] = torch.nn.functional.pad(batch[key], (0, extra), value=value)
        return batch.to("cuda"), length

    def extract(self, instruction, state, context, images, *, vision=True):
        started = time.monotonic()
        batch, length = self.encode(instruction, state, context, images, vision=vision)
        with self.torch.inference_mode():
            # Calling the backbone avoids allocating a sequence-by-vocabulary logits tensor.
            result = self.model.model(**batch, use_cache=False, return_dict=True)
            positions = batch["attention_mask"].nonzero(as_tuple=False)
            last = positions[positions[:, 0] == 0, 1].max()
            vector = result.last_hidden_state[0, last].float().cpu().numpy()
        if vector.shape != (2048,) or not np.isfinite(vector).all():
            raise ValueError("Expected a finite 2048-dimensional Qwen3.5 feature.")
        return vector, dict(tokens=length, seconds=time.monotonic() - started)
