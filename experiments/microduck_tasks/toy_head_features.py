"""Input-only frozen Qwen features for the current dialogue/goal contract."""

import hashlib
import time
from pathlib import Path

import numpy as np

from .head_features import FrozenFeatures
from .head_features import feature_config as legacy_config
from .toy_dataset import contract_digest, digest, dumps
from .toy_inputs import model_payload
from .toy_schema import SYSTEM

FEATURE_VERSION = "qwen35-toy-dialogue-input-1"


def messages_for(request, *, vision=True):
    """Serialize only the current request, never a supervised response."""
    content = []
    if vision:
        for camera in ("front", "near"):
            content.extend([{"type": "text", "text": f"Current camera: {camera}"}, {"type": "image"}])
    content.append({"type": "text", "text": dumps(model_payload(request))})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def feature_config(vision=True):
    config = legacy_config(vision)
    config.pop("labels_include")
    config.update(
        version=FEATURE_VERSION,
        implementation_sha256=digest(__file__),
        loader_sha256=digest(Path(__file__).with_name("head_features.py")),
        contract_sha256=contract_digest(),
        prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
        dimension=2048,
        answer_tokens_in_input=False,
        variable_length_batching="causal_right_padding",
    )
    return config


def input_digest(request):
    return hashlib.sha256(dumps(model_payload(request)).encode()).hexdigest()


def _images(images):
    from PIL import Image

    if set(images) != {"front", "near"}:
        raise ValueError("Require exactly front and near camera images.")
    result = []
    for name in ("front", "near"):
        image = images[name]
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                result.append(source.convert("RGB"))
        elif isinstance(image, Image.Image):
            result.append(image.convert("RGB"))
        else:
            result.append(Image.fromarray(np.asarray(image)).convert("RGB"))
    return result


def _right_padded_lengths(mask):
    """Reject left padding or holes: each pooled index is the final real input token."""
    import torch

    lengths = mask.sum(dim=1).long()
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
    if (lengths < 1).any() or not torch.equal(mask.bool(), expected):
        raise ValueError("Feature extraction requires nonempty, contiguous, right-padded input tokens.")
    return lengths


def _pool(hidden, mask):
    import torch

    lengths = _right_padded_lengths(mask)
    indices = torch.arange(len(lengths), device=hidden.device)
    return hidden[indices, lengths - 1].float().cpu().numpy().copy()


class ToyFrozenFeatures(FrozenFeatures):
    """Reuse the pinned NF4 loader without changing the historical v2 feature contract."""

    def encode_batch(self, requests, images, *, vision=True):
        import torch

        if not 1 <= len(requests) <= 4 or len(images) != len(requests):
            raise ValueError("Use one through four requests with matching camera dictionaries.")
        self.processor.tokenizer.padding_side = "right"
        texts = [
            self.processor.apply_chat_template(
                messages_for(request, vision=vision),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for request in requests
        ]
        pil = [image for cameras in images for image in _images(cameras)] if vision else []
        try:
            batch = dict(self.processor(text=texts, images=pil or None, return_tensors="pt", padding=True))
        finally:
            for image in pil:
                image.close()
        lengths = _right_padded_lengths(batch["attention_mask"])
        if int(lengths.max()) > 4096:
            raise ValueError(f"Input length {int(lengths.max())} exceeds 4096; truncation is disabled.")
        length = batch["input_ids"].shape[1]
        extra = ((length + 1023) // 1024) * 1024 - length
        for key, value in (
            ("input_ids", self.processor.tokenizer.pad_token_id),
            ("attention_mask", 0),
            ("mm_token_type_ids", 0),
            ("token_type_ids", 0),
        ):
            if key in batch:
                batch[key] = torch.nn.functional.pad(batch[key], (0, extra), value=value)
        return batch, lengths.tolist()

    def extract_batch(self, requests, images, *, vision=True):
        started = time.monotonic()
        batch, lengths = self.encode_batch(requests, images, vision=vision)
        batch = {key: value.to("cuda") for key, value in batch.items()}
        with self.torch.inference_mode():
            result = self.model.model(**batch, use_cache=False, return_dict=True)
            vectors = _pool(result.last_hidden_state, batch["attention_mask"])
        if vectors.shape != (len(lengths), 2048) or not np.isfinite(vectors).all():
            raise ValueError("Expected finite 2048-dimensional input-only Qwen3.5 features.")
        seconds = time.monotonic() - started
        timings = [
            {
                "tokens": length,
                "padded_tokens": int(batch["input_ids"].shape[1]),
                "batch_size": len(lengths),
                "batch_seconds": seconds,
                "seconds": seconds / len(lengths),
            }
            for length in lengths
        ]
        return vectors, timings

    def extract(self, request, images, *, vision=True):
        vectors, timings = self.extract_batch([request], [images], vision=vision)
        return vectors[0], timings[0]
