"""Generation boundaries and cache settings without allocating a GPU or model."""

import copy
import json
from contextlib import contextmanager

import numpy as np
import pytest

from .head_features import FrozenFeatures
from .toy_fixtures import request
from .toy_vlm_generation import ToyVLMGenerator


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.moves = []

    @property
    def shape(self):
        return self.values.shape

    def to(self, device):
        self.moves.append(device)
        return self

    def sum(self):
        return self.values.sum()

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def tolist(self):
        return self.values.tolist()


class Tokenizer:
    eos_token_id = 248046
    pad_token_id = 248044

    def __init__(self):
        self.decoded = []

    def decode(self, tokens, *, skip_special_tokens):
        self.decoded.append((list(tokens), skip_special_tokens))
        return f"completion-{tokens[0]}"


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.chats = []
        self.batches = []
        self.length = 3
        self.mask = None

    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.chats.append(copy.deepcopy(messages))
        return f"fresh-prompt-{len(self.chats)}"

    def __call__(self, *, text, images, return_tensors, padding):
        assert text == [f"fresh-prompt-{len(self.chats)}"]
        assert return_tensors == "pt" and padding is False
        assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255)]
        self.images = images
        offset = len(self.batches) * 100
        batch = {
            "input_ids": Tensor([np.arange(offset + 1, offset + self.length + 1)]),
            "attention_mask": Tensor([self.mask if self.mask is not None else np.ones(self.length, dtype=int)]),
            "pixel_values": Tensor([[0.25, 0.75]]),
            "image_grid_thw": Tensor([[1, 2, 2], [1, 2, 2]]),
            "mm_token_type_ids": Tensor([np.zeros(self.length, dtype=int)]),
        }
        self.batches.append(batch)
        return batch


class Torch:
    def __init__(self):
        self.inference_active = False

    @contextmanager
    def inference_mode(self):
        assert not self.inference_active
        self.inference_active = True
        try:
            yield
        finally:
            self.inference_active = False


class Model:
    def __init__(self, torch):
        self.torch = torch
        self.calls = []
        self.terminal = 248046

    def generate(self, **kwargs):
        assert self.torch.inference_active
        self.calls.append(kwargs)
        prefix = kwargs["input_ids"].tolist()[0]
        return Tensor([prefix + [900 + len(self.calls), self.terminal]])


@pytest.fixture(params=("full", "compact", "rules"))
def generator(monkeypatch, tmp_path, request):
    def load(self, model_dir):
        assert model_dir == tmp_path
        self.processor = Processor()
        self.torch = Torch()
        self.model = Model(self.torch)
        self.load_seconds = 0.0

    monkeypatch.setattr(FrozenFeatures, "__init__", load)
    return ToyVLMGenerator(tmp_path, prompt_variant=request.param)


@pytest.fixture
def images():
    pil = pytest.importorskip("PIL.Image")
    cameras = dict(front=pil.new("RGB", (2, 2), (255, 0, 0)), near=pil.new("RGB", (2, 2), (0, 0, 255)))
    yield cameras
    for image in cameras.values():
        image.close()


def test_encoding_preserves_request_two_images_and_valid_unpadded_tokens(generator, images):
    state = request("CURRENT_USER_SENTINEL", planned=True)
    original = copy.deepcopy(state)
    batch, length = generator.encode_request(state, images)
    assert length == 3 and batch["input_ids"].shape == (1, 3)
    assert state == original
    messages = generator.processor.chats[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "CURRENT_USER_SENTINEL" not in messages[0]["content"]
    content = messages[1]["content"]
    assert [item["text"].strip() for item in content[:4] if item["type"] == "text"] == [
        "Current camera: front",
        "Current camera: near",
    ]
    assert sum(item["type"] == "image" for item in content) == 2
    context = next(
        item["text"].strip()
        for item in content
        if item["type"] == "text" and item["text"].strip().startswith(("{", "HOST_CONTEXT_JSON="))
    )
    payload = json.loads(context.removeprefix("HOST_CONTEXT_JSON="))
    assert payload["event"]["text"] == "CURRENT_USER_SENTINEL"
    assert "revision" not in payload["execution"] and "target" not in payload
    assert all(tensor.moves == ["cuda"] for tensor in batch.values())
    assert images["front"].getpixel((0, 0)) == (255, 0, 0), "Caller images must remain open."
    for converted in generator.processor.images:
        with pytest.raises(ValueError, match="closed image"):
            converted.getpixel((0, 0))


def test_cached_decoding_is_bounded_and_two_calls_have_fresh_context(generator, images):
    first, first_timing = generator.generate(request("FIRST_USER"), images)
    second, second_timing = generator.generate(request("SECOND_USER"), images)
    assert (first, second) == ("completion-901", "completion-902")
    assert generator.processor.tokenizer.decoded == [([901, 248046], True), ([902, 248046], True)]
    assert generator.model.calls[0]["input_ids"].tolist() == [[1, 2, 3]]
    assert generator.model.calls[1]["input_ids"].tolist() == [[101, 102, 103]]
    assert "FIRST_USER" not in json.dumps(generator.processor.chats[1])
    for call in generator.model.calls:
        assert call["use_cache"] is True and call["logits_to_keep"] == 1
        assert call["do_sample"] is False
        assert call["max_new_tokens"] == 768 and call["max_time"] == 90.0
        assert call["eos_token_id"] == [248046, 248044] and call["pad_token_id"] == 248044
        assert "past_key_values" not in call and "labels" not in call
    for timing in (first_timing, second_timing):
        assert timing["input_tokens"] == 3 and timing["output_tokens"] == 2
        assert timing["finish_reason"] == "eos"
        assert timing["seconds"] >= timing["encoding_seconds"] >= 0
    assert generator.metadata["constrained_decoding"] is False
    assert generator.metadata["prompt"]["supervised_answers_in_context"] is (
        generator.metadata["prompt_variant"] == "compact"
    )
    if generator.metadata["prompt_variant"] == "compact":
        assert generator.metadata["prompt"]["evaluation_answers_in_context"] is False


def test_input_budget_rejects_before_gpu_transfer_or_generation(generator, images):
    generator.processor.length = generator.max_input_tokens + 1
    with pytest.raises(ValueError, match="truncation is disabled"):
        generator.generate(request("点头"), images)
    assert generator.model.calls == []
    assert all(not tensor.moves for tensor in generator.processor.batches[0].values())


def test_input_at_budget_is_accepted_without_padding(generator, images):
    generator.processor.length = generator.max_input_tokens
    batch, length = generator.encode_request(request("点头"), images)
    assert length == generator.max_input_tokens and batch["input_ids"].shape == (1, length)


@pytest.mark.parametrize("mask", [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
def test_padded_or_gapped_input_is_refused(generator, images, mask):
    generator.processor.mask = mask
    with pytest.raises(ValueError, match="unpadded input"):
        generator.generate(request("点头"), images)
    assert generator.model.calls == []


@pytest.mark.parametrize("terminal, expected", [(248044, "eos"), (777, "limit")])
def test_chat_end_and_model_end_both_stop_but_capped_output_is_distinguishable(generator, images, terminal, expected):
    generator.model.terminal = terminal
    _, timing = generator.generate(request("你好"), images)
    assert timing["finish_reason"] == expected


def test_processor_failure_closes_converted_images_without_generation(generator, images):
    processor = generator.processor

    def fail(*, text, images, return_tensors, padding):
        processor.images = images
        raise ValueError("processor failed")

    # A callable proxy preserves the real template while making preprocessing fail.
    class BrokenProcessor:
        apply_chat_template = processor.apply_chat_template
        __call__ = staticmethod(fail)

    generator.processor = BrokenProcessor()
    with pytest.raises(ValueError, match="processor failed"):
        generator.generate(request("你好"), images)
    assert generator.model.calls == []
    for converted in processor.images:
        with pytest.raises(ValueError, match="closed image"):
            converted.getpixel((0, 0))


def test_unknown_prompt_variant_is_rejected_before_loading_model(monkeypatch):
    def load(*args):
        pytest.fail("Invalid prompt variant must fail before GPU allocation.")

    monkeypatch.setattr(FrozenFeatures, "__init__", load)
    with pytest.raises(ValueError, match="prompt_variant"):
        ToyVLMGenerator("unused", prompt_variant="unknown")
