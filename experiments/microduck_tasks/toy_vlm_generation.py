"""Pinned Qwen generation with two cameras and no feature-extraction padding."""

import time
from pathlib import Path

from .head_features import FrozenFeatures
from .setup_skill_head import MODEL_REPO, MODEL_REVISION, WEIGHTS_SHA256, digest
from .toy_head_features import _images
from .toy_vlm_prompt import messages_for, prompt_config


class ToyVLMGenerator(FrozenFeatures):
    """Reuse the verified NF4 loader, but generate from fresh, unpadded requests."""

    def __init__(self, model_dir, *, prompt_variant="compact"):
        if prompt_variant == "compact":
            from .toy_vlm_prompt_compact import messages_for as compact_messages
            from .toy_vlm_prompt_compact import prompt_config as compact_config

            self.messages_for = compact_messages
            context_config = compact_config()
        elif prompt_variant == "full":
            self.messages_for = messages_for
            context_config = prompt_config()
        elif prompt_variant == "rules":
            from .toy_vlm_prompt_rules import messages_for as rules_messages
            from .toy_vlm_prompt_rules import prompt_config as rules_config

            self.messages_for = rules_messages
            context_config = rules_config()
        else:
            raise ValueError("Use prompt_variant='full', 'compact' or 'rules'.")
        super().__init__(model_dir)
        tokenizer = self.processor.tokenizer
        self.options = dict(
            do_sample=False,
            use_cache=True,
            logits_to_keep=1,
            max_new_tokens=768,
            max_time=90.0,
            eos_token_id=list(dict.fromkeys([tokenizer.eos_token_id, tokenizer.pad_token_id])),
            pad_token_id=tokenizer.pad_token_id,
        )
        self.max_input_tokens = 8192
        self.metadata = dict(
            model_repo=MODEL_REPO,
            model_revision=MODEL_REVISION,
            weights_sha256=WEIGHTS_SHA256,
            model_dir=str(Path(model_dir).resolve()),
            implementation_sha256=digest(__file__),
            loader_sha256=digest(Path(__file__).with_name("head_features.py")),
            prompt_variant=prompt_variant,
            prompt=context_config,
            generation=self.options.copy(),
            max_input_tokens=self.max_input_tokens,
            quantization="NF4-double; visual and lm_head remain bfloat16",
            constrained_decoding=False,
            validation="strict JSON and contextual protocol; no repairs or retries",
            load_seconds=self.load_seconds,
        )

    def encode_request(self, request, images):
        text = self.processor.apply_chat_template(
            self.messages_for(request), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        pil = _images(images)
        try:
            batch = self.processor(text=[text], images=pil, return_tensors="pt", padding=False)
        finally:
            for image in pil:
                image.close()
        length = int(batch["input_ids"].shape[1])
        if length > self.max_input_tokens:
            raise ValueError(f"Input length {length} exceeds {self.max_input_tokens}; truncation is disabled.")
        if int(batch["attention_mask"].sum()) != length:
            raise ValueError("Single-request generation requires unpadded input.")
        return {key: value.to("cuda") for key, value in batch.items()}, length

    def generate(self, request, images):
        started = time.monotonic()
        batch, length = self.encode_request(request, images)
        encoded_at = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(**batch, **self.options)
        tokens = output[0, length:].tolist()
        text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        finish = "eos" if tokens and tokens[-1] in self.options["eos_token_id"] else "limit"
        return text, dict(
            input_tokens=length,
            output_tokens=len(tokens),
            finish_reason=finish,
            encoding_seconds=encoded_at - started,
            seconds=time.monotonic() - started,
        )
