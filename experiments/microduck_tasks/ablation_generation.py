"""Load one pinned model for the whole replay matrix, with unchanged decoding."""

import time

from .ablation_prompt import messages_for
from .toy_head_features import _images
from .toy_vlm_generation import ToyVLMGenerator


class AblationGenerator(ToyVLMGenerator):
    def __init__(self, model_dir, *, grammar_max_whitespace=None):
        super().__init__(model_dir, prompt_variant="compact")
        self.grammar = None
        self.grammar_max_whitespace = grammar_max_whitespace
        self.metadata["grammar_max_whitespace"] = grammar_max_whitespace

    def encode_variant(self, request, images, variant):
        messages = messages_for(request, variant)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        pil = _images(images) if variant != "P6" else []
        try:
            batch = self.processor(text=[text], images=pil or None, return_tensors="pt", padding=False)
        finally:
            for image in pil:
                image.close()
        length = int(batch["input_ids"].shape[1])
        if length > self.max_input_tokens or int(batch["attention_mask"].sum()) != length:
            raise ValueError("Input must be unpadded and fit without truncation.")
        return {key: value.to("cuda") for key, value in batch.items()}, length, text

    def _processor(self):
        from xgrammar.contrib.hf import LogitsProcessor

        from .ablation_grammar import compile_grammar

        if self.grammar is None:
            self.grammar = compile_grammar(
                self.processor,
                self.model.config,
                self.options["eos_token_id"],
                max_whitespace_cnt=self.grammar_max_whitespace,
            )
        return LogitsProcessor(self.grammar)

    def generate_variant(self, request, images, variant):
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        batch, length, rendered = self.encode_variant(request, images, variant)
        options = self.options.copy()
        if variant == "P5":
            options["logits_processor"] = [self._processor()]
        with self.torch.inference_mode():
            output = self.model.generate(**batch, **options)
        self.torch.cuda.synchronize()
        tokens = output[0, length:].tolist()
        text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        return (
            text,
            dict(
                input_tokens=length,
                output_tokens=len(tokens),
                seconds=time.monotonic() - started,
                finish_reason="eos" if tokens and tokens[-1] in self.options["eos_token_id"] else "limit",
                peak_allocated_mib=self.torch.cuda.max_memory_allocated() / 1024**2,
                peak_reserved_mib=self.torch.cuda.max_memory_reserved() / 1024**2,
            ),
            rendered,
        )
