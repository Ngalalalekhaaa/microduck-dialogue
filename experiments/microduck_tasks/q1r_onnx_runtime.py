"""Pinned, CUDA-only Q1R generation without importing PyTorch or Transformers."""

import json
import threading
import time
from pathlib import Path

from .q1r_onnx_benchmark import MAX_NEW_TOKENS, digest, generate
from .q1r_prompt import messages_for
from .q1r_prompt import metadata as prompt_metadata


def refuse_template(message):
    raise ValueError(message)


def warmup_request():
    """An isolated synthetic greeting, never submitted to a dialogue session."""
    from .toy_catalog import SELF_ACTIONS
    from .toy_protocol import DEFAULT_MAX_GOALS, GOALS, VERSION

    return dict(
        schema_version=VERSION,
        identity=dict(name="小鸭"),
        event=dict(kind="user", text="你好"),
        history=[],
        observation=dict(
            age_s=0,
            entities=[],
            posture="standing",
            stationary=True,
            environment=dict(surface="simulation_floor", motion_allowed=False, edge_guard="unavailable"),
        ),
        execution=dict(state="idle", revision=0, active=None, pending=[], last_result=None),
        capabilities=dict(
            goal_types=list(GOALS),
            self_actions=list(SELF_ACTIONS),
            continuous_move=True,
            sit_hold=True,
            search_targets=[],
            max_goals=DEFAULT_MAX_GOALS,
            queue_control=True,
        ),
    )


class Q1RTextCodec:
    """Render the unchanged HF template and encode it with the Rust tokenizer."""

    def __init__(self, model_dir, tokenizer_dir=None):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        model_dir = Path(model_dir).expanduser().resolve()
        required = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        candidates = [Path(tokenizer_dir).expanduser().resolve()] if tokenizer_dir else [model_dir]
        folder = next((path for path in candidates if all((path / name).is_file() for name in required)), None)
        if folder is None:
            raise FileNotFoundError("Provide tokenizer.json, tokenizer_config.json and chat_template.jinja together.")
        if folder != model_dir and any(digest(folder / name) != digest(model_dir / name) for name in required):
            raise ValueError(
                "An explicit tokenizer directory must match all pinned tokenizer assets in the ONNX package."
            )
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        tokenizer_config = json.loads((folder / "tokenizer_config.json").read_text())
        special_ids = {}
        for name in ("eos_token", "pad_token"):
            token = tokenizer_config.get(name)
            token = token.get("content") if isinstance(token, dict) else token
            token_id = self.tokenizer.token_to_id(token) if isinstance(token, str) else None
            if token_id is None:
                raise ValueError(f"The tokenizer must define {name}; generation endings cannot be guessed.")
            special_ids[name] = token_id
        self.endings = list(dict.fromkeys(special_ids.values()))
        self.pad_token_id = special_ids["pad_token"]
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        environment.globals["raise_exception"] = refuse_template
        environment.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
        self.template = environment.from_string((folder / "chat_template.jinja").read_text())
        self.metadata = dict(
            directory=str(folder),
            files={name: digest(folder / name) for name in required},
            eos_token_ids=self.endings,
            pad_token_id=self.pad_token_id,
            enable_thinking=False,
        )

    def encode(self, request):
        prompt = self.template.render(messages=messages_for(request), add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        return dict(prompt=prompt, prefix_ids=ids)


def verify_package(model_dir):
    """Check the evaluated package manifest and graph bounds before allocating CUDA."""
    import onnx

    manifest_path = model_dir.parent / f"{model_dir.name}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("runtime_generation_pass"):
        raise ValueError("Select an ONNX package whose manifest records successful runtime generation.")
    required = {
        "model.onnx",
        "model.onnx.data",
        "genai_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
    if not required.issubset(manifest.get("files", {})):
        raise ValueError("The evaluated package manifest must pin the graph, external weights and tokenizer assets.")
    for name, expected in manifest["files"].items():
        path = (model_dir / name).resolve()
        if not path.is_relative_to(model_dir) or not path.is_file():
            raise ValueError(f"The manifest file is missing or outside the model package: {name}")
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"ONNX package file changed since evaluation: {name}")
    config = json.loads((model_dir / "genai_config.json").read_text())
    if config["model"]["type"] != "qwen3_5_text":
        raise ValueError("Use the evaluated Q1R Qwen3.5 text-only package.")
    context_length = int(config["model"]["context_length"])
    if context_length <= MAX_NEW_TOKENS:
        raise ValueError("ONNX context must leave room for both the full prompt and 1536 generated tokens.")
    graph = onnx.load(str(model_dir / "model.onnx"), load_external_data=False)
    initializers = {item.name: item for item in graph.graph.initializer}
    rope_lengths = {}
    for name in ("cos_cache", "sin_cache"):
        tensor = initializers.get(name)
        if tensor is None or len(tensor.dims) != 2 or tensor.dims[0] < context_length:
            raise ValueError(
                "The actual RoPE cache is shorter than model context; changing JSON alone is insufficient."
            )
        rope_lengths[name] = tensor.dims[0]
    for tensor in graph.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value not in manifest["files"]:
                raise ValueError(f"Graph references unpinned external weights: {entry.value}")
    source_path = model_dir.parent / "merge_report.json"
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    checkpoint = Path(source["selected"]) if source.get("selected") else None
    if checkpoint and (checkpoint / "identity.json").is_file():
        identity = json.loads((checkpoint / "identity.json").read_text())
        if identity["weights_sha256"] != source.get("adapter_sha256") or identity["prompt"] != prompt_metadata():
            raise ValueError("The source Q1R checkpoint identity or dialogue prompt changed.")
    return config, dict(
        manifest=str(manifest_path),
        manifest_sha256=digest(manifest_path),
        files=manifest["files"],
        precision=manifest.get("precision"),
        runtime_generation_pass=True,
        rope_cache_lengths=rope_lengths,
        source_checkpoint=str(checkpoint) if checkpoint else None,
        source_adapter_sha256=source.get("adapter_sha256"),
    )


class Q1ROnnxGenerator:
    def __init__(self, model_dir, tokenizer_dir=None):
        # The launcher supplies the isolated ONNX overlay and CUDA shared-library paths.
        import numpy as np
        import onnxruntime as ort
        import onnxruntime_genai as og

        self.model_dir = Path(model_dir).expanduser().resolve()
        config, package = verify_package(self.model_dir)
        self.codec = Q1RTextCodec(self.model_dir, tokenizer_dir)
        self.context_length = int(config["model"]["context_length"])
        self.maximum_input_tokens = min(4096, self.context_length - MAX_NEW_TOKENS)
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available or not og.is_cuda_available():
            raise RuntimeError("Q1R ONNX requires the CUDA ONNX overlay; CPU fallback is not allowed.")
        configuration = og.Config(str(self.model_dir))
        configuration.overlay(
            json.dumps(dict(model=dict(eos_token_id=self.codec.endings, pad_token_id=self.codec.pad_token_id)))
        )
        configuration.clear_providers()
        configuration.append_provider("cuda")
        started = time.monotonic()
        self.model = og.Model(configuration)
        if str(self.model.device_type).lower() != "cuda":
            raise RuntimeError(f"Expected a CUDA model, received {self.model.device_type}; refusing CPU fallback.")
        self.og, self.np = og, np
        self.lock = threading.Lock()
        self.last_generation = None
        self.metadata = dict(
            model_dir=str(self.model_dir),
            package=package,
            tokenizer=self.codec.metadata,
            prompt=prompt_metadata(),
            ort_version=ort.__version__,
            genai_version=og.__version__,
            device_type=self.model.device_type,
            providers=available,
            load_seconds=time.monotonic() - started,
            context_length=self.context_length,
            maximum_input_tokens=self.maximum_input_tokens,
            maximum_output_tokens=MAX_NEW_TOKENS,
            images_passed_to_model=False,
            reply_source="model_generation",
            fresh_generator_per_request=True,
        )
        self.metadata["warmup"] = self._warmup()

    def _warmup(self):
        # Materialize the lazy CUDA kernels before the launcher constructs MuJoCo/EGL.
        row = self.codec.encode(warmup_request())
        result = generate(
            self.og,
            self.np,
            self.model,
            self.codec.tokenizer,
            dict(id="initialization_warmup", prefix_ids=row["prefix_ids"]),
            self.codec.endings,
            context_length=self.context_length,
            stop_after_tokens=1,
        )
        return dict(
            synthetic=True,
            purpose="Initialize CUDA kernels before simulation construction",
            physical_executed=False,
            visible_in_chat=False,
            timing=result["timing"],
            sequence_validation=result["sequence_validation"],
            generated_ids=result["generated_ids"],
        )

    def generate(self, request, images):
        with self.lock:
            self.last_generation = None
            row = self.codec.encode(request)
            if len(row["prefix_ids"]) > self.maximum_input_tokens:
                raise ValueError(
                    f"The full request has {len(row['prefix_ids'])} tokens; this ONNX package allows "
                    f"{self.maximum_input_tokens} input tokens while reserving all {MAX_NEW_TOKENS} output tokens. "
                    "Start a shorter conversation or use a validated larger-context package; nothing was truncated."
                )
            result = generate(
                self.og,
                self.np,
                self.model,
                self.codec.tokenizer,
                dict(id="interactive", prefix_ids=row["prefix_ids"]),
                self.codec.endings,
                context_length=self.context_length,
            )
            self.last_generation = dict(
                sequence_validation=result["sequence_validation"],
                generated_ids=result["generated_ids"],
                input_tokens=len(row["prefix_ids"]),
            )
            return result["raw_text"], result["timing"]
