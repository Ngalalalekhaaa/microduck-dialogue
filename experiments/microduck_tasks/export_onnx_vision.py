"""Export Qwen3-VL vision and all three DeepStack features for the simulator cameras.

The graph deliberately fixes a single image to grid [1,16,20]. The runtime runs
it independently for each 320x240 RGB camera and rejects unsupported grids.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class VisionGraph(nn.Module):
    def __init__(self, visual, grid):
        super().__init__()
        self.visual = visual
        visual.config._attn_implementation = "eager"
        with torch.no_grad():
            self.register_buffer("position", visual.fast_pos_embed_interpolate(grid))
            rotary = visual.rot_pos_emb(grid)
            emb = torch.cat((rotary, rotary), dim=-1)
            self.register_buffer("cosine", emb.cos())
            self.register_buffer("sine", emb.sin())
        self.register_buffer("cu_seqlens", torch.tensor([0, int(grid.prod())], dtype=torch.int64))

    def forward(self, pixel_values):
        hidden = self.visual.patch_embed(pixel_values) + self.position
        deepstack = []
        for index, block in enumerate(self.visual.blocks):
            hidden = block(hidden, cu_seqlens=self.cu_seqlens, position_embeddings=(self.cosine, self.sine))
            if index in self.visual.deepstack_visual_indexes:
                merger = self.visual.deepstack_merger_list[self.visual.deepstack_visual_indexes.index(index)]
                deepstack.append(merger(hidden))
        return (self.visual.merger(hidden), *deepstack)


def load_visual(path):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = AutoConfig.from_pretrained(path, local_files_only=True).vision_config
    config._attn_implementation = "eager"
    with torch.device("meta"):
        visual = Qwen3VLVisionModel(config)
    state = {}
    files = sorted(path.glob("model*.safetensors"))
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith("model.visual."):
                    state[key.removeprefix("model.visual.")] = source.get_tensor(key).half()
    visual.load_state_dict(state, strict=True, assign=True)
    # Rotary frequency is a non-persistent buffer and therefore absent from weights.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding

    visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)
    return visual.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=Path(".cache/microduck_onnx/merged_model"))
    parser.add_argument("--output", type=Path, default=Path(".cache/microduck_onnx/model/vision.onnx"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    grid = torch.tensor([[1, 16, 20]], dtype=torch.int64)
    visual = load_visual(args.merged)
    graph = VisionGraph(visual, grid).eval()
    if args.image:
        from PIL import Image
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            args.merged, local_files_only=True, min_pixels=65536, max_pixels=98304
        )
        with Image.open(args.image) as image:
            processed = processor.image_processor(images=[image.convert("RGB")], return_tensors="pt")
        torch.testing.assert_close(processed["image_grid_thw"], grid)
        pixels = processed["pixel_values"].half()
    else:
        torch.manual_seed(127)
        pixels = torch.randn(320, 1536, dtype=torch.float16)
    with torch.inference_mode():
        reference, deepstack = visual(pixels, grid)
        reference = (reference, *deepstack)
        wrapped = graph(pixels)
        for original, actual in zip(reference, wrapped, strict=True):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        if args.device != "cpu":
            graph.to(args.device)
            pixels = pixels.to(args.device)
            reference = graph(pixels)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.verify_only:
            torch.onnx.export(
                graph,
                (pixels,),
                str(args.output),
                input_names=["pixel_values"],
                output_names=["image_embeds", "deepstack_0", "deepstack_1", "deepstack_2"],
                opset_version=18,
                dynamo=False,
                external_data=True,
            )
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    providers = ["CUDAExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=providers)
    if session.get_providers()[0] != providers[0]:
        raise RuntimeError("Requested ONNX execution provider did not load; check CUDA libraries.")
    actual = session.run(None, {"pixel_values": pixels.cpu().numpy()})
    errors = []
    for expected, result in zip(reference, actual, strict=True):
        expected = expected.float().cpu().numpy()
        np.testing.assert_allclose(result.astype(np.float32), expected, rtol=0.03, atol=0.15)
        errors.append(float(np.max(np.abs(expected - result))))
    report = {
        "status": "complete",
        "merged_model": str(args.merged.resolve()),
        "grid_thw": [1, 16, 20],
        "camera_rgb_size": [320, 240],
        "input_shape": [320, 1536],
        "output_shapes": [list(x.shape) for x in actual],
        "hf_wrapper_exact_match": True,
        "onnx_max_abs_errors": errors,
        "validation_image": str(args.image) if args.image else None,
        "dtype": "float16",
        "verification_device": args.device,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
