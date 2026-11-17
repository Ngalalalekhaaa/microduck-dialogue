"""Small manifest/graph checks; no model weights, torch forward or GPU needed."""

import copy
import struct
import tempfile
import unittest
from pathlib import Path

from experiments.microduck_tasks.bundle_onnx import (
    CASES,
    decoder_names,
    graph_files,
    local_file,
    validate_decoder,
    validate_vision,
)


def decoder_report():
    inputs, outputs = decoder_names(28)
    return {
        "inputs": inputs,
        "outputs": outputs,
        "original_hf_equivalence": [
            {"sequence": sequence, "past_length": past, "exact_match": True} for sequence, past in sorted(CASES)
        ],
        "verification": [
            {
                "sequence": sequence,
                "past_length": past,
                "finite": True,
                "max_abs_error": 0.01,
                "logits_max_abs_error": 0.01,
                "pytorch_top_token": 5,
                "onnx_top_token": 5,
            }
            for sequence, past in sorted(CASES)
        ],
    }


class BundleValidationTests(unittest.TestCase):
    def test_partial_tolerance_is_explicit_and_never_reported_passed(self):
        report = decoder_report()
        self.assertTrue(validate_decoder(report)["strict_tolerance_passed"])
        report["verification"][0]["outputs"] = {"present_value_27": {"tolerance_passed": False}}
        with self.assertRaisesRegex(ValueError, "exceed tolerance"):
            validate_decoder(report)
        assessment = validate_decoder(report, allow_pending=True)
        self.assertFalse(assessment["strict_tolerance_passed"])
        self.assertEqual(assessment["status"], "partial_awaiting_full_generation")

    def test_pending_mode_still_rejects_nonfinite_and_incomplete_cases(self):
        report = decoder_report()
        report["verification"][0]["finite"] = False
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_decoder(report, allow_pending=True)
        report = decoder_report()
        report["verification"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_decoder(report, allow_pending=True)

    def test_tiny_decoder_report_cannot_be_bundled_as_full_model(self):
        report = decoder_report()
        report["inputs"], report["outputs"] = decoder_names(3)
        with self.assertRaisesRegex(ValueError, "full 28-layer"):
            validate_decoder(report, allow_pending=True)

    def test_camera_grid_and_deepstack_are_checked(self):
        merged = Path("/tmp/example_merged")
        report = {
            "status": "complete",
            "merged_model": str(merged),
            "grid_thw": [1, 16, 20],
            "camera_rgb_size": [320, 240],
            "input_shape": [320, 1536],
            "output_shapes": [[80, 2048]] * 4,
            "hf_wrapper_exact_match": True,
            "dtype": "float16",
            "onnx_max_abs_errors": [0.01] * 4,
        }
        validate_vision(report, merged)
        wrong = copy.deepcopy(report)
        wrong["grid_thw"] = [1, 20, 16]
        with self.assertRaisesRegex(ValueError, "320x240"):
            validate_vision(wrong, merged)
        wrong = copy.deepcopy(report)
        wrong["output_shapes"].pop()
        with self.assertRaisesRegex(ValueError, "DeepStack"):
            validate_vision(wrong, merged)

    def test_external_weights_are_included_and_traversal_refused(self):
        import onnx

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tensor = onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT, [2], struct.pack("ff", 1, 2), raw=True)
            graph = onnx.helper.make_graph([], "fixture", [], [], [tensor])
            model = onnx.helper.make_model(graph)
            path = directory / "fixture.onnx"
            onnx.save_model(
                model,
                path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.bin",
                size_threshold=0,
            )
            self.assertEqual(graph_files(directory, ["fixture.onnx"]), ["fixture.onnx", "weights.bin"])
            model = onnx.load(path, load_external_data=False)
            for entry in model.graph.initializer[0].external_data:
                if entry.key == "location":
                    entry.value = "../weights.bin"
            onnx.save_model(model, path)
            with self.assertRaisesRegex(ValueError, "invalid bundle"):
                graph_files(directory, ["fixture.onnx"])

    def test_symlink_to_external_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle = directory / "model"
            bundle.mkdir()
            outside = directory / "outside.bin"
            outside.write_bytes(b"test")
            (bundle / "weights.bin").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "escapes"):
                local_file(bundle, "weights.bin")


if __name__ == "__main__":
    unittest.main()
