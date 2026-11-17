"""CPU checks for the GenAI terminal-EOS sequence contract."""

import pytest

from .q1r_onnx_benchmark import sequence_contract, sequence_diagnostic


@pytest.mark.parametrize("eos", [248046, 248044])
def test_completed_cuda_sequence_may_omit_only_terminal_eos(eos):
    prefix = [101, 102]
    output = [201, 202, eos]
    assert sequence_contract(prefix, output, prefix + output, [248046, 248044], True) == "exact"
    assert (
        sequence_contract(prefix, output, prefix + output[:-1], [248046, 248044], True)
        == "terminal_eos_omitted_by_genai"
    )


@pytest.mark.parametrize(
    ("output", "sequence", "done"),
    [
        ([201, 202, 248046], [101, 102, 201, 202], False),
        ([201, 202, 203], [101, 102, 201, 202], True),
        ([201, 202, 248046], [999, 102, 201, 202], True),
        ([201, 202, 248046], [101, 102, 201, 999], True),
        ([201, 202, 248046], [101, 102, 201], True),
        ([201, 202, 248046], [101, 102, 201, 202, 248044], True),
    ],
)
def test_sequence_validation_keeps_rejecting_other_differences(output, sequence, done):
    assert sequence_contract([101, 102], output, sequence, [248046, 248044], done) is None


def test_mismatch_diagnostic_reports_lengths_and_first_difference():
    diagnostic = sequence_diagnostic([101, 102], [201, 202, 248046], [101, 102, 201, 999])
    assert diagnostic["prefix_matches"]
    assert diagnostic["expected_sequence_length"] == 5
    assert diagnostic["actual_sequence_length"] == 4
    assert diagnostic["first_difference_index"] == 3
    assert diagnostic["first_difference_output_index"] == 1
    assert diagnostic["expected_token"] == 202
    assert diagnostic["actual_token"] == 999
