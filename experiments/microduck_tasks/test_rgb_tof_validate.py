"""Static validation retains failures and declares layouts before observations."""

import json
from types import SimpleNamespace

import pytest

from . import rgb_tof_validate as module


def test_protocol_precedes_cases_and_existing_evidence_cannot_be_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "validation"
    calls = []

    def fake_case(path, index, x, y, args):
        protocol = json.loads((output / "predeclared_protocol.json").read_text())
        assert protocol["layouts"][index]["red"] == [x, y]
        assert "rgb_tof_validate.py" in protocol["source_sha256"]
        calls.append(index)
        if index == 0:
            return dict(index=index, status="failed", error="simulated startup failure")
        return dict(
            index=index,
            status="completed",
            evaluation=dict(red_detected=index != 1, euclidean_error_mm=2 * index, tilt_absolute_error_deg=0.1),
            missing_ranges=dict(red_rejected=True),
            ranges_shifted_plus_0_3m=dict(red_rejected=index != 2),
        )

    monkeypatch.setattr(module, "_case", fake_case)
    args = SimpleNamespace(output=output, seed=17, noise=0.002, dropout=0.0)
    module.run(args)
    summary = json.loads((output / "summary.json").read_text())
    assert calls == list(range(8))
    assert summary["completed_cases"] == 7
    assert summary["red_detected_cases"] == 6
    assert summary["missing_ranges_rejected_cases"] == 7
    assert summary["shifted_ranges_rejected_cases"] == 6
    assert summary["max_position_error_mm"] == 14
    assert summary["failures"] == [dict(index=0, error="simulated startup failure")]
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.run(args)
    assert (output / "summary.json").read_bytes() == before


def test_controller_truth_guard_refuses_access():
    with pytest.raises(AssertionError, match="truth"):
        module._forbidden_truth()
