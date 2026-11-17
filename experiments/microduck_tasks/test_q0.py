"""Audit SFT boundaries and prove selective logits supervise the complete answer."""

import copy

import pytest
import torch

from .q0_data import choose, encode


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(".cache/microduck_head/model", local_files_only=True)


def test_full_answer_and_eos_are_supervised_but_no_prefix_tokens(processor):
    from .h0_data import generate

    rows = generate("train")
    for group in ("single", "multi3", "multi6", "boundary", "interaction"):
        row = next(r for r in rows if r["group"] == group)
        encoded = encode(processor, row)
        n = encoded["input_tokens"]
        assert encoded["labels"][:n] == [-100] * n
        assert encoded["labels"][n:] == encoded["answer_ids"]
        assert encoded["answer_ids"][-1] == processor.tokenizer.eos_token_id
        assert encoded["padded_tokens"] >= encoded["total_tokens"]
        different = copy.deepcopy(row)
        different["expected"] = dict(decision="respond", goals=[])
        assert encode(processor, different)["prefix_ids"] == encoded["prefix_ids"]


def test_selected_logit_loss_and_gradients_equal_full_shifted_masked_loss():
    torch.manual_seed(17)
    prefix, total, padded, vocab = 8, 14, 16, 19
    logits = torch.randn(1, padded, vocab, requires_grad=True)
    labels = torch.full((1, padded), -100, dtype=torch.long)
    labels[0, prefix:total] = torch.randint(0, vocab, (total - prefix,))
    full = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab), labels[:, 1:].reshape(-1))
    grad_full = torch.autograd.grad(full, logits, retain_graph=True)[0]
    positions = torch.arange(prefix - 1, total - 1)
    selected = torch.nn.functional.cross_entropy(logits[0, positions], labels[0, prefix:total])
    grad_selected = torch.autograd.grad(selected, logits)[0]
    torch.testing.assert_close(selected, full)
    torch.testing.assert_close(grad_selected, grad_full)
    assert (grad_selected[0, prefix - 1 : total - 1].abs().sum(-1) > 0).all()
    assert not grad_selected[0, : prefix - 1].any() and not grad_selected[0, total - 1 :].any()


def test_q0_subset_uses_only_training_rows_with_required_coverage():
    from .h0_data import generate

    rows = generate("train")
    encoded = [dict(id=r["id"], row=r, total_tokens=100 + (100 if r["group"] == "multi6" else 0)) for r in rows]
    picked = choose(encoded)
    assert len(picked) == len({r["id"] for r in picked}) == 32
    assert {r["row"]["split"] for r in picked} == {"train"}
    assert {0, 1, 3, 6} == {len(r["row"]["expected"]["goals"]) for r in picked}
    assert {"decline", "append", "clarify", "cancel", "replace"} == {r["row"]["expected"]["decision"] for r in picked}
