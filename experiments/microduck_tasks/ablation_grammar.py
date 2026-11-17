"""Structural constraints derived from the protocol; original validators stay authoritative."""

from .ablation_contract import plan_schema
from .toy_protocol import DEFAULT_MAX_GOALS


def grammar_schema(*, max_goals=DEFAULT_MAX_GOALS):
    """Remove the nonblank regex unsupported with length bounds by XGrammar 0.2.9.

    Its regex lowering treats ``\\S`` as a whole string match, rejecting legitimate
    multi-character attributes. Keep the length bounds, and check nonblank strings
    with the unmodified host validator after decoding. Never rewrite model output.
    """
    result = plan_schema(max_goals=max_goals)

    def walk(value):
        if isinstance(value, dict):
            if "pattern" in value:
                if value["pattern"] != r"\S":
                    raise ValueError("A new protocol regex needs explicit grammar compatibility review.")
                del value["pattern"]
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(result)
    return result


def compile_grammar(processor, config, stop_ids, *, max_whitespace_cnt=None, max_goals=DEFAULT_MAX_GOALS):
    import xgrammar as xgr

    if max_whitespace_cnt is not None and (type(max_whitespace_cnt) is not int or max_whitespace_cnt < 1):
        raise ValueError("max_whitespace_cnt must be a positive integer or None.")
    info = xgr.TokenizerInfo.from_huggingface(
        processor.tokenizer, vocab_size=config.text_config.vocab_size, stop_token_ids=stop_ids
    )
    return xgr.GrammarCompiler(info).compile_json_schema(
        grammar_schema(max_goals=max_goals), any_order=True, max_whitespace_cnt=max_whitespace_cnt
    )
