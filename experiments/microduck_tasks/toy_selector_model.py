"""Shared autoregressive semantic decoder on frozen, input-only VLM vectors."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .toy_protocol import DECISIONS
from .toy_selector_labels import BOS, EOS, FIELDS, MAX_GOALS, MOTION, TOKENS, decode_goal

MODEL_VERSION = "microduck-toy-factorized-selector-1"


class SelectorModel(nn.Module):
    def __init__(self, dim=2048, hidden=64):
        super().__init__()
        self.dim, self.hidden = dim, hidden
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.15))
        self.decision = nn.Linear(hidden, len(DECISIONS))
        self.token_embedding = nn.Embedding(len(TOKENS) + 1, 24)
        self.field_embeddings = nn.ModuleDict(
            {name: nn.Embedding(len(values) + 1, 6) for name, values in FIELDS.items()}
        )
        self.cell = nn.GRUCell(hidden + 24 + 6 * len(FIELDS), hidden)
        self.token = nn.Linear(hidden, len(TOKENS))
        self.fields = nn.ModuleDict({name: nn.Linear(hidden, len(values)) for name, values in FIELDS.items()})

    def _previous(self, tokens, fields):
        values = [self.token_embedding(tokens)]
        for name in FIELDS:
            values.append(self.field_embeddings[name]((fields[name] + 1).clamp_min(0)))
        return torch.cat(values, -1)

    def _predicted_fields(self, tokens, logits):
        predictions = {name: values.argmax(-1) for name, values in logits.items()}
        perform_end = sum(name.startswith("perform:") for name in TOKENS)
        move_end = perform_end + sum(name.startswith("move:") for name in TOKENS)
        turn_end = move_end + sum(name.startswith("turn:") for name in TOKENS)
        applicability = {
            "repeat": tokens < perform_end,
            "limit_kind": (tokens >= perform_end) & (tokens < move_end),
            "seconds": (tokens >= perform_end) & (tokens < move_end) & (predictions["limit_kind"] == 0),
            "angle_deg": (tokens >= move_end) & (tokens < turn_end),
            "color": (tokens == TOKENS.index("kick")) | (tokens == TOKENS.index("navigate")),
        }
        return {name: value.masked_fill(~applicability[name], -100) for name, value in predictions.items()}

    def forward(self, features, teacher=None):
        context = self.project((features - self.mean) / self.scale)
        hidden = context
        previous = torch.full((len(features),), BOS, dtype=torch.long, device=features.device)
        fields = {name: torch.full_like(previous, -100) for name in FIELDS}
        output = {name: [] for name in ("token", *FIELDS)}
        for position in range(MAX_GOALS + 1):
            hidden = self.cell(torch.cat((context, self._previous(previous, fields)), -1), hidden)
            token_logits = self.token(hidden)
            field_logits = {name: head(hidden) for name, head in self.fields.items()}
            output["token"].append(token_logits)
            for name, values in field_logits.items():
                output[name].append(values)
            if teacher is None:
                choices = token_logits.clone()
                if position == 0:
                    choices[:, EOS] = -torch.inf
                previous = choices.argmax(-1)
                fields = self._predicted_fields(previous, field_logits)
            else:
                previous = teacher["token"][:, position].clamp_min(0)
                fields = {name: teacher[name][:, position] for name in FIELDS}
        return dict(
            decision=self.decision(context), **{name: torch.stack(values, 1) for name, values in output.items()}
        )


def decode_outputs(output):
    decisions = output["decision"].argmax(-1).cpu().tolist()
    logits = output["token"].clone()
    logits[:, 0, EOS] = -torch.inf
    tokens = logits.argmax(-1).cpu().tolist()
    fields = {name: values.argmax(-1).cpu().tolist() for name, values in output.items() if name in FIELDS}
    result = []
    for index, decision in enumerate(decisions):
        goals = []
        if DECISIONS[decision] in MOTION:
            for position, token in enumerate(tokens[index][:MAX_GOALS]):
                if token == EOS:
                    break
                goal = decode_goal(token, {name: values[index][position] for name, values in fields.items()})
                goals.append(goal)
                if goal["type"] == "move" and goal["limit"]["kind"] == "until_cancelled":
                    break
        result.append(dict(decision=DECISIONS[decision], goals=goals))
    return result


class Selector:
    def __init__(self, model, metadata):
        self.model = model.cpu().eval()
        self.metadata = metadata

    def predict(self, features, *, batch_size=128):
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.model.dim or not np.isfinite(values).all():
            raise ValueError(f"Supply finite features with shape [N,{self.model.dim}].")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        result = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                result.extend(decode_outputs(self.model(torch.from_numpy(values[start : start + batch_size]))))
        return result


def load_selector(checkpoint, *, feature_config=None, contract_sha256=None):
    path = Path(checkpoint)
    value = torch.load(path / "selector.pt" if path.is_dir() else path, map_location="cpu", weights_only=True)
    if value["version"] != MODEL_VERSION:
        raise ValueError("Use a compatible toy selector checkpoint.")
    if feature_config is not None and value["feature_config"] != feature_config:
        raise ValueError("Feature configuration differs from selector training.")
    if contract_sha256 is not None and value["contract_sha256"] != contract_sha256:
        raise ValueError("Dialogue contract differs from selector training.")
    if value["decisions"] != list(DECISIONS):
        raise ValueError("Selector decision vocabulary changed; use its original implementation.")
    if value["tokens"] != list(TOKENS) or value["fields"] != {name: list(values) for name, values in FIELDS.items()}:
        raise ValueError("Selector label vocabulary changed; use its original implementation.")
    model = SelectorModel(value["dim"], value["hidden"])
    model.load_state_dict(value["state_dict"])
    return Selector(model, {name: item for name, item in value.items() if name != "state_dict"})


def training_loss(output, labels, decision_weights, token_weights, *, feedback_weight=0.15):
    ce = torch.nn.functional.cross_entropy
    example_weights = torch.where(labels["user"], 1.0, feedback_weight)
    decision_loss = ce(output["decision"], labels["decision"], weight=decision_weights, reduction="none")
    total = (decision_loss * example_weights).sum() / example_weights.sum().clamp_min(1)
    motion = (labels["decision"] == DECISIONS.index("replace")) | (labels["decision"] == DECISIONS.index("append"))
    for name in ("token", *FIELDS):
        mask = (labels[name] != -100) & motion[:, None]
        if mask.any():
            weight = token_weights if name == "token" else None
            loss = ce(output[name][mask], labels[name][mask], weight=weight)
            total = total + loss * (1.0 if name == "token" else 0.3)
    return total
