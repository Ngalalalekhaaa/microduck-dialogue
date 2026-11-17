"""H2 adds additive memory attention to each GRU step; decision stays global."""

import torch
from torch import nn

from .toy_selector_labels import BOS, EOS, FIELDS, MAX_GOALS
from .toy_selector_model import SelectorModel

MODEL_VERSION = "microduck-h2-attentive-gru-1"


class AttentionSelector(SelectorModel):
    def __init__(self, dim=2048, hidden=128):
        super().__init__(dim, hidden)
        self.register_buffer("memory_mean", torch.zeros(dim))
        self.register_buffer("memory_scale", torch.ones(dim))
        self.memory_key = nn.Linear(dim, hidden)
        self.memory_value = nn.Linear(dim, hidden)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.energy = nn.Linear(hidden, 1, bias=False)

    def forward(self, features, memory, mask, teacher=None):
        if mask.shape != memory.shape[:2] or not mask.any(-1).all():
            raise ValueError("Each memory row must contain at least one unmasked slot.")
        context = self.project((features - self.mean) / self.scale)
        normalised = (memory - self.memory_mean) / self.memory_scale
        keys, values = self.memory_key(normalised), self.memory_value(normalised)
        hidden = context
        previous = torch.full((len(features),), BOS, dtype=torch.long, device=features.device)
        fields = {name: torch.full_like(previous, -100) for name in FIELDS}
        output = {name: [] for name in ("token", *FIELDS)}
        for position in range(MAX_GOALS + 1):
            energies = self.energy(torch.tanh(keys + self.query(hidden)[:, None, :])).squeeze(-1)
            weights = energies.masked_fill(~mask, -torch.inf).softmax(-1)
            attended = torch.bmm(weights[:, None, :], values).squeeze(1)
            hidden = self.cell(torch.cat((attended, self._previous(previous, fields)), -1), hidden)
            token_logits = self.token(hidden)
            field_logits = {name: head(hidden) for name, head in self.fields.items()}
            output["token"].append(token_logits)
            for name, field_values in field_logits.items():
                output[name].append(field_values)
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
