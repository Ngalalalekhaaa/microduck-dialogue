"""Learned skill selection with explicit, unlearned target binding and duration policy."""

import time

import numpy as np
import torch

from .head_features import FrozenFeatures, feature_config
from .plan_inputs import BOUNDARY_CONTRACT, LEGACY_CONTRACT, contract_digest
from .plan_tasks import PlanTaskSession
from .selection_plan import to_plan
from .setup_skill_head import digest
from .train_skill_head import make_head


class SkillHeadBackend:
    supports_continuation = False

    def __init__(self, checkpoint, model_dir):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        contract = saved.get("input_contract", LEGACY_CONTRACT)
        if contract not in (BOUNDARY_CONTRACT, LEGACY_CONTRACT) or (
            contract == BOUNDARY_CONTRACT and saved.get("input_contract_sha256") != contract_digest()
        ):
            raise ValueError("The trained input/executor contract changed; use its pinned source or retrain the head.")
        if saved["feature_config"] != feature_config(saved["feature_config"]["vision"]):
            raise ValueError("Feature/model/prompt configuration changed; extract features and retrain this head.")
        self.vision = saved["feature_config"]["vision"]
        self.classes, self.mean, self.scale = saved["classes"], saved["mean"], saved["scale"]
        self.head = make_head(saved["dim"], len(self.classes), saved["hidden"])
        self.head.load_state_dict(saved["state_dict"])
        self.head.eval().requires_grad_(False)
        self.features = FrozenFeatures(model_dir)
        self.name = self.model = "qwen35-frozen-skill-head"
        self.metadata = dict(
            backbone_frozen=True,
            robot_finetuned=False,
            head_trained=True,
            training_input_contract=contract,
            runtime_input_contract=BOUNDARY_CONTRACT,
            input_contract_aligned=contract == BOUNDARY_CONTRACT,
            checkpoint_sha256=digest(checkpoint),
            feature_config=saved["feature_config"],
            target_binding="rule_parser_and_rgbd_entities",
            duration="fixed",
            answers="perception_template",
            supports_continuation=False,
        )
        self.last_trace = {}

    def decide(self, request):
        started = time.monotonic()
        self.last_trace = dict(model=self.name, input_state=request.state, input_context=request.context)
        vector, timing = self.features.extract(
            request.instruction, request.state, request.context, request.images, vision=self.vision
        )
        x = torch.from_numpy(vector.astype(np.float16).astype(np.float32))
        with torch.inference_mode():
            probabilities = self.head((x - self.mean) / self.scale).softmax(-1)
        selected = self.classes[int(probabilities.argmax())]
        self.last_trace.update(
            raw_prediction=selected,
            probabilities={c: float(p) for c, p in zip(self.classes, probabilities, strict=True)},
            latency_s=time.monotonic() - started,
            feature_timing=timing,
        )
        return to_plan(selected, request)


class SkillHeadTaskSession(PlanTaskSession):
    @property
    def controller_source(self):
        return "frozen_vlm_trained_skill_head"
