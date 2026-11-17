"""Explicit synthetic contexts and contract-checked Q1R row construction."""

import copy
import hashlib
import json
import random

from .toy_fixtures import active, history, kick, observed_ball, perform, posture, request
from .toy_inputs import validate_input
from .toy_protocol import VERSION, validate_output
from .toy_teacher import _goal_label

SOURCE = "q1r_hand_authored_synthetic_semantic_v1"
COLOR_NAMES = {"red": "红", "yellow": "黄", "blue": "蓝"}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def base_context(text, index=0, *, history_length=None, posture_name="standing", state="idle"):
    """Create a fictional snapshot, never a claim that these images were captured."""
    req = request(text, planned=True)
    req["identity"]["name"] = ("小鸭", "豆豆", "鸭鸭", "小黄鸭")[index % 4]
    req["observation"]["age_s"] = (0.0, 0.02, 0.1, 0.25, 0.49)[index % 5]
    req["observation"]["posture"] = posture_name
    req["observation"]["entities"] = [observed_ball(c, j) for j, c in enumerate(COLOR_NAMES) if (index + j) % 4]
    for j, item in enumerate(req["observation"]["entities"]):
        item["position_body_m"] = [round(0.4 + (index % 5) * 0.13 + j * 0.1, 2), round((j - 1) * 0.2, 2), 0.03]
    background = [
        ("你好呀", "你好，我在。"),
        ("你是机器人吗", "是的，我是和你互动的小鸭机器人。"),
        ("你会踢紫球吗", "当前只支持红球、黄球、蓝球，紫球还不能可靠识别。"),
        ("我刚才说的是踢方块，那个做不了对吧", "对，当前交互技能支持踢球，不支持踢方块。"),
        ("一会儿也许会让你踢黄球，现在先不做", "好，等你明确要求执行再开始。"),
        ("我还想看看蓝球，先聊一下", "可以，我们先聊，不启动新的动作。"),
    ]
    count = index % 13 if history_length is None else history_length
    all_history = history(*background)
    req["history"] = all_history[-count:] if count else []
    if state in ("running", "paused"):
        req["execution"] = active(perform("look_around"), paused=state == "paused", pending=[kick("yellow")])
        req["observation"]["stationary"] = state == "paused"
    elif state == "failed":
        req["execution"]["state"] = "failed"
        req["execution"]["last_result"] = dict(
            goal=kick("yellow"), status="failed", reason="target_lost", verified=False
        )
    req["execution"]["revision"] = index % 41
    return req


def acknowledgement(goals, index=0, *, append=False):
    labels = [_goal_label(g) for g in goals]
    prefix = ("好的，我会", "收到，接下来我会", "可以，我来", "明白，我会", "好呀，准备", "没问题，我会")[index % 6]
    if append:
        return prefix + "把" + "、".join(labels) + "接在当前任务后面。"
    return prefix + "，然后".join(labels) + "。"


class Builder:
    """Preserve independent family assignment and deduplicate complete inputs."""

    def __init__(self, seed=1611):
        self.rows = []
        self.rng = random.Random(seed)
        self.family_splits = {}
        self.seen = set()
        self.duplicate_skips = 0

    def add(
        self, family, split, group, req, decision="respond", goals=(), reply=None, rationale="", *, auto_stand=True
    ):
        if family in self.family_splits and self.family_splits[family] != split:
            raise ValueError(f"Family crossed splits: {family}")
        self.family_splits[family] = split
        req, goals = copy.deepcopy(req), copy.deepcopy(list(goals))
        if split == "val":
            # Keep the most recent referent/correction turns, while independently
            # exercising long background histories in held-out wording families.
            count = (int(hashlib.sha256(family.encode()).hexdigest()[:4], 16) + len(self.rows)) % 13
            keep = max(len(req["history"]), count)
            filler = base_context("background", history_length=12)["history"]
            req["history"] = (filler + req["history"])[-keep:] if keep else []
        if (
            auto_stand
            and decision == "replace"
            and goals
            and req["observation"]["posture"] == "sitting"
            and goals[0]["type"] != "posture"
        ):
            goals.insert(0, posture("standing"))
            rationale += " 当前坐姿已知，先加入站立前置动作，再执行本轮要求。"
        if reply is None:
            reply = acknowledgement(goals, len(self.rows), append=decision == "append")
        expected = dict(decision=decision, goals=goals, reply=reply)
        validate_input(req)
        validate_output(dict(schema_version=VERSION, **expected), req)
        key = digest(req)
        if key in self.seen:
            self.duplicate_skips += 1
            return
        self.seen.add(key)
        row = dict(
            id=f"q1r_{split}_{len(self.rows):06d}",
            split=split,
            group=group,
            family_id=family,
            request=req,
            expected=expected,
            rationale=rationale,
            source=SOURCE,
            physical_verified=False,
            human_reviewed=False,
        )
        self.rows.append(row)

    def commands(self, group, label, wordings, build, *, variants=12, val_variants=1, heldout=3):
        for p, wording in enumerate(wordings):
            split = "val" if p >= len(wordings) - heldout else "train"
            n = val_variants if split == "val" else variants
            for i in range(n):
                sample = i if split == "train" else p * 13 + i * 5
                text, goals = build(wording, sample)
                state = ("idle", "running", "paused")[i % 3]
                pose = "sitting" if i % 7 == 6 and state == "idle" else "standing"
                req = base_context(text, i, posture_name=pose, state=state)
                self.add(
                    f"{group}_{label}_{p:02d}",
                    split,
                    group,
                    req,
                    "replace",
                    goals,
                    rationale="手写话语与明确语义对应；最新完整指令替换旧队列，不从历史补入任何目标。",
                )
