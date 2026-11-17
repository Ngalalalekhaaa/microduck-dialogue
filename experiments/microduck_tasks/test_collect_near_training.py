"""Supplemental skill balancing uses independent scenes and held-out language."""

import math

from .collect_near_training import make_near_plan
from .collect_training import make_plan
from .language import parse_instruction


def test_supplemental_seed_partition_and_feet_are_predeclared():
    main_seeds, _ = make_plan()
    splits, plan = make_near_plan()
    assert [len(splits[split]) for split in ("train", "val", "test")] == [20, 5, 5]
    assert len(plan) == len({case["case_id"] for case in plan}) == 180
    assert not {seed for group in splits.values() for seed in group} & {
        seed for group in main_seeds.values() for seed in group
    }
    assert [case["expected_foot"] for case in plan[:3]] == ["left"] * 3
    assert {case["target_color"] for case in plan[:3]} == {"red", "yellow", "blue"}
    for case in plan:
        assert case["seed"] in splits[case["split"]]


def test_nearby_target_and_distractors_are_separated_and_reproducible():
    _, first = make_near_plan()
    _, second = make_near_plan()
    assert first == second
    for case in first:
        target = case["layout_world_xy_m"][case["target_color"]]
        assert 0.08 < math.hypot(*target) < 0.12
        assert (target[1] > 0) == (case["expected_foot"] == "left")
        for color, point in case["layout_world_xy_m"].items():
            if color != case["target_color"]:
                assert 0.35 <= math.hypot(*point) <= 0.6
                assert math.dist(point, target) >= 0.2
        intent = parse_instruction(case["instructions"][1])
        assert intent.kind == "kick" and intent.color == case["target_color"]
