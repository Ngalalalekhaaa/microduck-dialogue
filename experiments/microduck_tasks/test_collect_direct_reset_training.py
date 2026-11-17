"""A reset scene makes the requested language necessary for action selection."""

from .collect_direct_reset_training import make_reset_plan
from .language import parse_instruction


def test_reset_pairs_all_six_actions_with_each_scene_without_split_leakage():
    splits, plan = make_reset_plan()
    assert [len(splits[split]) for split in ("train", "val", "test")] == [10, 3, 3]
    assert len(plan) == len({case["case_id"] for case in plan}) == 96
    assert all(3000 <= case["seed"] <= 3015 for case in plan)
    for seed in range(3000, 3016):
        cases = [case for case in plan if case["seed"] == seed]
        assert len({case["split"] for case in cases}) == 1
        assert {case["expected_skill"] for case in cases} == {
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
            "turn_left",
            "turn_right",
        }
        for case in cases:
            assert len(case["instructions"]) == 1
            intent = parse_instruction(case["instructions"][0])
            assert f"{intent.kind}_{intent.direction}" == case["expected_skill"]
            assert case["seed"] in splits[case["split"]]
