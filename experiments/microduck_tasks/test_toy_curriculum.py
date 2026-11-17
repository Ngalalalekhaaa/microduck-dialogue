"""Review the requested semantics before any real data collection starts."""

import copy
import json
from collections import defaultdict

import pytest

from .toy_catalog import SELF_ACTIONS
from .toy_curriculum import SELF_PHRASES, build_curriculum, families
from .toy_fixtures import observed_ball, request
from .toy_teacher import ToyTeacher


def test_curriculum_is_finite_serializable_and_groups_all_paraphrases():
    episodes = build_curriculum()
    assert len(episodes) == len(families()) * 6
    assert 500 <= len(episodes) <= 800
    assert json.loads(json.dumps(episodes, ensure_ascii=False)) == episodes
    assert len({episode["episode_id"] for episode in episodes}) == len(episodes)
    assert len({episode["seed"] for episode in episodes}) == len(episodes)
    grouped = defaultdict(list)
    for episode in episodes:
        assert episode["metadata"]["training_ready"] is False
        assert "request" not in episode and "media" not in episode
        assert len(episode["commands"]) == len(episode["expected"])
        grouped[episode["split_group"]].append(episode)
    for group in grouped.values():
        assert len(group) == 6
        assert len({tuple(episode["commands"]) for episode in group}) == 6
        assert len({episode["metadata"]["family"] for episode in group}) == 1


def test_curriculum_covers_skills_queue_lengths_controls_and_negative_cases():
    episodes = build_curriculum()
    goals = [goal for ep in episodes for output in ep["expected"] for goal in output["goals"]]
    assert {goal["action"] for goal in goals if goal["type"] == "perform"} == set(SELF_ACTIONS)
    assert {goal["type"] for goal in goals} == {"perform", "move", "turn", "navigate", "kick", "posture"}
    lengths = {len(output["goals"]) for ep in episodes for output in ep["expected"] if output["decision"] == "replace"}
    assert lengths == {1, 2, 3, 4, 5, 6, 7}
    assert {output["decision"] for ep in episodes for output in ep["expected"]} == {
        "respond",
        "replace",
        "append",
        "cancel",
        "pause",
        "resume",
        "decline",
        "clarify",
    }
    assert {ep["kind"] for ep in episodes} == {"ordinary", "continuous_control", "append_cancel", "missing_target"}


def test_seven_supported_actions_keep_all_reviewed_paraphrases_as_complete_positive_plans():
    episodes = build_curriculum()
    actions = ("nod", "look_around", "open_beak", "close_beak", "body_sway", "groove", "caught_dancing")
    expected = [{"type": "perform", "action": action, "repeat": 1} for action in actions]
    seven_goal_cases = [episode for episode in episodes if episode["metadata"]["family"] == "sequence_seven_supported"]
    assert len(seven_goal_cases) == 6
    assert len({episode["split_group"] for episode in seven_goal_cases}) == 1
    assert all(episode["metadata"]["family"] != "decline_too_many_goals" for episode in episodes)
    for variant, episode in enumerate(seven_goal_cases):
        original_phrase = "先" + "，再".join(SELF_PHRASES[action][variant] for action in actions)
        assert episode["commands"][1] == original_phrase
        assert episode["expected"][1] == {"decision": "replace", "goals": expected}
        assert episode["metadata"]["training_ready"] is False


@pytest.mark.parametrize("episode", build_curriculum(), ids=lambda episode: episode["episode_id"])
@pytest.mark.parametrize("visible", [(), ("red",), ("blue",), ("red", "yellow", "blue")])
def test_every_reviewed_paraphrase_matches_independently_authored_goals(episode, visible):
    state = request("", planned=True)
    state["observation"]["entities"] = [observed_ball(color) for color in visible]
    teacher = ToyTeacher()
    for text, expected in zip(episode["commands"], episode["expected"], strict=True):
        state["event"]["text"] = text
        actual = teacher.decide(state)
        assert {"decision": actual["decision"], "goals": actual["goals"]} == expected, (text, actual)
        state["history"] = (
            state["history"] + [{"role": "user", "text": text}, {"role": "assistant", "text": actual["reply"]}]
        )[-12:]
        execution = state["execution"]
        decision = expected["decision"]
        if decision == "replace":
            if episode["kind"] in ("continuous_control", "append_cancel"):
                execution.update(
                    state="running",
                    active=copy.deepcopy(expected["goals"][0]),
                    pending=copy.deepcopy(expected["goals"][1:]),
                )
            else:
                for goal in expected["goals"]:
                    if goal["type"] == "posture":
                        state["observation"]["posture"] = goal["pose"]
        elif decision == "append":
            execution["pending"].extend(copy.deepcopy(expected["goals"]))
        elif decision == "pause":
            execution["state"] = "paused"
        elif decision == "resume":
            execution["state"] = "running"
        elif decision == "cancel":
            execution.update(state="idle", active=None, pending=[])


@pytest.mark.parametrize("repeats", [0, 3, 7, True, 4.5])
def test_unreviewed_or_infinite_variations_are_not_silently_generated(repeats):
    with pytest.raises(ValueError, match="reviewed"):
        build_curriculum(repeats=repeats)


def test_determinism_and_seed_independence():
    assert build_curriculum(repeats=4, base_seed=7) == build_curriculum(repeats=4, base_seed=7)
    left, right = build_curriculum(base_seed=7), build_curriculum(base_seed=10000)
    assert {ep["seed"] for ep in left}.isdisjoint(ep["seed"] for ep in right)
    assert [ep["split_group"] for ep in left] == [ep["split_group"] for ep in right]


def test_curriculum_does_not_use_a_teacher_or_fixture_to_construct_labels():
    import inspect

    from . import toy_curriculum

    source = inspect.getsource(toy_curriculum)
    assert "ToyTeacher" not in source
    assert "toy_fixtures" not in source
    assert 'training_ready": False' in source
