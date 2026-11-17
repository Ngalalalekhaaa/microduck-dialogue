"""Collection identifiers can merge across seeded batches without family leakage."""

from collections import defaultdict

from .collect_toy_bulk import _collection_specs
from .toy_curriculum import build_curriculum
from .toy_dataset import _split_episodes


def test_new_seeded_batches_have_disjoint_ids_and_preserve_curriculum_semantics():
    first = _collection_specs(repeats=6, base_seed=410000)
    second = _collection_specs(repeats=6, base_seed=510000)
    assert {spec["episode_id"] for spec in first}.isdisjoint(spec["episode_id"] for spec in second)
    assert len({spec["episode_id"] for spec in first + second}) == len(first) + len(second)
    originals = build_curriculum(repeats=6, base_seed=410000)
    for spec, original in zip(first, originals, strict=True):
        assert spec == dict(original, episode_id=f"seed410000_{original['episode_id']}")
    assert first == _collection_specs(repeats=6, base_seed=410000)
    assert {spec["episode_id"] for spec in first}.isdisjoint(spec["episode_id"] for spec in originals)


def test_merged_batches_keep_each_semantic_family_in_one_split():
    first = _collection_specs(repeats=6, base_seed=410000)
    second = _collection_specs(repeats=6, base_seed=510000)
    assert [spec["split_group"] for spec in first] == [spec["split_group"] for spec in second]
    splits = _split_episodes([(spec, []) for spec in first + second], three_way=True)
    families = defaultdict(set)
    for spec in first + second:
        families[spec["split_group"]].add(splits[spec["episode_id"]])
    assert set(splits.values()) == {"train", "val", "test"}
    assert all(len(assigned) == 1 for assigned in families.values())
