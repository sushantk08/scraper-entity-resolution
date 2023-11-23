"""Tests for the matcher's features, its veto, and its split."""

import numpy as np
import pandas as pd

import match


def rows(left_title, right_title, left_category="Poetry", right_category="Poetry",
         left_price=10.0, right_price=10.0):
    left = {
        "key": left_title,
        "title": left_title,
        "category": left_category,
        "price_listing_gbp": left_price,
    }
    right = {
        "key": right_title,
        "title": right_title,
        "category": right_category,
        "price_gbp": right_price,
    }
    return left, right


def test_identical_records_score_at_the_top():
    left, right = rows("the hunger games", "the hunger games")
    features = match.pair_features(left, right, cosine=1.0)
    assert features["sequence_ratio"] == 1.0
    assert features["word_jaccard"] == 1.0
    assert features["length_ratio"] == 1.0
    assert features["same_category"] == 1.0
    assert features["price_relative_diff"] == 0.0


def test_unrelated_records_score_low():
    left, right = rows("the hunger games", "sapiens", right_category="History",
                       right_price=40.0)
    features = match.pair_features(left, right, cosine=0.0)
    assert features["word_jaccard"] == 0.0
    assert features["same_category"] == 0.0
    assert features["price_relative_diff"] > 0.5


def test_blank_category_is_unknown_not_a_mismatch_signal():
    """An empty category must not count as agreement, even against another empty."""
    left, right = rows("x y z", "x y z", left_category="", right_category="")
    assert match.pair_features(left, right, cosine=1.0)["same_category"] == 0.0


def test_price_difference_is_relative_not_absolute():
    _, cheap = rows("a", "a", right_price=1.0)
    left, _ = rows("a", "a", left_price=2.0)
    features = match.pair_features(left, cheap, cosine=1.0)
    assert np.isclose(features["price_relative_diff"], 0.5)


def test_a_bare_number_is_part_of_the_name_not_a_volume():
    """This is the test that stops the veto refusing real matches.

    'orange: The Complete Collection 1' was a genuine false positive in error
    analysis, but its trailing 1 is not a volume marker. If it were read as one,
    the veto would start firing on titles that merely contain digits.
    """
    assert match.volumes("fruits basket, vol. 3") == {3}
    assert match.volumes("book 2 of the series") == {2}
    assert match.volumes("#11 in a row") == {11}
    assert match.volumes("orange: the complete collection 1") == set()
    assert match.volumes("1984") == set()
    assert match.volumes("") == set()


def test_disagreeing_volumes_conflict_and_agreeing_ones_do_not():
    left, right = rows("fruits basket vol. 1", "fruits basket vol. 3")
    conflicting = match.pair_features(left, right, cosine=0.95)
    assert conflicting["volume_conflict"] == 1.0
    assert conflicting["volume_match"] == 0.0

    left, right = rows("fruits basket vol. 3", "fruits basket vol. 3")
    agreeing = match.pair_features(left, right, cosine=1.0)
    assert agreeing["volume_conflict"] == 0.0
    assert agreeing["volume_match"] == 1.0


def test_a_volume_stated_on_only_one_side_is_not_a_conflict():
    """Unknown is not disagreement — the same rule as blank categories."""
    left, right = rows("fruits basket vol. 3", "fruits basket")
    features = match.pair_features(left, right, cosine=0.9)
    assert features["volume_conflict"] == 0.0
    assert features["volume_one_sided"] == 1.0


def test_veto_only_lowers_scores_and_only_on_conflicts():
    frame = pd.DataFrame({"volume_conflict": [1.0, 0.0, 1.0, 0.0]})
    scores = np.array([0.91, 0.91, 0.20, 0.10])
    vetoed = match.apply_veto(frame, scores)
    assert list(vetoed) == [0.0, 0.91, 0.0, 0.10]
    assert (vetoed <= scores).all(), "a veto must never raise a score"


def test_split_never_puts_a_right_record_in_both_folds():
    """This is the test that stops the numbers being flattering nonsense."""
    groups = np.array([f"R{index // 5}" for index in range(200)])
    train_index, test_index = match.grouped_split(groups, seed=0, test_fraction=0.3)
    assert set(groups[train_index]) & set(groups[test_index]) == set()
    assert len(train_index) + len(test_index) == len(groups)


def test_ablation_groups_only_name_real_features():
    """A misspelled feature name would silently change the experiment.

    Validated against ALL_PAIR_FEATURES, not FEATURES, because ablate.py
    deliberately studies the volume signals that the model is not trained on.
    """
    import ablate

    for name, features in ablate.GROUPS.items():
        assert features, f"{name} is empty"
        for feature in features:
            assert feature in match.ALL_PAIR_FEATURES, (
                f"{name} names unknown feature {feature!r}"
            )