"""Tests for the matcher's features and its split."""

import numpy as np

import match


def rows(left_title, right_title, left_category="Poetry", right_category="Poetry",
         left_price=10.0, right_price=10.0):
    left = {"key": left_title, "category": left_category, "price_listing_gbp": left_price}
    right = {"key": right_title, "category": right_category, "price_gbp": right_price}
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


def test_split_never_puts_a_right_record_in_both_folds():
    """This is the test that stops the numbers being flattering nonsense."""
    groups = np.array([f"R{index // 5}" for index in range(200)])
    train_index, test_index = match.grouped_split(groups, seed=0, test_fraction=0.3)
    assert set(groups[train_index]) & set(groups[test_index]) == set()
    assert len(train_index) + len(test_index) == len(groups)

def test_ablation_groups_only_name_real_features():
    """A misspelled feature name would silently change the experiment."""
    import ablate

    for name, features in ablate.GROUPS.items():
        assert features, f"{name} is empty"
        for feature in features:
            assert feature in match.FEATURES, f"{name} names unknown feature {feature!r}"