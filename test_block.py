"""Tests for blocking. Small hand-built frames, no files."""

import pandas as pd

import block


def frames():
    left = pd.DataFrame(
        {
            "left_id": ["L1", "L2", "L3"],
            "title": ["The Hunger Games", "Sharp Objects", "Sapiens"],
        }
    )
    right = pd.DataFrame(
        {
            "right_id": ["R1", "R2"],
            # R1 is a typo of L1; R2 is L2 exactly.
            "title": ["The Hunger Gmaes", "Sharp Objects"],
        }
    )
    import clean

    left["key"] = clean.normalise_title(left["title"])
    right["key"] = clean.normalise_title(right["title"])
    return left, right


def test_exact_key_finds_only_the_identical_pair():
    left, right = frames()
    assert block.exact_key_pairs(left, right) == {("L2", "R2")}


def test_nearest_neighbours_recovers_the_typo():
    """This is the whole point of blocking on similarity rather than equality."""
    left, right = frames()
    pairs = block.nearest_neighbour_pairs(left, right, k=1)
    assert ("L1", "R1") in pairs
    assert ("L2", "R2") in pairs


def test_more_neighbours_never_lowers_recall():
    left, right = frames()
    truth = {("L1", "R1"), ("L2", "R2")}
    recalls = [
        block.score(block.nearest_neighbour_pairs(left, right, k), truth, 6)[0]
        for k in (1, 2, 3)
    ]
    assert recalls == sorted(recalls), recalls


def test_score_reports_a_real_miss():
    """A blocking scheme that finds nothing must score zero, not crash."""
    recall, reduction = block.score(set(), {("L1", "R1")}, 6)
    assert recall == 0.0
    assert reduction == 1.0