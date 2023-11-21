"""Tests for the decision policies. Hand-built scores, no model, no files."""

import numpy as np
import pandas as pd

import decide


def frame():
    # Two right records, two candidates each.
    return pd.DataFrame(
        {
            "left_id": ["L1", "L2", "L1", "L3"],
            "right_id": ["R1", "R1", "R2", "R2"],
        }
    )


def test_threshold_only_can_accept_two_matches_for_one_record():
    """The behaviour we are trying to improve on — shown, not assumed."""
    scores = np.array([0.9, 0.8, 0.1, 0.1])
    keep = decide.threshold_only(frame(), scores, 0.5)
    assert list(keep) == [True, True, False, False]


def test_best_per_right_keeps_at_most_one_per_record():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    keep = decide.best_per_right(frame(), scores, 0.5)
    assert list(keep) == [True, False, True, False]


def test_best_per_right_still_respects_the_threshold():
    """A winner that clears nothing must be rejected, not accepted by default."""
    scores = np.array([0.2, 0.1, 0.9, 0.8])
    keep = decide.best_per_right(frame(), scores, 0.5)
    assert list(keep) == [False, False, True, False]


def test_mutual_best_rejects_a_left_record_wanted_more_elsewhere():
    """L1 is R2's favourite, but L1 prefers R1 — so R2/L1 is not mutual."""
    scores = np.array([0.95, 0.10, 0.90, 0.20])
    keep = decide.mutual_best(frame(), scores, 0.5)
    assert list(keep) == [True, False, False, False]


def test_no_policy_can_exceed_the_candidates_it_was_given():
    scores = np.array([0.99, 0.99, 0.99, 0.99])
    for policy in decide.POLICIES.values():
        assert policy(frame(), scores, 0.5).sum() <= len(scores)