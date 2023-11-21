"""Guards on the generated benchmark itself.

A benchmark can be broken in a way that makes a model look perfect. If every
right-hand record has a correct answer, "pick the best candidate" scores 1.000
without ever making a rejection and the threshold is never exercised. That
happened. These tests are why it will not happen again silently.
"""

from pathlib import Path

import pandas as pd

DATA = Path("data")


def load():
    for name in ("books_left.csv", "books_right.csv", "truth.csv"):
        assert (DATA / name).exists(), f"{name} missing — run perturb.py first"
    return (
        pd.read_csv(DATA / "books_left.csv"),
        pd.read_csv(DATA / "books_right.csv"),
        pd.read_csv(DATA / "truth.csv"),
    )


def test_some_right_records_have_no_correct_answer():
    """Without these, a one-to-one policy's precision is not measurable."""
    _, right, truth = load()
    unmatched = set(right["right_id"]) - set(truth["right_id"])
    assert len(unmatched) >= 50, f"only {len(unmatched)} right records lack a partner"


def test_some_left_records_have_no_correct_answer():
    left, _, truth = load()
    unmatched = set(left["upc"]) - set(truth["left_id"])
    assert len(unmatched) >= 50, f"only {len(unmatched)} left records lack a partner"


def test_truth_is_one_to_one():
    """The one-to-one decision policy is only valid if the truth is one-to-one."""
    _, _, truth = load()
    assert truth["left_id"].is_unique
    assert truth["right_id"].is_unique


def test_neither_side_is_the_whole_catalogue():
    left, right, truth = load()
    assert len(left) > len(truth), "left must hold records with no partner"
    assert len(right) > len(truth), "right must hold records with no partner"


def test_every_truth_id_exists_on_both_sides():
    left, right, truth = load()
    assert set(truth["left_id"]) <= set(left["upc"])
    assert set(truth["right_id"]) <= set(right["right_id"])