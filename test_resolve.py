"""Tests for the resolution layer. Hand-built records, an in-memory database, no model.

The flagship test here is `test_the_audit_set_of_policies_can_actually_fail`. The
original closure audit was wrong not because its arithmetic was wrong but because it
only ever ran under a policy where closure is provably a no-op, so it could not have
reported anything else. That test runs the audit's own policy list against input
where closure does damage, and fails if the list ever gets narrowed back down.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

import decide
import resolve
import sweep

LEFT = {"L1": 1, "L2": 2}
RIGHT = {"R1": 11, "R2": 12}
ALL_RECORDS = [1, 2, 11, 12]


def candidates():
    """Three candidate pairs. L1 is the top-scoring partner for both right records,
    which is the situation one-to-one assignment exists to resolve."""
    return pd.DataFrame(
        {
            "left_id": ["L1", "L1", "L2"],
            "right_id": ["R1", "R2", "R1"],
        }
    )


SCORES = np.array([0.9, 0.8, 0.1])

# L1 and R1 are the same book. R2 and L2 match nothing.
TRUTH_PAIRS = {("L1", "R1")}


def truth_of():
    entity_of, _ = resolve.truth_components(TRUTH_PAIRS, LEFT, RIGHT, ALL_RECORDS)
    return entity_of


def resolved(policy):
    keep = policy(candidates(), SCORES, 0.5)
    return resolve.resolve(candidates(), keep, LEFT, RIGHT, ALL_RECORDS)


# --------------------------------------------------------------------------- union


def test_union_joins_transitively():
    union = resolve.Union([1, 2, 3])
    union.join(1, 2)
    union.join(2, 3)
    assert union.components() == [[1, 2, 3]]


def test_find_compresses_the_path_it_walked():
    """Two joined trees leave a two-step path; find must flatten it, and must not
    change which component anything belongs to while doing so."""
    union = resolve.Union([1, 2, 3, 4])
    union.join(1, 2)
    union.join(3, 4)
    union.join(2, 4)                       # 4 -> 3 -> 1
    assert union.parent[4] == 3             # the two-step path exists
    assert union.find(4) == 1
    assert union.parent[4] == 1             # now direct, not via 3
    assert union.components() == [[1, 2, 3, 4]]


def test_every_record_appears_in_exactly_one_component():
    components, _ = resolved(decide.threshold_only)
    flattened = [record for members in components for record in members]
    assert sorted(flattened) == sorted(ALL_RECORDS)
    assert len(flattened) == len(set(flattened))


def test_a_record_that_matched_nothing_still_gets_an_entity():
    """A singleton is a resolution decision, not a record that fell out of the
    pipeline. Dropping them would silently shrink the catalogue."""
    keep = np.array([False, False, False])
    components, accepted = resolve.resolve(
        candidates(), keep, LEFT, RIGHT, ALL_RECORDS
    )
    assert accepted == set()
    assert sorted(components) == [[1], [2], [11], [12]]


# ---------------------------------------------------------------------- truth sets


def test_ground_truth_is_the_components_of_the_labels_not_the_labels():
    """L1-R1 and L1-R2 are both labelled, so R1 and R2 are the same book as each
    other even though no label says so. Judging merges against the rows instead of
    the components would score that pair wrong."""
    entity_of, sets = resolve.truth_components(
        {("L1", "R1"), ("L1", "R2")}, LEFT, RIGHT, ALL_RECORDS
    )
    assert entity_of[11] == entity_of[12]
    assert frozenset({1, 11, 12}) in sets


def test_a_record_in_no_label_is_a_ground_truth_singleton():
    entity_of, sets = resolve.truth_components(
        TRUTH_PAIRS, LEFT, RIGHT, ALL_RECORDS
    )
    assert entity_of[2] != entity_of[1]
    assert frozenset({2}) in sets
    assert frozenset({12}) in sets


# -------------------------------------------------------------------- closure cost


def test_mutual_best_makes_closure_a_no_op():
    """Not a fact about the data: mutual best accepts a pair only when each record
    is the other's best, so no record is in two accepted pairs, so the accepted set
    is already disjoint. Closure over a matching is the identity function."""
    components, accepted = resolved(decide.mutual_best)
    added, correct = sweep.closure_merges(components, accepted, truth_of())
    assert accepted == {frozenset((1, 11))}
    assert (added, correct) == (0, 0)


def test_a_permissive_policy_merges_a_pair_nobody_accepted():
    """Threshold-only lets L1 win both right records, which welds R1 and R2 into one
    entity - a right-right pair that blocking never even generated, because blocking
    only ever compares across sources."""
    components, accepted = resolved(decide.threshold_only)
    added, correct = sweep.closure_merges(components, accepted, truth_of())
    assert sorted(components) == [[1, 11, 12], [2]]
    assert frozenset((11, 12)) not in accepted
    assert added == 1


def test_the_merge_closure_invented_here_is_wrong():
    """The measured result on the real benchmark was 1,630 implied merges and zero
    correct. The mechanism is this one, in miniature."""
    components, accepted = resolved(decide.threshold_only)
    _, correct = sweep.closure_merges(components, accepted, truth_of())
    assert correct == 0


def test_a_two_record_entity_cannot_contain_a_closure_merge():
    """Its single internal pair is the pair that created it, so counting it would
    inflate the cost of closure by one per accepted pair."""
    added, _ = sweep.closure_merges(
        [[1, 11]], {frozenset((1, 11))}, truth_of()
    )
    assert added == 0


def test_the_audit_set_of_policies_can_actually_fail():
    """The original audit ran under the shipping policy alone and reported zero
    merges, which that policy guarantees regardless of input. At least one policy in
    POLICY_ORDER has to be capable of producing a merge, or the audit is unfalsifiable.
    """
    assert resolve.WRITE_POLICY in resolve.POLICY_ORDER
    merges = []
    for name in resolve.POLICY_ORDER:
        components, accepted = resolved(decide.POLICIES[name])
        added, _ = sweep.closure_merges(components, accepted, truth_of())
        merges.append(added)
    assert max(merges) > 0, "no policy in POLICY_ORDER can produce a closure merge"


# --------------------------------------------------------------------------- labels


def truth_file(tmp_path, monkeypatch, rows):
    path = tmp_path / "truth.csv"
    path.write_text("left_id,right_id\n" + "".join(f"{a},{b}\n" for a, b in rows))
    monkeypatch.setattr(resolve, "TRUTH_PATH", str(path))


def test_numeric_looking_ids_still_match_as_strings(tmp_path, monkeypatch):
    """Delete dtype=str from read_truth and this test fails: pandas types a numeric
    id column as int64, the ids stop comparing equal to the TEXT external_ids they
    came from, and the model trains on zero positive examples while printing a table
    of the usual shape."""
    truth_file(tmp_path, monkeypatch, [("1", "101")])
    left = pd.DataFrame({"left_id": ["1", "2"]})
    right = pd.DataFrame({"right_id": ["101", "102"]})
    assert resolve.read_truth(left, right) == {("1", "101")}


def test_labels_that_reach_no_stored_record_are_refused(tmp_path, monkeypatch):
    truth_file(tmp_path, monkeypatch, [("L9", "R9")])
    left = pd.DataFrame({"left_id": ["L1"]})
    right = pd.DataFrame({"right_id": ["R1"]})
    with pytest.raises(SystemExit):
        resolve.read_truth(left, right)


def test_partially_reachable_labels_are_refused(tmp_path, monkeypatch):
    """Half the labels landing is worse than none: the run would proceed and report
    recall against a denominator quietly missing rows the loader lost."""
    truth_file(tmp_path, monkeypatch, [("L1", "R1"), ("L2", "R2")])
    left = pd.DataFrame({"left_id": ["L1", "L2"]})
    right = pd.DataFrame({"right_id": ["R1"]})
    with pytest.raises(SystemExit):
        resolve.read_truth(left, right)


# ------------------------------------------------------------------------- database
# The schema below mirrors store.py's derived tables rather than importing them.
# A test that the two agree belongs in the test_store.py this repo does not have yet.


def schema():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE record (
            record_id   INTEGER PRIMARY KEY,
            source      TEXT,
            external_id TEXT,
            title       TEXT,
            category    TEXT,
            price       REAL
        );
        CREATE TABLE entity (
            entity_id INTEGER PRIMARY KEY,
            label     TEXT
        );
        CREATE TABLE record_entity (
            record_id INTEGER REFERENCES record (record_id),
            entity_id INTEGER REFERENCES entity (entity_id)
        );
        """
    )
    return conn


def populated():
    conn = schema()
    conn.executemany(
        "INSERT INTO record (record_id, source, external_id, title, category, price)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, resolve.LEFT_SOURCE, "L1", "Arena", "Fiction", 10.0),
            (2, resolve.LEFT_SOURCE, "L2", "Boxed Set", None, 20.0),
            (11, resolve.RIGHT_SOURCE, "R1", "Arnea", "Fiction", 11.0),
            (12, resolve.RIGHT_SOURCE, "R2", "Crown", "Poetry", 21.0),
        ],
    )
    return conn


def test_read_source_renames_the_columns_the_matcher_asks_for():
    """Storage normalises every catalogue's price to `price`; pair_features still
    speaks the CSVs' names. The adapter lives in one place on purpose."""
    frame = resolve.read_source(
        populated(), resolve.RIGHT_SOURCE, "right_id", "price_gbp"
    )
    assert list(frame["right_id"]) == ["R1", "R2"]
    assert list(frame["price_gbp"]) == [11.0, 21.0]
    assert "key" in frame.columns


def test_an_unknown_category_becomes_empty_not_nan():
    """bool(NaN) is True, so a missing category would read as a real value to the
    same_category feature. 78 of the 750 right-hand records genuinely have none."""
    frame = resolve.read_source(
        populated(), resolve.LEFT_SOURCE, "left_id", "price_listing_gbp"
    )
    assert list(frame["category"]) == ["Fiction", ""]
    assert not frame["category"].isna().any()


def test_a_missing_price_is_refused_rather_than_imputed():
    conn = populated()
    conn.execute("UPDATE record SET price = NULL WHERE record_id = 11")
    with pytest.raises(SystemExit):
        resolve.read_source(conn, resolve.RIGHT_SOURCE, "right_id", "price_gbp")


def test_a_source_the_loader_never_wrote_is_refused():
    with pytest.raises(SystemExit):
        resolve.read_source(schema(), resolve.LEFT_SOURCE, "left_id", "price")


def test_writing_entities_twice_leaves_the_same_tables():
    """Derived data: a bad matching run is a re-run of resolve.py, never a re-scrape.
    This also pins the DELETE order - record_entity holds the foreign keys and
    PRAGMA foreign_keys is ON, so clearing entity first would fail on the second run.
    """
    conn = populated()
    titles = {1: "Arena", 2: "Boxed Set", 11: "Arnea", 12: "Crown"}
    components = [[1, 11], [2], [12]]

    def counts():
        return (
            conn.execute("SELECT count(*) FROM entity").fetchone()[0],
            conn.execute("SELECT count(*) FROM record_entity").fetchone()[0],
        )

    resolve.write_entities(conn, components, titles)
    first = counts()
    resolve.write_entities(conn, components, titles)
    assert counts() == first == (3, 4)


def test_every_record_in_a_component_gets_a_row():
    conn = populated()
    titles = {1: "Arena", 2: "Boxed Set", 11: "Arnea", 12: "Crown"}
    resolve.write_entities(conn, [[1, 11, 12], [2]], titles)
    rows = conn.execute(
        "SELECT record_id, entity_id FROM record_entity ORDER BY record_id"
    ).fetchall()
    assert [record_id for record_id, _ in rows] == [1, 2, 11, 12]
    assert len({entity_id for _, entity_id in rows}) == 2
