"""Turn accepted pairs into entities, and price what transitive closure costs.

    python resolve.py

Reads records from the database rather than from the CSVs. That is the point of
having a storage layer: if the loader dropped or mangled anything the matcher
needs, this is where it shows up, because the same features get computed from
stored rows instead of from the file they came from.

WHAT THE FIRST VERSION OF THIS SCRIPT GOT WRONG. It audited closure under one
decision policy - mutual best - and reported that closure merged nothing: 592
accepted pairs, 592 two-record entities, zero pairs merged below the bar, zero
entities holding two records from one source. That was not a fact about the data.
Mutual best accepts a pair only when each record is the other's best available
option, so no record can appear in two accepted pairs, so the accepted pairs are
already disjoint, so union-find has nothing to join. CLOSURE OVER A MATCHING IS THE
IDENTITY FUNCTION. That audit would have printed zero on any input at all, including
deliberately broken input, which makes it a measurement with no failure mode.

So the audit now runs under all three policies. Best-per-right lets one left record
win several rights, and threshold-only lets everything match everything, and both
produce components larger than two - which is where closure starts making decisions
nobody scored. Note especially that a right-right merge is a pair blocking never even
generated, because blocking only ever compares across sources.

HOW A MERGE IS JUDGED. truth.csv labels left-right pairs, so a right-right merge looks
unlabelled and would count as wrong by default - which is not true: if L is the same
book as R1 and as R2, then R1 and R2 are the same book as each other. Ground truth is
therefore the connected components of truth.csv, not its rows, and two records are
correctly merged when they share one. Records appearing in no label are singletons in
ground truth, so merging them into anything is wrong.

No percentages in the audit, only counts. The first version printed 591 of 592 correct
as "100%", rounding a false positive out of existence.

Entities are written for one policy only. The entity counts cover every pair, including
the training fold, so they are optimistic; the per-policy precision and recall are
computed on the held-out fold, and those are the figures that mean something.

record_entity and entity are DELETEd and rebuilt on every run. They are derived data -
a bad matching run is a re-run of this script, never a re-scrape.
"""

import os
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

import clean
import decide
import match
import store

LEFT_SOURCE = "books_left"
RIGHT_SOURCE = "books_right"
TRUTH_PATH = os.path.join("data", "truth.csv")

K = match.K
THRESHOLD = decide.DEFAULT_THRESHOLD
POLICY_ORDER = ["threshold only", "best per right record", "mutual best"]
WRITE_POLICY = "mutual best"


class Union:
    """Union-find with path compression. Small enough to read in one sitting."""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:                      # compress on the way back down
            self.parent[item], item = root, self.parent[item]
        return root

    def join(self, one, other):
        first, second = self.find(one), self.find(other)
        if first != second:
            self.parent[second] = first

    def components(self):
        groups = defaultdict(list)
        for item in self.parent:
            groups[self.find(item)].append(item)
        return [sorted(members) for members in groups.values()]


def read_source(conn, source, id_column, price_column):
    """Rebuild the frame the matcher expects, from stored rows.

    The renames are the adapter between two vocabularies: storage normalises every
    catalogue's price to `price`, while pair_features still speaks the CSVs' names.
    Better to keep the adapter visible in one place than to teach the schema a
    supplier's column names.
    """
    frame = pd.read_sql_query(
        "SELECT record_id, external_id, title, category, price FROM record "
        "WHERE source = ? ORDER BY record_id",
        conn, params=(source,),
    )
    if frame.empty:
        raise SystemExit(
            f"no {source} records in {store.DB_PATH}. Run store.py first - and if it "
            f"reported that source as absent, run perturb.py before it."
        )

    blank_price = int(frame["price"].isna().sum())
    if blank_price:
        raise SystemExit(
            f"{source}: {blank_price} records have no price. pair_features would turn "
            f"those into NaN features and the model would train on them silently. Fix "
            f"the load rather than imputing here."
        )

    frame[id_column] = frame["external_id"].astype(str)
    # bool(NaN) is True, which would make "unknown" read as a real value to the
    # same_category feature. block.load() does this too, for the same reason - and
    # 78 of the 750 right-hand records genuinely have no category.
    frame["category"] = frame["category"].fillna("")
    frame["key"] = clean.normalise_title(frame["title"])
    frame[price_column] = frame["price"].astype(float)
    print(f"  {source:<12}{len(frame):>5} records from the database")
    return frame


def read_truth(left, right):
    """Labels, with a hard check that they refer to records that exist.

    dtype=str is load-bearing. external_id is TEXT in the database, and pandas would
    read a numeric right_id as int64 - so the two would never compare equal, is_match
    would be zero everywhere, and the model would train on no positives at all while
    printing a table that looks fine.
    """
    truth = pd.read_csv(TRUTH_PATH, dtype=str)
    pairs = set(zip(truth["left_id"], truth["right_id"]))
    left_ids, right_ids = set(left["left_id"]), set(right["right_id"])
    reachable = {
        pair for pair in pairs if pair[0] in left_ids and pair[1] in right_ids
    }
    print(f"  truth       {len(pairs):>5} labelled pairs, "
          f"{len(reachable)} with both records stored")
    if not reachable:
        raise SystemExit(
            "not one labelled pair maps onto the database. truth.csv ids and "
            "record.external_id values are not the same strings, which would otherwise "
            "surface as a model trained on zero positive examples."
        )
    if len(reachable) < len(pairs):
        raise SystemExit(
            f"{len(pairs) - len(reachable)} labelled pairs refer to records that are "
            f"not in the database. The load lost rows; fix that before measuring "
            f"anything against these labels."
        )
    return reachable


def truth_components(truth_pairs, record_of_left, record_of_right, all_records):
    """Ground truth is the components of the labels, not the labels themselves."""
    union = Union(all_records)
    for left_id, right_id in truth_pairs:
        union.join(record_of_left[left_id], record_of_right[right_id])
    components = union.components()
    entity_of = {}
    for index, members in enumerate(components):
        for record_id in members:
            entity_of[record_id] = index
    return entity_of, {frozenset(members) for members in components}


def resolve(table, keep, record_of_left, record_of_right, all_records):
    """Union-find over accepted pairs. Every record gets an entity, including the
    ones nothing matched - a singleton is a resolution decision too."""
    union = Union(all_records)
    accepted = set()
    for left_id, right_id in zip(
        table.loc[keep, "left_id"], table.loc[keep, "right_id"]
    ):
        one, other = record_of_left[left_id], record_of_right[right_id]
        union.join(one, other)
        accepted.add(frozenset((one, other)))
    return union.components(), accepted


def audit(name, components, accepted, candidate_scores, truth_entity, truth_sets,
          source_of, test_metrics):
    sizes = Counter(len(members) for members in components)
    shape = "  ".join(f"{size}->{count}" for size, count in sorted(sizes.items()))
    print(f"\n  {name}")
    print(f"    {len(accepted)} accepted pairs, {len(components)} entities, "
          f"sizes {shape}")

    counts, correct, same_source = Counter(), Counter(), Counter()
    rejected_scores = []
    for members in components:
        for one, other in combinations(members, 2):
            pair = frozenset((one, other))
            if pair in accepted:
                bucket = "accepted by the matcher"
            elif pair in candidate_scores:
                bucket = "scored, below the bar"
                rejected_scores.append(candidate_scores[pair])
            else:
                bucket = "never a candidate"
            counts[bucket] += 1
            correct[bucket] += int(truth_entity[one] == truth_entity[other])
            same_source[bucket] += int(source_of[one] == source_of[other])

    for bucket in ("accepted by the matcher", "scored, below the bar",
                   "never a candidate"):
        total = counts[bucket]
        if not total:
            print(f"      {bucket:<26}{0:>6}")
            continue
        print(f"      {bucket:<26}{total:>6}   correct {correct[bucket]}, "
              f"wrong {total - correct[bucket]}, same-source {same_source[bucket]}")
    if rejected_scores:
        scores = np.asarray(rejected_scores)
        print(f"        merged-but-rejected scored {scores.min():.3f} to "
              f"{scores.max():.3f}, mean {scores.mean():.3f}, against a bar of "
              f"{THRESHOLD}")

    broken = sum(
        1 for members in components
        if len(members) > len({source_of[record_id] for record_id in members})
    )
    exact = sum(1 for members in components if frozenset(members) in truth_sets)
    print(f"      entities holding 2+ records from one source: {broken}")
    print(f"      entities exactly equal to a ground-truth entity: "
          f"{exact} of {len(truth_sets)}")
    precision, recall, f1, false_positive, false_negative = test_metrics
    print(f"      held-out fold: precision {precision:.3f}  recall {recall:.3f}  "
          f"F1 {f1:.3f}   ({false_positive} FP, {false_negative} FN)")


def write_entities(conn, components, titles):
    """Rebuild the derived tables from scratch. record_entity goes first: it holds
    the foreign keys, and PRAGMA foreign_keys is ON."""
    conn.execute("DELETE FROM record_entity")
    conn.execute("DELETE FROM entity")
    for members in components:
        entity_id = conn.execute(
            "INSERT INTO entity (label) VALUES (?)", (titles[members[0]],)
        ).lastrowid
        conn.executemany(
            "INSERT INTO record_entity (record_id, entity_id) VALUES (?, ?)",
            [(record_id, entity_id) for record_id in members],
        )
    conn.commit()
    print(f"  wrote {len(components)} entities covering "
          f"{sum(len(members) for members in components)} records")


def main():
    unknown = [name for name in POLICY_ORDER if name not in decide.POLICIES]
    if unknown:
        raise SystemExit(
            f"decide.POLICIES has no {unknown}; it has {sorted(decide.POLICIES)}"
        )

    conn = store.connect()
    print("what came out of the database:")
    left = read_source(conn, LEFT_SOURCE, "left_id", "price_listing_gbp")
    right = read_source(conn, RIGHT_SOURCE, "right_id", "price_gbp")
    truth_pairs = read_truth(left, right)

    record_of_left = dict(zip(left["left_id"], left["record_id"]))
    record_of_right = dict(zip(right["right_id"], right["record_id"]))
    titles = dict(zip(left["record_id"], left["title"]))
    titles.update(zip(right["record_id"], right["title"]))
    source_of = {record_id: LEFT_SOURCE for record_id in left["record_id"]}
    source_of.update({record_id: RIGHT_SOURCE for record_id in right["record_id"]})
    all_records = list(source_of)

    truth_entity, truth_sets = truth_components(
        truth_pairs, record_of_left, record_of_right, all_records
    )
    print(f"  ground truth {len(truth_sets):>4} entities over "
          f"{len(all_records)} records")

    table = match.build_table(left, right, truth_pairs, K)
    train_index, test_index = match.grouped_split(table["right_id"])
    scorer = match.fit_scorer(table.iloc[train_index])
    scores = scorer(table)
    print(f"\nblocking and scoring (k={K}, bar {THRESHOLD}):")
    print(f"  {len(table)} candidate pairs, {int(table['is_match'].sum())} of "
          f"{len(truth_pairs)} true pairs reached the matcher")

    candidate_scores = {
        frozenset((record_of_left[left_id], record_of_right[right_id])): score
        for left_id, right_id, score in zip(
            table["left_id"], table["right_id"], scores
        )
    }
    test_truth = table.iloc[test_index]["is_match"].to_numpy()

    print("\nwhat closure costs under each decision policy:")
    chosen = None
    for name in POLICY_ORDER:
        keep = decide.POLICIES[name](table, scores, THRESHOLD)
        components, accepted = resolve(
            table, keep, record_of_left, record_of_right, all_records
        )
        precision, recall, f1, false_positive, false_negative, _, _ = decide.measure(
            test_truth, keep[test_index]
        )
        audit(name, components, accepted, candidate_scores, truth_entity, truth_sets,
              source_of, (precision, recall, f1, false_positive, false_negative))
        if name == WRITE_POLICY:
            chosen = components

    print(f"\nwriting entities for '{WRITE_POLICY}':")
    write_entities(conn, chosen, titles)
    print("  cluster counts above cover every pair, including the training fold, so")
    print("  they are optimistic. The held-out precision and recall are not.")
    conn.close()


if __name__ == "__main__":
    main()