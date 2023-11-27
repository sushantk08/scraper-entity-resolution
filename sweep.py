"""Is 0.997 a measurement, or an accident of one split?

    python sweep.py

Single-digit counts are what the policies differ by. There are roughly 225 right
records and 174 true pairs in a 30% test fold, so one extra false negative moves
recall from 0.994 to 0.983 and one false positive moves precision by about six
thousandths. So: refit over many splits and report a band.

Blocking and the feature table are computed once, because neither depends on the
split - only the fold assignment and the logistic fit change. That also keeps the
comparison paired, which matters more than the medians: asking how often mutual best
beats best-per-right ON THE SAME SPLIT is a much sharper question than asking whether
its median is higher.

ENTITY EXACTNESS IS MEASURED PER HELD-OUT RIGHT RECORD, not per entity. For each right
record in the test fold, its predicted component either equals its ground-truth
component or does not. That gives a denominator that does not move between policies,
and it avoids resolve.py's mistake of counting entities over records the model trained
on. One caveat remains: a held-out record's component can be dragged out of shape by a
neighbouring training-fold record, via a chain. That is a real property of clustering
rather than a flaw in the metric - the same contamination would exist in production -
but this number is not as cleanly held-out as the pair metrics beside it.

Nothing here writes to the database. resolve.py owns the entity tables.
"""

import json
import os
from itertools import combinations

import numpy as np

import decide
import match
import resolve
import store

SEEDS = [match.SEED + offset for offset in range(20)]


def plain(value):
    """numpy scalars are not JSON-serialisable and np.int64 is not a subclass of int,
    so a count raises here rather than being silently coerced to something else."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"{type(value).__name__} in the results: {value!r}")


def band(values, decimals):
    array = np.asarray(values, dtype=float)
    return (f"median {np.median(array):>7.{decimals}f}   "
            f"range {array.min():.{decimals}f} to {array.max():.{decimals}f}")


def closure_merges(components, accepted, truth_of):
    """Pairs inside an entity the matcher never accepted, and how many were right
    anyway. Components smaller than three cannot contain one: a two-record component
    has exactly one internal pair, and that pair is why it exists."""
    added = correct = 0
    for members in components:
        if len(members) < 3:
            continue
        for one, other in combinations(members, 2):
            if frozenset((one, other)) not in accepted:
                added += 1
                correct += int(truth_of[one] == truth_of[other])
    return added, correct


def main():
    conn = store.connect()
    print("what came out of the database:")
    left = resolve.read_source(
        conn, resolve.LEFT_SOURCE, "left_id", "price_listing_gbp"
    )
    right = resolve.read_source(
        conn, resolve.RIGHT_SOURCE, "right_id", "price_gbp"
    )
    truth_pairs = resolve.read_truth(left, right)
    conn.close()

    record_of_left = dict(zip(left["left_id"], left["record_id"]))
    record_of_right = dict(zip(right["right_id"], right["record_id"]))
    all_records = list(left["record_id"]) + list(right["record_id"])

    _, truth_sets = resolve.truth_components(
        truth_pairs, record_of_left, record_of_right, all_records
    )
    truth_of = {
        record_id: members for members in truth_sets for record_id in members
    }

    table = match.build_table(left, right, truth_pairs, resolve.K)
    print(f"\n{len(SEEDS)} splits, seeds {SEEDS[0]} to {SEEDS[-1]}, grouped by "
          f"right record")
    print(f"  {len(table)} candidate pairs and their features computed once; "
          f"only the fold assignment and the fit change")

    results = {name: [] for name in resolve.POLICY_ORDER}
    for seed in SEEDS:
        train_index, test_index = match.grouped_split(table["right_id"], seed=seed)
        scorer = match.fit_scorer(table.iloc[train_index])
        scores = scorer(table)
        test = table.iloc[test_index]
        test_truth = test["is_match"].to_numpy()
        held_out = {record_of_right[right_id] for right_id in test["right_id"]}

        for name in resolve.POLICY_ORDER:
            keep = decide.POLICIES[name](table, scores, resolve.THRESHOLD)
            components, accepted = resolve.resolve(
                table, keep, record_of_left, record_of_right, all_records
            )
            precision, recall, f1, false_positive, false_negative, _, _ = \
                decide.measure(test_truth, keep[test_index])
            entity_of = {
                record_id: frozenset(members)
                for members in components for record_id in members
            }
            exact = sum(
                1 for record_id in held_out
                if entity_of[record_id] == truth_of[record_id]
            )
            added, added_correct = closure_merges(components, accepted, truth_of)
            results[name].append({
                "accepted": len(accepted),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "exact": exact / len(held_out),
                "closure": added,
                "closure_correct": added_correct,
            })
        if seed == SEEDS[0]:
            print(f"  {len(held_out)} right records and "
                  f"{int(test_truth.sum())} true pairs in the first test fold")

    # README.md's sweep tables are rendered from this file by tables.py, so a
    # number in the README cannot drift from the run that produced it. Raw per-split
    # values, not the medians, so a different summary needs no re-run. Deliberately
    # no timestamp: an unchanged run must produce an unchanged file, or the diff
    # stops answering the only question worth asking of it.
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "sweep.json"), "w", newline="\n") as handle:
        json.dump(
            {
                "seeds": SEEDS,
                "k": resolve.K,
                "threshold": resolve.THRESHOLD,
                "policy_order": resolve.POLICY_ORDER,
                "write_policy": resolve.WRITE_POLICY,
                "candidate_pairs": len(table),
                "results": results,
            },
            handle,
            default=plain,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"\n  wrote results/sweep.json - {len(SEEDS)} splits x "
          f"{len(resolve.POLICY_ORDER)} policies")

    for name in resolve.POLICY_ORDER:
        runs = results[name]
        print(f"\n  {name}")
        for label, key, decimals in (
            ("accepted pairs", "accepted", 0),
            ("held-out precision", "precision", 3),
            ("held-out recall", "recall", 3),
            ("held-out F1", "f1", 3),
            ("held-out entity exact", "exact", 3),
            ("false positives", "false_positive", 0),
            ("false negatives", "false_negative", 0),
            ("closure merges", "closure", 0),
        ):
            print(f"    {label:<22}{band([run[key] for run in runs], decimals)}")
        total_added = sum(run["closure"] for run in runs)
        total_correct = sum(run["closure_correct"] for run in runs)
        print(f"    of {total_added} closure merges across all splits, "
              f"{total_correct} were correct")

    print(f"\n  paired on the same splits, against '{resolve.WRITE_POLICY}':")
    reference = results[resolve.WRITE_POLICY]
    for name in resolve.POLICY_ORDER:
        if name == resolve.WRITE_POLICY:
            continue
        for key in ("f1", "exact"):
            wins = sum(
                other[key] > mine[key]
                for other, mine in zip(results[name], reference)
            )
            losses = sum(
                other[key] < mine[key]
                for other, mine in zip(results[name], reference)
            )
            print(f"    {name} {key}: better on {wins} splits, worse on {losses}, "
                  f"tied on {len(SEEDS) - wins - losses}")


if __name__ == "__main__":
    main()