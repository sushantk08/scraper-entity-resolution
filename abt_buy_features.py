"""Does any feature past character similarity earn its place on Abt-Buy?

    python abt_buy_features.py

One run of match_abt_buy.py put all ten features (F1 0.688) BELOW every subset of
them, including text-only (0.700) and all-minus-description_known (0.702). Seven
of seven subsets beat the full model, which is either a real finding or the trap
this repo has already hit twice: reading one split's noise as an effect. A 0.014
F1 delta on 3,280 pairs is exactly the size of thing that turned out to be tuner
noise in step 36.

So hold the candidate table and the feature code fixed, resample the grouped
train/test split 20 times, and ask how often each subset beats the full set. A
feature group that wins on 10 of 20 splits is noise however large its mean delta
looks. The sign test here is over splits, which ARE independent draws - unlike the
seven nested subsets, which share one split and cannot be tested against each other.

Imputation is refit inside the loop. Computing the fill values once outside it
would leak every test fold's mean into every comparison.

Reports discordant DECISIONS as well as F1, per the rule from step 36: 0.977 ->
0.983 sounded like a result and was 6 changed decisions split 4-2.
"""

import json
from math import comb

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import abt_buy
import decide
import match
import match_abt_buy as mab

SPLITS = 20
SEED = 20260827
TEST_FRACTION = 0.3
THRESHOLD = mab.THRESHOLD
REFERENCE = "all features"
RESULTS = "results/abt_features.json"
# summarise() is called once per layer and writes its rows in here, so the
# call sites stay two lines long instead of growing an assignment.
LAYERS = {}


def impute(train, test, columns):
    """Same as mab.impute_from_train, without the per-split printing."""
    train, test = train.copy(), test.copy()
    for column in columns:
        fill = train[column].mean()
        fill = 0.0 if np.isnan(fill) else fill
        train[column] = train[column].fillna(fill)
        test[column] = test[column].fillna(fill)
    return train, test


def sign_test(wins, losses):
    """Exact two-sided p for wins-vs-losses under a fair coin. Ties dropped."""
    total = wins + losses
    if total == 0:
        return 1.0
    gap = abs(wins - losses)
    return sum(comb(total, k) for k in range(total + 1)
               if abs(2 * k - total) >= gap) / 2 ** total


def summarise(label, results, reference_name, key=None):
    recorded = {}
    reference = np.asarray(results[reference_name], dtype=float)
    print(f"\n{label}")
    print(f"  {'feature set':<30}{'mean F1':>9}{'sd':>7}{'worst':>8}{'best':>7}"
          f"{'beats all':>11}{'p':>7}")
    for name, values in results.items():
        values = np.asarray(values, dtype=float)
        if name == reference_name:
            wins = losses = None
            wins_column, p_column = f"{'(reference)':>11}", f"{'-':>7}"
        else:
            wins = int((values > reference + 1e-12).sum())
            losses = int((values < reference - 1e-12).sum())
            wins_column = f"{f'{wins}/{SPLITS}':>11}"
            p_column = f"{sign_test(wins, losses):>7.3f}"
        print(f"  {name:<30}{values.mean():>9.3f}{values.std(ddof=1):>7.3f}"
              f"{values.min():>8.3f}{values.max():>7.3f}{wins_column}{p_column}")
        recorded[name] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "worst": float(values.min()),
            "best": float(values.max()),
            "wins": wins,
            "losses": losses,
            "p": None if wins is None else sign_test(wins, losses),
        }
    if key is not None:
        LAYERS[key] = recorded
    return recorded


def main():
    left, right, truth = abt_buy.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    table = mab.build_table(left, right, truth_pairs, mab.K)
    print(f"\n{len(table):,} candidate pairs built once and held fixed; "
          f"resampling the grouped split {SPLITS} times")

    pair_f1 = {name: [] for name in mab.GROUPS}
    one_to_one_f1 = {name: [] for name in mab.GROUPS}
    fixed = {name: 0 for name in mab.GROUPS}
    broke = {name: 0 for name in mab.GROUPS}
    baseline_f1 = []
    decisions = 0

    splitter = GroupShuffleSplit(
        n_splits=SPLITS, test_size=TEST_FRACTION, random_state=SEED
    )
    for fold, (train_index, test_index) in enumerate(
        splitter.split(table, groups=table["right_id"]), start=1
    ):
        train, test = impute(
            table.iloc[train_index], table.iloc[test_index], mab.IMPUTED
        )
        y_train = train["is_match"].to_numpy()
        y_test = test["is_match"].to_numpy()
        decisions += len(test)

        cut = match.best_threshold(y_train, train["cosine"].to_numpy())
        baseline_f1.append(
            mab.metrics(y_test, test["cosine"].to_numpy() >= cut)[2]
        )

        keeps = {}
        for name, features in mab.GROUPS.items():
            _, scorer = mab.fit(train, features)
            scores = scorer(test)
            keeps[name] = scores >= THRESHOLD
            pair_f1[name].append(mab.metrics(y_test, keeps[name])[2])
            one_to_one_f1[name].append(
                mab.metrics(y_test, decide.mutual_best(test, scores, THRESHOLD))[2]
            )

        # Which individual decisions changed relative to the full feature set,
        # and did the change agree with the label or disagree with it?
        truth_mask = y_test == 1
        reference_keep = keeps[REFERENCE]
        for name, keep in keeps.items():
            changed = keep != reference_keep
            fixed[name] += int((changed & (keep == truth_mask)).sum())
            broke[name] += int((changed & (keep != truth_mask)).sum())

        print(f"  split {fold:>2}/{SPLITS}", end="\r")
    print(" " * 40, end="\r")

    print(f"\ncosine-threshold baseline, no model: mean F1 "
          f"{np.mean(baseline_f1):.3f} (sd {np.std(baseline_f1, ddof=1):.3f})")
    summarise("pair classification at 0.5 — the row comparable to published work:",
              pair_f1, REFERENCE, key="pair")
    summarise("under mutual-best one-to-one assignment at 0.5:",
              one_to_one_f1, REFERENCE, key="one_to_one")

    print(f"\nchanged decisions vs '{REFERENCE}', pooled over "
          f"{decisions:,} test decisions:")
    print(f"  {'feature set':<30}{'fixed':>8}{'broke':>8}{'net':>7}{'p':>8}")
    for name in mab.GROUPS:
        if name == REFERENCE:
            continue
        net = fixed[name] - broke[name]
        print(f"  {name:<30}{fixed[name]:>8}{broke[name]:>8}{net:>+7}"
              f"{sign_test(fixed[name], broke[name]):>8.3f}")

    print("\nHow to read this: a subset that wins on 10-12 of 20 splits with p > 0.05")
    print("is indistinguishable from the full feature set, and the honest claim is")
    print("that the extra features buy nothing measurable - not that they hurt.")
    # Reference first, then GROUPS order. No other curation: the README's
    # version moved "cosine alone" to the bottom, which is a judgement about
    # presentation that a recorded run should not be making.
    order = [REFERENCE] + [n for n in mab.GROUPS if n != REFERENCE]
    if sorted(order) != sorted(mab.GROUPS):
        raise SystemExit(f"order lost a group: {order}")
    for key in ("pair", "one_to_one"):
        if sorted(LAYERS.get(key, {})) != sorted(mab.GROUPS):
            raise SystemExit(f"{key} recorded {sorted(LAYERS.get(key, {}))}")
    with open(RESULTS, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "baseline": {
                    "mean": float(np.mean(baseline_f1)),
                    "sd": float(np.std(baseline_f1, ddof=1)),
                },
                "blocking_k": mab.K,
                "candidate_pairs": len(table),
                "decision_changes": {
                    name: {
                        "fixed": fixed[name],
                        "broke": broke[name],
                        "net": fixed[name] - broke[name],
                        "p": sign_test(fixed[name], broke[name]),
                    }
                    for name in mab.GROUPS
                    if name != REFERENCE
                },
                "decisions": decisions,
                "layers": LAYERS,
                "order": order,
                "reference": REFERENCE,
                "seed": SEED,
                "sizes": {name: len(f) for name, f in mab.GROUPS.items()},
                "splits": SPLITS,
                "test_fraction": TEST_FRACTION,
                "threshold": THRESHOLD,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"wrote {RESULTS} - {len(LAYERS)} layers, {len(order)} sets")


if __name__ == "__main__":
    main()