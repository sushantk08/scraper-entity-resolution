"""Decide which candidate pairs are the same book, and measure how well.

    python match.py

Two things make the numbers here trustworthy rather than flattering:

  1. The split groups by right_id, so every candidate for one right-hand record
     lands in the same fold. Without that, the model trains on near-copies of
     the rows it is tested on.
  2. Everything is compared against a baseline that thresholds the similarity
     score alone. A classifier that cannot beat one number is not earning its
     place in the pipeline.

Volume numbers are handled as a CONSTRAINT, not a feature, and that was measured
rather than assumed (volume_feature.py). Feeding volume_conflict/match/one_sided
to the model as three extra features scored WORSE than applying one veto after
scoring — 5 false positives against 4 — because adding features means refitting,
and refitting moves the boundary on pairs that have no volume numbers at all.
The veto touches only the 134 candidate pairs whose stated volumes disagree, and
0 of those have ever been a true match.
"""

import json
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import block

K = 5
SEED = 20260826
TEST_FRACTION = 0.3
RESULTS = "results/match.json"
# Stable row keys. README.md's labels are built from these plus the two
# recorded thresholds, so a retuned threshold cannot leave a stale label
# sitting in the table the way "threshold tuned" did.
BASELINE, PLAIN, SHIPPED, TUNED = "baseline", "classifier", "shipped", "tuned"
RECORDED = {}

VOLUME_MARKER = re.compile(r"(?:vol\.?|volume|book|part|no\.?|#)\s*(\d+)", re.I)

# What the model is trained on.
FEATURES = [
    "cosine",
    "sequence_ratio",
    "word_jaccard",
    "length_ratio",
    "same_category",
    "price_relative_diff",
]

# Computed on every pair but deliberately NOT given to the model. CONSTRAINTS
# drive apply_veto(); DIAGNOSTIC exists so ablate.py can put a number on what
# they would have been worth as features. Splitting these out keeps
# test_match.py's misspelled-name guard working without letting it reject a
# feature we chose not to train on.
CONSTRAINTS = ["volume_conflict"]
DIAGNOSTIC = ["volume_match", "volume_one_sided"]
ALL_PAIR_FEATURES = FEATURES + CONSTRAINTS + DIAGNOSTIC


def title_of(row):
    """Books call it 'title', Abt-Buy calls it 'name'. Accept either, tolerate neither."""
    return str(row.get("title") or row.get("name") or "")


def volumes(title):
    """Every explicitly MARKED volume number in a title, as a set of ints.

    Only marked numbers count ('Vol. 3', 'Book 2', '#11'). A bare trailing number
    is usually part of the name — 'orange: The Complete Collection 1' and '1984'
    must both come back empty, or the veto starts refusing real matches.
    """
    return {int(number) for number in VOLUME_MARKER.findall(str(title))}


def pair_features(left_row, right_row, cosine):
    left_key, right_key = left_row["key"], right_row["key"]
    left_words, right_words = set(left_key.split()), set(right_key.split())
    union = left_words | right_words

    left_price = float(left_row["price_listing_gbp"])
    right_price = float(right_row["price_gbp"])
    largest = max(left_price, right_price, 0.01)

    left_volumes = volumes(title_of(left_row))
    right_volumes = volumes(title_of(right_row))
    both_stated = bool(left_volumes) and bool(right_volumes)
    volumes_agree = bool(left_volumes & right_volumes)

    return {
        "cosine": cosine,
        "sequence_ratio": SequenceMatcher(None, left_key, right_key).ratio(),
        "word_jaccard": len(left_words & right_words) / len(union) if union else 0.0,
        "length_ratio": min(len(left_key), len(right_key))
        / max(len(left_key), len(right_key), 1),
        # Blank on either side means "unknown", which is not "differs".
        "same_category": float(
            bool(left_row["category"])
            and bool(right_row["category"])
            and left_row["category"] == right_row["category"]
        ),
        "price_relative_diff": abs(left_price - right_price) / largest,
        # Constraint + diagnostics. Not in FEATURES on purpose.
        "volume_conflict": float(both_stated and not volumes_agree),
        "volume_match": float(both_stated and volumes_agree),
        "volume_one_sided": float(bool(left_volumes) != bool(right_volumes)),
    }


def apply_veto(frame, scores):
    """Force a score to 0 where the two titles state disagreeing volume numbers.

    Applied AFTER scoring, so it cannot disturb any pair it does not fire on —
    which is exactly why it beat the learned version. Monotone by construction:
    it can only ever lower a score, never raise one.
    """
    vetoed = np.asarray(scores, dtype=float).copy()
    vetoed[frame["volume_conflict"].to_numpy() == 1.0] = 0.0
    return vetoed


def build_table(left, right, truth_pairs, k):
    left_rows = left.set_index("left_id").to_dict("index")
    right_rows = right.set_index("right_id").to_dict("index")

    rows = []
    for left_id, right_id, cosine in block.nearest_neighbours(left, right, k):
        features = pair_features(left_rows[left_id], right_rows[right_id], cosine)
        features["left_id"] = left_id
        features["right_id"] = right_id
        features["is_match"] = int((left_id, right_id) in truth_pairs)
        rows.append(features)
    return pd.DataFrame(rows)


def grouped_split(groups, seed=SEED, test_fraction=TEST_FRACTION):
    """Split so that no group appears in both halves."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def evaluate(y_true, scores, threshold):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, scores >= threshold, average="binary", zero_division=0
    )
    return precision, recall, f1


def best_threshold(y_true, scores):
    """Pick a threshold on the TRAINING half only."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.50, 0.999, 80)))
    return max(candidates, key=lambda t: evaluate(y_true, scores, t)[2])


def fit_scorer(train, features=None):
    """Train on the training fold; return a function that scores any frame.

    The returned scorer applies the volume veto, so callers get the shipped
    configuration rather than the raw model.
    """
    features = FEATURES if features is None else features
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(train[features]), train["is_match"])

    def score_rows(frame):
        raw = model.predict_proba(scaler.transform(frame[features]))[:, 1]
        return apply_veto(frame, raw)

    return score_rows


def main():
    left, right, truth = block.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))

    table = build_table(left, right, truth_pairs, K)
    positives = int(table["is_match"].sum())
    print(f"{len(table)} candidate pairs at k={K}, {positives} of them true "
          f"({positives / len(table):.1%} positive)")

    blocking_recall = positives / len(truth_pairs)
    print(f"blocking recall  : {blocking_recall:.3f} "
          f"({len(truth_pairs) - positives} true pairs never reach the matcher)")

    conflicts = table["volume_conflict"] == 1.0
    print(f"volume conflicts : {int(conflicts.sum())} pairs the veto will refuse, "
          f"{int(table.loc[conflicts, 'is_match'].sum())} of them true\n")

    train_index, test_index = grouped_split(table["right_id"].to_numpy())
    train, test = table.iloc[train_index], table.iloc[test_index]
    overlap = set(train["right_id"]) & set(test["right_id"])
    if overlap:
        raise SystemExit(f"ERROR: {len(overlap)} right_ids in both folds — split is leaking")
    print(f"train {len(train)} pairs / test {len(test)} pairs, no shared right_id")

    y_train = train["is_match"].to_numpy()
    y_test = test["is_match"].to_numpy()

    # Baseline: the similarity score alone, threshold chosen on train.
    baseline_threshold = best_threshold(y_train, train["cosine"].to_numpy())
    precision, recall, f1 = evaluate(y_test, test["cosine"].to_numpy(), baseline_threshold)
    print(f"\nbaseline (cosine >= {baseline_threshold:.3f})")
    print(f"  precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}")
    RECORDED[BASELINE] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    scaler = StandardScaler().fit(train[FEATURES])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(train[FEATURES]), y_train)
    train_raw = model.predict_proba(scaler.transform(train[FEATURES]))[:, 1]
    test_raw = model.predict_proba(scaler.transform(test[FEATURES]))[:, 1]
    test_shipped = apply_veto(test, test_raw)
    tuned = best_threshold(y_train, apply_veto(train, train_raw))

    print("\nlogistic regression")
    for key, label, scores, threshold in (
        (PLAIN, "model alone, at 0.5", test_raw, 0.5),
        (SHIPPED, "+ volume veto, at 0.5", test_shipped, 0.5),
        (TUNED, f"+ veto, at {tuned:.3f} (tuned on train)", test_shipped, tuned),
    ):
        precision, recall, f1 = evaluate(y_test, scores, threshold)
        RECORDED[key] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        print(f"  {label:<34} precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}")
    print("  (the tuned row is kept visible because it scores WORSE than 0.5 —")
    print("   a threshold picked on one fold does not transfer to another)")

    print("\n  coefficients (NOT importance — features are correlated; see ablate.py):")
    for name, weight in sorted(
        zip(FEATURES, model.coef_[0]), key=lambda item: -abs(item[1])
    ):
        print(f"    {name:<22}{weight:+.2f}")

    # Of all true pairs in the test fold, how many came out the far end?
    predicted = test_shipped >= 0.5
    test_right_ids = set(test["right_id"])
    truth_in_test = {pair for pair in truth_pairs if pair[1] in test_right_ids}
    found = {
        (row.left_id, row.right_id)
        for row, keep in zip(test.itertuples(), predicted)
        if keep and (row.left_id, row.right_id) in truth_in_test
    }
    print(f"\nend-to-end recall on the test fold: {len(found)}/{len(truth_in_test)} "
          f"= {len(found) / len(truth_in_test):.3f}")
    print("  (blocking losses included — this is the honest headline number)")
    order = [BASELINE, PLAIN, SHIPPED, TUNED]
    unrecorded = [name for name in order if name not in RECORDED]
    if unrecorded:
        raise SystemExit(f"ERROR: nothing recorded for {unrecorded}")
    with open(RESULTS, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "k": K,
                "seed": SEED,
                "test_fraction": TEST_FRACTION,
                "candidate_pairs": int(len(table)),
                "candidates_true": positives,
                "true_pairs": len(truth_pairs),
                "blocking_recall": float(blocking_recall),
                "blocking_lost": len(truth_pairs) - positives,
                "volume_conflicts": int(conflicts.sum()),
                "volume_conflicts_true": int(table.loc[conflicts, "is_match"].sum()),
                "train_pairs": int(len(train)),
                "test_pairs": int(len(test)),
                "test_true": int(y_test.sum()),
                "baseline_threshold": float(baseline_threshold),
                "tuned_threshold": float(tuned),
                "threshold": 0.5,
                "shipped": SHIPPED,
                "order": order,
                "results": RECORDED,
                "end_to_end": {
                    "found": len(found),
                    "truth_in_test": len(truth_in_test),
                    "recall": len(found) / len(truth_in_test),
                },
                "coefficients": {
                    name: float(weight)
                    for name, weight in zip(FEATURES, model.coef_[0])
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"\nwrote {RESULTS} - {len(order)} rows")


if __name__ == "__main__":
    main()