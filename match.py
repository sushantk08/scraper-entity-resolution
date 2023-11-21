"""Decide which candidate pairs are the same book, and measure how well.

    python match.py

Two things make the numbers here trustworthy rather than flattering:

  1. The split groups by right_id, so every candidate for one right-hand record
     lands in the same fold. Without that, the model trains on near-copies of
     the rows it is tested on.
  2. Everything is compared against a baseline that thresholds the similarity
     score alone. A classifier that cannot beat one number is not earning its
     place in the pipeline.
"""

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

FEATURES = [
    "cosine",
    "sequence_ratio",
    "word_jaccard",
    "length_ratio",
    "same_category",
    "price_relative_diff",
]


def pair_features(left_row, right_row, cosine):
    left_key, right_key = left_row["key"], right_row["key"]
    left_words, right_words = set(left_key.split()), set(right_key.split())
    union = left_words | right_words

    left_price = float(left_row["price_listing_gbp"])
    right_price = float(right_row["price_gbp"])
    largest = max(left_price, right_price, 0.01)

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
    }


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


def fit_scorer(train, features):
    """Train on the training fold; return a function that scores any frame."""
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(train[features]), train["is_match"])

    def score_rows(frame):
        return model.predict_proba(scaler.transform(frame[features]))[:, 1]

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
          f"({len(truth_pairs) - positives} true pairs never reach the matcher)\n")

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

    scaler = StandardScaler().fit(train[FEATURES])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(train[FEATURES]), y_train)
    train_scores = model.predict_proba(scaler.transform(train[FEATURES]))[:, 1]
    test_scores = model.predict_proba(scaler.transform(test[FEATURES]))[:, 1]
    tuned = best_threshold(y_train, train_scores)

    print("\nlogistic regression")
    for label, threshold in (("at 0.5", 0.5), (f"at {tuned:.3f} (tuned on train)", tuned)):
        precision, recall, f1 = evaluate(y_test, test_scores, threshold)
        print(f"  {label:<28} precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}")

    print("\n  coefficients (NOT importance — features are correlated; see ablate.py):")
    for name, weight in sorted(
        zip(FEATURES, model.coef_[0]), key=lambda item: -abs(item[1])
    ):
        print(f"    {name:<22}{weight:+.2f}")

    # Of all true pairs in the test fold, how many came out the far end?
    predicted = test_scores >= 0.5
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


if __name__ == "__main__":
    main()