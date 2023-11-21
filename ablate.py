"""Measure what each feature group is actually worth.

    python ablate.py

Logistic-regression coefficients are not feature importance when features are
correlated — cosine, sequence_ratio and word_jaccard all measure title
similarity, so they share credit arbitrarily and signs can flip. The honest way
to ask "does this feature matter" is to remove it and re-measure.

This matters here because price agreement is an artifact of how perturb.py
generates data: it shifts prices by at most 15%, while two unrelated books can
differ by £50. A model leaning on price would score well on our data and badly
on a real catalogue, so we need to know how much of the score is title.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import block
import match

GROUPS = {
    "all features": match.FEATURES,
    "title similarity only": [
        "cosine",
        "sequence_ratio",
        "word_jaccard",
        "length_ratio",
    ],
    "everything except price": [
        feature for feature in match.FEATURES if feature != "price_relative_diff"
    ],
    "everything except category": [
        feature for feature in match.FEATURES if feature != "same_category"
    ],
    "cosine alone": ["cosine"],
    "price alone": ["price_relative_diff"],
}


def fit_and_score(train, test, features, threshold=0.5):
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(train[features]), train["is_match"])
    scores = model.predict_proba(scaler.transform(test[features]))[:, 1]
    return match.evaluate(test["is_match"].to_numpy(), scores, threshold)


def main():
    left, right, truth = block.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    table = match.build_table(left, right, truth_pairs, match.K)

    train_index, test_index = match.grouped_split(table["right_id"].to_numpy())
    train, test = table.iloc[train_index], table.iloc[test_index]
    print(f"train {len(train)} / test {len(test)} pairs, split grouped by right_id\n")

    print(f"{'feature set':<28}{'precision':>11}{'recall':>9}{'f1':>8}{'vs all':>9}")
    reference = None
    for name, features in GROUPS.items():
        precision, recall, f1 = fit_and_score(train, test, features)
        if reference is None:
            reference = f1
        print(
            f"{name:<28}{precision:>11.3f}{recall:>9.3f}{f1:>8.3f}{f1 - reference:>+9.3f}"
        )

    print("\nHow to read this: if 'title similarity only' lands close to 'all")
    print("features', the pipeline works on titles and price is a bonus. If it")
    print("lands far below — or if 'price alone' scores well — then the headline")
    print("number is mostly price agreement, which is an artifact of our")
    print("generator and would not survive on a real catalogue.")


if __name__ == "__main__":
    main()