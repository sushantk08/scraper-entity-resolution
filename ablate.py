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

The last two rows exist to justify a decision rather than to explore. The volume
signals are NOT in match.FEATURES; they are applied as a veto after scoring.
These rows show what they would have been worth as trained features, so "we chose
a constraint over a coefficient" is a measured claim and not a preference.
"""

import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import block
import match

VOLUME = match.CONSTRAINTS + match.DIAGNOSTIC
RESULTS = os.path.join("results", "ablation.json")
REFERENCE = "all features"
SHIPPED = "all features + volume veto"
THRESHOLD = 0.5

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
    "all + volume as features": match.FEATURES + VOLUME,
    "volume signals alone": VOLUME,
}


def fit_and_score(train, test, features, threshold=0.5, veto=False):
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(train[features]), train["is_match"])
    scores = model.predict_proba(scaler.transform(test[features]))[:, 1]
    if veto:
        scores = match.apply_veto(test, scores)
    return match.evaluate(test["is_match"].to_numpy(), scores, threshold)


def main():
    left, right, truth = block.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    table = match.build_table(left, right, truth_pairs, match.K)

    train_index, test_index = match.grouped_split(table["right_id"].to_numpy())
    train, test = table.iloc[train_index], table.iloc[test_index]
    print(f"train {len(train)} / test {len(test)} pairs, split grouped by right_id")
    print("all rows at a fixed threshold of 0.5 — a per-row tuned threshold would")
    print("mix tuner noise into every comparison\n")

    print(f"{'feature set':<28}{'precision':>11}{'recall':>9}{'f1':>8}{'vs all':>9}")
    reference = None
    recorded = {}
    if next(iter(GROUPS)) != REFERENCE:
        raise SystemExit(
            f"the 'vs all' column is measured against whichever group comes first, "
            f"but that is no longer {REFERENCE!r} - this table and the README's "
            f"delta column would silently disagree"
        )
    for name, features in GROUPS.items():
        precision, recall, f1 = fit_and_score(train, test, features)
        recorded[name] = {
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
        }
        if reference is None:
            reference = f1
        print(
            f"{name:<28}{precision:>11.3f}{recall:>9.3f}{f1:>8.3f}{f1 - reference:>+9.3f}"
        )

    # Not a feature set — a decision rule on top of the first row. This is what
    # actually ships.
    precision, recall, f1 = fit_and_score(train, test, match.FEATURES, veto=True)
    recorded[SHIPPED] = {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }
    print(
        f"{SHIPPED:<28}{precision:>11.3f}{recall:>9.3f}"
        f"{f1:>8.3f}{f1 - reference:>+9.3f}   <- shipped"
    )

    os.makedirs("results", exist_ok=True)
    # No timestamp on purpose: an unchanged run has to produce an unchanged file, or
    # the diff stops being able to answer "did any number move?".
    with open(RESULTS, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "seed": match.SEED,
                "k": match.K,
                "threshold": THRESHOLD,
                "train_pairs": len(train),
                "test_pairs": len(test),
                "reference": REFERENCE,
                "shipped": SHIPPED,
                "order": list(GROUPS) + [SHIPPED],
                "results": recorded,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"\nwrote {RESULTS} - {len(recorded)} feature sets")
    print("\nHow to read this: if 'title similarity only' lands close to 'all")
    print("features', the pipeline works on titles and price is a bonus. If it")
    print("lands far below — or if 'price alone' scores well — then the headline")
    print("number is mostly price agreement, which is an artifact of our")
    print("generator and would not survive on a real catalogue.")
    print("\nThe last three rows are the argument for a constraint over a")
    print("coefficient: 'volume signals alone' is near-worthless, 'all + volume as")
    print("features' beats plain 'all features' by refitting the whole boundary,")
    print("and the veto beats both by touching only the pairs it is about.")


if __name__ == "__main__":
    main()