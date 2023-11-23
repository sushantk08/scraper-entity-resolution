"""Look at the pairs the pipeline gets wrong, one by one.

    python error_analysis.py

Aggregate metrics tell you how much is wrong. They never tell you what kind of
wrong, and the kind is what decides what to build next. In particular this
separates two failures that a recall number lumps together:

  - blocking never produced the pair, so the classifier never saw it
  - blocking produced it and the classifier scored it too low

Only the second is a modelling problem. The first is fixed by a bigger k or a
better similarity function, and no amount of classifier work will touch it.
"""

import block
import decide
import match

MAX_EXAMPLES = 8


def title_of(row):
    return str(row.get("title") or row.get("key") or "")


def show_pair(left_rows, right_rows, left_id, right_id, score, features=None):
    left_row, right_row = left_rows[left_id], right_rows[right_id]
    print(f"    score {score:.3f}")
    print(f"      left  [{left_id}] {title_of(left_row)!r}")
    print(f"              category {str(left_row.get('category') or '(blank)')!r}, "
          f"£{float(left_row['price_listing_gbp']):.2f}")
    print(f"      right [{right_id}] {title_of(right_row)!r}")
    print(f"              category {str(right_row.get('category') or '(blank)')!r}, "
          f"£{float(right_row['price_gbp']):.2f}")
    if features is not None:
        pieces = [f"{name}={features[name]:.3f}" for name in match.FEATURES]
        print(f"      {'  '.join(pieces)}")


def main():
    left, right, truth = block.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    left_rows = left.set_index("left_id").to_dict("index")
    right_rows = right.set_index("right_id").to_dict("index")

    table = match.build_table(left, right, truth_pairs, match.K)
    candidates = set(zip(table["left_id"], table["right_id"]))

    train_index, test_index = match.grouped_split(table["right_id"].to_numpy())
    train = table.iloc[train_index].reset_index(drop=True)
    test = table.iloc[test_index].reset_index(drop=True)

    score_rows = match.fit_scorer(train, match.FEATURES)
    train_scores = score_rows(train)
    scores = score_rows(test)

    test_right_ids = set(test["right_id"])
    truth_in_test = {pair for pair in truth_pairs if pair[1] in test_right_ids}

    # 1. Pairs blocking never generated. No classifier can recover these.
    never_seen = sorted(truth_in_test - candidates)
    print(f"=== true pairs blocking never generated: {len(never_seen)}")
    print("    (raise k in block.py to fix these; the classifier is blameless)")
    for left_id, right_id in never_seen[:MAX_EXAMPLES]:
        print()
        show_pair(left_rows, right_rows, left_id, right_id, 0.0)

    # 2. Pairs blocking found but the classifier scored below threshold.
    threshold = 0.5
    accepted = scores >= threshold
    rows = list(test.itertuples())

    misses = [
        (row, score)
        for row, score, keep in zip(rows, scores, accepted)
        if row.is_match and not keep
    ]
    print(f"\n\n=== seen but scored too low at {threshold}: {len(misses)}")
    print("    (these ARE modelling failures)")
    for row, score in sorted(misses, key=lambda item: -item[1])[:MAX_EXAMPLES]:
        print()
        show_pair(left_rows, right_rows, row.left_id, row.right_id, score,
                  {name: getattr(row, name) for name in match.FEATURES})

    # 3. Wrong pairs accepted. On this data these are near-misses, not nonsense.
    false_positives = [
        (row, score)
        for row, score, keep in zip(rows, scores, accepted)
        if keep and not row.is_match
    ]
    print(f"\n\n=== wrong pairs accepted at {threshold}: {len(false_positives)}")
    print("    (read the titles: are they genuinely confusable, or is a feature lying?)")
    for row, score in sorted(false_positives, key=lambda item: -item[1])[:MAX_EXAMPLES]:
        print()
        show_pair(left_rows, right_rows, row.left_id, row.right_id, score,
                  {name: getattr(row, name) for name in match.FEATURES})

    # 4. What the one-to-one decision rule cleans up, and what it cannot.
    tuned = decide.tune(decide.best_per_right, train, train_scores,
                        train["is_match"].to_numpy())
    kept = decide.best_per_right(test, scores, tuned)
    survivors = [
        (row, score)
        for row, score, keep in zip(rows, scores, kept)
        if keep and not row.is_match
    ]
    print(f"\n\n=== after one-to-one assignment (threshold {tuned:.3f})")
    print(f"    false positives remaining: {len(survivors)} "
          f"(was {len(false_positives)} with a threshold alone)")
    for row, score in sorted(survivors, key=lambda item: -item[1])[:MAX_EXAMPLES]:
        print()
        show_pair(left_rows, right_rows, row.left_id, row.right_id, score,
                  {name: getattr(row, name) for name in match.FEATURES})

    still_missing = len(truth_in_test) - sum(
        1 for row, keep in zip(rows, kept) if keep and row.is_match
    )
    print(f"\n    true pairs still not recovered: {still_missing} of {len(truth_in_test)}")
    print(f"    of which {len(never_seen)} were never generated by blocking")


if __name__ == "__main__":
    main()