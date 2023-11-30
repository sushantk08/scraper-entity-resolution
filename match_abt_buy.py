"""Run the matcher on Abt-Buy, the published human-labelled benchmark.

    python match_abt_buy.py

Everything in match.py is measured on data perturb.py generated, which writes both
the noise and the labels. This file is the check on that: real product names from
two retailers, human-labelled pairs, and the same positive rate as the generated
benchmark - 0.093% against 0.094%, so the difficulty is not sparsity.

It deliberately does NOT reuse match.pair_features. abt_buy.py blocks on a
SQUASHED key with separators deleted ('CLI-8C' -> 'cli8c'), so character
similarity should read that string, but word overlap cannot - a string with no
spaces has one token. Word features read the spaced key, character features read
the squashed one. Sharing one function would have hidden that choice.

WHICH NUMBER IS COMPARABLE TO PUBLISHED WORK. Published Abt-Buy results score
pair classification over a fixed labelled candidate set. That is the
'classifier at 0.5' row here. The one-to-one rows score much higher, and the
reason is structural rather than clever: every Abt and Buy record in this
benchmark has at least one true partner, so a rule that accepts each record's
top candidate can barely be punished for accepting. It measures ranking, not
rejection. Quote the pair-classifier row when comparing to anyone else.

Missing values are handled explicitly - price is blank on 61%/46% of records, so
a blank arriving at the model as 0.0 would read as "prices agree perfectly", the
same bug class as a blank category counting as agreement. Unknowns become NaN and
are filled with the TRAINING fold's mean.

description_known WAS a feature and is now diagnostic only. abt_buy_features.py
resampled the grouped split 20 times: dropping it won 18 of 20 splits at the pair
layer (p 0.000) and tied at the one-to-one layer (9 of 20, p 1.000), so it is
dominated. Two supporting facts. Its purpose was to stop a missing description
reading as agreement, and the train-fold fill value is 0.077 - already near the
low end, so a blank reads as weak disagreement without needing a flag. And the
true-match rate is identical (9.8%) whether descriptions are present or missing,
which is measured below and was the death of my theory that its -0.37 weight
meant it encoded a per-record prior. price_known and manufacturer_known stay:
their fill values are 0.477 and 0.582, nowhere near as safe, and neither was
tested.
"""

import json
import re

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import abt_buy
import block
import decide
import match

K = 10
THRESHOLD = 0.5
RESULTS = "results/abt_classifier.json"
# Row names, so a rename cannot orphan the recording. The recorded run says
# "blocking_k", never "k": the k-agreement test compares every run declaring a
# "k" and this corpus has its own, unrelated to the books pipeline's.
BASELINE = "cosine baseline"
COMPARABLE = "classifier at 0.5"
ONE_TO_ONE = ("best per right record", "mutual best")
RECORDED = {}

TEXT = ["cosine", "sequence_ratio", "length_ratio", "word_jaccard"]
FEATURES = TEXT + [
    "price_relative_diff",
    "price_known",
    "description_jaccard",
    "manufacturer_in_name",
    "manufacturer_known",
]
# Computed on every pair, deliberately not trained on. Measured and rejected.
DIAGNOSTIC = ["description_known"]
ALL_PAIR_FEATURES = FEATURES + DIAGNOSTIC

GROUPS = {
    "all features": FEATURES,
    "cosine alone": ["cosine"],
    "character similarity only": ["cosine", "sequence_ratio", "length_ratio"],
    "text only": TEXT,
    "text + price": TEXT + ["price_relative_diff", "price_known"],
    "text + description": TEXT + ["description_jaccard"],
    "text + manufacturer": TEXT + ["manufacturer_in_name", "manufacturer_known"],
    "all + description_known": FEATURES + DIAGNOSTIC,
}

# Columns carrying NaN for "could not compute", filled from the train fold only.
IMPUTED = ["price_relative_diff", "description_jaccard", "manufacturer_in_name"]

# A misspelled name in GROUPS would otherwise surface as a KeyError halfway
# through a run. The books side catches this in pytest; here it cannot, because
# benchmark/ is gitignored and CI has no data to import, so the guard lives here.
_unknown = {name for group in GROUPS.values() for name in group}
_unknown -= set(ALL_PAIR_FEATURES)
if _unknown:
    raise SystemExit(f"ablation names features that do not exist: {sorted(_unknown)}")

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text):
    return set(TOKEN.findall(str(text).lower()))


def jaccard(left_tokens, right_tokens):
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def pair_features(left_row, right_row, cosine):
    left_key, right_key = left_row["key"], right_row["key"]

    features = {
        "cosine": cosine,
        "sequence_ratio": SequenceMatcher(None, left_key, right_key).ratio(),
        "length_ratio": min(len(left_key), len(right_key))
        / max(len(left_key), len(right_key), 1),
        "word_jaccard": jaccard(
            tokens(left_row["key_spaced"]), tokens(right_row["key_spaced"])
        ),
    }

    left_price, right_price = float(left_row["price"]), float(right_row["price"])
    if np.isnan(left_price) or np.isnan(right_price):
        features["price_relative_diff"] = np.nan
        features["price_known"] = 0.0
    else:
        features["price_relative_diff"] = abs(left_price - right_price) / max(
            left_price, right_price, 0.01
        )
        features["price_known"] = 1.0

    left_description = tokens(left_row["description"])
    right_description = tokens(right_row["description"])
    if left_description and right_description:
        features["description_jaccard"] = jaccard(left_description, right_description)
        features["description_known"] = 1.0
    else:
        features["description_jaccard"] = np.nan
        features["description_known"] = 0.0

    manufacturer = re.sub(r"[^a-z0-9]+", "", str(right_row["manufacturer"]).lower())
    if manufacturer:
        features["manufacturer_in_name"] = float(manufacturer in left_key)
        features["manufacturer_known"] = 1.0
    else:
        features["manufacturer_in_name"] = np.nan
        features["manufacturer_known"] = 0.0

    return features


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


def impute_from_train(train, test, columns):
    """Fill NaN with the training fold's mean. Never the whole table's."""
    train, test = train.copy(), test.copy()
    for column in columns:
        fill = train[column].mean()
        fill = 0.0 if np.isnan(fill) else fill
        share = test[column].isna().mean()
        print(f"  {column:<24} filled {int(test[column].isna().sum()):>5} of "
              f"{len(test):>5} test rows ({share:.0%}) with train mean {fill:.3f}")
        train[column] = train[column].fillna(fill)
        test[column] = test[column].fillna(fill)
    return train, test


def fit(train, features):
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(train[features]), train["is_match"])
    return model, lambda frame: model.predict_proba(
        scaler.transform(frame[features])
    )[:, 1]


def metrics(y_true, keep):
    true_positive = int((keep & (y_true == 1)).sum())
    false_positive = int((keep & (y_true == 0)).sum())
    false_negative = int((~keep & (y_true == 1)).sum())
    accepted = true_positive + false_positive
    precision = true_positive / accepted if accepted else 0.0
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, false_positive, false_negative


def show(name, y_true, keep, end_to_end=None, record=None):
    precision, recall, f1, fp, fn = metrics(y_true, keep)
    tail = "" if end_to_end is None else f"{end_to_end:>12.3f}"
    print(f"{name:<30}{precision:>10.3f}{recall:>9.3f}{f1:>8.3f}"
          f"{fp:>6}{fn:>6}{tail}")
    if record is not None:
        RECORDED[record] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "end_to_end": end_to_end,
        }
    return f1


def main():
    left, right, truth = abt_buy.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    possible = len(left) * len(right)
    print(f"{len(left)} Abt x {len(right)} Buy = {possible:,} possible pairs, "
          f"{len(truth_pairs):,} true ({len(truth_pairs) / possible:.3%} positive)\n")

    table = build_table(left, right, truth_pairs, K)
    reached = int(table["is_match"].sum())
    print(f"blocking at k={K}: {len(table):,} candidate pairs, {reached:,} true "
          f"({reached / len(table):.1%} positive)")
    print(f"blocking recall: {reached / len(truth_pairs):.3f} — "
          f"{len(truth_pairs) - reached} true pairs never reach the matcher\n")

    train_index, test_index = match.grouped_split(table["right_id"].to_numpy())
    train, test = table.iloc[train_index], table.iloc[test_index]
    print("missing-value handling (train mean, computed before the model sees test):")
    train, test = impute_from_train(train, test, IMPUTED)

    y_train, y_test = train["is_match"].to_numpy(), test["is_match"].to_numpy()

    # End-to-end denominator: every true pair whose Buy record is in the test
    # fold, INCLUDING the ones blocking never generated. The recall column below
    # cannot see those, so it flatters the pipeline unless this is reported too.
    test_right_ids = set(test["right_id"])
    truth_in_test = {pair for pair in truth_pairs if pair[1] in test_right_ids}
    candidate_pairs = list(zip(test["left_id"], test["right_id"]))
    blocking_lost = len(truth_in_test) - int(y_test.sum())
    print(f"\ntrain {len(train):,} / test {len(test):,} pairs, grouped by Buy id")
    print(f"{int(y_test.sum())} true pairs reached the matcher; {len(truth_in_test)} "
          f"exist in this fold, so blocking already lost {blocking_lost}")

    def end_to_end(keep):
        found = {pair for pair, k in zip(candidate_pairs, keep) if k}
        return len(found & truth_in_test) / len(truth_in_test)

    model, scorer = fit(train, FEATURES)
    test_scores = scorer(test)

    print(f"\n{'configuration':<30}{'precision':>10}{'recall':>9}{'F1':>8}"
          f"{'FP':>6}{'FN':>6}{'end-to-end':>12}")
    cosine_threshold = match.best_threshold(y_train, train["cosine"].to_numpy())
    baseline_keep = test["cosine"].to_numpy() >= cosine_threshold
    show(f"cosine >= {cosine_threshold:.3f}", y_test, baseline_keep,
         end_to_end(baseline_keep), record=BASELINE)

    pair_keep = test_scores >= THRESHOLD
    show("classifier at 0.5  <- COMPARE", y_test, pair_keep,
         end_to_end(pair_keep), record=COMPARABLE)
    for name in ONE_TO_ONE:
        keep = decide.POLICIES[name](test, test_scores, THRESHOLD)
        show(f"{name} at 0.5", y_test, keep, end_to_end(keep), record=name)
    print("  the marked row is the one comparable to published results; the")
    print("  one-to-one rows exploit the fact that every record here has a partner")
    print("  F1 and end-to-end recall rank these in OPPOSITE orders - the one-to-one")
    print("  constraint trades found links for precision, so the objective decides")

    print("\nwhich features earn their place, this split only "
          "(F1 at 0.5, pair / mutual best):")
    for name, features in GROUPS.items():
        _, group_scorer = fit(train, features)
        group_scores = group_scorer(test)
        pair_f1 = metrics(y_test, group_scores >= THRESHOLD)[2]
        one_to_one_f1 = metrics(
            y_test, decide.mutual_best(test, group_scores, THRESHOLD)
        )[2]
        marker = "  <- shipped" if features is FEATURES else ""
        print(f"  {name:<30}{pair_f1:>8.3f}{one_to_one_f1:>14.3f}{marker}")
    print("  one split cannot separate these - see abt_buy_features.py, which")
    print("  resamples the split 20 times and counts wins instead of means")

    # The measurement that killed my theory about description_known's -0.37
    # weight: I claimed it encoded WHICH RECORDS rarely match rather than
    # anything about the pair. If that were true these two rates would diverge.
    print("\nwas description_known measuring similarity, or record identity?")
    rates = {}
    for value, label in ((1.0, "description on both sides"), (0.0, "one side blank")):
        subset = test[test["description_known"] == value]
        rate = subset["is_match"].mean() if len(subset) else float("nan")
        print(f"  {label:<28}{len(subset):>6} pairs, {rate:>6.1%} true")
        rates[label] = {"pairs": int(len(subset)), "true_rate": float(rate)}
    print("  neither - the rates match, so it carried no information about the pair")
    print("  and its coefficient was collinearity. Now diagnostic only.")

    print("\n  coefficients (NOT importance — correlated features):")
    for name, weight in sorted(
        zip(FEATURES, model.coef_[0]), key=lambda item: -abs(item[1])
    ):
        print(f"    {name:<24}{weight:+.2f}")

    # These three numbers were "16", "5" and "0.995" typed into the strings
    # below. abt_buy.py already derives the Buy count and the ceiling, so two
    # scripts were printing the same fact, one measured and one by hand.
    left_partners, right_partners = {}, {}
    for left_id, right_id in truth_pairs:
        left_partners[left_id] = left_partners.get(left_id, 0) + 1
        right_partners[right_id] = right_partners.get(right_id, 0) + 1
    left_several = sum(1 for n in left_partners.values() if n > 1)
    right_several = sum(1 for n in right_partners.values() if n > 1)
    ceiling = len(right_partners) / len(truth_pairs)
    print(f"\n{left_several} Abt and {right_several} Buy records have two true "
          f"partners, so one-to-one assignment carries a measured recall "
          f"ceiling of {ceiling:.3f} on this data.")
    for name in (BASELINE, COMPARABLE) + ONE_TO_ONE:
        if name not in RECORDED:
            raise SystemExit(f"{name} not measured - {sorted(RECORDED)}")
    if any(v["true_rate"] != v["true_rate"] for v in rates.values()):
        raise SystemExit(f"empty description subset - {rates}")
    with open(RESULTS, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "blocking_k": K,
                "blocking_lost": blocking_lost,
                "blocking_recall": reached / len(truth_pairs),
                "candidate_pairs": len(table),
                "candidates_true": reached,
                "comparable": COMPARABLE,
                "cosine_threshold": cosine_threshold,
                "description_known": rates,
                "left_records": len(left),
                "left_with_several_partners": left_several,
                "order": [BASELINE, COMPARABLE] + list(ONE_TO_ONE),
                "recall_ceiling": ceiling,
                "results": RECORDED,
                "right_records": len(right),
                "right_with_several_partners": right_several,
                "test_pairs": len(test),
                "test_true": int(y_test.sum()),
                "threshold": THRESHOLD,
                "train_pairs": len(train),
                "true_pairs": len(truth_pairs),
                "truth_in_test": len(truth_in_test),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"wrote {RESULTS} - {len(RECORDED)} rows")


if __name__ == "__main__":
    main()