"""Was the volume feature worth it? Measured against the pipeline we ship.

    python volume_feature.py

The first version of this experiment compared 6 features against 9 at a fixed 0.5
threshold and called it a win: F1 0.977 -> 0.983, false positives 7 -> 5. Then
decide.py — the configuration that actually ships — got WORSE: best-per-right F1
0.997 -> 0.991. Two mistakes caused that.

  1. I measured against an intermediate layer. error_analysis.py had already shown
     that one-to-one assignment removed all 7 false positives, series confusion
     included. So the feature was buying a fix the decision layer gave away free,
     and I kept its side effects anyway.

  2. Adding a feature means refitting, and refitting moves the boundary everywhere.
     The two NEW false positives ('Can You Keep a Secret?' vs 'Me Keep Posted')
     contain no volume numbers at all — every volume feature is 0.0 on that pair.
     They are collateral from the refit, not from the feature firing.

So this compares four configurations on one split, each reported raw AND through
one-to-one assignment. The veto configurations do not refit, so they cannot
produce collateral: they can only delete a pair whose stated volumes disagree, and
no such pair has ever been a true match here (0 of 134).

Thresholds are fixed at 0.5 throughout, deliberately. decide.py tunes per policy
on the train fold, and we already know that does not transfer — tuner noise would
be larger than the effect we are trying to see.
"""

from math import comb

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import block
import match

VOLUME = ["volume_conflict", "volume_match", "volume_one_sided"]
BASE = [feature for feature in match.FEATURES if feature not in VOLUME]
LEARNED = BASE + VOLUME
THRESHOLD = 0.5


def fit(train, features):
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(train[features]), train["is_match"])

    def score_rows(frame):
        return model.predict_proba(scaler.transform(frame[features]))[:, 1]

    return score_rows


def veto(frame, scores):
    """Zero any pair whose stated volume numbers disagree. No refit, so no collateral."""
    out = scores.copy()
    out[frame["volume_conflict"].to_numpy() == 1.0] = 0.0
    return out


def threshold_only(frame, scores, threshold):
    return scores >= threshold


def one_to_one(frame, scores, threshold):
    """Best-scoring candidate per right record, and only if it clears the threshold."""
    work = frame.reset_index(drop=True)
    winners = np.zeros(len(work), dtype=bool)
    order = work.assign(_score=scores).groupby("right_id")["_score"].idxmax()
    winners[order.to_numpy()] = True
    return winners & (scores >= threshold)


def measure(y_true, keep):
    true_positive = int((keep & (y_true == 1)).sum())
    false_positive = int((keep & (y_true == 0)).sum())
    false_negative = int((~keep & (y_true == 1)).sum())
    precision = true_positive / (true_positive + false_positive) if keep.any() else 0.0
    recall = true_positive / (true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1, false_positive, false_negative


def wrong_pairs(frame, y_true, keep):
    """The specific pairs each configuration gets wrong, so we can diff them."""
    ids = list(zip(frame["left_id"], frame["right_id"]))
    return (
        {pair for pair, k, y in zip(ids, keep, y_true) if k and y == 0},
        {pair for pair, k, y in zip(ids, keep, y_true) if not k and y == 1},
    )


def coin_flip_p(fixed, broken):
    """Exact two-sided p for a fixed/broken split under 'the change did nothing'.

    Only the decisions that CHANGED carry information. Everything else is agreement
    between the two configurations and tells us nothing about which is better.
    """
    total = fixed + broken
    if total == 0:
        return 1.0
    gap = abs(fixed - broken)
    return sum(
        comb(total, k) for k in range(total + 1) if abs(2 * k - total) >= gap
    ) / 2 ** total


def main():
    left, right, truth = block.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    table = match.build_table(left, right, truth_pairs, match.K)

    train_index, test_index = match.grouped_split(table["right_id"].to_numpy())
    train, test = table.iloc[train_index], table.iloc[test_index]
    y_test = test["is_match"].to_numpy()

    conflicts = test["volume_conflict"] == 1.0
    print(f"test fold: {len(test)} pairs, {int(y_test.sum())} true")
    print(f"volume conflicts in test: {int(conflicts.sum())} pairs, "
          f"{int(test.loc[conflicts, 'is_match'].sum())} of them true\n")

    base_scorer = fit(train, BASE)
    learned_scorer = fit(train, LEARNED)
    base_scores = base_scorer(test)
    learned_scores = learned_scorer(test)

    configurations = {
        "6 features": base_scores,
        "6 features + volume veto": veto(test, base_scores),
        "9 features (volume learned)": learned_scores,
        "9 features + veto": veto(test, learned_scores),
    }

    header = f"{'configuration':<30}{'P':>7}{'R':>7}{'F1':>7}{'FP':>5}{'FN':>4}"
    for policy_name, policy in (("threshold 0.5 only", threshold_only),
                                ("one-to-one at 0.5", one_to_one)):
        print(f"{policy_name}\n{header}")
        for name, scores in configurations.items():
            keep = policy(test, scores, THRESHOLD)
            precision, recall, f1, fp, fn = measure(y_test, keep)
            print(f"{name:<30}{precision:>7.3f}{recall:>7.3f}{f1:>7.3f}{fp:>5}{fn:>4}")
        print()

    # Only the decisions that differ carry information about which is better.
    print("what changed, against the 6-feature baseline (one-to-one at 0.5):")
    baseline_keep = one_to_one(test, base_scores, THRESHOLD)
    baseline_fp, baseline_fn = wrong_pairs(test, y_test, baseline_keep)
    baseline_errors = baseline_fp | baseline_fn

    for name in ("6 features + volume veto", "9 features (volume learned)",
                 "9 features + veto"):
        keep = one_to_one(test, configurations[name], THRESHOLD)
        false_positive, false_negative = wrong_pairs(test, y_test, keep)
        errors = false_positive | false_negative
        fixed = len(baseline_errors - errors)
        broken = len(errors - baseline_errors)
        print(f"  {name:<30} fixed {fixed}, broke {broken}   "
              f"(coin-flip p = {coin_flip_p(fixed, broken):.2f})")

    print("\nHow to read this: a high p means the two configurations disagree on so")
    print("few pairs that chance alone explains the split, and the F1 difference is")
    print("not evidence. In that case choose on grounds you can defend out loud —")
    print("fewer moving parts, or errors that are easier to explain — and say")
    print("plainly that the metric did not decide it.")


if __name__ == "__main__":
    main()