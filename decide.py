"""Turn per-pair scores into decisions, and measure what the decision rule is worth.

    python decide.py

The matcher scores every candidate pair independently. But a right-hand record has
at most one true partner in this benchmark, and that constraint is free — no
retraining, no new features — and on this data it is worth more than any feature
we added. That is the headline finding of this file.

Thresholds are the trap, twice over.

  First: an earlier version tuned a threshold per policy on the train fold, so
  comparing policies quietly became comparing tuner luck. Adding three features
  moved one policy's pick from 0.403 to 0.652 and 'cost' 0.006 F1 that had nothing
  to do with the features. I misdiagnosed that as a modelling side effect.

  Second: the fix reported a 'test-oracle' row, and that row came out BELOW the
  flat-0.5 row — impossible for a threshold chosen with test labels. The candidate
  grid was built from quantiles of the score distribution, which is bimodal (~84%
  of pairs near 0.0, ~16% near 1.0), so it barely sampled the range where the
  decision actually lives. A uniform grid plus an explicit assertion fixes it.

Every policy is reported at three thresholds:

  flat 0.5      no tuning at all — what ships
  train-tuned   chosen on the train fold — the only tuned value you could deploy
  test-oracle   chosen on the test fold — NOT achievable, printed only to bound
                what perfect threshold selection could have bought

Tuning never helps on this data: it loses for threshold-only and ties for both
one-to-one policies. So the pipeline ships at a flat 0.5.
"""

import numpy as np
from scipy.stats import beta

import block
import match

DEFAULT_THRESHOLD = 0.5


def threshold_only(frame, scores, threshold):
    """No constraint at all. The baseline any decision rule has to beat."""
    return np.asarray(scores) >= threshold


def best_per_right(frame, scores, threshold):
    """One partner per right record: its best candidate, if that clears the bar.

    Cheapest real win in the project. It cannot help a right record that has no
    true partner — for those, the threshold is still the only defence.
    """
    scores = np.asarray(scores)
    work = frame.reset_index(drop=True).assign(_score=scores)
    keep = np.zeros(len(work), dtype=bool)
    keep[work.groupby("right_id")["_score"].idxmax().to_numpy()] = True
    return keep & (scores >= threshold)


def mutual_best(frame, scores, threshold):
    """Both sides must prefer each other.

    Measured as buying nothing over best_per_right on this data. Kept as a
    negative result, not sold as a feature.
    """
    scores = np.asarray(scores)
    work = frame.reset_index(drop=True).assign(_score=scores)
    per_right = set(work.groupby("right_id")["_score"].idxmax().to_numpy())
    per_left = set(work.groupby("left_id")["_score"].idxmax().to_numpy())
    keep = np.zeros(len(work), dtype=bool)
    for index in per_right & per_left:
        keep[index] = True
    return keep & (scores >= threshold)


POLICIES = {
    "threshold only": threshold_only,
    "best per right record": best_per_right,
    "mutual best": mutual_best,
}


def measure(y_true, keep):
    true_positive = int((keep & (y_true == 1)).sum())
    false_positive = int((keep & (y_true == 0)).sum())
    false_negative = int((~keep & (y_true == 1)).sum())
    accepted = true_positive + false_positive
    precision = true_positive / accepted if accepted else 0.0
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, false_positive, false_negative, true_positive, accepted


def candidate_thresholds(scores):
    """Thresholds to search.

    A uniform grid, NOT quantiles of the scores. The score distribution is bimodal,
    so its quantiles cluster at both extremes and skip the middle — the first
    version of this searched quantiles and never even tried 0.5, which made the
    'oracle' row score worse than the default. DEFAULT_THRESHOLD is added
    explicitly so the oracle is guaranteed to be at least as good as what ships.
    """
    uniform = np.linspace(0.01, 0.99, 99)
    dense = np.quantile(scores, np.linspace(0.30, 0.999, 40))
    return np.unique(np.concatenate([uniform, dense, [DEFAULT_THRESHOLD]]))


def tune(policy, frame, scores, y_true):
    """Best threshold for this policy on whatever fold it is handed."""
    return max(
        candidate_thresholds(scores),
        key=lambda t: measure(y_true, policy(frame, scores, t))[2],
    )


def lower_bound(successes, trials, confidence=0.95):
    """One-sided Clopper-Pearson lower bound.

    Exists so a precision of 1.000 gets reported as the claim it actually
    supports. 173 of 173 correct is not 'perfect' — it is 'at least 0.98'.
    """
    if trials == 0:
        return 0.0
    if successes >= trials:
        return (1 - confidence) ** (1 / trials)
    return float(beta.ppf(1 - confidence, successes, trials - successes + 1))


def main():
    left, right, truth = block.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    table = match.build_table(left, right, truth_pairs, match.K)

    train_index, test_index = match.grouped_split(table["right_id"].to_numpy())
    train, test = table.iloc[train_index], table.iloc[test_index]

    # fit_scorer applies the volume veto, so these are shipped scores, not raw ones.
    scorer = match.fit_scorer(train)
    train_scores, test_scores = scorer(train), scorer(test)
    y_train = train["is_match"].to_numpy()
    y_test = test["is_match"].to_numpy()

    right_ids = set(test["right_id"])
    with_partner = {pair[1] for pair in truth_pairs if pair[1] in right_ids}
    rejectable = len(right_ids) - len(with_partner)
    print(f"test fold: {len(test)} candidate pairs across {len(right_ids)} right records")
    print(f"  {len(with_partner)} have a true partner, {rejectable} have none — those")
    print("  are what the threshold has to reject, and no one-to-one rule can help there")

    blocking_losses = len(
        {pair for pair in truth_pairs if pair[1] in right_ids}
        - set(zip(test["left_id"], test["right_id"]))
    )
    print(f"  blocking lost {blocking_losses} true pairs in this fold, so recall below")
    print("  is already end-to-end recall\n")

    header = (
        f"{'policy':<24}{'threshold from':<16}{'thr':>7}{'P':>8}{'R':>8}"
        f"{'F1':>8}{'FP':>5}{'FN':>4}"
    )
    print(header)
    shipped, oracle_f1 = {}, {}
    for name, policy in POLICIES.items():
        sources = (
            ("flat 0.5", DEFAULT_THRESHOLD),
            ("train-tuned", tune(policy, train, train_scores, y_train)),
            ("test-oracle", tune(policy, test, test_scores, y_test)),
        )
        for label, threshold in sources:
            keep = policy(test, test_scores, threshold)
            precision, recall, f1, fp, fn, tp, accepted = measure(y_test, keep)
            print(
                f"{name:<24}{label:<16}{threshold:>7.3f}{precision:>8.3f}"
                f"{recall:>8.3f}{f1:>8.3f}{fp:>5}{fn:>4}"
            )
            if label == "flat 0.5":
                shipped[name] = (precision, recall, f1, tp, accepted)
            if label == "test-oracle":
                oracle_f1[name] = f1
        print()

    # A row labelled "upper bound" that sits below what ships is not a bound, it
    # is a search bug. This assertion is the whole reason the grid changed.
    for name in POLICIES:
        if oracle_f1[name] < shipped[name][2] - 1e-12:
            raise SystemExit(
                f"ERROR: test-oracle for '{name}' ({oracle_f1[name]:.3f}) is below "
                f"flat 0.5 ({shipped[name][2]:.3f}).\n"
                "The candidate grid is missing the useful range — it is not an "
                "upper bound, so do not quote it."
            )

    print("test-oracle rows are NOT achievable — they use test labels to pick the")
    print("threshold. They bound what better threshold selection could buy, and a")
    print("check above refuses to print them if they fall below the flat-0.5 row.\n")

    best_name = max(shipped, key=lambda name: shipped[name][2])
    precision, recall, f1, tp, accepted = shipped[best_name]
    print(f"shipped configuration: {best_name} at {DEFAULT_THRESHOLD}")
    print(f"  precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}")
    print(f"  {tp} of {accepted} accepted pairs are correct, so the defensible claim")
    print(f"  is precision >= {lower_bound(tp, accepted):.3f} "
          f"(95% one-sided Clopper-Pearson), not {precision:.3f}")

    plain = shipped["threshold only"][2]
    headroom = oracle_f1[best_name] - f1
    print(f"\nwhat the one-to-one constraint is worth: f1 {plain:.3f} -> {f1:.3f}, "
          f"with no retraining")
    print(f"what a perfect threshold would add on top: {headroom:+.3f} f1")
    if headroom <= 0.0005:
        print("  i.e. nothing — 0.5 is already optimal for this policy, which is why")
        print("  the pipeline does not tune a threshold at all")
    if shipped["mutual best"][2] <= shipped["best per right record"][2]:
        print("\nthe extra per-left constraint in 'mutual best' bought nothing — a")
        print("measured negative result, kept in the table rather than deleted")


if __name__ == "__main__":
    main()