"""Turn pairwise scores into decisions, and compare decision policies.

    python decide.py

The classifier scores each candidate on its own, so it can accept two different
left records for the same right record. Here that is always wrong, and since the
problem is precision-limited rather than recall-limited, forbidding it should
pay — with no retraining, because it is a decision change, not a model change.

Caveat worth keeping honest: "at most one partner" holds because perturb.py
built the data that way. A real catalogue can carry two legitimate editions of
the same book, so this constraint has to be argued per dataset, not assumed.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import block
import match


def threshold_only(frame, scores, threshold):
    """Every pair judged independently — the behaviour we are improving on."""
    return scores >= threshold


def best_per_right(frame, scores, threshold):
    """At most one match per right record — its highest-scoring candidate."""
    work = pd.DataFrame({"right_id": frame["right_id"].to_numpy(), "score": scores})
    winners = work.groupby("right_id")["score"].idxmax().to_numpy()
    keep = np.zeros(len(work), dtype=bool)
    keep[winners] = True
    return keep & (scores >= threshold)


def mutual_best(frame, scores, threshold):
    """Keep a pair only if it is the best available option for BOTH sides."""
    work = pd.DataFrame(
        {
            "left_id": frame["left_id"].to_numpy(),
            "right_id": frame["right_id"].to_numpy(),
            "score": scores,
        }
    )
    best_for_right = set(work.groupby("right_id")["score"].idxmax())
    best_for_left = set(work.groupby("left_id")["score"].idxmax())
    keep = np.zeros(len(work), dtype=bool)
    keep[list(best_for_right & best_for_left)] = True
    return keep & (scores >= threshold)


POLICIES = {
    "threshold only": threshold_only,
    "best per right record": best_per_right,
    "mutual best": mutual_best,
}


def measure(y_true, keep):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, keep, average="binary", zero_division=0
    )
    return precision, recall, f1


def tune(policy, frame, scores, y_true):
    """Choose this policy's threshold on the training fold only."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.30, 0.999, 60)))
    return max(candidates, key=lambda t: measure(y_true, policy(frame, scores, t))[2])


def main():
    left, right, truth = block.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    table = match.build_table(left, right, truth_pairs, match.K)

    train_index, test_index = match.grouped_split(table["right_id"].to_numpy())
    train = table.iloc[train_index].reset_index(drop=True)
    test = table.iloc[test_index].reset_index(drop=True)

    score_rows = match.fit_scorer(train, match.FEATURES)
    train_scores = score_rows(train)
    test_scores = score_rows(test)

    y_train = train["is_match"].to_numpy()
    y_test = test["is_match"].to_numpy()

    test_right_ids = set(test["right_id"])
    truth_in_test = {pair for pair in truth_pairs if pair[1] in test_right_ids}
    with_partner = {pair[1] for pair in truth_pairs}
    rejectable = sum(1 for right_id in test_right_ids if right_id not in with_partner)

    print(f"test fold: {len(test)} candidate pairs across {len(test_right_ids)} right records")
    print(f"  {len(truth_in_test)} have a true partner, {rejectable} have none "
          f"— those are what the threshold has to reject\n")

    print(f"{'policy':<24}{'threshold':>11}{'precision':>11}{'recall':>9}"
          f"{'f1':>8}{'end-to-end':>12}")

    for name, policy in POLICIES.items():
        threshold = tune(policy, train, train_scores, y_train)
        keep = policy(test, test_scores, threshold)
        precision, recall, f1 = measure(y_test, keep)

        accepted = {
            (row.left_id, row.right_id)
            for row, kept in zip(test.itertuples(), keep)
            if kept
        }
        end_to_end = len(accepted & truth_in_test) / len(truth_in_test)

        print(f"{name:<24}{threshold:>11.3f}{precision:>11.3f}{recall:>9.3f}"
              f"{f1:>8.3f}{end_to_end:>12.3f}")


if __name__ == "__main__":
    main()