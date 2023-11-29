"""Generate candidate pairs cheaply, and measure what that costs.

    python block.py

Two numbers decide whether a blocking scheme is any good:

  blocking recall - the fraction of true matches that survive. Anything lost
                    here can never be recovered by a later matching step.
  reduction ratio - the fraction of all possible pairs we avoided comparing.

They trade off against each other, so the useful output is the trade-off table,
not a single score.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

import clean

DATA_DIR = Path("data")
RESULTS = Path("results") / "blocking.json"

# The k the rest of the pipeline blocks at. match.py, ablate.py and
# sweep.py all use it, so recording it lets a test assert they agree.
SHIPPED_K = 5
NEIGHBOURS = (1, 3, 5, 10, 20)
SHIPPED = f"nearest neighbours k={SHIPPED_K}"


def load():
    left = pd.read_csv(DATA_DIR / "books_left.csv").rename(columns={"upc": "left_id"})
    right = pd.read_csv(DATA_DIR / "books_right.csv")
    truth = pd.read_csv(DATA_DIR / "truth.csv")
    # A missing category must read back as "", not NaN: bool(NaN) is True, which
    # would make "unknown" look like a real value to the same_category feature.
    for frame in (left, right):
        frame["category"] = frame["category"].fillna("")
    left["key"] = clean.normalise_title(left["title"])
    right["key"] = clean.normalise_title(right["title"])
    return left, right, truth


def exact_key_pairs(left, right):
    """The cheapest blocking there is: identical normalised titles."""
    merged = right.merge(left, on="key", suffixes=("_r", "_l"))
    return set(zip(merged["left_id"], merged["right_id"]))


def nearest_neighbours(left, right, k):
    """The k closest left records for each right record, with similarities.

    Character n-grams rather than words, because the noise we care about is
    typos and reordering — both of which break word-level matching.
    """
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    left_matrix = vectorizer.fit_transform(left["key"])
    right_matrix = vectorizer.transform(right["key"])

    index = NearestNeighbors(
        n_neighbors=min(k, len(left)), metric="cosine", algorithm="brute"
    )
    index.fit(left_matrix)
    distances, neighbours = index.kneighbors(right_matrix)

    left_ids = left["left_id"].to_numpy()
    right_ids = right["right_id"].to_numpy()
    return [
        (left_ids[neighbours[row][slot]], right_ids[row], 1.0 - distances[row][slot])
        for row in range(len(right_ids))
        for slot in range(neighbours.shape[1])
    ]


def nearest_neighbour_pairs(left, right, k):
    return {(left_id, right_id) for left_id, right_id, _ in nearest_neighbours(left, right, k)}


def score(pairs, truth_pairs, total_possible):
    recall = len(pairs & truth_pairs) / len(truth_pairs)
    reduction = 1 - len(pairs) / total_possible
    return recall, reduction


def main():
    left, right, truth = load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    total_possible = len(left) * len(right)

    print(f"{len(left)} x {len(right)} = {total_possible:,} possible pairs")
    print(f"{len(truth_pairs)} of them are true matches\n")
    print(f"{'scheme':<26}{'pairs':>10}{'recall':>10}{'reduction':>12}")

    schemes = [("exact normalised key", exact_key_pairs(left, right))]
    for k in NEIGHBOURS:
        schemes.append((f"nearest neighbours k={k}", nearest_neighbour_pairs(left, right, k)))

    recorded = {}
    best = None
    for name, pairs in schemes:
        recall, reduction = score(pairs, truth_pairs, total_possible)
        print(f"{name:<26}{len(pairs):>10,}{recall:>10.3f}{reduction:>12.4%}")
        recorded[name] = {
            "pairs": len(pairs),
            "recall": float(recall),
            "reduction": float(reduction),
        }
        if best is None or recall > best[1]:
            best = (pairs, recall, name)

    # The pairs blocking threw away are the ones worth looking at by eye.
    missed = truth_pairs - best[0]
    print(f"\n{len(missed)} true pairs missed by '{best[2]}'")
    right_titles = dict(zip(right["right_id"], right["title"]))
    left_titles = dict(zip(left["left_id"], left["title"]))
    for left_id, right_id in list(missed)[:5]:
        print(f"  {left_titles[left_id]}\n    vs {right_titles[right_id]}")

    if SHIPPED not in recorded:
        raise SystemExit(f"{SHIPPED} never measured - {sorted(recorded)}")
    # No timestamp: an unchanged run must produce an unchanged file, or
    # the diff stops answering "did any number move?".
    with open(RESULTS, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "k": SHIPPED_K,
                "left_records": len(left),
                "neighbours": list(NEIGHBOURS),
                "order": [name for name, _ in schemes],
                "possible_pairs": total_possible,
                "results": recorded,
                "right_records": len(right),
                "shipped": SHIPPED,
                "true_pairs": len(truth_pairs),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"\nwrote {RESULTS} - {len(recorded)} schemes")


if __name__ == "__main__":
    main()