"""Make a deliberately messy second copy of the catalogue, with known answers.

    python perturb.py

Duplicates created by corrupting known records give ground truth that is exact
by construction. That proves the matcher works on noise WE chose — it is not a
substitute for a human-labelled benchmark, and the README must say so.

Only 70% of the left-hand records get a partner, so there are records with no
correct answer. Without those, precision cannot be measured.
"""

import random
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
SOURCE = DATA_DIR / "books_clean.csv"
RIGHT_OUT = DATA_DIR / "books_right.csv"
TRUTH_OUT = DATA_DIR / "truth.csv"

SEED = 20260826  # fixed, so the numbers are reproducible
MATCHED_FRACTION = 0.7

ABBREVIATIONS = {"and": "&", "volume": "vol.", "first": "1st", "second": "2nd"}


def drop_subtitle(title, rng):
    """'Title: A Memoir' -> 'Title'. Very common in real catalogues."""
    return re.split(r"[:(\[]", title)[0].strip()


def swap_first_two_words(title, rng):
    words = title.split()
    if len(words) >= 2:
        words[0], words[1] = words[1], words[0]
    return " ".join(words)


def typo(title, rng):
    """A dropped or transposed character — what a human keying data does."""
    letters = list(title)
    if len(letters) < 4:
        return title
    position = rng.randrange(1, len(letters) - 1)
    if rng.random() < 0.5:
        del letters[position]
    else:
        letters[position], letters[position + 1] = letters[position + 1], letters[position]
    return "".join(letters)


def abbreviate(title, rng):
    return " ".join(ABBREVIATIONS.get(word.lower(), word) for word in title.split())


def strip_punctuation(title, rng):
    return re.sub(r"[^\w\s]", "", title)


PERTURBATIONS = [
    drop_subtitle,
    swap_first_two_words,
    typo,
    abbreviate,
    strip_punctuation,
]


def perturb_title(title, rng):
    changed = title
    for change in rng.sample(PERTURBATIONS, rng.randint(1, 3)):
        changed = change(changed, rng)
    changed = changed.strip()
    # A perturbation can eat a short title entirely; keep the original then.
    return changed or title


def main():
    if not SOURCE.exists():
        raise SystemExit(f"ERROR: {SOURCE} missing — run clean.py first")

    frame = pd.read_csv(SOURCE)
    if not frame["upc"].is_unique:
        raise SystemExit("ERROR: upc is not unique — it cannot be the record id")

    rng = random.Random(SEED)
    matched = frame.sample(frac=MATCHED_FRACTION, random_state=SEED)

    rows = []
    truth = []
    for position, (_, record) in enumerate(matched.iterrows()):
        right_id = f"R{position:04d}"
        rows.append(
            {
                "right_id": right_id,
                "title": perturb_title(str(record["title"]), rng),
                "category": record["category"],
                "price_gbp": record["price_listing_gbp"],
            }
        )
        truth.append({"left_id": record["upc"], "right_id": right_id})

    right = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED)
    right.to_csv(RIGHT_OUT, index=False)
    pd.DataFrame(truth).to_csv(TRUTH_OUT, index=False)

    unchanged = sum(
        1
        for row, (_, record) in zip(rows, matched.iterrows())
        if row["title"] == str(record["title"])
    )
    print(f"left side       : {len(frame)} records")
    print(f"right side      : {len(right)} records ({len(frame) - len(right)} unmatchable)")
    print(f"true pairs      : {len(truth)}")
    print(f"identical titles: {unchanged} (the easy ones)")
    print("\nexamples:")
    for row, (_, record) in list(zip(rows, matched.iterrows()))[:5]:
        print(f"  {record['title']}\n    -> {row['title']}")


if __name__ == "__main__":
    main()