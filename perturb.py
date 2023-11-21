"""Build a two-sided benchmark from the catalogue, with known answers.

    python perturb.py

Splits the books three ways:

  shared      -> in both catalogues, with a perturbed copy on the right
  left only   -> in the left catalogue, no correct answer
  right only  -> in the right catalogue, no correct answer

All three are necessary. If every right-hand record has a partner, then a
one-to-one policy scores 1.000 by ranking alone and never makes a rejection, so
the threshold is never tested. That is exactly what happened before this split
existed.

Duplicates created by corrupting known records give ground truth that is exact
by construction. It proves the matcher works on noise WE chose — it is not a
substitute for a human-labelled benchmark, and the README must say so.

Price and category are corrupted too. Copied across untouched, they would agree
perfectly on every true pair and the matcher would score well by reading a field
that only agrees because this script made it agree.
"""

import random
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
SOURCE = DATA_DIR / "books_clean.csv"
LEFT_OUT = DATA_DIR / "books_left.csv"
RIGHT_OUT = DATA_DIR / "books_right.csv"
TRUTH_OUT = DATA_DIR / "truth.csv"

SEED = 20260826  # fixed, so the numbers are reproducible
SHARED_FRACTION = 0.60
LEFT_ONLY_FRACTION = 0.25  # the remaining 15% becomes right-only

LEFT_COLUMNS = ["upc", "title", "category", "price_listing_gbp"]
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


def perturb_price(price, rng):
    """Two sources rarely quote the same price. Shift 60% by up to 15%."""
    if rng.random() < 0.6:
        return round(float(price) * (1 + rng.uniform(-0.15, 0.15)), 2)
    return float(price)


def perturb_category(category, categories, rng):
    """One in five records has a missing or wrong category."""
    roll = rng.random()
    if roll < 0.10:
        return ""
    if roll < 0.20:
        return rng.choice(categories)
    return category


def main():
    if not SOURCE.exists():
        raise SystemExit(f"ERROR: {SOURCE} missing — run clean.py first")

    frame = pd.read_csv(SOURCE)
    if not frame["upc"].is_unique:
        raise SystemExit("ERROR: upc is not unique — it cannot be the record id")

    rng = random.Random(SEED)
    categories = sorted(frame["category"].dropna().unique())

    shuffled = frame.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    shared_end = int(len(shuffled) * SHARED_FRACTION)
    left_only_end = shared_end + int(len(shuffled) * LEFT_ONLY_FRACTION)

    shared = shuffled.iloc[:shared_end]
    left_only = shuffled.iloc[shared_end:left_only_end]
    right_only = shuffled.iloc[left_only_end:]

    left = pd.concat([shared, left_only])[LEFT_COLUMNS]

    rows = []
    truth = []
    for source, has_partner in ((shared, True), (right_only, False)):
        for _, record in source.iterrows():
            right_id = f"R{len(rows):04d}"
            rows.append(
                {
                    "right_id": right_id,
                    "title": perturb_title(str(record["title"]), rng),
                    "category": perturb_category(record["category"], categories, rng),
                    "price_gbp": perturb_price(record["price_listing_gbp"], rng),
                }
            )
            if has_partner:
                truth.append({"left_id": record["upc"], "right_id": right_id})

    right = pd.DataFrame(rows)
    left.to_csv(LEFT_OUT, index=False)
    right.sample(frac=1.0, random_state=SEED).to_csv(RIGHT_OUT, index=False)
    pd.DataFrame(truth).to_csv(TRUTH_OUT, index=False)

    print(f"catalogue        : {len(frame)} books")
    print(f"left side        : {len(left)} records "
          f"({len(left_only)} with no correct answer)")
    print(f"right side       : {len(right)} records "
          f"({len(right_only)} with no correct answer)")
    print(f"true pairs       : {len(truth)}")
    print(f"possible pairs   : {len(left) * len(right):,}")
    print("\nexamples:")
    for row, (_, record) in list(zip(rows, shared.iterrows()))[:4]:
        print(f"  {record['title']}  [{record['category']}, "
              f"£{record['price_listing_gbp']}]")
        print(f"    -> {row['title']}  [{row['category']}, £{row['price_gbp']}]")


if __name__ == "__main__":
    main()