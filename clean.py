"""Turn the scraped CSV into typed, validated columns.

    python clean.py

Reads data/books.csv, writes data/books_clean.csv — but refuses to write
anything if a conversion silently failed on too many rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
SOURCE = DATA_DIR / "books.csv"
OUTPUT = DATA_DIR / "books_clean.csv"

# More than this fraction unparseable means the source format changed,
# not that a few rows are odd.
MAX_FAILURE_RATE = 0.02

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def money_to_float(series):
    """'£51.77' -> 51.77. Also survives 'Â£51.77', which is what a
    mis-decoded pound sign looks like."""
    digits = series.astype(str).str.replace(r"[^0-9.]", "", regex=True)
    return pd.to_numeric(digits, errors="coerce")


def stock_to_int(series):
    """'In stock (22 available)' -> 22."""
    found = series.astype(str).str.extract(r"\((\d+)\s+available\)")[0]
    return pd.to_numeric(found, errors="coerce")


def normalise_title(series):
    """A comparison key: lowercase, punctuation gone, spaces collapsed.
    This is the column the deduplication step will match on later."""
    return (
        series.astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def clean(frame):
    out = pd.DataFrame()
    out["title"] = frame["title"].str.strip()
    out["title_key"] = normalise_title(frame["title"])
    out["category"] = frame["category"].str.strip()
    out["upc"] = frame["upc"].str.strip()
    out["rating_stars"] = frame["rating"].map(RATING_WORDS)
    out["price_listing_gbp"] = money_to_float(frame["price"])
    out["price_excl_tax_gbp"] = money_to_float(frame["price_excl_tax"])
    out["price_incl_tax_gbp"] = money_to_float(frame["price_incl_tax"])
    out["tax_gbp"] = money_to_float(frame["tax"])
    out["stock_count"] = stock_to_int(frame["stock"])
    out["in_stock"] = out["stock_count"].fillna(0) > 0
    out["review_count"] = pd.to_numeric(frame["review_count"], errors="coerce")
    out["description_words"] = frame["description"].fillna("").str.split().str.len()
    out["detail_url"] = frame["detail_url"]
    return out


def validate(out):
    """Report on the cleaning and return a list of things that look broken."""
    problems = []

    for column in [
        "rating_stars",
        "price_listing_gbp",
        "price_incl_tax_gbp",
        "stock_count",
    ]:
        missing = int(out[column].isna().sum())
        rate = missing / len(out)
        print(f"  {column:20s} unparseable: {missing:5d}  ({rate:.2%})")
        if rate > MAX_FAILURE_RATE:
            problems.append(f"{column}: {rate:.1%} unparseable — source format changed")

    # Two independent sources for the same number: the listing page and the
    # book's own page. If they disagree, the two-phase crawl is misaligned.
    both = out[["price_listing_gbp", "price_incl_tax_gbp"]].dropna()
    disagree = int(
        (~np.isclose(both["price_listing_gbp"], both["price_incl_tax_gbp"])).sum()
    )
    print(f"  listing price vs detail price disagree: {disagree} of {len(both)}")
    if disagree > len(both) * 0.05:
        problems.append(f"{disagree} books have different prices on the two pages")

    # Arithmetic that must hold inside a single page.
    triple = out[["price_excl_tax_gbp", "tax_gbp", "price_incl_tax_gbp"]].dropna()
    bad_sum = int(
        (
            ~np.isclose(
                triple["price_excl_tax_gbp"] + triple["tax_gbp"],
                triple["price_incl_tax_gbp"],
            )
        ).sum()
    )
    print(f"  excl + tax != incl: {bad_sum} of {len(triple)}")
    if bad_sum:
        problems.append(f"{bad_sum} books fail the excl + tax = incl check")

    return problems


def describe(out):
    """Print what the data actually looks like, including useless columns."""
    print(f"  categories: {out['category'].nunique()}")
    print(f"  price range: £{out['price_listing_gbp'].min():.2f} "
          f"– £{out['price_listing_gbp'].max():.2f} "
          f"(mean £{out['price_listing_gbp'].mean():.2f})")
    print(f"  books with no description: "
          f"{int((out['description_words'] == 0).sum())}")

    for column in ["tax_gbp", "review_count", "in_stock"]:
        values = out[column].dropna().unique()
        if len(values) == 1:
            print(f"  note: '{column}' is {values[0]} for every book "
                  "— it carries no information")


def main():
    if not SOURCE.exists():
        raise SystemExit(f"ERROR: {SOURCE} missing — run parse.py first")

    frame = pd.read_csv(SOURCE)
    print(f"read {len(frame)} rows from {SOURCE}")

    out = clean(frame)
    problems = validate(out)
    describe(out)

    if problems:
        print("\nREFUSING TO WRITE — cleaning looks broken:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    out.to_csv(OUTPUT, index=False)
    print(f"\nwrote {len(out)} rows x {len(out.columns)} columns to {OUTPUT}")


if __name__ == "__main__":
    main()