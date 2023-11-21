"""Look at the raw HTML behind a record, when a parsed field looks wrong.

    python show.py "Add a comment"       # the first few books in that category
    python show.py a897fe39b1053632      # one book, by UPC
"""

import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

import sites

DETAIL_DIR = Path("pages/books_detail")
CLEAN = Path("data/books_clean.csv")
LIMIT = 3


def rows_for(frame, query):
    by_upc = frame[frame["upc"] == query]
    if not by_upc.empty:
        return by_upc
    return frame[frame["category"] == query].head(LIMIT)


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: python show.py "<category>" | <upc>')

    frame = pd.read_csv(CLEAN)
    rows = rows_for(frame, sys.argv[1])
    if rows.empty:
        raise SystemExit(f"nothing matches {sys.argv[1]!r}")

    for _, row in rows.iterrows():
        path = DETAIL_DIR / f"{sites.slug_for(row['detail_url'])}.html"
        print(f"\n=== {row['title']}")
        print(f"    parsed category : {row['category']!r}")
        print(f"    detail url      : {row['detail_url']}")
        if not path.exists():
            print("    (detail page not saved)")
            continue
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        print("    breadcrumb links:")
        for anchor in soup.select("ul.breadcrumb li a"):
            print(f"      {anchor.get_text(strip=True)!r} -> {anchor.get('href')}")


if __name__ == "__main__":
    main()