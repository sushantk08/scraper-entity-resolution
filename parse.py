"""Read saved HTML from disk and write records to CSV. No browser, no network."""

import csv
from pathlib import Path

from bs4 import BeautifulSoup

PAGES_DIR = Path("pages")
OUTPUT_FILE = Path("quotes.csv")
FIELDS = ["author", "text", "tags"]


def parse_quotes(html):
    """Turn one page of HTML into a list of records."""
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for quote in soup.select("div.quote"):
        tags = [tag.get_text(strip=True) for tag in quote.select("a.tag")]
        records.append(
            {
                "author": quote.select_one("small.author").get_text(strip=True),
                "text": quote.select_one("span.text").get_text(strip=True),
                "tags": ";".join(tags),
            }
        )
    return records


def parse_all():
    paths = sorted(PAGES_DIR.glob("*.html"))
    if not paths:
        raise SystemExit(f"ERROR: no HTML files in {PAGES_DIR}/ — run fetch.py first")

    records = []
    for path in paths:
        page_records = parse_quotes(path.read_text(encoding="utf-8"))
        print(f"{path.name}: {len(page_records)} quotes")
        records.extend(page_records)
    return records


def save_csv(records, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


records = parse_all()

if not records:
    raise SystemExit("ERROR: parsed 0 records — the page structure probably changed")

save_csv(records, OUTPUT_FILE)
print(f"saved {len(records)} records to {OUTPUT_FILE}")