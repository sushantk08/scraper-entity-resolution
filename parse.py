"""Read saved HTML and write records to CSV. No browser, no network.

    python parse.py books
"""

import csv
import sys
from pathlib import Path

import sites

PAGES_ROOT = Path("pages")
DATA_DIR = Path("data")


def parse_saved_pages(site):
    page_dir = PAGES_ROOT / site["name"]
    paths = sorted(page_dir.glob("*.html"))
    if not paths:
        raise SystemExit(f"ERROR: no HTML in {page_dir}/ — run fetch.py first")

    records = []
    for path in paths:
        records.extend(site["parse"](path.read_text(encoding="utf-8")))
    print(f"{len(paths)} pages -> {len(records)} records")
    return records


def merge_details(site, records):
    """Add each record's detail fields. Returns the number still missing."""
    detail_dir = PAGES_ROOT / f"{site['name']}_detail"
    blank = {field: "" for field in site["detail_fields"]}
    missing = 0
    for record in records:
        path = detail_dir / f"{sites.slug_for(record['detail_url'])}.html"
        if path.exists():
            record.update(site["detail_parse"](path.read_text(encoding="utf-8")))
        else:
            record.update(blank)
            missing += 1
    return missing


def save_csv(site, records, fields):
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{site['name']}.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python parse.py <books|quotes>")
    site = sites.get_site(sys.argv[1])

    records = parse_saved_pages(site)
    if not records:
        raise SystemExit("ERROR: parsed 0 records — the page structure probably changed")

    fields = list(site["csv_fields"])
    detail_dir = PAGES_ROOT / f"{site['name']}_detail"
    if site["detail_parse"] and detail_dir.exists():
        missing = merge_details(site, records)
        fields += site["detail_fields"]
        print(f"detail pages: {len(records) - missing} merged, {missing} missing")

    path = save_csv(site, records, fields)
    print(f"saved {len(records)} records to {path}")


if __name__ == "__main__":
    main()