"""Download each record's own page.

    python fetch_details.py books 5     # just the first 5, to check the parser
    python fetch_details.py books       # all of them

Safe to interrupt with Ctrl-C: pages already on disk are skipped, so re-running
continues where it stopped instead of starting over.
"""

import csv
import sys
import time
from pathlib import Path

import sites
from fetch import DELAY_SECONDS, browser_fetch, make_driver, plain_fetch

DATA_DIR = Path("data")
PAGES_ROOT = Path("pages")


def detail_urls(site):
    path = DATA_DIR / f"{site['name']}.csv"
    if not path.exists():
        raise SystemExit(f"ERROR: {path} missing — run fetch.py then parse.py first")
    with open(path, newline="", encoding="utf-8") as handle:
        return [row["detail_url"] for row in csv.DictReader(handle) if row["detail_url"]]


def fetch_details(site, limit=None):
    out_dir = PAGES_ROOT / f"{site['name']}_detail"
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = detail_urls(site)
    if limit:
        urls = urls[:limit]

    driver = make_driver() if site["needs_browser"] else None
    fetched = 0
    skipped = 0
    failures = []
    try:
        for number, url in enumerate(urls, start=1):
            path = out_dir / f"{sites.slug_for(url)}.html"
            if path.exists():
                skipped += 1
                continue

            try:
                if driver is None:
                    html = plain_fetch(url)
                else:
                    html = browser_fetch(driver, url, site["detail_selector"])
            except Exception as error:
                # One bad page must not end a thousand-page crawl.
                failures.append(f"{url} -> {type(error).__name__}: {error}")
                continue

            if not html:
                failures.append(f"{url} -> empty response")
                continue

            path.write_text(html, encoding="utf-8", newline="")
            fetched += 1
            if fetched % 25 == 0:
                print(f"  {number}/{len(urls)} — {fetched} fetched")
            time.sleep(DELAY_SECONDS)
    finally:
        if driver is not None:
            driver.quit()
    return fetched, skipped, failures


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python fetch_details.py <site> [limit]")
    site = sites.get_site(sys.argv[1])
    if not site["detail_parse"]:
        raise SystemExit(f"{site['name']} has no per-record pages")

    limit = int(sys.argv[2]) if len(sys.argv) == 3 else None
    fetched, skipped, failures = fetch_details(site, limit)

    print(f"fetched {fetched}, already had {skipped}, failed {len(failures)}")
    for failure in failures[:10]:
        print(f"  FAILED {failure}")
    if fetched == 0 and skipped == 0:
        raise SystemExit("ERROR: nothing was fetched")


if __name__ == "__main__":
    main()