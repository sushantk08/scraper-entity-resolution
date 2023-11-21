"""Check a live site still has the structure we expect.

    python canary.py books

This tests the WEBSITE, not our code. It is meant to fail when the site changes.
"""

import sys

import sites
from fetch import browser_fetch, make_driver, plain_fetch


def fetch_one(site, url, selector):
    if not site["needs_browser"]:
        return plain_fetch(url)
    driver = make_driver()
    try:
        return browser_fetch(driver, url, selector)
    finally:
        driver.quit()


def check_fields(records, fields, where):
    problems = []
    for field in fields:
        empty = sum(1 for record in records if not record[field])
        if empty:
            problems.append(
                f"{where}: {empty} of {len(records)} records have an empty '{field}'"
            )
    return problems


def check(site):
    url = site["urls"][0]
    html = fetch_one(site, url, site["record_selector"])
    if not html:
        return [f"no HTML returned from {url}"]

    records = site["parse"](html)
    if not records:
        return [f"parsed 0 records from {url}"]

    problems = check_fields(records, site["required_fields"], "listing")

    if site["detail_parse"]:
        detail_url = records[0]["detail_url"]
        if not detail_url:
            problems.append("listing: first record has no detail_url")
        else:
            detail_html = fetch_one(site, detail_url, site["detail_selector"])
            if not detail_html:
                problems.append(f"no HTML returned from {detail_url}")
            else:
                problems += check_fields(
                    [site["detail_parse"](detail_html)],
                    site["detail_required_fields"],
                    "detail",
                )

    return problems


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python canary.py <books|quotes>")
    site = sites.get_site(sys.argv[1])
    problems = check(site)
    if problems:
        print(f"CANARY FAILED for {site['name']}:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print(f"canary OK — {site['name']} structure unchanged")


if __name__ == "__main__":
    main()