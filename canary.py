"""Check a live site still has the structure we expect.

    python canary.py books

This tests the WEBSITE, not our code. It is meant to fail when the site changes.
"""

import sys

import sites
from fetch import browser_fetch, make_driver, plain_fetch


def check(site):
    url = site["urls"][0]
    if site["needs_browser"]:
        driver = make_driver()
        try:
            html = browser_fetch(driver, url, site["record_selector"])
        finally:
            driver.quit()
    else:
        html = plain_fetch(url)

    if not html:
        return [f"no HTML returned from {url}"]

    records = site["parse"](html)
    problems = []
    if not records:
        problems.append(f"parsed 0 records from {url}")
    for field in site["required_fields"]:
        empty = sum(1 for record in records if not record[field])
        if empty:
            problems.append(
                f"{empty} of {len(records)} records have an empty '{field}'"
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