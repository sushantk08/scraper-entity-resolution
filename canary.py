"""Check the live website still has the structure we expect.

This is NOT a test of our code — it's a test of the website. It is meant to
fail when the site changes its markup, which is exactly when we need to know.
Run it on a schedule, not on every commit.
"""

from fetch import get_html, make_driver
from parse import parse_quotes

URL = "https://quotes.toscrape.com/js/page/1/"


def check():
    driver = make_driver()
    try:
        html = get_html(driver, URL)
    finally:
        driver.quit()

    problems = []

    if html is None:
        # Can't check anything else, so report and stop here.
        return ["page never rendered any div.quote — selector or site changed"]

    records = parse_quotes(html)

    if not records:
        problems.append("parsed 0 records from the live page")

    for field in ["author", "text"]:
        if any(not record[field] for record in records):
            problems.append(f"some records have an empty '{field}'")

    return problems


problems = check()

if problems:
    print("CANARY FAILED — the site's structure has probably changed:")
    for problem in problems:
        print(f"  - {problem}")
    raise SystemExit(1)

print("canary OK — live site structure unchanged")