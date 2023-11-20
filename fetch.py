"""Download a site's pages and save the HTML to disk.

    python fetch.py books
    python fetch.py quotes

Uses a plain download when the content is already in the HTML, and a real
browser only when JavaScript is needed to build the page.
"""

import sys
import time
import urllib.request
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import sites

PAGES_ROOT = Path("pages")
DELAY_SECONDS = 1
USER_AGENT = "PortfolioLearningScraper/0.1 (personal study project)"


def plain_fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def make_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument(f"--user-agent={USER_AGENT}")
    return webdriver.Chrome(options=options)


def browser_fetch(driver, url, record_selector):
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, record_selector))
        )
    except TimeoutException:
        return None
    return driver.page_source


def fetch_site(site):
    out_dir = PAGES_ROOT / site["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    driver = make_driver() if site["needs_browser"] else None
    saved = 0
    try:
        for number, url in enumerate(site["urls"], start=1):
            if driver is None:
                html = plain_fetch(url)
            else:
                html = browser_fetch(driver, url, site["record_selector"])

            # Parsing here is only to answer "did this page have anything?",
            # which is how we know we've run past the last page.
            if html is None or not site["parse"](html):
                print(f"page {number}: no records, stopping")
                break

            path = out_dir / f"page-{number:03d}.html"
            path.write_text(html, encoding="utf-8", newline="")
            saved += 1
            print(f"saved {path}")
            time.sleep(DELAY_SECONDS)
    finally:
        if driver is not None:
            driver.quit()
    return saved


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python fetch.py <books|quotes>")
    site = sites.get_site(sys.argv[1])
    saved = fetch_site(site)
    if saved == 0:
        raise SystemExit(f"ERROR: saved 0 pages for {site['name']}")
    print(f"done: {saved} pages in {PAGES_ROOT / site['name']}/")


if __name__ == "__main__":
    main()