"""Download pages and save their HTML to disk. Run this when you want fresh data."""

import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL = "https://quotes.toscrape.com/js/page/{page}/"
PAGES_DIR = Path("pages")

DELAY_SECONDS = 1
MAX_PAGES = 20


def make_driver():
    options = Options()
    options.add_argument("--headless=new")
    return webdriver.Chrome(options=options)


def get_html(driver, url):
    """Load one page. Returns the HTML, or None if no quotes ever appeared."""
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.quote"))
        )
    except TimeoutException:
        return None
    return driver.page_source


def fetch_all():
    PAGES_DIR.mkdir(exist_ok=True)
    driver = make_driver()
    saved = 0
    try:
        for page in range(1, MAX_PAGES + 1):
            html = get_html(driver, BASE_URL.format(page=page))
            if html is None:
                print(f"page {page}: empty, stopping")
                break
            path = PAGES_DIR / f"page-{page:02d}.html"
            # newline="" keeps the file byte-faithful to what the browser gave us.
            path.write_text(html, encoding="utf-8", newline="")
            saved += 1
            print(f"saved {path}")
            time.sleep(DELAY_SECONDS)
    finally:
        driver.quit()
    return saved


saved = fetch_all()

if saved == 0:
    raise SystemExit("ERROR: saved 0 pages — check the URL or the page structure")

print(f"done: {saved} pages in {PAGES_DIR}/")