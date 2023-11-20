import csv
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL = "https://quotes.toscrape.com/js/page/{page}/"
OUTPUT_FILE = "quotes.csv"
FIELDS = ["author", "text", "tags"]

DELAY_SECONDS = 1   # pause between page requests
MAX_PAGES = 20      # safety bound so a bug can't loop forever


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
        # Past the last page there are no quotes to wait for.
        return None
    return driver.page_source


def parse_quotes(html):
    """Turn HTML into a list of records. No browser involved, no network."""
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


def scrape_all():
    """Walk pages until one has no quotes, collecting records as we go."""
    driver = make_driver()
    records = []
    try:
        for page in range(1, MAX_PAGES + 1):
            html = get_html(driver, BASE_URL.format(page=page))
            if html is None:
                print(f"page {page}: empty, stopping")
                break
            page_records = parse_quotes(html)
            records.extend(page_records)
            print(f"page {page}: {len(page_records)} quotes")
            time.sleep(DELAY_SECONDS)
    finally:
        driver.quit()
    return records


def save_csv(records, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


records = scrape_all()

if not records:
    raise SystemExit("ERROR: scraped 0 records — the page structure probably changed")

save_csv(records, OUTPUT_FILE)
print(f"saved {len(records)} records to {OUTPUT_FILE}")