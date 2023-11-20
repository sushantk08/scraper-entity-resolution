import csv

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

URL = "https://quotes.toscrape.com/js/"
OUTPUT_FILE = "quotes.csv"
FIELDS = ["author", "text", "tags"]


def get_html(url):
    """Open a real browser, let the page's JavaScript run, return the finished HTML."""
    options = Options()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.quote"))
        )
        return driver.page_source
    finally:
        # Runs even if something above fails, so we never leave a stray
        # Chrome process running in the background.
        driver.quit()


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


def save_csv(records, path):
    """Write records to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


html = get_html(URL)
records = parse_quotes(html)
save_csv(records, OUTPUT_FILE)
print(f"saved {len(records)} records to {OUTPUT_FILE}")