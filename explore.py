"""Check whether a site can be scraped politely, before writing any parser for it.

For each URL, reports: whether a plain download works, whether a browser is
needed, and whether the site is serving a bot challenge instead of content.
"""

import urllib.error
import urllib.request

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

USER_AGENT = "PortfolioLearningScraper/0.1 (personal study project)"

# A bot gate almost always announces itself somewhere in the HTML.
GATE_MARKERS = [
    "verify_human",
    "verify you are human",
    "captcha",
    "cf-challenge",
    "just a moment",
    "access denied",
    "enable javascript and cookies",
]

URLS = [
    # Positive control: we know this one works. If it reports a problem,
    # the probe itself is broken, not the site.
    "https://quotes.toscrape.com/js/",
    "https://books.toscrape.com/",
    "https://www.scrapethissite.com/pages/forms/",
]


def find_gates(html):
    lowered = html.lower()
    return [marker for marker in GATE_MARKERS if marker in lowered]


def plain_download(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, ""
    except Exception as error:
        # A probe should report failures, never crash on them.
        return f"error: {type(error).__name__}", ""


def main():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument(f"--user-agent={USER_AGENT}")
    driver = webdriver.Chrome(options=options)
    try:
        for url in URLS:
            print(f"\n=== {url}")
            status, plain = plain_download(url)
            print(f"  plain download status : {status}")
            print(f"  plain size            : {len(plain)} chars")

            driver.get(url)
            rendered = driver.page_source
            print(f"  browser size          : {len(rendered)} chars")

            gates = find_gates(plain) + find_gates(rendered)
            if gates:
                print(f"  VERDICT: bot challenge ({set(gates)}) — do not scrape")
            elif not plain:
                print("  VERDICT: plain download failed — browser required")
            elif abs(len(plain) - len(rendered)) < len(rendered) * 0.2:
                print("  VERDICT: usable, no browser needed (plain HTML)")
            else:
                print("  VERDICT: usable, browser needed (JavaScript-rendered)")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()