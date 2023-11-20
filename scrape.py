from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

URL = "https://quotes.toscrape.com/js/"

# Headless means the browser runs without opening a window.
options = Options()
options.add_argument("--headless=new")

driver = webdriver.Chrome(options=options)
driver.get(URL)

# Wait until the JavaScript has actually put the quotes on the page.
# Without this, we'd sometimes read an empty page and get zero results.
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "div.quote"))
)

html = driver.page_source
driver.quit()

# Selenium's job is done: it ran the JavaScript and handed us the finished HTML.
# BeautifulSoup's job starts here: pull the pieces we want out of that HTML.
soup = BeautifulSoup(html, "html.parser")

for quote in soup.select("div.quote"):
    text = quote.select_one("span.text").get_text(strip=True)
    author = quote.select_one("small.author").get_text(strip=True)
    print(f"{author} — {text}")