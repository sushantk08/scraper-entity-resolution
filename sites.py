"""One dictionary per website, describing how to fetch and read it."""

from urllib.parse import urljoin

from bs4 import BeautifulSoup


def text_of(element):
    """Text of an element, or empty string if it wasn't found."""
    return element.get_text(strip=True) if element else ""


def parse_books(html):
    soup = BeautifulSoup(html, "html.parser")
    base = "https://books.toscrape.com/catalogue/"
    records = []
    for block in soup.select("article.product_pod"):
        link = block.select_one("h3 a")

        # The rating is encoded in the class name, e.g. "star-rating Three".
        rating_element = block.select_one("p.star-rating")
        rating_classes = [
            name
            for name in (rating_element.get("class", []) if rating_element else [])
            if name != "star-rating"
        ]

        records.append(
            {
                # The link's visible text is truncated with an ellipsis;
                # the title attribute holds the full title.
                "title": link.get("title", "").strip() if link else "",
                "price": text_of(block.select_one("p.price_color")),
                "rating": rating_classes[0] if rating_classes else "",
                "availability": text_of(block.select_one("p.availability")),
                "detail_url": (
                    urljoin(base, link["href"])
                    if link and link.has_attr("href")
                    else ""
                ),
            }
        )
    return records


def parse_quotes(html):
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for block in soup.select("div.quote"):
        records.append(
            {
                "author": text_of(block.select_one("small.author")),
                "text": text_of(block.select_one("span.text")),
                "tags": ";".join(
                    tag.get_text(strip=True) for tag in block.select("a.tag")
                ),
            }
        )
    return records


BOOKS = {
    "name": "books",
    "needs_browser": False,  # content is in the initial HTML
    "record_selector": "article.product_pod",
    "urls": [
        f"https://books.toscrape.com/catalogue/page-{page}.html"
        for page in range(1, 51)
    ],
    "csv_fields": ["title", "price", "rating", "availability", "detail_url"],
    "required_fields": ["title", "price"],
    "parse": parse_books,
}

QUOTES = {
    "name": "quotes",
    "needs_browser": True,  # JavaScript builds the list
    "record_selector": "div.quote",
    "urls": [
        f"https://quotes.toscrape.com/js/page/{page}/" for page in range(1, 21)
    ],
    "csv_fields": ["author", "text", "tags"],
    "required_fields": ["author", "text"],
    "parse": parse_quotes,
}

ALL_SITES = {"books": BOOKS, "quotes": QUOTES}


def get_site(name):
    if name not in ALL_SITES:
        choices = ", ".join(sorted(ALL_SITES))
        raise SystemExit(f"unknown site '{name}' — choose one of: {choices}")
    return ALL_SITES[name]