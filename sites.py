"""One dictionary per website, describing how to fetch and read it."""

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def text_of(element):
    """Text of an element, or empty string if it wasn't found."""
    return element.get_text(strip=True) if element else ""


def slug_for(url):
    """A safe, unique filename stem for a detail-page URL.

    .../catalogue/a-light-in-the-attic_1000/index.html  ->  a-light-in-the-attic_1000
    """
    path = urlparse(url).path.strip("/").removesuffix("/index.html")
    return (path.split("/")[-1] or "index")[:120]


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


def parse_book_detail(html):
    """Read the extra fields that only exist on a single book's own page."""
    soup = BeautifulSoup(html, "html.parser")

    # The product information table is a list of label/value rows.
    info = {}
    for row in soup.select("table.table-striped tr"):
        label = text_of(row.select_one("th"))
        if label:
            info[label] = text_of(row.select_one("td"))

    # The description is the paragraph *after* the heading, not inside it.
    description = ""
    heading = soup.select_one("#product_description")
    if heading:
        description = text_of(heading.find_next_sibling("p"))

    # Breadcrumb is Home / Books / <category> / <title>, and only the first
    # three are links — so the last link is the category.
    crumbs = [text_of(anchor) for anchor in soup.select("ul.breadcrumb li a")]
    category = crumbs[-1] if len(crumbs) >= 3 else ""

    return {
        "upc": info.get("UPC", ""),
        "price_excl_tax": info.get("Price (excl. tax)", ""),
        "price_incl_tax": info.get("Price (incl. tax)", ""),
        "tax": info.get("Tax", ""),
        "stock": info.get("Availability", ""),
        "review_count": info.get("Number of reviews", ""),
        "category": category,
        "description": description,
    }


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
    # Each record links to its own page with more fields on it.
    "detail_selector": "div.product_main",
    "detail_fields": [
        "upc",
        "price_excl_tax",
        "price_incl_tax",
        "tax",
        "stock",
        "review_count",
        "category",
        "description",
    ],
    "detail_required_fields": ["upc", "category"],
    "detail_parse": parse_book_detail,
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
    # No per-record pages on this site.
    "detail_parse": None,
}

ALL_SITES = {"books": BOOKS, "quotes": QUOTES}


def get_site(name):
    if name not in ALL_SITES:
        choices = ", ".join(sorted(ALL_SITES))
        raise SystemExit(f"unknown site '{name}' — choose one of: {choices}")
    return ALL_SITES[name]