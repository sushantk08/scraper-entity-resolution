"""Parser tests. No browser, no network — just frozen pages from fixtures/."""

from pathlib import Path

import pytest

import sites

FIXTURES = {
    "books": Path("fixtures/books_page.html"),
    "quotes": Path("fixtures/quotes_page.html"),
}


def load(site_name):
    return FIXTURES[site_name].read_text(encoding="utf-8")


@pytest.mark.parametrize("site_name", sorted(FIXTURES))
def test_finds_records(site_name):
    assert len(sites.get_site(site_name)["parse"](load(site_name))) > 0


@pytest.mark.parametrize("site_name", sorted(FIXTURES))
def test_required_fields_are_filled(site_name):
    site = sites.get_site(site_name)
    for record in site["parse"](load(site_name)):
        for field in site["required_fields"]:
            assert record[field], f"empty '{field}' in {record}"


@pytest.mark.parametrize("site_name", sorted(FIXTURES))
def test_empty_page_yields_nothing(site_name):
    assert sites.get_site(site_name)["parse"]("<html></html>") == []


def test_books_fields_look_right():
    records = sites.get_site("books")["parse"](load("books"))
    assert len(records) == 20
    first = records[0]
    assert first["rating"] in {"One", "Two", "Three", "Four", "Five"}
    assert first["detail_url"].startswith("https://books.toscrape.com/catalogue/")


def test_books_titles_are_not_truncated():
    """The link text ends in an ellipsis for long titles, so we read the
    title attribute instead. This test is what stops that regressing."""
    records = sites.get_site("books")["parse"](load("books"))
    assert not any(record["title"].endswith("...") for record in records)


DETAIL_FIXTURE = Path("fixtures/books_detail.html")


def test_detail_page_fields():
    record = sites.parse_book_detail(DETAIL_FIXTURE.read_text(encoding="utf-8"))
    assert len(record["upc"]) == 16, record["upc"]
    assert record["category"], "no category found in the breadcrumb"
    assert record["price_incl_tax"], "no post-tax price found in the info table"
    assert "In stock" in record["stock"], record["stock"]
    assert len(record["description"]) > 20, record["description"]


def test_detail_without_a_description_does_not_crash():
    """Some books have no description. That must give an empty string, not an
    exception, so the canary can report it instead of blowing up."""
    html = """
    <table class="table table-striped"><tr><th>UPC</th><td>abc123</td></tr></table>
    """
    record = sites.parse_book_detail(html)
    assert record["upc"] == "abc123"
    assert record["description"] == ""
    assert record["category"] == ""