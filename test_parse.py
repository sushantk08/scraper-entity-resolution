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