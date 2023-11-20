"""Tests for the parser. No browser, no network — just a frozen page from fixtures/."""

from pathlib import Path

from parse import parse_quotes

FIXTURE = Path("fixtures/sample_page.html")


def load_fixture():
    return FIXTURE.read_text(encoding="utf-8")


def test_finds_quotes():
    records = parse_quotes(load_fixture())
    assert len(records) == 10


def test_every_field_is_filled():
    records = parse_quotes(load_fixture())
    for record in records:
        assert record["author"], f"empty author in {record}"
        assert record["text"], f"empty text in {record}"


def test_parses_a_known_quote():
    """A specific, real value from the fixture — proves we extract content,
    not just non-empty strings."""
    records = parse_quotes(load_fixture())
    authors = [record["author"] for record in records]
    assert "Albert Einstein" in authors


def test_empty_page_yields_nothing():
    """The positive control's opposite: given no quotes, we must return
    an empty list rather than inventing records or crashing."""
    assert parse_quotes("<html><body></body></html>") == []