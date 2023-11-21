"""Tests for the cleaning step. Pure functions, no files, no network."""

import numpy as np
import pandas as pd

import clean


def test_money_handles_currency_and_mojibake():
    series = pd.Series(["£51.77", "Â£13.99", "0.00", "", "nonsense"])
    result = clean.money_to_float(series)
    assert result[0] == 51.77
    assert result[1] == 13.99
    assert result[2] == 0.00
    assert np.isnan(result[3])
    assert np.isnan(result[4])


def test_stock_count_extracted_or_missing():
    series = pd.Series(["In stock (22 available)", "In stock", "Out of stock"])
    result = clean.stock_to_int(series)
    assert result[0] == 22
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_title_key_ignores_case_and_punctuation():
    series = pd.Series(["The Hunger Games (Book #1)", "the   hunger games book 1"])
    result = clean.normalise_title(series)
    assert result[0] == result[1] == "the hunger games book 1"


def test_validate_accepts_good_data():
    good = pd.DataFrame(
        {
            "rating_stars": [3, 4],
            "price_listing_gbp": [10.0, 20.0],
            "price_excl_tax_gbp": [10.0, 20.0],
            "price_incl_tax_gbp": [10.0, 20.0],
            "tax_gbp": [0.0, 0.0],
            "stock_count": [5, 7],
        }
    )
    assert clean.validate(good) == []


def test_validate_catches_a_broken_parse():
    """If a selector breaks, the column comes back empty. The validator must
    say so rather than letting a file of blanks be written."""
    broken = pd.DataFrame(
        {
            "rating_stars": [np.nan, np.nan],
            "price_listing_gbp": [np.nan, np.nan],
            "price_excl_tax_gbp": [10.0, 20.0],
            "price_incl_tax_gbp": [10.0, 20.0],
            "tax_gbp": [0.0, 0.0],
            "stock_count": [np.nan, np.nan],
        }
    )
    problems = clean.validate(broken)
    assert len(problems) >= 3, problems


def test_validate_catches_prices_that_disagree():
    """The listing page and the detail page are independent sources for the
    same number. Silent disagreement would mean the crawl is misaligned."""
    mismatched = pd.DataFrame(
        {
            "rating_stars": [3, 4],
            "price_listing_gbp": [10.0, 20.0],
            "price_excl_tax_gbp": [99.0, 99.0],
            "price_incl_tax_gbp": [99.0, 99.0],
            "tax_gbp": [0.0, 0.0],
            "stock_count": [5, 7],
        }
    )
    problems = clean.validate(mismatched)
    assert any("different prices" in problem for problem in problems), problems