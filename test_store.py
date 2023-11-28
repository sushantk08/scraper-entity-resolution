"""Tests for the storage layer. In-memory databases, hand-built frames, no CSVs.

store.py carries the README's largest design argument - records are immutable, entities
are derived, and a bad matching run is a DELETE rather than a re-scrape - and until now
had no tests at all. The claims worth pinning are the ones that fail quietly:

  - an older record table is refused, because CREATE TABLE IF NOT EXISTS does nothing
    when the table already exists with different columns
  - UNIQUE (source, external_id), not UNIQUE (external_id): books and books_left both
    key on upc, so the same UPC in two catalogues is genuinely two records
  - COALESCE on update, so re-running from a CSV that has lost a column does not delete
    what an earlier run loaded
  - blanks are NULL, never 0.0 and never "", so a missing value cannot read as agreement
    once the matcher gets hold of it
  - a missing price is not quietly taken from the redundant price column, which an
    earlier version of this loader did

NO TEST HERE MAY CHOOSE A DATABASE PATH. store.DB_PATH is relative, so a test that
forgot to redirect it would write into the repo's data/ directory and could overwrite
the database the README's numbers were measured on. Everything below either passes
":memory:" explicitly or monkeypatches store.connect, which takes the decision away.
"""
import os
import sqlite3

import pandas as pd
import pytest

import store
import test_resolve


def database():
    conn = store.connect(":memory:")
    store.create(conn)
    return conn


def books(**overrides):
    """A books_clean.csv-shaped frame, minus the columns store.py ignores anyway."""
    frame = pd.DataFrame({
        "title": ["Arena", "Boxed Set"],
        "category": ["Fiction", "Poetry"],
        "upc": ["u1", "u2"],
        "rating_stars": [4, 2],
        "price_listing_gbp": [10.0, 20.0],
        "price_incl_tax_gbp": [10.0, 20.0],
        "detail_url": ["http://a", "http://b"],
        "description": ["a book", "another book"],
    })
    for name, values in overrides.items():
        frame[name] = values
    return frame


def left():
    """Same UPCs as books(), which is the point of test_two_sources_may_share_a_key."""
    return pd.DataFrame({
        "upc": ["u1", "u2"],
        "title": ["Arena", "Boxed Set"],
        "category": ["Fiction", "Poetry"],
        "price_listing_gbp": [10.0, 20.0],
    })


def right():
    return pd.DataFrame({
        "right_id": ["r1", "r2"],
        "title": ["Arnea", "Boxed Set"],
        "category": ["Fiction", ""],
        "price_gbp": [11.0, 21.0],
    })


def load(conn, frame, source="books"):
    store.load_records(conn, frame, source, store.SOURCES[source])


def count(conn):
    return conn.execute("SELECT count(*) FROM record").fetchone()[0]


def ids(conn):
    return [row[0] for row in
            conn.execute("SELECT external_id FROM record ORDER BY record_id")]


def value(conn, column, external_id):
    return conn.execute(
        f"SELECT {column} FROM record WHERE external_id = ?", (external_id,)
    ).fetchone()[0]


def columns_of(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


# ---------------------------------------------------------------------- the schema


def test_an_older_record_table_is_refused_rather_than_left_alone():
    """CREATE TABLE IF NOT EXISTS silently does nothing when the table exists with
    different columns, which is exactly how a schema change becomes a column full of
    NULLs that nothing complains about."""
    conn = store.connect(":memory:")
    conn.execute("CREATE TABLE record (record_id INTEGER PRIMARY KEY, title TEXT)")
    with pytest.raises(SystemExit):
        store.create(conn)


def test_create_can_run_against_its_own_output():
    conn = database()
    store.create(conn)
    assert columns_of(conn, "record") == store.EXPECTED_RECORD_COLUMNS


def test_dropping_every_decision_leaves_every_record():
    """The design claim the whole schema exists to support. If the loader merged rows
    in place, throwing away a bad matching run would mean scraping the site again."""
    conn = database()
    load(conn, books())
    conn.execute("INSERT INTO entity (entity_id, label) VALUES (1, 'Arena')")
    conn.executemany(
        "INSERT INTO record_entity (record_id, entity_id) VALUES (?, 1)",
        list(conn.execute("SELECT record_id FROM record")),
    )
    conn.commit()
    assert count(conn) == 2
    assert conn.execute("SELECT count(*) FROM record_entity").fetchone()[0] == 2
    before = conn.execute("SELECT count(*), sum(price) FROM record").fetchone()
    conn.execute("DELETE FROM record_entity")
    conn.execute("DELETE FROM entity")
    assert conn.execute("SELECT count(*), sum(price) FROM record").fetchone() == before


def test_a_decision_cannot_point_at_a_missing_entity():
    """SQLite leaves foreign keys OFF by default, so this passes only because connect()
    turns them on. Delete that PRAGMA and record_entity becomes free-text."""
    conn = database()
    load(conn, books())
    record_id = conn.execute("SELECT record_id FROM record").fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO record_entity (record_id, entity_id) VALUES (?, 99)",
            (record_id,),
        )


def test_one_source_cannot_hold_the_same_external_id_twice():
    conn = database()
    load(conn, books())
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO record (source, external_id, title)"
            " VALUES ('books', 'u1', 'Arena')"
        )


def test_two_sources_may_share_a_key():
    """books and books_left both key on upc, so the same UPC really is two records.
    A UNIQUE on external_id alone would make the benchmark impossible to load, and
    would quietly merge two catalogues at load time - before the matcher ever ran."""
    conn = database()
    load(conn, books())
    load(conn, left(), "books_left")
    assert count(conn) == 4


# ---------------------------------------------------------------- blanks and prices


def test_a_blank_category_is_null_not_an_empty_string():
    """resolve.py's same_category feature compares these. An empty string would be a
    value that two records can agree on by both being unknown."""
    conn = database()
    load(conn, right(), "books_right")
    assert value(conn, "category", "r2") is None


def test_a_missing_price_is_not_quietly_taken_from_the_redundant_column():
    """An earlier version of this loader fell back to a different price column when the
    expected one was missing, without saying so. price_listing_gbp still holds 10.00 for
    this row; price must be NULL anyway, because the map names price_incl_tax_gbp."""
    conn = database()
    load(conn, books(price_incl_tax_gbp=[None, 20.0]))
    assert value(conn, "price", "u1") is None
    assert value(conn, "price", "u2") == 20.0


def test_a_whitespace_title_is_refused_by_the_schema():
    """_text() turns "   " into None and title is NOT NULL, so a record with nothing
    to match on cannot be stored at all. Pinned because the alternative - storing ""
    - would give the matcher a title that compares equal to every other blank one."""
    conn = database()
    with pytest.raises(sqlite3.IntegrityError):
        load(conn, books(title=["   ", "Boxed Set"]))


def test_a_source_with_no_title_is_refused():
    conn = database()
    with pytest.raises(SystemExit):
        load(conn, books().drop(columns=["title"]))


def test_disagreeing_price_columns_are_reported(capsys):
    conn = database()
    load(conn, books(price_listing_gbp=[10.0, 99.0]))
    assert "1 rows disagree" in capsys.readouterr().out


def test_agreeing_price_columns_say_that_storing_one_loses_nothing(capsys):
    conn = database()
    load(conn, books())
    assert "all rows agree to the penny" in capsys.readouterr().out


# ------------------------------------------------------------------ column review


def test_a_missing_mapped_column_is_announced_and_stored_as_null(capsys):
    conn = database()
    load(conn, books().drop(columns=["description"]))
    assert "ABSENT:  description" in capsys.readouterr().out
    assert value(conn, "description", "u1") is None


def test_an_unexplained_column_is_a_warning(capsys):
    """Every CSV column is stored, explained, or the key. A dump of ignored names is
    not a review, so anything else has to be a choice somebody wrote down."""
    conn = database()
    load(conn, books(isbn=["i1", "i2"]))
    assert "WARNING: isbn" in capsys.readouterr().out


def test_the_key_column_is_not_reported_as_unexplained(capsys):
    """right_id is books_right's only identifier and is deliberately not a stored
    field, so it must not read as an oversight."""
    conn = database()
    load(conn, right(), "books_right")
    printed = capsys.readouterr().out
    assert "key:     right_id" in printed
    assert "WARNING" not in printed


def test_each_source_maps_its_own_price_column_to_price():
    """price_listing_gbp and price_gbp are the same field under two names - a small
    live example of the problem this whole project is about, and the reason resolve.py
    can read one column instead of three."""
    conn = database()
    load(conn, left(), "books_left")
    load(conn, right(), "books_right")
    assert dict(conn.execute(
        "SELECT source, count(price) FROM record GROUP BY source"
    )) == {"books_left": 2, "books_right": 2}


# ------------------------------------------------------------------- re-running it


def test_reloading_the_same_rows_inserts_nothing(capsys):
    conn = database()
    load(conn, books())
    load(conn, books())
    assert count(conn) == 2
    assert "0 inserted, 2 updated in place" in capsys.readouterr().out


def test_reloading_without_a_column_does_not_erase_what_it_loaded():
    """COALESCE on update. The cost is that correcting a value to NULL needs an
    explicit DELETE; the benefit is that a CSV which lost a column cannot delete
    data by being re-run, which is a far more likely accident."""
    conn = database()
    load(conn, books())
    load(conn, books().drop(columns=["description"]))
    assert value(conn, "description", "u1") == "a book"


def test_a_duplicated_key_is_rejected_and_the_fallback_is_announced(capsys):
    """Row position invalidates every stored decision if the input is ever reordered,
    so falling back to it is reported rather than taken quietly."""
    conn = database()
    load(conn, books(upc=["u1", "u1"]))
    printed = capsys.readouterr().out
    assert "REJECTED" in printed and "row position" in printed
    assert ids(conn) == ["row-0", "row-1"]


def test_a_missing_key_column_falls_back_to_row_position(capsys):
    conn = database()
    load(conn, books().drop(columns=["upc"]))
    assert ids(conn) == ["row-0", "row-1"]
    assert "no upc column" in capsys.readouterr().out


# ------------------------------------------------------------------------- main()
# connect is patched rather than chdir'd: DB_PATH is relative and its default is bound
# at definition, so monkeypatching the constant would not redirect anything.


def patched(monkeypatch):
    conn = store.connect(":memory:")
    monkeypatch.setattr(store, "connect", lambda *a, **k: conn)
    return conn


def test_the_raw_scrape_is_never_loaded_in_place_of_the_clean_file(
        tmp_path, monkeypatch):
    """The refusal that matters most. books.csv is prices-as-text with nothing
    validated; loading it would put data in the database that clean.py never agreed
    to write, which is the one thing the validation layer exists to prevent."""
    patched(monkeypatch)
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    books().to_csv(os.path.join("data", "books.csv"), index=False)
    with pytest.raises(SystemExit) as refusal:
        store.main()
    assert "clean.py" in str(refusal.value)


def test_a_missing_optional_source_is_skipped_rather_than_fatal(
        tmp_path, monkeypatch, capsys):
    """books_left and books_right are built by perturb.py, so a fresh clone that has
    not run it must still load the catalogue instead of refusing."""
    patched(monkeypatch)
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    books().to_csv(os.path.join("data", "books_clean.csv"), index=False)
    store.main()
    printed = capsys.readouterr().out
    assert "loaded: books" in printed
    assert "absent, skipping" in printed


# ------------------------------------------------------- the duplicated test schema


def test_the_resolution_tests_declare_no_column_this_loader_never_creates():
    """test_resolve.py builds its own three-table schema rather than importing this
    one, and a comment there says a test that the two agree belongs here. A subset is
    fine - resolve.py reads six columns. A column declared there and missing here
    would let those tests pass against a database store.py cannot produce."""
    real, fake = database(), test_resolve.schema()
    for table in ("record", "entity", "record_entity"):
        assert columns_of(fake, table) <= columns_of(real, table), table


def test_the_columns_resolve_py_reads_are_all_stored():
    assert {"record_id", "source", "external_id", "title", "category", "price"} \
        <= columns_of(database(), "record")
