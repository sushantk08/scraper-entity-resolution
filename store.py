"""Normalised storage for the pipeline's records and its matching decisions.

    python store.py

SQLite, deliberately, and not Postgres yet. The part worth designing is the schema
and the loader; SQLite lets both be tested offline in milliseconds like everything
else here, because "no network in CI" is a rule this project has kept since its
first commit. Only connect() has to change to point this at Postgres.

Three tables, and the separation between the first two is the entire design:

    record         one row per scraped record, per source. Nothing merged, ever.
    entity         one row per real-world thing that several records refer to.
    record_entity  which records the matcher decided are the same thing.

A pipeline that overwrites its scraped rows with merged ones can never be asked
"why did you merge these two", and can never be re-run with a better matcher. So
raw rows are immutable and the decisions live in a separate table that can be
dropped and rebuilt from nothing. A bad matching run is then a DELETE, not a
re-scrape.

THREE SOURCES, NOT ONE. This used to load a single CSV, which made the source
column decorative: 1000 books with 1000 distinct UPCs have nothing to resolve, so
union-find over them would produce 1000 entities of one record each. The two sides
of the generated benchmark are now loaded as their own sources, which is what
gives resolve.py something real to merge and truth.csv something to score it
against. The catalogue stays loaded as a third source that should merge nothing -
a negative control, in the same spirit as the canary's positive one.

Each source carries its own column map, because price_incl_tax_gbp,
price_listing_gbp and price_gbp are the same field under three names. That is a
small live example of why entity resolution is a problem in the first place.

UNIQUE (source, external_id) is load-bearing rather than decorative: it makes
re-running this script idempotent, and it turns "the scrape produced the same
product twice" from a silent duplicate into a reported count.

Blanks are stored as NULL, never 0.0 and never an empty string - the same rule the
matcher needed on Abt-Buy, enforced one layer earlier.

NOTHING IS SILENTLY SUBSTITUTED. An earlier version fell back to the raw
data/books.csv when the clean file was absent, and fell back to a different price
column when the expected one was missing. Both were quiet. A missing required file
is now a refusal naming the script to run, and a missing mapped column is a loud
ABSENT line saying which database column will be NULL for every row.

EVERY COLUMN IN THE CSV IS EITHER STORED, EXPLAINED, OR THE KEY. The first version
of this file listed the columns it wanted and reported the rest as "ignored", which
correctly caught that it had silently stored 1000 NULL prices - clean.py names them
price_incl_tax_gbp, not price. A dump of ignored names is not a review, though, so
an unexplained column is a warning: either map it or write down why not.
"""

import os
import sqlite3

import pandas as pd

DB_PATH = os.path.join("data", "pipeline.db")

# One entry per source. path is the only file that source is ever read from;
# columns maps that file's names onto the schema's; external_id names the natural
# key; build_with is the script that produces the file, quoted in the error when
# it is missing.
SOURCES = {
    "books": {
        "path": os.path.join("data", "books_clean.csv"),
        "raw_path": os.path.join("data", "books.csv"),
        "build_with": "clean.py",
        "required": True,
        "external_id": "upc",
        "columns": {
            "title": "title",
            "category": "category",
            "upc": "upc",
            "price_incl_tax_gbp": "price",
            "rating_stars": "rating",
            "detail_url": "source_url",
            "description": "description",
        },
        "ignored": {
            "title_key": "derived; block.py recomputes it, storing it would go stale",
            "description_words": "derived from the description text, which is stored",
            "price_listing_gbp": "redundant with price_incl_tax_gbp; audited below",
            "price_excl_tax_gbp": "redundant; tax is 0 here, which the audit checks",
            "tax_gbp": "constant 0 across all 1000 books",
            "in_stock": "constant true across all 1000 books",
            "review_count": "constant 0 across all 1000 books",
            "stock_count": "availability when scraped, not a property of the book",
        },
    },
    "books_left": {
        "path": os.path.join("data", "books_left.csv"),
        "build_with": "perturb.py",
        "required": False,
        "external_id": "upc",
        "columns": {
            "title": "title",
            "category": "category",
            "upc": "upc",
            "price_listing_gbp": "price",
        },
        "ignored": {},
    },
    "books_right": {
        "path": os.path.join("data", "books_right.csv"),
        "build_with": "perturb.py",
        "required": False,
        # right_id is this side's only identifier. It is the key rather than a
        # stored field, which is why it is not in columns and not a warning.
        "external_id": "right_id",
        "columns": {
            "title": "title",
            "category": "category",
            "price_gbp": "price",
        },
        "ignored": {},
    },
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS record (
    record_id   INTEGER PRIMARY KEY,
    source      TEXT    NOT NULL,
    external_id TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    category    TEXT,
    upc         TEXT,
    price       REAL,
    rating      INTEGER,
    source_url  TEXT,
    description TEXT,
    UNIQUE (source, external_id)
);

CREATE INDEX IF NOT EXISTS record_by_source   ON record (source);
CREATE INDEX IF NOT EXISTS record_by_category ON record (source, category);

CREATE TABLE IF NOT EXISTS entity (
    entity_id INTEGER PRIMARY KEY,
    label     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS record_entity (
    record_id INTEGER NOT NULL REFERENCES record (record_id),
    entity_id INTEGER NOT NULL REFERENCES entity (entity_id),
    PRIMARY KEY (record_id, entity_id)
);
"""

STORED = ["title", "category", "upc", "price", "rating", "source_url", "description"]
EXPECTED_RECORD_COLUMNS = {"record_id", "source", "external_id", *STORED}


def connect(path=DB_PATH):
    """The only function that knows what database this is. Swap it for Postgres."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite leaves this OFF by default
    return conn


def create(conn):
    """Create the tables, and refuse to run against an older shape of them.

    CREATE TABLE IF NOT EXISTS silently does nothing when the table exists with
    different columns, which is exactly how a schema change turns into a column
    full of NULLs. This database is derived data and rebuilds in a second, so the
    honest fix is to say so and stop. A database holding anything irreplaceable
    would need a migration here instead.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(record)")}
    if existing and existing != EXPECTED_RECORD_COLUMNS:
        missing = sorted(EXPECTED_RECORD_COLUMNS - existing)
        extra = sorted(existing - EXPECTED_RECORD_COLUMNS)
        raise SystemExit(
            f"{DB_PATH} has an older schema (missing {missing}, unexpected {extra}).\n"
            f"It is derived data - delete it and re-run:\n"
            f"    rm {DB_PATH}            (Git Bash)\n"
            f"    Remove-Item {DB_PATH}   (PowerShell)"
        )
    conn.executescript(SCHEMA)
    conn.commit()


def _text(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _number(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def review_columns(frame, config):
    """Every CSV column is stored, explained, or the key. Warn about the rest."""
    mapping = config["columns"]
    ignored = config["ignored"]
    key = config["external_id"]

    stored = [name for name in mapping if name in frame.columns]
    absent = [name for name in mapping if name not in frame.columns]
    print(f"  stored:  {', '.join(f'{n} -> {mapping[n]}' for n in stored)}")
    for name in absent:
        print(f"  ABSENT:  {name} -> {mapping[name]} will be NULL for every row")
    if key in frame.columns and key not in mapping:
        print(f"  key:     {key} identifies the record; not stored as a field")

    for name in sorted(ignored):
        if name in frame.columns:
            print(f"  skipped: {name} - {ignored[name]}")

    unexplained = [
        name for name in frame.columns
        if name not in mapping and name not in ignored and name != key
    ]
    for name in unexplained:
        print(f"  WARNING: {name} is neither stored, explained, nor the key. Map it,")
        print(f"           or give this source's 'ignored' a reason, so the next")
        print(f"           person knows it was a choice and not an oversight.")


def audit_prices(frame):
    """Three price columns should be identical here. Check rather than assume."""
    names = [
        name for name in
        ("price_listing_gbp", "price_excl_tax_gbp", "price_incl_tax_gbp")
        if name in frame.columns
    ]
    if len(names) < 2:
        return
    prices = frame[names].apply(pd.to_numeric, errors="coerce")
    spread = prices.max(axis=1) - prices.min(axis=1)
    disagreeing = int((spread > 0.005).sum())
    print(f"  price audit: {', '.join(names)}")
    if disagreeing:
        worst = spread.idxmax()
        detail = ", ".join(f"{name}={prices.at[worst, name]:.2f}" for name in names)
        print(f"    {disagreeing} rows disagree by more than a penny. Worst: {detail}")
        print("    storing one of them therefore loses information - decide which.")
    else:
        print("    all rows agree to the penny, so storing one loses nothing")


def choose_external_id(frame, config):
    """Use the source's declared key. Fall back to row position, and say so.

    Row position changes if the scrape order or the benchmark split ever changes,
    and every stored decision then points at the wrong record, so the fallback is
    reported loudly rather than taken quietly.
    """
    column = config["external_id"]
    if column in frame.columns:
        values = frame[column].astype(str).str.strip()
        blank = int(((values == "") | (values == "nan")).sum())
        duplicated = int(values.duplicated().sum())
        if not blank and not duplicated:
            print(f"  external_id: {column}, {len(values)} values, all unique")
            return values
        print(f"  external_id: {column} REJECTED - {duplicated} duplicated, "
              f"{blank} blank.")
    else:
        print(f"  external_id: no {column} column in this file.")
    print("               Falling back to row position, which invalidates every")
    print("               stored decision if the input is ever reordered.")
    return pd.Series([f"row-{i}" for i in range(len(frame))], index=frame.index)


def load_records(conn, frame, source, config):
    review_columns(frame, config)
    audit_prices(frame)
    if "title" not in frame.columns:
        raise SystemExit(f"refusing to load {source}: no title column, "
                         f"nothing to match on")

    mapping = config["columns"]
    reverse = {database: csv for csv, database in mapping.items()}
    external_id = choose_external_id(frame, config)

    rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        values = []
        for column in STORED:
            source_column = reverse.get(column)
            raw = row.get(source_column) if source_column else None
            values.append(
                _number(raw) if column in ("price", "rating") else _text(raw)
            )
        rows.append((source, external_id.iloc[position], *values))

    # COALESCE on update, deliberately: re-running from a CSV that has lost a
    # column must not delete data an earlier run loaded. The cost is that
    # genuinely correcting a value to NULL needs an explicit DELETE.
    assignments = ",\n            ".join(
        f"{column} = COALESCE(excluded.{column}, record.{column})"
        for column in STORED
    )
    statement = (
        f"INSERT INTO record (source, external_id, {', '.join(STORED)}) "
        f"VALUES ({', '.join('?' * (2 + len(STORED)))}) "
        f"ON CONFLICT (source, external_id) DO UPDATE SET\n            {assignments}"
    )

    def stored_count():
        return conn.execute(
            "SELECT COUNT(*) FROM record WHERE source = ?", (source,)
        ).fetchone()[0]

    before = stored_count()
    conn.executemany(statement, rows)
    conn.commit()
    inserted = stored_count() - before
    print(f"  {len(rows)} rows offered, {inserted} inserted, "
          f"{len(rows) - inserted} updated in place")


def report(conn):
    total = conn.execute("SELECT COUNT(*) FROM record").fetchone()[0]
    print(f"\nwhat is in the database now: {total} records")
    sources = list(conn.execute(
        "SELECT source, COUNT(*) FROM record GROUP BY source ORDER BY 1"
    ))
    for source, count in sources:
        print(f"    {source:<14}{count:>6}")

    print("\n  how much of each column is populated, per source. A column full for")
    print("  one source and empty for another is the normal state of affairs here,")
    print("  not a bug: two catalogues rarely carry the same fields.")
    for source, count in sources:
        print(f"\n    {source}  ({count} records)")
        for column in STORED:
            filled = conn.execute(
                f"SELECT COUNT({column}) FROM record WHERE source = ?", (source,)
            ).fetchone()[0]
            flag = "  <- empty" if filled == 0 else ""
            print(f"      {column:<14}{filled:>6} of {count}{flag}")
        prices = conn.execute(
            "SELECT MIN(price), AVG(price), MAX(price) FROM record "
            "WHERE source = ? AND price IS NOT NULL", (source,)
        ).fetchone()
        if prices[0] is not None:
            print(f"      price range   {prices[0]:>6.2f} to {prices[2]:.2f}, "
                  f"mean {prices[1]:.2f}")

    print("\n  NULL, not 0.0 and not an empty string: a blank must never be able to")
    print("  read as agreement once the matcher gets hold of it.")

    print("\n  largest categories across all sources:")
    for name, count in conn.execute(
        "SELECT COALESCE(category, '(none)'), COUNT(*) FROM record "
        "GROUP BY category ORDER BY 2 DESC LIMIT 5"
    ):
        print(f"    {name:<26}{count:>6}")

    print("\n  the entity tables stay empty until resolve.py writes to them:")
    for table in ("entity", "record_entity"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"    {table:<16}{count:>6}")


def main():
    conn = connect()
    create(conn)

    loaded = []
    for source, config in SOURCES.items():
        path = config["path"]
        if not os.path.exists(path):
            raw = config.get("raw_path")
            if config["required"] and raw and os.path.exists(raw):
                raise SystemExit(
                    f"refusing to load: {path} is missing, but {raw} is here.\n"
                    f"{raw} is the raw scrape - prices as text, nothing validated,\n"
                    f"no checks run. Loading it would put data in the database that\n"
                    f"{config['build_with']} never agreed to write, which is the one\n"
                    f"thing the validation layer exists to prevent. Build it first:\n"
                    f"    python {config['build_with']}"
                )
            if config["required"]:
                raise SystemExit(
                    f"refusing to load: {path} does not exist.\n"
                    f"Build it with {config['build_with']} - the order is in the README."
                )
            print(f"\n{source}: {path} absent, skipping. "
                  f"Build it with {config['build_with']} to resolve against it.")
            continue

        frame = pd.read_csv(path)
        print(f"\n{source}: {path}, {len(frame)} rows, {len(frame.columns)} columns")
        load_records(conn, frame, source, config)
        loaded.append(source)

    report(conn)
    conn.close()
    print(f"\nwritten to {DB_PATH}, gitignored because it is derived data -")
    print(f"rebuild it with this script rather than committing a binary blob.")
    print(f"loaded: {', '.join(loaded)}")


if __name__ == "__main__":
    main()