"""Report the true shape of a downloaded entity-resolution benchmark.

    python inspect_pairs.py benchmark/Abt.csv benchmark/Buy.csv benchmark/abt_buy_perfectMapping.csv

Written because I am about to build a loader against this data and would
otherwise be guessing at id types, blank rates and encodings. It also runs the
two checks that cost us a whole step on our own generated benchmark:

  - how many records on EACH side have no correct answer (if zero on either
    side, a one-to-one policy cannot produce a false positive there, so
    precision would be measuring ranking rather than decisions)
  - how many records have MORE than one correct answer (if any do, the
    best-partner-per-record rule is wrong here and caps recall by construction)
"""

import sys
from pathlib import Path

import pandas as pd

SAMPLE_ROWS = 2


def read_any(path):
    """These files are frequently latin-1, not utf-8. Try, then fall back."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
            print(f"  read with encoding={encoding}")
            return frame
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"ERROR: could not decode {path} as utf-8 or latin-1")


def describe(path):
    print(f"\n=== {path}")
    if not Path(path).exists():
        raise SystemExit(f"ERROR: no such file: {path}")

    frame = read_any(path)
    print(f"  {len(frame)} rows x {len(frame.columns)} columns")
    print(f"  {'column':<16}{'blank':>8}{'unique':>9}{'maxlen':>8}  example")
    for column in frame.columns:
        values = frame[column].astype(str).str.strip()
        blank = (values == "").mean()
        example = next((v for v in values if v), "")
        if len(example) > 40:
            example = example[:37] + "..."
        print(f"  {column:<16}{blank:>7.1%}{values.nunique():>9}"
              f"{values.str.len().max():>8}  {example}")

    print(f"\n  first {SAMPLE_ROWS} rows in full:")
    for position, (_, row) in enumerate(frame.head(SAMPLE_ROWS).iterrows()):
        print(f"    --- row {position}")
        for column in frame.columns:
            value = str(row[column])
            if len(value) > 160:
                value = value[:160] + f"... [{len(str(row[column]))} chars]"
            print(f"      {column}: {value}")
    return frame


def check_join(left, right, mapping):
    print("\n=== joining the mapping to both tables")
    left_ids = set(left["id"].astype(str).str.strip())
    right_ids = set(right["id"].astype(str).str.strip())

    map_left, map_right = mapping.columns[0], mapping.columns[1]
    pairs = {
        (str(a).strip(), str(b).strip())
        for a, b in zip(mapping[map_left], mapping[map_right])
    }
    resolved = sum(1 for a, b in pairs if a in left_ids and b in right_ids)
    print(f"  {len(pairs)} unique pairs, {resolved} resolve against both id columns")
    if resolved < len(pairs):
        print("  ^ some ids do not resolve — stray whitespace or a mismatched file")

    print("\n=== records with NO correct answer")
    matched_left = {a for a, _ in pairs}
    matched_right = {b for _, b in pairs}
    lonely_left = len(left_ids - matched_left)
    lonely_right = len(right_ids - matched_right)
    print(f"  left  ({map_left}) : {lonely_left} of {len(left_ids)} "
          f"({lonely_left / len(left_ids):.1%}) have no partner")
    print(f"  right ({map_right}): {lonely_right} of {len(right_ids)} "
          f"({lonely_right / len(right_ids):.1%}) have no partner")
    if lonely_left == 0 or lonely_right == 0:
        print("  WARNING: one side is fully matched — same trap as our own benchmark")
    else:
        print("  good — both sides carry unmatchable records")

    print("\n=== records with MORE THAN ONE correct answer")
    for label, ids in ((map_left, [a for a, _ in pairs]),
                       (map_right, [b for _, b in pairs])):
        counts = pd.Series(ids).value_counts()
        print(f"  {label:<8} max partners {counts.max()}, "
              f"{(counts > 1).sum()} records have 2 or more")
    print("  (any nonzero count means best_per_right is WRONG on this dataset)")

    print("\n=== difficulty preview: pairs per record if we block at k=5")
    print(f"  {len(left_ids)} x {len(right_ids)} = {len(left_ids) * len(right_ids):,} "
          f"possible pairs, {len(pairs)} of them true "
          f"({len(pairs) / (len(left_ids) * len(right_ids)):.5%})")


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: python inspect_pairs.py <left.csv> <right.csv> <mapping.csv>")
    left = describe(sys.argv[1])
    right = describe(sys.argv[2])
    mapping = describe(sys.argv[3])
    check_join(left, right, mapping)


if __name__ == "__main__":
    main()