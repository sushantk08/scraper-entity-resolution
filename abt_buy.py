"""Load Abt-Buy into the same shape our own pipeline already uses.

    python abt_buy.py

The point of this file is that block.py and match.py should not need to change.
If our candidate generation only works on the data we generated ourselves, it
does not work.

Blocking key chosen by measurement, not taste (see normalise_compare.py and
block_multi.py):

  normalise  'CLI-8C' -> 'cli 8c'   separators become spaces
  squash     'CLI-8C' -> 'cli8c'    separators deleted            <- WINNER
  split      'CLI-8C' -> 'cli 8 c'  letter/digit boundaries split  <- REJECTED

squash beats normalise at every k for identical cost (0.981 vs 0.976 at k=6),
because deleting the hyphen makes 'CLI-8C' and 'CLI8C' identical without
breaking a contiguous part number like 'KXTG6700B' apart. Splitting letter-digit
boundaries was tested and lost 5 pairs to gain 2: those long part numbers are
the rarest substrings in the data, and fragmenting them destroys the signal.
Unioning squash with normalise was also tested and rejected — at an equal pair
budget it never won, because two variants of the same string are not diverse
enough to be worth two passes.
"""

import re

import pandas as pd

import block
from inspect_pairs import read_any

ABT = "benchmark/Abt.csv"
BUY = "benchmark/Buy.csv"
MAPPING = "benchmark/abt_buy_perfectMapping.csv"


def normalise(text):
    """Lowercase, keep letters and digits, collapse everything else to spaces.

    Kept for comparison scripts and for readable debug output. NOT the blocking
    key — squash measured better.
    """
    lowered = str(text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", lowered)).strip()


def squash(text):
    """Lowercase, keep letters and digits, delete every separator.

    This is the blocking key. Deleting rather than spacing is what makes
    'CLI-8C' and 'CLI8C' collide.
    """
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def money(text):
    """'$399.00' -> 399.0 ; '' -> NaN. Blank must stay missing, not become 0."""
    digits = re.sub(r"[^0-9.]", "", str(text))
    try:
        return float(digits) if digits else float("nan")
    except ValueError:
        return float("nan")


def load():
    """Return (left, right, truth) with the same columns block.py expects."""
    abt = read_any(ABT)
    buy = read_any(BUY)
    mapping = read_any(MAPPING)

    left = pd.DataFrame({
        "left_id": abt["id"].astype(str).str.strip(),
        "name": abt["name"].astype(str).str.strip(),
        "key": abt["name"].map(squash),
        "key_spaced": abt["name"].map(normalise),
        "description": abt["description"].astype(str).str.strip(),
        # Abt has no manufacturer column at all, so this side is blank by
        # construction. Left in place so the shape matches; any feature using it
        # is inert on this dataset until a brand is derived from the name.
        "manufacturer": "",
        "price": abt["price"].map(money),
    })

    right = pd.DataFrame({
        "right_id": buy["id"].astype(str).str.strip(),
        "name": buy["name"].astype(str).str.strip(),
        "key": buy["name"].map(squash),
        "key_spaced": buy["name"].map(normalise),
        "description": buy["description"].astype(str).str.strip(),
        "manufacturer": buy["manufacturer"].astype(str).str.strip(),
        "price": buy["price"].map(money),
    })

    truth = pd.DataFrame({
        "left_id": mapping["idAbt"].astype(str).str.strip(),
        "right_id": mapping["idBuy"].astype(str).str.strip(),
    })
    return left, right, truth


def main():
    left, right, truth = load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    total_possible = len(left) * len(right)

    print(f"\n{len(left)} left x {len(right)} right = {total_possible:,} possible pairs")
    print(f"{len(truth_pairs)} true pairs "
          f"({len(truth_pairs) / total_possible:.5%} of all pairs)")
    print(f"blank prices: left {left['price'].isna().mean():.1%}, "
          f"right {right['price'].isna().mean():.1%}")
    print(f"blank descriptions: left {(left['description'] == '').mean():.1%}, "
          f"right {(right['description'] == '').mean():.1%}")

    ceiling = len(set(truth["right_id"])) / len(truth_pairs)
    print(f"\nrecall ceiling for a one-partner-per-right-record rule: {ceiling:.3f}")
    print("  (5 right records genuinely have two true partners here)")

    exact = block.exact_key_pairs(left, right)
    print(f"\nblocking key = squash (chosen by measurement, see module docstring)")
    print(f"{'method':<22}{'recall':>9}{'pairs kept':>13}{'% of space':>12}")
    print(f"{'exact key match':<22}{len(exact & truth_pairs) / len(truth_pairs):>9.3f}"
          f"{len(exact):>13,}{len(exact) / total_possible:>11.2%}")

    for k in (1, 3, 5, 6, 10, 20):
        pairs = block.nearest_neighbour_pairs(left, right, k)
        recall = len(pairs & truth_pairs) / len(truth_pairs)
        print(f"{'k = ' + str(k):<22}{recall:>9.3f}{len(pairs):>13,}"
              f"{len(pairs) / total_possible:>11.2%}")

    missed = sorted(truth_pairs - block.nearest_neighbour_pairs(left, right, 10))
    print(f"\n{len(missed)} true pairs unreachable at k=10. The remaining ceiling is")
    print("not a string-metric problem — some Buy titles carry no product identity:")
    left_names = dict(zip(left["left_id"], left["name"]))
    right_names = dict(zip(right["right_id"], right["name"]))
    for left_id, right_id in missed[:6]:
        print(f"\n  left  {left_names.get(left_id, '?')!r}")
        print(f"  right {right_names.get(right_id, '?')!r}")


if __name__ == "__main__":
    main()