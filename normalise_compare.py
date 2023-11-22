"""Test one hypothesis about blocking misses before changing the pipeline.

    python normalise_compare.py

Hypothesis: many unreachable pairs are part numbers written 'CLI8C' on one site
and 'CLI-8C' on the other. Our normaliser strips punctuation but never splits a
letter-digit boundary, so those two produce different tokens.

The discipline that matters here is checking BOTH directions. A change that
recovers 12 pairs and quietly loses 9 is nearly worthless, and looking only at
what improved would hide that completely.
"""

import re

import abt_buy
import block

KS = (1, 3, 5, 10, 20)
INSPECT_AT = 10


def split_alnum(text):
    """Like abt_buy.normalise, but also splits letter<->digit boundaries.

    'CLI-8C'  -> 'cli 8 c'
    'CLI8C'   -> 'cli 8 c'   <- these two now agree
    'EZXS88W' -> 'ezxs 88 w' (still distinct from 'ezxs 55 w')
    """
    lowered = str(text).lower()
    spaced = re.sub(r"[^a-z0-9]+", " ", lowered)
    spaced = re.sub(r"(?<=[a-z])(?=\d)", " ", spaced)
    spaced = re.sub(r"(?<=\d)(?=[a-z])", " ", spaced)
    return re.sub(r"\s+", " ", spaced).strip()


VARIANTS = {
    "current": abt_buy.normalise,
    "split letter/digit": split_alnum,
}


def rekey(frame, normaliser):
    out = frame.copy()
    out["key"] = frame["name"].map(normaliser)
    return out


def main():
    left, right, truth = abt_buy.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))

    keyed = {name: (rekey(left, fn), rekey(right, fn)) for name, fn in VARIANTS.items()}

    print("\nexample keys, to see what each variant actually does:")
    for name, (keyed_left, _) in keyed.items():
        example = keyed_left.loc[keyed_left["name"].str.contains("CLI8C"), "key"]
        print(f"  {name:<20}{example.iloc[0]!r}" if len(example) else f"  {name}: (none)")

    print(f"\n{'k':>4}" + "".join(f"{name:>22}" for name in VARIANTS))
    found = {}
    for k in KS:
        row = f"{k:>4}"
        for name, (keyed_left, keyed_right) in keyed.items():
            pairs = block.nearest_neighbour_pairs(keyed_left, keyed_right, k)
            if k == INSPECT_AT:
                found[name] = pairs & truth_pairs
            row += f"{len(pairs & truth_pairs) / len(truth_pairs):>22.3f}"
        print(row)

    before, after = found["current"], found["split letter/digit"]
    gained = sorted(after - before)
    lost = sorted(before - after)

    print(f"\nat k={INSPECT_AT}: recovered {len(gained)} pairs, lost {len(lost)} pairs, "
          f"net {len(gained) - len(lost):+d}")

    left_names = dict(zip(left["left_id"], left["name"]))
    right_names = dict(zip(right["right_id"], right["name"]))

    for label, pairs in (("RECOVERED", gained), ("LOST", lost)):
        print(f"\n=== {label} ({len(pairs)})")
        for left_id, right_id in pairs[:6]:
            print(f"  left  {left_names.get(left_id, '?')!r}")
            print(f"  right {right_names.get(right_id, '?')!r}\n")


if __name__ == "__main__":
    main()