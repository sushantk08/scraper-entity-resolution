"""Does blocking on two different keys beat blocking harder on one?

    python block_multi.py

normalise_compare.py established that splitting letter-digit boundaries hurts:
a contiguous part number like 'KXTG6700B' is the rarest, most informative
substring available, and fragmenting it destroys more than it repairs.

So instead of changing the key, add a second one and union the candidates:

    normalise : 'CLI-8C' -> 'cli 8c'   (words preserved, separators become space)
    squash    : 'CLI-8C' -> 'cli8c'    (separators deleted, nothing fragmented)

A union can only increase recall, so recall alone would be a dishonest way to
judge it. The comparison that means something is at an equal pair budget: does
union-at-k beat a single pass at the k that keeps roughly the same number of
pairs? If not, the second key is just an expensive way to raise k.
"""

import re

import abt_buy
import block

KS = (1, 3, 5, 6, 7, 10, 20)


def squash(text):
    """Delete every separator rather than turning it into a space."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def rekey(frame, normaliser):
    out = frame.copy()
    out["key"] = frame["name"].map(normaliser)
    return out


def main():
    left, right, truth = abt_buy.load()
    truth_pairs = set(zip(truth["left_id"], truth["right_id"]))
    total = len(left) * len(right)

    plain = (rekey(left, abt_buy.normalise), rekey(right, abt_buy.normalise))
    tight = (rekey(left, squash), rekey(right, squash))

    print("\nwhat the two keys do to the same name:")
    sample = left.loc[left["name"].str.contains("CLI8C"), "name"].iloc[0]
    print(f"  name     {sample!r}")
    print(f"  normalise {abt_buy.normalise(sample)!r}")
    print(f"  squash    {squash(sample)!r}")

    results = {}
    print(f"\n{'strategy':<28}{'k':>4}{'pairs kept':>13}{'% of space':>12}{'recall':>9}")
    for k in KS:
        for label, frames in (("normalise only", plain), ("squash only", tight)):
            pairs = block.nearest_neighbour_pairs(*frames, k)
            results[(label, k)] = pairs
            print(f"{label:<28}{k:>4}{len(pairs):>13,}"
                  f"{len(pairs) / total:>11.2%}"
                  f"{len(pairs & truth_pairs) / len(truth_pairs):>9.3f}")

        union = results[("normalise only", k)] | results[("squash only", k)]
        results[("union", k)] = union
        print(f"{'union of both':<28}{k:>4}{len(union):>13,}"
              f"{len(union) / total:>11.2%}"
              f"{len(union & truth_pairs) / len(truth_pairs):>9.3f}")
        print()

    print("=== the comparison that actually decides it: equal pair budget")
    print(f"{'strategy':<28}{'pairs kept':>13}{'recall':>9}")
    for label, k in (("union", 5), ("normalise only", 10), ("squash only", 10)):
        pairs = results[(label, k)]
        print(f"{label + f' at k={k}':<28}{len(pairs):>13,}"
              f"{len(pairs & truth_pairs) / len(truth_pairs):>9.3f}")
    print("\n  union at k=5 and a single pass at k=10 cost about the same number of")
    print("  pairs. Whichever has the higher recall there is the one to keep.")

    gained = sorted(results[("union", 5)] - results[("normalise only", 5)])
    gained_true = [pair for pair in gained if pair in truth_pairs]
    print(f"\nthe second key added {len(gained):,} candidate pairs at k=5, "
          f"{len(gained_true)} of them true")
    left_names = dict(zip(left["left_id"], left["name"]))
    right_names = dict(zip(right["right_id"], right["name"]))
    for left_id, right_id in gained_true[:6]:
        print(f"\n  left  {left_names.get(left_id, '?')!r}")
        print(f"  right {right_names.get(right_id, '?')!r}")


if __name__ == "__main__":
    main()