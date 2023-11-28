"""The README's generated tables, rendered from the runs recorded under results/.

Four numbers in this README went stale once, and three findings that were really
one-split artefacts got published as results. Both failures were possible because the
tables were typed by hand from console output. These three are not: sweep.py records
every per-split value it computed, this renders the tables from that record, and
test_tables.py fails when the committed README disagrees with what this renders.

WHAT THAT BUYS IS NARROWER THAN IT SOUNDS. It proves the README agrees with the last
recorded sweep. It does not prove the sweep was right - nothing here re-derives a
median from anything except sweep.py's own output. The defensible claim for the README
is "generated from a recorded run", not "verified".

    python tables.py           print the tables
    python tables.py --write   splice them into README.md

The README calls the policies something friendlier than the code does, so DISPLAY is
the one place those two vocabularies are reconciled. A generator that emitted the
internal names would fail its own drift test forever.
"""
import io
import json
import os
import statistics
import sys

RESULTS = os.path.join("results", "sweep.json")
ABLATION = os.path.join("results", "ablation.json")
README = "README.md"

# Only "cosine alone" actually differs, but all nine are listed so that a new feature
# group cannot reach the README under its internal name by defaulting to itself.
FEATURE_SETS = {
    "all features": "all features",
    "all features + volume veto": "all features + volume veto",
    "all + volume as features": "all + volume as features",
    "everything except category": "everything except category",
    "everything except price": "everything except price",
    "title similarity only": "title similarity only",
    "cosine alone": "similarity score alone",
    "price alone": "price alone",
    "volume signals alone": "volume signals alone",
}

DISPLAY = {
    "threshold only": "threshold only",
    "best per right record": "best partner per record",
    "mutual best": "mutually-best partners",
}


def load_sweep():
    with io.open(RESULTS, encoding="utf-8") as handle:
        data = json.load(handle)
    missing = set(data["policy_order"]) - set(DISPLAY)
    if missing:
        sys.exit(f"no README name for {sorted(missing)} - add it to DISPLAY")
    return data


def load():
    """Every recorded run the README needs, keyed by source. Each BLOCKS entry names
    the one it reads, so a builder cannot quietly render from the wrong file."""
    data = {"sweep": load_sweep()}
    with io.open(ABLATION, encoding="utf-8") as handle:
        data["ablation"] = json.load(handle)
    missing = set(data["ablation"]["order"]) - set(FEATURE_SETS)
    if missing:
        sys.exit(f"no README name for {sorted(missing)} - add it to FEATURE_SETS")
    return data


def values(data, name, key):
    return [run[key] for run in data["results"][name]]


def cell(numbers, decimals):
    """median [min, max]. The band is the point: a single split is one draw, and the
    reference split happened to land near the favourable end of twenty of them."""
    low, high = min(numbers), max(numbers)
    middle = statistics.median(numbers)
    return (f"{middle:.{decimals}f} "
            f"[{low:.{decimals}f}, {high:.{decimals}f}]")


def render(header, rows):
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]

    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    rule = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(header), rule] + [line(row) for row in rows])


def policy_comparison(data):
    rows = []
    for name in data["policy_order"]:
        f1 = cell(values(data, name, "f1"), 3)
        if name == data["write_policy"]:
            f1 = f"**{f1}**"
        rows.append([
            DISPLAY[name],
            cell(values(data, name, "precision"), 3),
            cell(values(data, name, "recall"), 3),
            f1,
            cell(values(data, name, "false_positive"), 0),
            cell(values(data, name, "exact"), 3),
        ])
    return render(
        ["policy", "precision", "recall", "F1", "false positives",
         "entity exactness"],
        rows,
    )


def head_to_head(data):
    """Paired counts, not bands. The bands for the two one-to-one policies overlap
    heavily and read as indistinguishable; asking which won ON THE SAME SPLIT is what
    shows the effect is real."""
    reference = data["results"][data["write_policy"]]
    rows = []
    for name in data["policy_order"]:
        if name == data["write_policy"]:
            continue
        row = [DISPLAY[name]]
        for key in ("f1", "exact"):
            mine = [run[key] for run in reference]
            theirs = [run[key] for run in data["results"][name]]
            better = sum(a > b for a, b in zip(theirs, mine))
            worse = sum(a < b for a, b in zip(theirs, mine))
            row.append(f"{better} / {worse} / {len(mine) - better - worse}")
        rows.append(row)
    return render(
        [f"head-to-head vs {DISPLAY[data['write_policy']]}",
         "F1 (better / worse / tied)", "entity exactness"],
        rows,
    )


def closure_cost(data):
    splits = len(data["seeds"])
    rows = []
    for name in data["policy_order"]:
        merges = values(data, name, "closure")
        total = sum(merges)
        correct = sum(values(data, name, "closure_correct"))
        if total == 0:
            # Nothing to be right or wrong about. A 0 here would read as a finding
            # when it is the policy's definition: closure over a matching is identity.
            share = "-"
        else:
            share = f"**{correct:,}**" if correct == 0 else f"{correct:,}"
        rows.append([DISPLAY[name], cell(merges, 0), f"{total:,}", share])
    return render(
        ["policy", "closure merges per split", f"across {splits} splits",
         "of those, correct"],
        rows,
    )


# Located by marker once the markers exist. Before that, by a predicate on the header
# row - two of these tables start with "| policy", so the first column is not enough.
def ablation(data):
    """Ordered by F1 descending, which is the README's order and not GROUPS' order.

    The delta is measured BEFORE rounding. Rounding first gives +0.009 for the shipped
    row where the honest answer is +0.008, and two rows of the committed table would
    then disagree by one in the last digit for a reason nobody could find.
    """
    reference = data["results"][data["reference"]]["f1"]
    rows = []
    for name in sorted(data["order"], key=lambda n: -data["results"][n]["f1"]):
        f1 = data["results"][name]["f1"]
        # "-", not "+0.000": a row's distance from itself is not information.
        delta = "-" if name == data["reference"] else f"{f1 - reference:+.3f}"
        rows.append([FEATURE_SETS[name], f"{f1:.3f}", delta])
    return render(["feature set", "F1", "delta"], rows)


BLOCKS = [
    ("ablation", "ablation", ablation,
     # Two README tables have a "| feature set" header - the other is the Abt-Buy
     # ablation at ~line 573, whose columns are "pair F1" / "beats all".
     lambda line: line.startswith("| feature set") and "delta" in line),
    ("policy-comparison", "sweep", policy_comparison,
     lambda line: line.startswith("| policy") and "entity exactness" in line),
    ("head-to-head", "sweep", head_to_head,
     lambda line: "head-to-head vs" in line),
    ("closure-cost", "sweep", closure_cost,
     lambda line: line.startswith("| policy") and "closure merges" in line),
]

END = "<!-- end -->"


def blocks(data):
    return [(name, builder(data[source])) for name, source, builder, _ in BLOCKS]


def splice(lines, name, table, matches):
    begin = f"<!-- generated by tables.py: {name} -->"
    replacement = [begin] + table.split("\n") + [END]
    if begin in lines:
        start = lines.index(begin)
        return lines[:start] + replacement + lines[lines.index(END, start) + 1:]
    found = [index for index, line in enumerate(lines) if matches(line)]
    if len(found) != 1:
        sys.exit(f"refusing: {len(found)} candidate headers for {name}, expected 1")
    start = found[0]
    stop = start
    while stop < len(lines) and lines[stop].startswith("|"):
        stop += 1
    return lines[:start] + replacement + lines[stop:]


def write(data):
    text = io.open(README, encoding="utf-8", newline="").read()
    ending = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(ending)
    for (name, table), (_, _, _, matches) in zip(blocks(data), BLOCKS):
        lines = splice(lines, name, table, matches)
    io.open(README, "w", encoding="utf-8", newline="").write(ending.join(lines))
    print(f"spliced {len(BLOCKS)} tables into {README}")


def main():
    data = load()
    if "--write" in sys.argv:
        write(data)
        return
    for name, table in blocks(data):
        print(f"\n<!-- generated by tables.py: {name} -->")
        print(table)
        print(END)


if __name__ == "__main__":
    main()
