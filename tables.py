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
BLOCKING = os.path.join("results", "blocking.json")
VOLUME = os.path.join("results", "volume.json")
ABT_BLOCKING = os.path.join("results", "abt_blocking.json")
ABT_CLASSIFIER = os.path.join("results", "abt_classifier.json")
MATCH_RESULTS = os.path.join("results", "match.json")
MATCH_ROWS = ("baseline", "classifier", "shipped", "tuned")
ABT_FEATURES = os.path.join("results", "abt_features.json")
# match_abt_buy.GROUPS, which abt_buy_features.py imports wholesale. Listed
# here rather than imported because importing that module needs the benchmark
# data, and benchmark/ is gitignored so CI has none.
ABT_GROUPS = (
    "all features",
    "cosine alone",
    "character similarity only",
    "text only",
    "text + price",
    "text + description",
    "text + manufacturer",
    "all + description_known",
)
# Row names as match_abt_buy.py records them. The baseline's README label
# carries the tuned threshold, so it is built from the recorded number rather
# than listed here - typing "0.497" into this file would reintroduce exactly
# the defect the whole module exists to remove.
ABT_BASELINE = "cosine baseline"
ABT_ROWS = (
    ABT_BASELINE,
    "classifier at 0.5",
    "best per right record",
    "mutual best",
)
README = "README.md"

# Only "cosine alone" actually differs, but all nine are listed so that a new feature
# group cannot reach the README under its internal name by defaulting to itself.
# All six listed, not just the four the hand-typed table showed, so a scheme
# cannot reach the README under block.py's internal name.
SCHEMES = {
    "exact normalised key": "exact key match",
    "nearest neighbours k=1": "k = 1",
    "nearest neighbours k=3": "k = 3",
    "nearest neighbours k=5": "k = 5",
    "nearest neighbours k=10": "k = 10",
    "nearest neighbours k=20": "k = 20",
}

# A tuple, not a rename map: the README uses volume_feature.py's own labels.
# It exists so a fifth configuration cannot reach the README unnoticed.
CONFIGURATIONS = (
    "6 features",
    "6 features + volume veto",
    "9 features (volume learned)",
    "9 features + veto",
)
# A tuple, not a rename map: abt_buy.py already prints the README's labels. It
# exists so a scheme cannot reach the README unnoticed - the hand-typed table
# showed six of the seven that script measures.
ABT_SCHEMES = (
    "exact key match",
    "k = 1",
    "k = 3",
    "k = 5",
    "k = 6",
    "k = 10",
    "k = 20",
)

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


def read_run(path, names, label):
    """One recorded run, with every group name checked against the README's
    vocabulary. A builder emitting an internal name would fail its own drift
    test forever, so refuse at load time and name the map to add it to.
    """
    with io.open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    missing = set(data["order"]) - set(names)
    if missing:
        sys.exit(f"no README name for {sorted(missing)} - add it to {label}")
    return data


def load():
    """Every recorded run the README needs, keyed by source. Each BLOCKS entry names
    the one it reads, so a builder cannot quietly render from the wrong file."""
    return {
        "sweep": load_sweep(),
        "ablation": read_run(ABLATION, FEATURE_SETS, "FEATURE_SETS"),
        "blocking": read_run(BLOCKING, SCHEMES, "SCHEMES"),
        "volume": read_run(VOLUME, CONFIGURATIONS, "CONFIGURATIONS"),
        "abt_blocking": read_run(ABT_BLOCKING, ABT_SCHEMES, "ABT_SCHEMES"),
        "abt_classifier": read_run(ABT_CLASSIFIER, ABT_ROWS, "ABT_ROWS"),
        "abt_features": read_layers(ABT_FEATURES, ABT_GROUPS),
        "match": read_run(MATCH_RESULTS, MATCH_ROWS, "MATCH_ROWS"),
    }


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
def volume(data):
    """Only the policy the README tabulates. The other one the script measures is
    reported in the prose below the table instead - four identical rows are not a
    table - and a test checks that sentence against this same recorded run.
    """
    rows = []
    for name in data["order"]:
        run = data["policies"][data["tabulated"]][name]
        f1 = f"{run['f1']:.3f}"
        if name == data["shipped"]:
            f1 = f"**{f1}**"
        rows.append([
            name,
            f"{run['precision']:.3f}",
            f"{run['recall']:.3f}",
            f1,
            str(run["false_positives"]),
        ])
    return render(
        ["configuration", "precision", "recall", "F1", "false positives"], rows
    )


def read_layers(path, names):
    """abt_features.json is the one recording with two result layers instead
    of a single "results" map, because every feature set is measured twice -
    at a fixed threshold and under one-to-one assignment. read_run checks one
    map and cannot see this shape.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("pair", "one_to_one"):
        missing = [name for name in names if name not in data["layers"][key]]
        if missing:
            raise SystemExit(f"{path} {key} layer is missing {missing}")
    if sorted(data["order"]) != sorted(names):
        raise SystemExit(f"{path} order is {data['order']}, wanted {list(names)}")
    return data
def abt_features(data):
    """The Abt-Buy feature ablation over 20 paired resamples. The reference
    row's label carries its own feature count, taken from the recording, so it
    cannot drift when a feature is added or dropped - which is exactly what
    went wrong here. The published version of this table was measured against
    a ten-feature baseline the repo had stopped shipping, so every win count
    in it answered the opposite question, while every mean in it was correct.
    Rows follow the recorded order, reference first; the hand-typed version
    moved "cosine alone" to the bottom, which is a presentation judgement a
    recorded run should not be making.
    """
    reference = data["reference"]
    splits = data["splits"]
    rows = []
    for name in data["order"]:
        pair = data["layers"]["pair"][name]
        one = data["layers"]["one_to_one"][name]
        rows.append([
            f"all {data['sizes'][name]} features" if name == reference else name,
            f"{pair['mean']:.3f}",
            "-" if name == reference else f"{pair['wins']}/{splits}",
            f"{one['mean']:.3f}",
            "-" if name == reference else f"{one['wins']}/{splits}",
        ])
    marked = sum(1 for row in rows if row[2] == "-")
    if marked != 1:
        raise SystemExit(f"{marked} rows claim to be the reference, wanted 1")
    return render(
        ["feature set", "pair F1", "beats all", "one-to-one F1", "beats all"], rows
    )
def classifier(data):
    """The books classifier comparison. Both thresholds go into the labels
    rather than being typed, because the hand-typed row read "threshold tuned"
    without saying tuned to what - 0.816 appeared only in a paragraph twelve
    lines below the table. Bold marks the shipped configuration, not the
    maximum, and this refuses to render if any other row scores a higher F1,
    which would mean the repo ships something it has already measured as
    worse. The shipped row is allowed to tie.
    """
    threshold = data["threshold"]
    label = {
        "baseline": f"baseline: similarity >= {data['baseline_threshold']:.3f}",
        "classifier": f"classifier at {threshold}",
        "shipped": f"classifier + volume veto at {threshold}",
        "tuned": f"classifier + veto at {data['tuned_threshold']:.3f} (tuned)",
    }
    shipped = data["shipped"]
    top = max(row["f1"] for row in data["results"].values())
    if data["results"][shipped]["f1"] < top - 1e-12:
        raise SystemExit(f"{shipped} is not the best F1 measured: {top:.4f}")
    rows = []
    for name in data["order"]:
        row = data["results"][name]
        f1 = f"{row['f1']:.3f}"
        rows.append([
            label[name],
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"**{f1}**" if name == shipped else f1,
        ])
    return render(["model", "precision", "recall", "F1"], rows)
def abt_classifier(data):
    """The Abt-Buy classifier table. Two cells are bold and, unlike the
    blocking table, they ARE column maxima - best F1 and best end-to-end
    recall. That is deliberate: the paragraph under the table exists because
    the two maxima sit in DIFFERENT rows, which is the finding. So this
    builder refuses to render if one row ever wins both, since a silent
    re-render would then publish a table contradicting its own prose. It also
    refuses on a tie, where the choice of which cell to embolden would be
    arbitrary and would migrate on its own.
    """
    rows = []
    for name in data["order"]:
        run = data["results"][name]
        if name == ABT_BASELINE:
            label = f"baseline: cosine >= {data['cosine_threshold']:.3f}"
        else:
            label = DISPLAY.get(name, name)
        rows.append(
            [label, run["precision"], run["recall"], run["f1"], run["end_to_end"]]
        )
    best_f1 = max(row[3] for row in rows)
    best_end = max(row[4] for row in rows)
    if sum(1 for row in rows if row[3] == best_f1) != 1:
        raise SystemExit(f"F1 ties at {best_f1:.6f} - bolding one is arbitrary")
    if sum(1 for row in rows if row[4] == best_end) != 1:
        raise SystemExit(f"end-to-end ties at {best_end:.6f}")
    f1_row = [row[3] for row in rows].index(best_f1)
    end_row = [row[4] for row in rows].index(best_end)
    if f1_row == end_row:
        raise SystemExit(
            f"{rows[f1_row][0]} now wins both F1 and end-to-end recall, so the "
            "README's 'opposite orders' paragraph is no longer true - rewrite "
            "the prose before regenerating this table"
        )
    out = []
    for index, row in enumerate(rows):
        f1, end = f"{row[3]:.3f}", f"{row[4]:.3f}"
        out.append([
            row[0],
            f"{row[1]:.3f}",
            f"{row[2]:.3f}",
            f"**{f1}**" if index == f1_row else f1,
            f"**{end}**" if index == end_row else end,
        ])
    return render(
        ["configuration", "precision", "recall", "F1", "end-to-end recall"], out
    )
def abt_blocking(data):
    """Every scheme abt_buy.py measures. The hand-typed table omitted k = 3,
    which is where recall is still climbing steeply - 0.888, 0.958, 0.977 - so
    leaving it out flattened the curve the table exists to show. Nothing is
    bold, as in the hand-typed original: this table is the evidence for
    choosing a k, not the record of which one ships.
    """
    rows = []
    for name in data["order"]:
        run = data["results"][name]
        rows.append([
            name,
            f"{run['recall']:.3f}",
            f"{run['pairs']:,}",
            f"{run['share']:.2%}",
        ])
    return render(["method", "recall", "pairs kept", "% of search space"], rows)
def blocking(data):
    """Every scheme block.py measures, including the two the hand-typed table
    left out. Recall is monotonic in k - the k=5 neighbour set is a subset of
    the k=20 one - so those two must also read 1.000, and printing them is what
    makes k=5 the smallest sufficient k rather than merely a sufficient one.
    """
    rows = []
    for name in data["order"]:
        recall = f"{data['results'][name]['recall']:.3f}"
        # The shipped k is bold, not the best recall: three schemes tie at 1.000.
        if name == data["shipped"]:
            recall = f"**{recall}**"
        rows.append([SCHEMES[name], recall])
    return render(["method", "recall of true pairs"], rows)


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
    ("blocking", "blocking", blocking,
     # The Abt-Buy blocking table also starts "| method"; its second column
     # is "recall", not "recall of true pairs".
     lambda line: line.startswith("| method") and "recall of true" in line),
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
    ("volume-comparison", "volume", volume,
     # The Abt-Buy classifier table also starts "| configuration"; its last
     # column is "end-to-end recall".
     lambda line: line.startswith("| configuration") and "false pos" in line),
    ("abt-blocking", "abt_blocking", abt_blocking,
     # The books blocking table also starts "| method" and has no pairs column.
     lambda line: line.startswith("| method") and "pairs kept" in line),
    ("abt-classifier", "abt_classifier", abt_classifier,
     # The volume table also starts "| configuration"; this is the only one
     # reporting end-to-end recall.
     lambda line: line.startswith("| configuration") and "end-to-end" in line),
    ("abt-features", "abt_features", abt_features,
     # The books ablation table also starts "| feature set"; this is the only
     # one with a pair F1 column.
     lambda line: line.startswith("| feature set") and "pair F1" in line),
    ("classifier", "match", classifier,
     lambda line: line.startswith("| model") and "precision" in line),
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
