[![tests](https://github.com/sushantk08/scraper-entity-resolution/actions/workflows/tests.yml/badge.svg)](https://github.com/sushantk08/scraper-entity-resolution/actions/workflows/tests.yml)

# Resilient scraper → entity resolution pipeline

Scrapes a book catalogue, types and validates the result, then matches records
that refer to the same book across two dirty copies of the data — with every
claim below measured rather than asserted.

Python, BeautifulSoup, Selenium, pandas, scikit-learn, pytest. 36 tests.

## What is actually here

```
sites.py          one dict per site: URLs, selectors, parse function
fetch.py          listing pages -> pages/<site>/     (urllib, or Selenium if JS-rendered)
fetch_details.py  detail pages   -> pages/<site>_detail/   resumable, failure-tolerant
parse.py          saved HTML -> data/<site>.csv
clean.py          typing + validation; refuses to write if the data looks wrong
canary.py         hits the LIVE site and fails if the markup has drifted
perturb.py        builds a labelled matching benchmark from the clean data
block.py          candidate generation, measured by recall
match.py          pairwise classifier vs a real baseline, leak-guarded split
ablate.py         which features actually earn their place
decide.py         turning pair scores into one-to-one decisions
show.py           print the HTML behind a suspect field
explore.py        probe whether a new site is scrapeable at all
```

## Results

Benchmark: 1000 scraped books split into 850 "left" records and 750 "right"
records with 600 true pairs among 637,500 possible ones. 250 left and 150 right
records have **no correct answer at all** — without those, a one-to-one matcher
scores perfectly by ranking alone and never has to reject anything.

**Candidate generation** (character n-gram TF-IDF, k nearest neighbours):

| method              | recall of true pairs |
|---------------------|----------------------|
| exact key match     | 0.235                |
| k = 1               | 0.965                |
| k = 3               | 0.995                |
| k = 5               | **1.000**            |

Reduction ratio is deliberately not reported as a result: fixed-k emits exactly
k×|right| pairs, so reduction is mechanically `1 − k/|left|`. It is a knob, not
a finding. Only recall carries information here.

**Matching** (logistic regression; train/test split grouped by record id so no
record's near-duplicates straddle the split):

| model                          | precision | recall | F1        |
|--------------------------------|-----------|--------|-----------|
| baseline: similarity ≥ 0.622   | 0.876     | 0.891  | 0.883     |
| classifier, threshold 0.5      | 0.961     | 0.994  | **0.977** |
| classifier, threshold tuned    | 0.976     | 0.948  | 0.962     |

End-to-end recall, blocking losses included: **173 of 174 = 0.994**.

The tuned threshold (0.847, chosen on the training fold) scored *worse* on the
test fold than a naive 0.5. Tuning on one fold does not automatically transfer.

**Which features earn their place** (F1, and change vs. all features):

| feature set              | F1    | Δ      |
|--------------------------|-------|--------|
| all features             | 0.977 | —      |
| everything except category | 0.947 | −0.030 |
| everything except price   | 0.916 | −0.061 |
| title similarity only    | 0.889 | −0.088 |
| similarity score alone   | 0.875 | −0.102 |
| price alone              | 0.653 | −0.324 |

Coefficients are printed but explicitly **not** labelled as importance — the
features are correlated, and `length_ratio` carries a negative weight that would
be nonsense to read that way. Ablation is the honest version of that question.

**Decisions, not just scores.** The classifier judges each pair alone, so it can
accept two different left records for the same right record. Forbidding that
requires no retraining:

| policy                     | precision | recall | F1    |
|----------------------------|-----------|--------|-------|
| threshold only             | 0.971     | 0.966  | 0.968 |
| best partner per record    | 1.000     | 0.994  | **0.997** |
| mutually-best partners     | 1.000     | 0.994  | 0.997 |

In whole pairs: threshold-only accepted 168 of 174 true pairs plus 5 false ones;
best-partner accepted 173 and none false. The largest single improvement in the
project came from changing the decision rule, not the model.

## Things I am not going to overclaim

- **That precision figure is 0 false positives out of 173 accepted pairs.** With
  that sample size the defensible claim is ≥0.98, not 1.000.
- **`mutually-best` bought nothing** — identical to best-partner on every figure.
  It is kept as a measured negative result, not sold as a feature.
- **The benchmark is generated, not human-labelled.** `perturb.py` writes both
  the noise and the labels, and book titles are long enough that a corrupted copy
  is still far from any other book. Real catalogues are harder. **0.889 (title
  features only) is the number I would expect to survive contact with real data,
  not 0.977.**
- **One-to-one matching is an assumption this data satisfies by construction.** A
  real catalogue can carry two legitimate editions of the same book, which would
  cap recall by design.
- **Three scraped fields carry zero information** (tax, review count, in-stock
  flag are constant across all 1000 books). 1000 extra HTTP requests bought
  exactly three useful fields: category, description, UPC. That is reported
  rather than hidden.
- **The source data is dirty in a way that matters.** 22% of the catalogue sits
  in non-genre categories (`Default`: 152 books, `Add a comment`: 67), verified
  against the site's own breadcrumb URLs, so the parser is right and the site is
  wrong. Category agreement is therefore weak evidence on common values and
  strong on rare ones — IDF weighting is the principled fix, and is not done yet.

## Two things here that most scraper projects skip

**A schema-change canary.** `canary.py` fetches a live page and fails loudly if
the expected fields come back empty. It tests the *website*, not the code, so it
is designed to fail one day. It carries a positive control, because a check that
can only pass is indistinguishable from no check.

**Fetching is separated from parsing.** Pages are saved to disk, then parsed from
frozen fixtures. Parser tests run offline in milliseconds and are deterministic,
the detail crawl resumes by checking the filesystem rather than holding state in
memory, and per-request failures are collected and reported instead of raised —
5 of 1000 requests failed transiently and a plain re-run fetched exactly those 5.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate
pip install selenium beautifulsoup4 pandas scikit-learn pytest

python fetch.py books          # 50 listing pages -> pages/books/
python fetch_details.py books  # 1000 detail pages, resumable
python parse.py books          # -> data/books.csv
python clean.py                # typing + validation, refuses to write if wrong
python canary.py books         # live check: has the markup drifted?

python perturb.py              # build the labelled benchmark
python block.py                # candidate generation, measured
python match.py                # classifier vs baseline
python ablate.py               # what the features are worth
python decide.py               # scores -> one-to-one decisions

python -m pytest -q            # 36 tests, all offline
```

## Not built yet

Raw document store, normalised relational schema, a UI, CI wiring the canary in
on a schedule, and a published benchmark (Abt–Buy or Amazon–Google) so the
precision and recall above become comparable to published baselines instead of
measured on a distribution I generated myself.