[![tests](https://github.com/sushantk08/scraper-entity-resolution/actions/workflows/tests.yml/badge.svg)](https://github.com/sushantk08/scraper-entity-resolution/actions/workflows/tests.yml)

# Resilient scraper -> entity resolution pipeline

Scrapes a book catalogue, types and validates the result, loads it into a
normalised database, then matches records that refer to the same thing - across two
dirty copies of the scraped data, and across two real retailer catalogues from a
published benchmark - with every claim below measured rather than asserted,
including the measurements that came out against me.

Python, BeautifulSoup, Selenium, pandas, scikit-learn, SQLite, pytest. 40 tests,
all offline.

## What is actually here

```
sites.py             one dict per site: URLs, selectors, parse function
fetch.py             listing pages -> pages/<site>/ (urllib, or Selenium if JS-rendered)
fetch_details.py     detail pages -> pages/<site>_detail/, resumable, failure-tolerant
parse.py             saved HTML -> data/<site>.csv
clean.py             typing + validation; refuses to write if the data looks wrong
store.py             cleaned CSV -> SQLite: immutable records, derived decisions
canary.py            hits the LIVE site and fails if the markup has drifted
perturb.py           builds a labelled matching benchmark from the clean data
block.py             candidate generation, measured by recall
normalise_compare.py tests one hypothesis about blocking misses before changing anything
block_multi.py       the rejected experiment: two blocking keys instead of one
inspect_pairs.py     reports the true shape of a downloaded benchmark before trusting it
match.py             pairwise classifier vs a real baseline, leak-guarded split
ablate.py            which features actually earn their place
decide.py            turning pair scores into one-to-one decisions
error_analysis.py    the individual pairs that go wrong, and why
volume_feature.py    the experiment behind the volume veto, kept in the repo
abt_buy.py           loads a published benchmark into the same shape
match_abt_buy.py     the classifier on that benchmark, with missing values handled
abt_buy_features.py  resamples the split 20 times to test whether a feature matters
show.py              print the HTML behind a suspect field
explore.py           probe whether a new site is scrapeable at all
```

## Results

Benchmark: 1000 scraped books split into 850 "left" records and 750 "right"
records with 600 true pairs among 637,500 possible ones. 250 left and 150 right
records have **no correct answer at all** - without those, a one-to-one matcher
scores perfectly by ranking alone and never has to reject anything. An earlier
version of this benchmark lacked them and reported precision 1.000 at a
threshold that was never exercised.

**Candidate generation** (character n-gram TF-IDF, k nearest neighbours):

| method              | recall of true pairs |
|---------------------|----------------------|
| exact key match     | 0.235                |
| k = 1               | 0.965                |
| k = 3               | 0.995                |
| k = 5               | **1.000**            |

Reduction ratio is deliberately not reported as a result: fixed-k emits exactly
k x |right| pairs, so reduction is mechanically `1 - k/|left|`. It is a knob, not
a finding. Only recall carries information here.

**Matching** (logistic regression; train/test split grouped by record id so no
record's near-duplicates straddle the split):

| model                             | precision | recall | F1        |
|-----------------------------------|-----------|--------|-----------|
| baseline: similarity >= 0.622     | 0.876     | 0.891  | 0.883     |
| classifier at 0.5                 | 0.961     | 0.994  | 0.977     |
| classifier + volume veto at 0.5   | 0.977     | 0.994  | **0.986** |
| classifier + veto, threshold tuned| 0.988     | 0.960  | 0.974     |

End-to-end recall, blocking losses included: **173 of 174 = 0.994**.

The tuned threshold (0.816, chosen on the training fold) scored *worse* on the
test fold than a naive 0.5. Tuning on one fold does not automatically transfer,
and this project stopped tuning thresholds because of it - see the measurement
traps below, where widening the search made tuning worse rather than better.

**Which features earn their place** (F1 at a fixed 0.5, and change vs. all
features):

| feature set                 | F1    | delta  |
|-----------------------------|-------|--------|
| all features + volume veto  | 0.986 | +0.008 |
| all + volume as features    | 0.983 | +0.006 |
| all features                | 0.977 | -      |
| everything except category  | 0.947 | -0.030 |
| everything except price     | 0.916 | -0.061 |
| title similarity only       | 0.889 | -0.088 |
| similarity score alone      | 0.875 | -0.102 |
| price alone                 | 0.653 | -0.324 |
| volume signals alone        | 0.312 | -0.666 |

Every row uses the same fixed threshold on purpose. Letting each row pick its own
threshold turns an ablation into a comparison of tuner luck, which is exactly the
mistake documented further down.

Coefficients are printed but explicitly **not** labelled as importance - the
features are correlated, and `length_ratio` carries a negative weight that would
be nonsense to read that way. Ablation is the honest version of that question.

**Decisions, not just scores.** The classifier judges each pair alone, so it can
accept two different left records for the same right record. Forbidding that
requires no retraining, and is the largest single improvement in the project:

| policy                     | precision | recall | F1        | false positives |
|----------------------------|-----------|--------|-----------|-----------------|
| threshold only             | 0.977     | 0.994  | 0.986     | 4               |
| best partner per record    | 1.000     | 0.994  | **0.997** | 0               |
| mutually-best partners     | 1.000     | 0.994  | 0.997     | 0               |

In whole pairs: of 225 right-hand records in the test fold, 174 have a true
partner and 51 have none. Best-partner accepted 173 pairs and got all 173 right.

`decide.py` also prints each policy at a threshold tuned on the train fold and at
a threshold chosen with test labels - an oracle that is not achievable, printed
only to bound what better threshold selection could buy. **The oracle equals the
flat 0.5 row for all three policies, so a perfect threshold would add +0.000 F1.**
Shipping an untuned 0.5 is a measured result here, not laziness.

## Storage: immutable records, derived decisions

`store.py` loads the cleaned CSV into SQLite. Three tables, and the split between
the first two is the entire design:

| table           | holds                                                   |
|-----------------|---------------------------------------------------------|
| `record`        | one row per scraped record, per source. Never merged.   |
| `entity`        | one row per real-world thing several records refer to.  |
| `record_entity` | which records the matcher decided are the same thing.   |

A pipeline that overwrites its scraped rows with merged ones can never be asked
*why did you merge these two*, and can never be re-run with a better matcher. So
raw rows are immutable and the decisions live in a table that can be dropped and
rebuilt from nothing: a bad matching run is a `DELETE`, not a re-scrape.

`UNIQUE (source, external_id)` is load-bearing rather than decorative. It makes
re-running the loader idempotent, and turns "the scrape produced the same product
twice" from a silent duplicate into a reported count. `external_id` prefers the
natural key - `upc`, unique across all 1000 books - and falls back to row position
only while saying so loudly, because row position changes if the scrape order
changes, and every stored decision would then point at the wrong book.

Re-loading after adding four columns reported **1000 rows offered, 0 inserted, 1000
updated in place**. That line is the point of the whole design: `record_id` values
survived, so anything stored in `record_entity` still points where it did before.
An upsert reporting 1000 *inserted* would have meant fresh ids and silently
orphaned every decision built on the old ones.

The update uses `COALESCE(excluded.col, record.col)`, so re-running from a CSV that
has lost a column cannot delete data an earlier run loaded. The cost, stated
because it is a real one: genuinely correcting a value *to* NULL needs an explicit
`DELETE`.

**Blanks are NULL, never 0.0 and never an empty string** - the same rule the
matcher needed on Abt-Buy, enforced one layer earlier. Two of the 1000 books have
no description on the site and arrive as NULL rather than `""`, so a blank can
never read as agreement once description similarity is computed over them.

`CREATE TABLE IF NOT EXISTS` silently does nothing when the table already exists
with different columns, which is exactly how a schema change becomes a column full
of NULLs. The loader compares `PRAGMA table_info` against the schema it expects and
refuses to run against an older shape, pointing at a database that rebuilds in a
second. `PRAGMA foreign_keys = ON` is set explicitly, because SQLite leaves it off.

Every column in the CSV is either stored or explained: an unmapped, unexplained
column is a warning, because "I did not notice it" and "it carries nothing" look
identical in a list of ignored names.

The descriptions it stores are long - mean 1,432 characters, longest 8,646. That
matters for what comes next rather than now: the `description_jaccard` feature that
earned +0.24 on Abt-Buy was measured on short retail blurbs, and word overlap
between two 1,400-character documents drifts toward a floor set by common English.
The Abt-Buy feature weights are not assumed to transfer here.

SQLite deliberately, and not Postgres yet. The parts worth designing are the schema
and the loader, and SQLite keeps both testable offline in milliseconds - "no network
in CI" is a rule this project has kept since its first commit. `connect()` is the
only function that knows what database this is.

## Volume numbers: a constraint, not a feature

Error analysis showed that 5 of 7 false positives were series confusion - Fruits
Basket Vol. 1 matched to Vol. 3. Titles state volume numbers, and when two stated
numbers disagree the pair is never a match: that signal fires on 134 of 3,750
candidate pairs with **0 true matches among them**.

The interesting question was how to use it. Three volume features fed to the
model, or one veto applied after scoring:

| configuration                | precision | recall | F1        | false positives |
|------------------------------|-----------|--------|-----------|-----------------|
| 6 features                   | 0.961     | 0.994  | 0.977     | 7               |
| 6 features + volume veto     | 0.977     | 0.994  | **0.986** | 4               |
| 9 features (volume learned)  | 0.972     | 0.994  | 0.983     | 5               |
| 9 features + veto            | 0.972     | 0.994  | 0.983     | 5               |

The veto wins, and the reason is worth stating: adding features means refitting,
and refitting moves the decision boundary on pairs the new features say nothing
about. Both false positives that the 9-feature model introduced contain no volume
numbers at all. A veto touches only the 134 pairs it is actually about, and can
only ever lower a score.

**And under one-to-one assignment, all four configurations are identical** -
0.997 F1, 0 false positives, and not one differing decision between any pair of
them. One-to-one assignment was already cleaning up series confusion for free.
So the veto is cheap insurance for data where one-to-one is unavailable, and is
reported that way rather than as a headline.

A test pins the part that could quietly go wrong: only *marked* numbers count.
`Vol. 3`, `Book 2` and `#11` are volumes; `1984` and `orange: The Complete
Collection 1` are not. Without that, the veto would start refusing real matches
whose titles merely contain digits.

## Measurement traps this project hit

These are in the README because they were expensive and none of them are visible
in the final numbers.

**A benchmark that could not fail.** The first version derived every right-hand
record from a left-hand book, so each had exactly one true partner and all the
unmatchable records sat on one side. A one-to-one policy could not produce a
false positive, and `decide.py` duly reported precision 1.000 at a threshold that
was never exercised - it measured ranking and printed it as precision. Fixed with
a three-way split and a test suite over the generated data itself. Any metric
that reads 1.000 now gets audited before it gets quoted.

**A tuner leaking into a comparison.** `decide.py` used to tune a threshold per
policy on the train fold. When three features were added, one policy's pick moved
from 0.403 to 0.652 and F1 "fell" from 0.997 to 0.991. I attributed that to the
new features. It was entirely the tuner: at the shipping layer the features
changed zero decisions. Every comparison now runs at a fixed threshold, because
the tuner noise here was larger than any real effect being measured.

**An upper bound that did not bound.** The fix above added an oracle row to show
what a perfect threshold would be worth - and the oracle came out *below* the
shipped configuration, which is impossible. Its candidate thresholds were
quantiles of the score distribution, and that distribution is bimodal (~84% of
pairs near 0.0, ~16% near 1.0), so the grid clustered at both extremes and never
tried 0.5. Now the grid is uniform and an assertion refuses to print an oracle
that fails to bound the shipped row. Widening the grid also made train-tuning
look *worse*, not better: a finer search lets a threshold overfit the train fold
harder.

**Reporting deltas instead of decisions.** F1 0.977 -> 0.983 sounds like progress.
It was 6 changed decisions out of 1,125, split 4 fixed and 2 broken, which a coin
flip reproduces about 69% of the time. `volume_feature.py` now prints the
discordant counts and that probability, so a difference has to be worth
something before it gets called an improvement.

**A subset that beat the full model.** Every one of seven feature subsets scored
above the full ten-feature model on Abt-Buy, by up to 0.014 F1 - and seven of
seven pointing the same way is suggestive enough to act on. It is also untestable:
nested subsets measured on one shared split are not independent draws, so no
statistic over those seven rows means anything. Resampling the split 20 times and
counting per-split wins is testable, and it turned a suggestive ordering into a
signed result that *reverses* depending on which layer you measure. One feature
was then dropped on evidence instead of on the shape of the table.

**Two silent data losses, caught by making the code say what it stored.**
`store.py`'s first run stored 1000 NULL prices, because `clean.py` names the
column `price_incl_tax_gbp` and the loader asked for `price`. Its populated-column
report turned that into a visible `0 of 1000` instead of a plausible-looking
database. The same report then showed `description 0 of 1000`, which was not a
loader bug at all: `clean.py` had been reading each description and keeping only
its word count, discarding the text - the one field the residual Abt-Buy failures
point at. Neither loss would have raised an exception, and neither would have
shown up in any metric. There was a third in the same family: for about an hour
the loader printed *"a count of a description this CSV does not carry"* as its
reason for skipping `description_words`, after that had stopped being true. A
reason the program says out loud gets caught. A comment rots quietly.

## Does it generalise? Abt-Buy

Everything above is measured on a benchmark `perturb.py` generated, so the
obvious objection is that it only works on data built to be matchable. Abt-Buy
is a published benchmark with human-labelled pairs: 1,081 Abt products against
1,092 Buy products, 1,097 true pairs among 1,180,452 possible ones - **0.093%
positive, against 16% in the generated benchmark.**

`abt_buy.py` loads it into the same `(left, right, truth)` shape the rest of the
pipeline already uses, so `block.py` runs on it unchanged. Candidate generation
survives real data, and degrades honestly:

| method          | recall | pairs kept | % of search space |
|-----------------|--------|------------|-------------------|
| exact key match | 0.015  | 16         | 0.00%             |
| k = 1           | 0.888  | 1,092      | 0.09%             |
| k = 5           | 0.977  | 5,460      | 0.46%             |
| k = 6           | 0.981  | 6,552      | 0.56%             |
| k = 10          | 0.988  | 10,920     | 0.93%             |
| k = 20          | 0.995  | 21,840     | 1.85%             |

Exact key matching collapses from 0.235 to **0.015** - sixteen pairs out of
1,097. Two retailers essentially never write a product name identically, which
is the entire justification for fuzzy blocking, now measured rather than
asserted.

**The blocking key was chosen by experiment, and two alternatives were rejected.**

*Splitting letter-digit boundaries* so `CLI8C` and `CLI-8C` agree: recovered 2
pairs, lost 5, net worse at every k. Contiguous part numbers like `KXTG6700B`
are the rarest substrings in the data and fragmenting them destroys more signal
than the hyphen repairs.

*Deleting separators instead* (`CLI-8C` -> `cli8c`) works, because it makes those
two collide without breaking anything apart: 0.981 vs 0.976 at k=6, for the same
number of pairs. This is now the key.

*Unioning two keys* rather than choosing one: rejected. At an equal pair budget
it never won - 6,429 pairs for 0.980 against 6,552 for 0.981. The two keys differ
only in separator handling, so their candidate sets overlap heavily; 969 extra
pairs bought 8 true ones. Multiple blocking passes pay off when the keys are
*diverse*, not when they are spelling variants of each other.

**What this dataset does and does not exercise.** Every record on both sides has
a partner, so nothing ever needs to be rejected - precision here would measure
wrong-partner selection, not rejection, and those are different claims. And 5 Buy
records have two true Abt partners (16 Abt records have two Buy partners), which
puts a hard, measured recall ceiling of **0.995** on the one-partner-per-record
rule that scored 0.997 on the generated data. That assumption is provably wrong
here, by a known amount.

The 13 pairs unreachable at k=10 are not a string-metric problem. Abt sells
CLI8C, CLI8M, CLI8Y, CLI8R and CL41CL, so the discriminating substring is about
two characters in twenty-five and the shared boilerplate outweighs it. Others
carry no identity in the title at all - `LG 25.0 Cu.Ft. Total Capacity`. Getting
past ~0.99 needs the description column, not a better distance function.

### The matcher on real data

`match_abt_buy.py` runs the classifier on it. Test fold: 3,280 candidate pairs
over Buy records held out as whole groups. 322 true pairs reached the matcher and
328 exist in the fold, so blocking had already lost 6 before scoring began. The
last column is end-to-end recall, whose denominator is every true pair in the
fold *including* the ones blocking never generated - the recall column cannot see
those, and flatters the pipeline on its own.

| configuration                | precision | recall | F1        | end-to-end recall |
|------------------------------|-----------|--------|-----------|-------------------|
| baseline: cosine >= 0.497    | 0.578     | 0.770  | 0.660     | 0.756             |
| classifier at 0.5            | 0.564     | 0.929  | 0.702     | **0.912**         |
| best partner per record      | 0.928     | 0.885  | 0.906     | 0.869             |
| mutually-best partners       | 0.962     | 0.873  | **0.915** | 0.857             |

**Only the `classifier at 0.5` row is comparable to published Abt-Buy results.**
Published figures score pair classification over a fixed labelled candidate set.
The one-to-one rows score far higher for a structural reason rather than a clever
one: every record in this benchmark has a true partner, so a rule that accepts
each record's top candidate can barely be punished for accepting, and the number
measures ranking rather than rejection. That is the benchmark-that-could-not-fail
failure mode arriving from a different direction, in a dataset I did not build.

**F1 and end-to-end recall rank those policies in opposite orders**, which is the
most useful thing in the table. The one-to-one constraint trades about 6 points of
found links for about 40 points of precision. Neither number is more correct; they
answer different questions. A review queue a human works through wants the 0.912;
an automatic merge nobody checks wants the 0.962 precision. Any single F1 quoted
here hides that choice, so the objective goes next to the figure.

**`mutually-best` beats best-partner here after buying literally nothing on the
generated data** - 0.915 against 0.906, false positives 22 down to 11. Both sides
of Abt-Buy are real catalogues competing for partners rather than one side derived
from the other. A measured negative result did not transfer, which is the best
argument in this repo for keeping negative results instead of deleting them.

And the classifier only just beats one number: 0.702 against 0.660 for
`cosine >= 0.497`, and it gets there by trading precision away for recall. On the
generated benchmark the same design beat its baseline 0.986 to 0.883.

**Missing values are handled rather than defaulted.** Price is blank on 61% of Abt
records and 46% of Buy records. A blank arriving at the model as 0.0 would read as
"these prices agree perfectly" - the same bug class as a blank category counting
as agreement. Unknowns become NaN and are filled with the **training fold's** mean;
imputing from the whole table would leak. On the test fold that fills 2,625 of
3,280 rows for price (80%), 1,290 for description (39%) and 10 for manufacturer.

### Which features earn their place on real data: almost none

On the generated benchmark, dropping price cost 0.061 F1. On Abt-Buy, price,
description and manufacturer features together buy between 0.000 and 0.004 and
character similarity is the whole model. `abt_buy_features.py` builds the
candidate table once, resamples the grouped split 20 times, and counts how often
each subset beats the full set. The splits are paired, so per-split win counts see
what comparing means cannot: the spread across splits (sd 0.012 to 0.021) is
larger than every difference being measured.

| feature set                 | pair F1 | beats all | one-to-one F1 | beats all |
|-----------------------------|---------|-----------|---------------|-----------|
| all ten features            | 0.679   | -         | 0.906         | -         |
| character similarity only   | 0.686   | 17/20     | 0.899         | 3/20      |
| text only                   | 0.687   | 17/20     | 0.898         | 3/20      |
| text + price                | 0.691   | 17/20     | 0.899         | 3/20      |
| text + description          | 0.676   | 4/20      | 0.905         | 7/20      |
| text + manufacturer         | 0.685   | 15/20     | 0.898         | 2/20      |
| all minus description_known | 0.690   | 18/20     | 0.906         | 9/20      |
| cosine alone                | 0.589   | 0/20      | 0.865         | 0/20      |

**The effect is real at the pair layer and reverses under one-to-one assignment.**
Subsets win 17 of 20 splits at a fixed threshold and lose 17 of 20 once decisions
are made per record. One feature group causes both, in opposite directions: the
description features worsen the fixed-0.5 cut and improve the per-record ranking.
That is coherent - they shift where probability mass sits without changing which
candidate comes out on top, so a hard threshold gets miscalibrated while an argmax
gets better. Every other row in both columns falls out of that one mechanism.
Which makes the layer you ship the only one whose verdict counts - the same lesson
as the tuner trap, in a new costume. `description_known` was dropped because it is
the one row that is *dominated*: 18 of 20 better at the pair layer, and a dead tie
at the one-to-one layer (9 of 20, p 1.00). Better at one layer, indistinguishable
at the other. That table was produced with it still in; `abt_buy_features.py`
imports its feature groups from `match_abt_buy.py`, so re-running it now asks the
inverted question.

**A theory of mine that the data killed.** `description_known` carried a -0.37
weight, larger in magnitude than `description_jaccard` at +0.29 - "this record has
a description at all" reading as evidence *against* a match. My explanation: 40%
of Buy descriptions are blank and the split is grouped by Buy id, so the feature
could stand in for "this particular record rarely matches" - a per-record prior
that would not survive a different catalogue. Measured true-match rate: **9.8%
among pairs with descriptions on both sides, 9.8% among pairs missing one.**
Identical. It carried no information about the pair at all, and the coefficient was
collinearity. The theory was wrong and the conclusion happened to be right, which
is worth keeping separate. The four-line check still runs and still prints.

Dropping the indicator is safe here specifically because the train-fold fill for
`description_jaccard` is 0.077, near the low end, so a missing description already
reads as weak disagreement without a flag. `price_known` and `manufacturer_known`
stay: their fills are 0.477 and 0.582, nowhere near as safe, and neither was
tested. Four coefficients carry weight and five sit at or below 0.09, and that is
not a reason to prune further - over 20 splits, text-only lost 3 of 20 at the layer
that ships.

One limit on all of it: the 20 splits resample one fixed candidate table. That
establishes an effect is not split luck. It cannot establish it is not this
dataset.

## Things I am not going to overclaim

- **That precision figure is 0 false positives out of 173 accepted pairs.**
  `decide.py` computes and prints the Clopper-Pearson floor rather than letting me
  write one by hand: the defensible claim is **>= 0.983**, not 1.000.
- **`mutually-best` bought nothing on the generated benchmark** - identical to
  best-partner on every figure. It was kept as a measured negative result rather
  than deleted, and on Abt-Buy it turned out to be the best policy in the file. A
  negative result belongs to the data it was measured on.
- **The 0.915 on Abt-Buy is not comparable to published numbers**, and is not
  quoted as though it were. The comparable figure is **0.702**, for the structural
  reason given in that section.
- **The volume veto also buys nothing once one-to-one assignment is in place.**
  Zero changed decisions. It earns its place only on the threshold-only
  configuration, and that is what the tables say.
- **The generated benchmark is generated, not human-labelled.** `perturb.py`
  writes both the noise and the labels, and book titles are long enough that a
  corrupted copy is still far from any other book. Real catalogues are harder.
  **0.889 (title features only) is the number I would expect to survive contact
  with real data, not 0.986** - and Abt-Buy's 0.702 is evidence for exactly that.
- **One-to-one matching is an assumption the generated data satisfies by
  construction.** A real catalogue can carry two legitimate editions of the same
  book. Abt-Buy shows the cost: a measured ceiling of 0.995.
- **The one remaining false negative on books is a real weakness, not noise.**
  'Arena' vs 'Arnea' scores 0.106 on character n-gram cosine while sequence
  matching scores 0.800. Character n-grams are length-sensitive, and on very short
  titles a transposition is close to fatal. Not fixed.
- **Three scraped fields carry zero information** (tax, review count, in-stock
  flag are constant across all 1000 books). 1000 extra HTTP requests bought
  exactly three useful fields: category, description, UPC. That is reported
  rather than hidden.
- **The source data is dirty in a way that matters.** 22% of the catalogue sits
  in non-genre categories (`Default`: 152 books, `Add a comment`: 67), verified
  against the site's own breadcrumb URLs, so the parser is right and the site is
  wrong. Category agreement is therefore weak evidence on common values and
  strong on rare ones - IDF weighting is the principled fix, and is not done yet.
- **The database has never held a resolution decision.** `entity` and
  `record_entity` exist, are indexed, and are empty. The schema is designed and
  loaded; the layer that fills it is not written yet, and this README does not
  imply otherwise.

## Two things here that most scraper projects skip

**A schema-change canary.** `canary.py` fetches a live page and fails loudly if
the expected fields come back empty. It tests the *website*, not the code, so it
is designed to fail one day. It carries a positive control, because a check that
can only pass is indistinguishable from no check. It runs on its own weekly
schedule rather than in CI, so the tests badge only ever goes red for a code
problem.

**Fetching is separated from parsing.** Pages are saved to disk, then parsed from
frozen fixtures. Parser tests run offline in milliseconds and are deterministic,
the detail crawl resumes by checking the filesystem rather than holding state in
memory, and per-request failures are collected and reported instead of raised -
5 of 1000 requests failed transiently and a plain re-run fetched exactly those 5.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python fetch.py books          # 50 listing pages -> pages/books/
python fetch_details.py books  # 1000 detail pages, resumable
python parse.py books          # -> data/books.csv
python clean.py                # typing + validation -> data/books_clean.csv
python store.py                # -> data/pipeline.db, needs clean.py to have run
python canary.py books         # live check: has the markup drifted?
python perturb.py              # build the labelled benchmark
python block.py                # candidate generation, measured
python match.py                # classifier vs baseline
python ablate.py               # what the features are worth
python decide.py               # scores -> one-to-one decisions
python error_analysis.py       # the pairs that go wrong, individually
python -m pytest -q            # 40 tests, all offline
```

`data/books.csv` is committed, so every step from `clean.py` onwards runs without
touching the live site. `data/books_clean.csv` and `data/pipeline.db` are **not**
committed: both are derived, both rebuild offline in seconds, and tracking the
clean CSV as well as the raw one would add two multi-megabyte blobs to history on
every re-scrape instead of one. So `clean.py` has to run before `store.py` on a
fresh clone - and `store.py` refuses, naming the script to run, rather than quietly
loading the unvalidated raw scrape if you skip it.

The Abt-Buy scripts need that benchmark's three CSVs, which are not redistributed
here. Put `Abt.csv`, `Buy.csv` and `abt_buy_perfectMapping.csv` in `benchmark/`
(gitignored) and then:

```bash
python match_abt_buy.py        # the classifier on real, human-labelled data
python abt_buy_features.py     # 20 resampled splits: does a feature matter?
```

## Not built yet

A raw document store and a UI. The normalised schema now exists and is loaded;
what is missing is the layer that fills it - union-find over the pairs the matcher
accepted, plus an audit of how many within-entity pairs the matcher scored *below*
threshold. Transitive closure will merge records the matcher explicitly rejected,
and that count is the honest price of closure, so it belongs in the output rather
than buried.

Blocking recall of 0.988 is now the binding cap on Abt-Buy recall, and the
residual misses point at the description column rather than at a better string
metric - so `description` as a genuinely *diverse* second blocking key is the next
experiment worth running. The rejected union-of-keys result says spelling variants
do not pay; a description is not a spelling variant of a product name.

Also outstanding: IDF weighting of category agreement, a length-aware treatment of
cosine for the 'Arena'/'Arnea' case, and generating these result tables from a
script with a drift test, so they cannot go stale by hand. Four of them already
did once.