# F1 Podium Prediction

*Predicting Formula 1 podium finishes from pre-race data, with leakage-safe evaluation.*

## Overview

This project asks whether a Formula 1 podium finish can be predicted using only
information available before the race starts: starting grid position, qualifying
pace, and recent form. The data is one row per driver per race across the
1950-2024 seasons, assembled from the Kaggle "Formula 1 World Championship"
tables.

Each entry carries three blocks of pre-race features:

- **Grid** — the confirmed starting slot and the field size.
- **Qualifying pace** — the driver's lap-time gap to the fastest qualifier, as a
  percentage, plus gaps to the cars immediately ahead and behind and to the
  teammate. Qualifying features are restricted to 2003 onward, the first season
  with complete coverage.
- **Recent form** — podium rate, championship-points share, and car-completion
  rate over the driver's and team's previous five races.

An optional enrichment tier pulls sector-time deltas, speed-trap percentiles, and
long-run practice pace from the FastF1 timing API for 2018-2024, to test whether
modern telemetry adds signal beyond the public data.

Three hypotheses are tested:

- **H1** — a model on pre-race features beats the common heuristic "the top three
  qualifiers make the podium."
- **H2** — predictability has risen over time.
- **H3** — what makes a race hard to predict is cars breaking down, not close
  racing; as cars grew more reliable, the model's errors should shift from
  mechanical retirements to drivers simply beaten on track.

## Highlights

- **Leakage-safe by construction.** Every form feature is aggregated to one row
  per race and shifted by one race before use, so no feature can see the race it
  predicts. The guards are asserted in the self-tests, not just promised.
- **Forward-chained evaluation.** Models train only on seasons strictly before
  the test season, never on a random split that would leak an era's future.
- **Honest baselines.** The model is measured against random guessing, the
  top-three-grid heuristic, and the empirical podium rate per grid slot.
- **Uncertainty quantified.** Race-clustered bootstrap confidence intervals, a
  within-race permutation test, and a McNemar test against the grid heuristic.

## Tech stack

Python 3, pandas and NumPy for the data, scikit-learn for the models
(logistic regression and histogram gradient boosting), SciPy for the statistics,
matplotlib for the figures, and the optional FastF1 package for telemetry.

## Repository structure

```
.
├── f1_config.py              constants, eras, plot styling, feature blocks
├── f1_data.py                CSV loading, leak-free features, FastF1 enrichment
├── f1_models.py              models, metrics, baselines, forward-chaining
├── f1_experiments.py         H1/H2/H3, the FastF1 ablation, significance tests
├── f1_viz.py                 the result figures
├── f1_podium_prediction.py   CLI, orchestration, self-tests
├── kaggle_f1_dataset/        the Kaggle CSVs (not committed; see Data)
├── requirements.txt
└── README.md
```

## Getting started

```bash
pip install -r requirements.txt
```

The FastF1 dependency is only needed for the optional telemetry tier; the main
analysis runs on the Kaggle CSVs alone.

## Usage

```bash
# Kaggle-only analysis (no network needed)
python f1_podium_prediction.py

# Self-tests: leakage guards, metric correctness, lap-time parsing
python f1_podium_prediction.py --selftest

# Fetch the FastF1 telemetry tier (respects the 500 calls/hour limit)
python f1_podium_prediction.py --with-fastf1 --auto-resume --fetch-only

# Audit the fetched FastF1 data for completeness and join integrity
python f1_podium_prediction.py --audit-fastf1
```

The run prints the full results tables and writes three figures (the model-vs-
baseline showdown, predictability by decade, and the error-cause shift).

## Data

The analysis uses the **Kaggle "Formula 1 World Championship (1950-2020)"**
dataset (updated through 2024), a set of relational CSV files covering races,
results, qualifying, standings, and status. It is not committed. Download it from
Kaggle and place the CSVs under `kaggle_f1_dataset/`:

```
kaggle_f1_dataset/{races,results,qualifying,constructors,
    constructor_standings,status,drivers}.csv
```

The optional `fastf1_features.csv` is regenerated from the FastF1 library rather
than committed; `f1_data.py` builds it (`build_fastf1_features`), and
`f1_podium_prediction.py --with-fastf1` drives the pull. Both the dataset
directory and the feature cache are gitignored.

Preprocessing applies two fixes that matter: the Indianapolis 500 rounds
(1950-1960) are dropped, since they were contested by Indy-car constructors and
would corrupt grid-based features, and constructor identity is normalised across
engine suffixes so "Lotus-Climax", "Lotus-Ford" and "Team Lotus" collapse to one
team. Because the points system changed several times, the form features use
scale-invariant shares rather than raw points.

## Results

Forward-chained over the complete-qualifying era, the full model reaches about
**66% precision@3** — two of every three podium slots called correctly. That is
statistically indistinguishable from the top-three-grid heuristic (65.7%,
p ≈ 0.74): the model improves overall ranking quality (PR-AUC 0.679 versus 0.143
for random guessing, and +0.031 over the per-slot grid base rate), but starting
position alone carries nearly all of the signal needed to name the podium. Adding
FastF1 telemetry on the recent seasons gave no measurable gain (Δ precision@3
≈ +0.4%, 95% CI crossing zero).

Predictability rose from **44.7% in the 1960s to 66% in the 2020s**. The cause is
reliability, not closer racing: the mechanical/incident retirement rate among
starters fell from ~46% in the early eras to ~12% today, and the model's wrong
podium picks shifted accordingly — from drivers whose cars broke down to drivers
who were simply outpaced on track.

## License

Released under the [MIT License](LICENSE).
