"""Predict F1 podium finishes from pre-race data (1950-2024).

Entry point and orchestration. The work is split across modules:

    f1_config.py        constants, era definitions, plot styling, feature blocks
    f1_data.py          CSV loading, leak-free feature engineering, and the
                        optional FastF1 enrichment pull/audit (2018-2024)
    f1_models.py        models, metrics, baselines, forward-chained evaluation
    f1_experiments.py   H1/H2/H3, the FastF1 ablation, significance testing,
                        and the printed stats block
    f1_viz.py           the three result figures
    f1_podium_prediction.py   this file: CLI, orchestration, self-tests

Usage:

    # Kaggle-only analysis (~4 min, no network needed)
    python f1_podium_prediction.py

    # Check the plumbing before the long FastF1 pull (~3 min)
    python f1_podium_prediction.py --with-fastf1 --seasons 2024

    # Fetch every FastF1 season, waiting out the 500 calls/hour limit
    python f1_podium_prediction.py --with-fastf1 --auto-resume --fetch-only

    # Check the fetched FastF1 data is complete and correctly joined
    python f1_podium_prediction.py --audit-fastf1

    # Self-tests
    python f1_podium_prediction.py --selftest

Requirements:
    pip install pandas numpy scikit-learn matplotlib scipy
    pip install "fastf1>=3.3"        # only for --with-fastf1

Expected layout:
    f1_*.py
    kaggle_f1_dataset/*.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from f1_config import DATA_DIR, FASTF1_CACHE, FASTF1_START, OUT_DIR


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--with-fastf1",
        action="store_true",
        help="fetch FastF1 features before analysing (needs network)",
    )
    p.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(range(FASTF1_START, 2025)),
        help="seasons to pull from FastF1 (default 2018-2024)",
    )
    p.add_argument(
        "--ff1-cache",
        default=os.path.join(OUT_DIR, "ff1_cache"),
        help="FastF1 HTTP cache directory",
    )
    p.add_argument(
        "--force-refetch",
        action="store_true",
        help="ignore per-season cache files and pull again",
    )
    p.add_argument(
        "--auto-resume",
        action="store_true",
        help="when the API rate limit is hit, wait and continue "
        "automatically until every season is fetched",
    )
    p.add_argument(
        "--wait-minutes",
        type=int,
        default=61,
        help="minutes to wait after a rate limit (default 61)",
    )
    p.add_argument(
        "--max-waits",
        type=int,
        default=12,
        help="give up after this many waits (default 12)",
    )
    p.add_argument(
        "--fetch-only",
        action="store_true",
        help="fetch FastF1 data and exit without running the analysis",
    )
    p.add_argument(
        "--no-fp2",
        action="store_true",
        help="skip FP2 long-run pace; roughly halves the API calls per "
        "season, which matters against the 500 calls/hour limit",
    )
    p.add_argument(
        "--audit-fastf1",
        action="store_true",
        help="audit the fetched FastF1 data for completeness and join "
        "integrity, then exit",
    )
    p.add_argument(
        "--selftest", action="store_true", help="run the built-in tests and exit"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested stage.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv``.

    Returns:
        A process exit code (0 on success; the number of failures for the
        self-tests and the audit).
    """
    args = build_parser().parse_args(argv)

    if args.with_fastf1:
        from f1_data import build_fastf1_features

        build_fastf1_features(
            args.seasons,
            args.ff1_cache,
            FASTF1_CACHE,
            force=args.force_refetch,
            want_fp2=not args.no_fp2,
            auto_resume=args.auto_resume,
            wait_minutes=args.wait_minutes,
            max_waits=args.max_waits,
        )
        if args.fetch_only:
            return 0

    from f1_data import load_dataset

    df, fastf1_cols, fastf1_years = load_dataset()

    if args.selftest:
        return run_selftest(df)

    if args.audit_fastf1:
        from f1_data import audit_fastf1_features

        return audit_fastf1_features(
            df, pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
        )

    from f1_experiments import (
        championship_ceiling,
        print_key_stats,
        run_fastf1_ablation,
        run_h1,
        run_h2,
        run_h3,
        run_h3_by_decade,
        run_significance,
    )
    from f1_viz import (
        viz_h1_model_showdown,
        viz_h2_decade_predictability,
        viz_h3_error_shift,
    )

    res = run_h1(df)
    run_fastf1_ablation(df, fastf1_cols, fastf1_years)
    championship_ceiling(pd.read_csv(os.path.join(DATA_DIR, "races.csv")))
    # Return values (best model frame, McNemar p, ...) no longer feed a figure,
    # but the printed permutation/bootstrap/McNemar block is still report text.
    run_significance(res)

    full_scored, per_season = run_h2(res["full"])
    _, anatomy_pct, dnf_rate_by_era = run_h3(full_scored, res["full"])
    decade_anatomy = run_h3_by_decade(full_scored)

    viz_h1_model_showdown(res["summary"])
    viz_h2_decade_predictability(per_season)
    viz_h3_error_shift(decade_anatomy)

    print_key_stats(res, per_season, anatomy_pct, dnf_rate_by_era)
    return 0


##### Self-tests
#
# Not accuracy checks. These prove the two things that would silently invalidate
# every number in the report: that no feature can see the race it is predicting,
# and that the evaluation never trains and tests on the same race.


def _toy() -> pd.DataFrame:
    """Three races, four drivers. Grid order equals finishing order."""
    rows = []
    for race in (1, 2, 3):
        for pos, drv in enumerate([1, 2, 3, 4], start=1):
            rows.append(
                {
                    "raceId": race,
                    "driverId": drv,
                    "grid_pos": pos,
                    "podium": int(pos <= 3),
                    "prob": 1.0 / pos,
                    "year": 2000 + race,
                }
            )
    return pd.DataFrame(rows)


class _Spy:
    """Records which seasons it was trained on."""

    seen = []

    def fit(self, X, y):
        _Spy.seen.append(sorted(X["year"].unique()))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.5)])


def run_selftest(df: pd.DataFrame) -> int:
    from f1_config import QUALI_START
    from f1_data import parse_lap, prior_by_race
    from f1_models import (
        baseline_grid_rate,
        baseline_grid_rule,
        bootstrap_precision_ci,
        forward_chain,
        precision_at_3,
        top3_mask,
    )

    checks, failures = 0, []

    def check(name, condition):
        nonlocal checks
        checks += 1
        if condition:
            print(f"  [pass] {name}")
        else:
            failures.append(name)
            print(f"  [FAIL] {name}")

    print("Metric correctness")
    toy = _toy()
    p3, exact = precision_at_3(toy)
    check(
        "perfect ranking scores 1.0", abs(p3 - 1.0) < 1e-9 and abs(exact - 1.0) < 1e-9
    )

    rev = toy.copy()
    rev["prob"] = rev["grid_pos"]
    p3r, exr = precision_at_3(rev)
    check("reversed ranking scores 2/3", abs(p3r - 2 / 3) < 1e-9 and exr == 0.0)

    check(
        "top-3 mask picks exactly three per race",
        toy[top3_mask(toy)].groupby("raceId").size().eq(3).all(),
    )

    short = pd.DataFrame(
        {"raceId": [9, 9], "driverId": [1, 2], "podium": [1, 1], "prob": [0.9, 0.8]}
    )
    check(
        "short field does not over-pick or crash",
        top3_mask(short).sum() == 2 and abs(precision_at_3(short)[0] - 1.0) < 1e-9,
    )

    tied = toy.copy()
    tied["prob"] = 0.5
    check("tied probabilities stay finite", np.isfinite(precision_at_3(tied)[0]))

    print("Baselines")
    prec_t, rec_t, _ = baseline_grid_rule(toy)
    check(
        "grid rule is perfect when grid equals result",
        abs(prec_t - 1.0) < 1e-9 and abs(rec_t - 1.0) < 1e-9,
    )

    unseen_train = pd.DataFrame(
        {"grid_pos": [1, 2], "podium": [1, 1], "raceId": [1, 1]}
    )
    unseen_test = pd.DataFrame({"grid_pos": [17], "podium": [0], "raceId": [2]})
    check(
        "unseen grid slot falls back to the prior, not NaN",
        baseline_grid_rate(unseen_train, unseen_test)["prob"].notna().all(),
    )

    print("Leakage guards")
    d = pd.DataFrame(
        {
            "driverId": [1] * 4,
            "podium": [1, 1, 1, 0],
            "date": pd.date_range("2020-01-01", periods=4),
        }
    ).sort_values("date")
    rolled = d.groupby("driverId")["podium"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    check("form feature is NaN before a driver's first race", pd.isna(rolled.iloc[0]))
    check("form feature excludes the current race", abs(rolled.iloc[3] - 1.0) < 1e-9)

    team = (
        pd.DataFrame(
            {
                "raceId": [1, 1, 2, 2],
                "constructorId": ["A", "A", "A", "A"],
                "driverId": [1, 2, 1, 2],
                "podium": [1, 0, 0, 1],
                "points": [25, 18, 10, 8],
                "date": pd.to_datetime(
                    ["2020-01-01", "2020-01-01", "2020-01-08", "2020-01-08"]
                ),
            }
        )
        .sort_values(["date", "raceId"])
        .reset_index(drop=True)
    )
    team["race_seq"] = team.groupby("raceId", sort=False).ngroup()
    team_rate = prior_by_race(team, "constructorId", "podium", window=5)
    team_pts = prior_by_race(team, "constructorId", "points", agg="sum")
    check(
        "team-level form feature is NaN for both teammates in the opening race",
        pd.isna(team_rate[0]) and pd.isna(team_rate[1]),
    )
    check(
        "team-level form feature never leaks a teammate's same-race result",
        abs(team_rate[2] - team_rate[3]) < 1e-9
        and abs(team_pts[2] - team_pts[3]) < 1e-9,
    )
    check(
        "team-level cumulative points sum both teammates' prior race, not one",
        abs(team_pts[2] - 43.0) < 1e-9,
    )

    _Spy.seen = []
    chain_data = pd.DataFrame(
        {
            "year": np.repeat([2000, 2001, 2002, 2003], 4),
            "raceId": np.repeat([1, 2, 3, 4], 4),
            "driverId": list(range(1, 5)) * 4,
            "podium": [1, 1, 1, 0] * 4,
        }
    )
    chained = forward_chain(chain_data, ["year"], _Spy, burn_in=1)
    check(
        "forward chaining never trains on the test season",
        all(max(tr) < te for tr, te in zip(_Spy.seen, [2001, 2002, 2003])),
    )
    check(
        "forward chaining honours the burn-in",
        set(chained["year"]) == {2001, 2002, 2003},
    )

    print("Feature construction on the real data")
    check(
        "no Indianapolis 500 rows survive 1950-1960",
        not (
            (df["name"].str.contains("Indianapolis", case=False, na=False))
            & (df["year"] <= 1960)
        ).any(),
    )
    check("pit-lane starts are recoded, never zero", (df["grid_pos"] > 0).all())
    check(
        "grid slots beyond the field size are rare and pre-1990",
        (((df["grid"] > df["n_starters"]) & (df["grid"] > 0)).mean() < 0.001)
        and df.loc[(df["grid"] > df["n_starters"]) & (df["grid"] > 0), "year"].max()
        < 1990,
    )
    check(
        "podium label matches positionOrder",
        (df["podium"] == (df["positionOrder"] <= 3).astype(int)).all(),
    )
    check(
        "at most three podium finishers per race from 1961 onwards",
        df[df["year"] >= 1961].groupby("raceId")["podium"].sum().le(3).all(),
    )
    check(
        "extra podium rows are confined to the shared-drive era",
        df.loc[df.groupby("raceId")["podium"].transform("sum") > 3, "year"].max()
        <= 1960,
    )
    check(
        "shared-drive podium rows never exceed seven",
        df.groupby("raceId")["podium"].sum().le(7).all(),
    )
    q_era = df[df["year"] >= QUALI_START]
    check(
        "session-normalised gap is non-negative and clipped",
        q_era["gap_to_session_best_pct"].dropna().between(0, 20).all(),
    )
    check(
        "Q3 drivers sit closer to the session best than Q1 drivers",
        q_era.loc[q_era["quali_stage"] == 3, "gap_to_session_best_pct"].mean()
        < q_era.loc[q_era["quali_stage"] == 1, "gap_to_session_best_pct"].mean(),
    )
    check(
        "qualifying coverage is essentially complete from 2003",
        q_era["quali_pos"].isna().mean() < 0.02,
    )

    print("Lap-time parsing")
    cases = [
        ("1:15.962", 75.962),
        ("0:59.100", 59.1),
        ("59.100", 59.1),
        ("\\N", None),
        ("", None),
        (None, None),
        (float("nan"), None),
    ]
    ok = True
    for raw, want in cases:
        got = parse_lap(raw)
        ok &= np.isnan(got) if want is None else abs(got - want) < 1e-9
    check("parse_lap handles all sentinel and format cases", ok)

    print("Bootstrap interval")

    ci_lo, ci_hi = bootstrap_precision_ci(toy, seed=0)
    check(
        "precision@3 bootstrap CI collapses to a point when every race is identical",
        abs(ci_lo - 1.0) < 1e-9 and abs(ci_hi - 1.0) < 1e-9,
    )

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED: " + "; ".join(failures))
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
