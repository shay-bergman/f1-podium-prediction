"""The experiments: H1 model comparison, the FastF1 ablation, the championship
ceiling check, significance testing, the H2/H3 derived tables, and the printed
stats block that summarises the results.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import average_precision_score

from f1_config import (
    ABLATION_SEEDS,
    BURN_IN_FASTF1,
    BURN_IN_FULL,
    BURN_IN_QUALI,
    DATA_DIR,
    ERAS,
    FASTF1_START,
    FEATURE_SETS,
    QUALI_START,
    RNG_SEED,
)
from f1_models import (
    MODELS,
    baseline_grid_rate,
    baseline_grid_rule,
    bootstrap_precision_ci,
    evaluate,
    forward_chain,
    make_hgb,
    make_logreg,
    per_race_hits,
    precision_at_3,
    top3_mask,
)

N_PERM = 2000
N_BOOT = 2000


##### H1: model versus baselines


def run_h1(df: pd.DataFrame) -> dict:
    """Forward-chained comparison over the complete-qualifying era.

    Returns a dict carrying everything the later stages and the figures need.
    """
    full = df[df["year"] >= 1950].copy()
    quali = df[df["year"] >= QUALI_START].copy()
    print(
        f"[prep] full tier {full['year'].min()}-{full['year'].max()}: {len(full)} rows, "
        f"{full['raceId'].nunique()} races, podium prevalence {full['podium'].mean():.3f}"
    )
    print(
        f"[prep] quali tier {quali['year'].min()}-{quali['year'].max()}: {len(quali)} rows, "
        f"{quali['raceId'].nunique()} races, podium prevalence {quali['podium'].mean():.3f}"
    )

    print("\n--- H1: model vs baselines, forward-chained over 2003-2024 ---")
    scored = {}
    for set_name, feats in FEATURE_SETS.items():
        for model_name, factory in MODELS.items():
            key = f"{model_name} | {set_name}"
            scored[key] = forward_chain(quali, feats, factory, BURN_IN_QUALI)

    ref = scored["Gradient boosting | Qualifying only"]
    test_years = sorted(ref["year"].unique())
    holdout = quali[quali["year"].isin(test_years)]

    # Every row gets a race-clustered bootstrap CI on precision@3, not just the
    # single "best" model. Several rows (B1, B2, grid-only) sit within a
    # rounding error of each other, and a bare point estimate hides that.
    rows = []
    b1_mask = (holdout["grid_pos"] <= 3).values
    prec, rec, _ = baseline_grid_rule(holdout)
    rows.append(
        {
            "Method": "B1 grid <= 3 rule",
            "PR-AUC": np.nan,
            "ROC-AUC": np.nan,
            "Brier": np.nan,
            "Precision@3": prec,
            "Exact podium": np.nan,
            "n": len(holdout),
            "Precision@3 CI": bootstrap_precision_ci(holdout, mask=b1_mask),
        }
    )

    b0 = holdout.copy()
    b0["prob"] = holdout["podium"].mean()
    b0_scores = evaluate(b0)
    b0_scores["Precision@3"] = holdout["podium"].mean()  # ties -> expected value
    b0_scores["Exact podium"] = 0.0

    b0_scores["Precision@3 CI"] = (np.nan, np.nan)
    rows.append({"Method": "B0 majority class", **b0_scores})

    b2 = pd.concat(
        [
            baseline_grid_rate(quali[quali["year"] < y], quali[quali["year"] == y])
            for y in test_years
        ],
        ignore_index=True,
    )
    b2_scores = evaluate(b2)
    b2_scores["Precision@3 CI"] = bootstrap_precision_ci(b2)
    rows.append({"Method": "B2 grid base rate", **b2_scores})

    for key, frame in scored.items():
        s = evaluate(frame)
        s["Precision@3 CI"] = bootstrap_precision_ci(frame)
        rows.append({"Method": key, **s})

    summary = pd.DataFrame(rows).set_index("Method")
    printable = summary.copy()
    printable["Precision@3 CI"] = printable["Precision@3 CI"].apply(
        lambda t: "-" if pd.isna(t[0]) else f"[{t[0]:.3f}, {t[1]:.3f}]"
    )
    print(printable.round(3).to_string())

    return {
        "full": full,
        "quali": quali,
        "scored": scored,
        "summary": summary,
        "holdout": holdout,
        "test_years": test_years,
        "b2": b2,
        "b1_precision": prec,
        "b1_recall": rec,
    }


##### FastF1 ablation (only when the enrichment file is present)


def run_fastf1_ablation(
    df: pd.DataFrame, fastf1_cols: list[str], fastf1_years: list[int]
) -> pd.DataFrame | None:
    """Paired by construction: both models are forward-chained over the same
    seasons and scored on the same rows, so the only difference is the block."""
    BLOCK_Q = FEATURE_SETS["Qualifying only"]
    BLOCK_C = [c for c in FEATURE_SETS["Qualifying + form"] if c not in BLOCK_Q]
    FASTF1_COLS, FASTF1_YEARS = fastf1_cols, fastf1_years
    ff1_summary = None
    if FASTF1_COLS:
        ff_tier = df[df["year"].isin(FASTF1_YEARS)].copy()
        for col in FASTF1_COLS:
            ff_tier[col] = pd.to_numeric(ff_tier[col], errors="coerce")
        # Keep only columns that actually came back populated: a half-empty column
        # adds imputation noise and nothing else.
        usable = [c for c in FASTF1_COLS if ff_tier[c].notna().mean() > 0.30]
        dropped = [c for c in FASTF1_COLS if c not in usable]
        if dropped:
            print(f"[ff1] dropping sparse columns (<30% coverage): {dropped}")
        if not usable:
            print("[ff1] no column cleared the coverage threshold, ablation skipped")
        elif len(FASTF1_YEARS) <= BURN_IN_FASTF1:
            print(
                f"[ff1] only {len(FASTF1_YEARS)} season(s) fetched; the forward-chained "
                f"ablation needs more than {BURN_IN_FASTF1}. Re-run with "
                f"--with-fastf1 (no --seasons) to pull {FASTF1_START}-2024."
            )
        else:
            # Refit under every seed in ABLATION_SEEDS and report the
            # spread, not just the canonical-seed point estimate.
            pr_deltas, p3_deltas = [], []
            base_ff, with_ff = None, None
            for seed in ABLATION_SEEDS:
                factory = lambda s=seed: make_hgb(s)
                base = forward_chain(
                    ff_tier, BLOCK_Q + BLOCK_C, factory, BURN_IN_FASTF1
                )
                withf = forward_chain(
                    ff_tier, BLOCK_Q + BLOCK_C + usable, factory, BURN_IN_FASTF1
                )
                b_eval, w_eval = evaluate(base), evaluate(withf)
                pr_deltas.append(w_eval["PR-AUC"] - b_eval["PR-AUC"])
                p3_deltas.append(w_eval["Precision@3"] - b_eval["Precision@3"])
                if seed == RNG_SEED:
                    base_ff, with_ff = base, withf

            ff1_summary = pd.DataFrame(
                [
                    {"Method": "Qualifying + form (Kaggle only)", **evaluate(base_ff)},
                    {"Method": "Qualifying + form + FastF1", **evaluate(with_ff)},
                ]
            ).set_index("Method")
            print(
                f"\n--- FastF1 ablation, forward-chained over "
                f"{int(with_ff['year'].min())}-{int(with_ff['year'].max())} "
                f"({len(usable)} extra features) ---"
            )
            print(ff1_summary.round(3).to_string())
            print(
                f"  PR-AUC delta from FastF1 features (seed {RNG_SEED}): "
                f"{ff1_summary.loc['Qualifying + form + FastF1', 'PR-AUC'] - ff1_summary.loc['Qualifying + form (Kaggle only)', 'PR-AUC']:+.3f}"
            )

            pr_deltas, p3_deltas = np.array(pr_deltas), np.array(p3_deltas)
            n_pos_pr = int((pr_deltas > 0).sum())
            n_pos_p3 = int((p3_deltas > 0).sum())
            print(
                f"  Seed stability over {len(ABLATION_SEEDS)} seeds {ABLATION_SEEDS}:"
            )
            print(
                f"    PR-AUC delta:      mean {pr_deltas.mean():+.3f} (sd {pr_deltas.std():.3f}), "
                f"positive in {n_pos_pr}/{len(ABLATION_SEEDS)}"
            )
            print(
                f"    Precision@3 delta: mean {p3_deltas.mean():+.3f} (sd {p3_deltas.std():.3f}), "
                f"positive in {n_pos_p3}/{len(ABLATION_SEEDS)}"
            )
            if n_pos_pr not in (0, len(ABLATION_SEEDS)) or n_pos_p3 not in (
                0,
                len(ABLATION_SEEDS),
            ):
                print(
                    "    => sign is not stable across seeds: treat the single-seed CI as "
                    "optimistic, not as proof of a real effect."
                )
            ff1_summary.attrs["seed_stability"] = {
                "seeds": ABLATION_SEEDS,
                "pr_auc_delta": pr_deltas,
                "precision_at_3_delta": p3_deltas,
            }

            rng_ff = np.random.default_rng(RNG_SEED)
            N_BOOT_FF = 2000
            base_mask, with_mask = top3_mask(base_ff), top3_mask(with_ff)
            hits_base, k_base = per_race_hits(base_ff, base_mask)
            hits_with, k_with = per_race_hits(with_ff, with_mask)
            n_races_ff = len(hits_base)
            idx = rng_ff.integers(0, n_races_ff, size=(N_BOOT_FF, n_races_ff))
            p3_diff_boot = hits_with[idx].sum(axis=1) / k_with[idx].sum(
                axis=1
            ) - hits_base[idx].sum(axis=1) / k_base[idx].sum(axis=1)
            p3_d_lo, p3_d_hi = np.percentile(p3_diff_boot, [2.5, 97.5])

            # PR-AUC isn't computed per race, so this half resamples races but
            # pulls in every row belonging to each resampled race (same pattern
            # as the paired bootstrap against B2 in run_significance).
            pair = base_ff[["raceId", "driverId", "podium"]].copy()
            pair["prob_base"] = base_ff["prob"].values
            pair["prob_with"] = (
                with_ff.set_index(["raceId", "driverId"])
                .loc[list(zip(base_ff["raceId"], base_ff["driverId"])), "prob"]
                .values
            )
            race_index = {r: i for i, r in enumerate(pair["raceId"].unique())}
            groups = pair["raceId"].map(race_index).values
            pr_diff_boot = np.empty(N_BOOT_FF)
            for i in range(N_BOOT_FF):
                pick = rng_ff.integers(0, len(race_index), size=len(race_index))
                sel = np.concatenate([np.flatnonzero(groups == g) for g in pick])
                s = pair.iloc[sel]
                if s["podium"].nunique() < 2:
                    pr_diff_boot[i] = 0.0
                    continue
                pr_diff_boot[i] = average_precision_score(
                    s["podium"], s["prob_with"]
                ) - average_precision_score(s["podium"], s["prob_base"])
            pr_d_lo, pr_d_hi = np.percentile(pr_diff_boot, [2.5, 97.5])

            print(
                f"  Paired bootstrap over races (seed {RNG_SEED}; captures test-set "
                f"sampling variance the seed sweep above does not):"
            )
            print(
                f"    PR-AUC delta:      {pr_diff_boot.mean():+.3f} "
                f"95% CI [{pr_d_lo:+.3f}, {pr_d_hi:+.3f}]"
            )
            print(
                f"    Precision@3 delta: {p3_diff_boot.mean():+.3f} "
                f"95% CI [{p3_d_lo:+.3f}, {p3_d_hi:+.3f}]"
            )
            ff1_summary.attrs["paired_bootstrap"] = {
                "pr_auc_delta_mean": pr_diff_boot.mean(),
                "pr_auc_delta_ci": (pr_d_lo, pr_d_hi),
                "precision_at_3_delta_mean": p3_diff_boot.mean(),
                "precision_at_3_delta_ci": (p3_d_lo, p3_d_hi),
            }

    return ff1_summary


##### Championship ceiling check


def championship_ceiling(races: pd.DataFrame) -> list[dict]:
    champ_std = pd.read_csv(os.path.join(DATA_DIR, "constructor_standings.csv"))
    champ = champ_std.merge(races[["raceId", "year", "round"]], on="raceId")
    rounds_per_year = champ.groupby("year")["round"].max()

    ceiling_rows = []
    for frac in (0.25, 0.33, 0.50):
        hits, total, by_era = 0, 0, {}
        for yr, n_rounds in rounds_per_year.items():
            if yr < 1958:  # constructors' title starts in 1958
                continue
            season = champ[champ["year"] == yr]
            cut = int(np.ceil(frac * n_rounds))
            early = season[season["round"] == cut]
            final = season[season["round"] == n_rounds]
            if early.empty or final.empty:
                continue
            leader = early.loc[early["points"].idxmax(), "constructorId"]
            winner = final.loc[final["points"].idxmax(), "constructorId"]
            hits += int(leader == winner)
            total += 1
            era = next((l for s, e, l, _ in ERAS if s <= yr <= e), None)
            if era:
                a, b = by_era.get(era, (0, 0))
                by_era[era] = (a + int(leader == winner), b + 1)
        ceiling_rows.append(
            {
                "fraction": frac,
                "accuracy": hits / total,
                "n_seasons": total,
                "by_era": by_era,
            }
        )

    print("\n--- Championship ceiling: does the early leader win the title? ---")
    for row in ceiling_rows:
        print(
            f"  after {row['fraction'] * 100:.0f}% of the season: "
            f"{row['accuracy'] * 100:.1f}% ({row['n_seasons']} seasons)"
        )
    print("  by era, at one-third distance:")
    for era, (a, b) in ceiling_rows[1]["by_era"].items():
        print(f"    {era:22s} {a}/{b} = {a / b * 100:.0f}%")
    print(
        "  => a single feature already explains most of the outcome, so a classifier\n"
        "     on 67 seasons cannot demonstrate learning above it. Task scoped out."
    )

    return ceiling_rows


##### Significance


def run_significance(res: dict) -> tuple:
    """Permutation test, bootstrap CIs, a paired bootstrap against B2 and McNemar
    against the grid rule. Returns the best-scoring frame for downstream use."""
    scored, b2 = res["scored"], res["b2"]
    rng = np.random.default_rng(RNG_SEED)
    best_key = max(scored, key=lambda k: evaluate(scored[k])["Precision@3"])
    best = scored[best_key]
    obs_p3, obs_exact = precision_at_3(best)

    # Permutation test: shuffle the podium labels *within* each race, preserving exactly
    # three positives, then re-score the same predictions. Tests whether the ordering
    # carries information beyond the per-race class structure.
    best_mask = top3_mask(best)
    n_podium = best.groupby("raceId")["podium"].transform("sum").values
    denominator = per_race_hits(best, best_mask)[1].sum()
    perm_stats = np.empty(N_PERM)
    for i in range(N_PERM):
        # Rank a random key within each race and hand the podium to the lowest n_podium
        # entries: an exact within-race shuffle that preserves three positives per race.
        key = pd.Series(rng.random(len(best)))
        order = key.groupby(best["raceId"].values).rank(method="first").values
        perm_stats[i] = np.sum((order <= n_podium) & best_mask) / denominator
    p_value = (np.sum(perm_stats >= obs_p3) + 1) / (N_PERM + 1)
    print(
        f"\nPermutation test on {best_key}: observed precision@3 = {obs_p3:.3f}, "
        f"null mean = {perm_stats.mean():.3f} (sd {perm_stats.std():.3f}), p = {p_value:.4f}"
    )

    # Bootstrap CI over races (races are the independent unit, not rows).
    hits_arr, k_arr = per_race_hits(best, best_mask)
    n_races = len(hits_arr)
    idx = rng.integers(0, n_races, size=(N_BOOT, n_races))
    boot = hits_arr[idx].sum(axis=1) / k_arr[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"Bootstrap 95% CI for precision@3: [{lo:.3f}, {hi:.3f}]")

    pair = best[["raceId", "podium"]].copy()
    pair["prob_model"] = best["prob"].values
    pair["prob_b2"] = (
        b2.set_index(["raceId", "driverId"])
        .loc[list(zip(best["raceId"], best["driverId"])), "prob"]
        .values
    )
    race_index = {r: i for i, r in enumerate(pair["raceId"].unique())}
    groups = pair["raceId"].map(race_index).values
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(race_index), size=len(race_index))
        sel = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        s = pair.iloc[sel]
        if s["podium"].nunique() < 2:
            diffs[i] = 0.0
            continue
        diffs[i] = average_precision_score(
            s["podium"], s["prob_model"]
        ) - average_precision_score(s["podium"], s["prob_b2"])
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
    print(
        f"Paired bootstrap PR-AUC (model - B2 grid base rate): "
        f"{diffs.mean():+.3f} 95% CI [{d_lo:+.3f}, {d_hi:+.3f}]"
    )

    # McNemar on the top-3 picks: where do the model and the grid rule disagree?
    b1_pick = (best["grid_pos"] <= 3).values
    model_pick = best_mask
    b1_right = b1_pick == best["podium"].values
    model_right = model_pick == best["podium"].values
    n01 = int(np.sum(~b1_right & model_right))
    n10 = int(np.sum(b1_right & ~model_right))
    mcnemar_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) else 0.0
    mcnemar_p = 1 - chi2.cdf(mcnemar_stat, 1)
    print(
        f"McNemar model vs B1: model-only correct={n01}, B1-only correct={n10}, "
        f"chi2={mcnemar_stat:.2f}, p={mcnemar_p:.4f}"
    )

    return best, best_mask, mcnemar_p


##### H2


def run_h2(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    BLOCK_G = ["grid_pos", "grid_norm", "pit_lane_start", "n_starters"]
    BLOCK_C = [
        c
        for c in FEATURE_SETS["Qualifying + form"]
        if c not in FEATURE_SETS["Qualifying only"]
    ]
    full_scored = forward_chain(full, BLOCK_G + BLOCK_C, make_logreg, BURN_IN_FULL)
    per_season = []
    for yr, grp in full_scored.groupby("year"):
        p3, exact = precision_at_3(grp)
        b1p, _, _ = baseline_grid_rule(grp)

        avg_n = grp.groupby("raceId")["n_starters"].first().mean()
        chance = 3 / avg_n
        per_season.append(
            {
                "year": yr,
                "precision_at_3": p3,
                "exact": exact,
                "b1_precision": b1p,
                "avg_field_size": avg_n,
                "chance_level": chance,
                "lift_over_chance": p3 - chance,
            }
        )
    per_season = pd.DataFrame(per_season).sort_values("year").reset_index(drop=True)

    return full_scored, per_season


##### H3: what does the model get wrong, and why?


def run_h3(full_scored: pd.DataFrame, full: pd.DataFrame) -> tuple:
    """Returns (anatomy, anatomy_pct, dnf_rate_by_era)."""
    misses = full_scored.copy()
    misses["pred_top3"] = top3_mask(misses).astype(int)
    wrong = misses[(misses["pred_top3"] == 1) & (misses["podium"] == 0)]

    era_of = {}
    for start, end, label, _ in ERAS:
        for y in range(start, end + 1):
            era_of[y] = label
    wrong = wrong[wrong["year"].isin(era_of)].copy()
    wrong["era"] = wrong["year"].map(era_of)
    anatomy = wrong.groupby(["era", "finish_class"]).size().unstack(fill_value=0)
    anatomy = anatomy.reindex([l for _, _, l, _ in ERAS])
    anatomy_pct = anatomy.div(anatomy.sum(axis=1), axis=0) * 100

    starters = full[(full["grid"] > 0) & full["year"].isin(era_of)].copy()
    starters["era"] = starters["year"].map(era_of)
    dnf_rate_by_era = (starters.groupby("era")["dnf"].mean() * 100).reindex(
        [l for _, _, l, _ in ERAS]
    )

    return anatomy, anatomy_pct, dnf_rate_by_era


##### H3, decade view

H3_DECADES = [
    (1960, 1969, "1960s"),
    (1970, 1979, "1970s"),
    (1980, 1989, "1980s"),
    (1990, 1999, "1990s"),
    (2000, 2009, "2000s"),
    (2010, 2019, "2010s"),
    (2020, 2024, "2020s"),
]


def run_h3_by_decade(full_scored: pd.DataFrame) -> pd.DataFrame:
    misses = full_scored.copy()
    misses["pred_top3"] = top3_mask(misses).astype(int)
    wrong = misses[(misses["pred_top3"] == 1) & (misses["podium"] == 0)]

    decade_of = {}
    for start, end, label in H3_DECADES:
        for y in range(start, end + 1):
            decade_of[y] = label
    wrong = wrong[wrong["year"].isin(decade_of)].copy()
    wrong["decade"] = wrong["year"].map(decade_of)

    anatomy = wrong.groupby(["decade", "finish_class"]).size().unstack(fill_value=0)
    anatomy = anatomy.reindex([label for _, _, label in H3_DECADES])
    anatomy_pct = anatomy.div(anatomy.sum(axis=1), axis=0) * 100
    mech = anatomy_pct.get("mechanical", 0)

    return pd.DataFrame({"mechanical": mech, "other_combined": 100 - mech})


##### The printed stats block that summarises the results


def print_key_stats(
    res: dict,
    per_season: pd.DataFrame,
    anatomy_pct: pd.DataFrame,
    dnf_rate_by_era: pd.Series,
) -> None:
    full, scored = res["full"], res["scored"]
    print("\n--- Key stats for narrative ---")

    print("\nBaseline B1 (grid <= 3 rule) precision by era:")
    for start, end, label, _ in ERAS:
        d = full[(full["year"] >= start) & (full["year"] <= end) & (full["grid"] > 0)]
        p, r, _ = baseline_grid_rule(d)
        print(
            f"  {label:22s} {start}-{end}: precision={p:.3f} recall={r:.3f} n={len(d)}"
        )

    print("\nWhere podium finishers started from:")
    for lo_y, hi_y in [(1958, 1969), (2003, 2024)]:
        d = full[
            (full["year"] >= lo_y)
            & (full["year"] <= hi_y)
            & (full["grid"] > 0)
            & (full["podium"] == 1)
        ]
        band = pd.cut(
            d["grid_pos"],
            [0, 3, 6, 10, 99],
            labels=["P1-P3", "P4-P6", "P7-P10", "P11+"],
        )
        share = band.value_counts(normalize=True).reindex(
            ["P1-P3", "P4-P6", "P7-P10", "P11+"]
        )
        print(
            f"  {lo_y}-{hi_y}: "
            + ", ".join(f"{k} {v * 100:.1f}%" for k, v in share.items())
        )

    print("\nAblation (gradient boosting, forward-chained 2008-2024):")
    for set_name in FEATURE_SETS:
        s = evaluate(scored[f"Gradient boosting | {set_name}"])
        print(
            f"  {set_name:20s} PR-AUC={s['PR-AUC']:.3f}  precision@3={s['Precision@3']:.3f}  "
            f"exact podium={s['Exact podium']:.3f}"
        )

    print("\nPredictability trend (model precision@3, decade means):")
    for lo_y, hi_y in [
        (1960, 1969),
        (1970, 1979),
        (1980, 1989),
        (1990, 1999),
        (2000, 2009),
        (2010, 2019),
        (2020, 2024),
    ]:
        d = per_season[(per_season["year"] >= lo_y) & (per_season["year"] <= hi_y)]
        if len(d):
            print(f"  {lo_y}-{hi_y}: {d['precision_at_3'].mean() * 100:.1f}%")

    print(
        "\nSame trend, net of chance-level drift from changing field size "
        "(precision@3 minus 3/avg field size, decade means):"
    )
    for lo_y, hi_y in [
        (1960, 1969),
        (1970, 1979),
        (1980, 1989),
        (1990, 1999),
        (2000, 2009),
        (2010, 2019),
        (2020, 2024),
    ]:
        d = per_season[(per_season["year"] >= lo_y) & (per_season["year"] <= hi_y)]
        if len(d):
            print(
                f"  {lo_y}-{hi_y}: avg field size {d['avg_field_size'].mean():.1f}, "
                f"chance {d['chance_level'].mean() * 100:.1f}%, "
                f"lift {d['lift_over_chance'].mean() * 100:.1f}pp"
            )

    print("\nError anatomy (% of wrong podium picks):")
    print(anatomy_pct.round(1).to_string())

    print(
        "\nModel-free check: mechanical/incident DNF rate among actual starters, "
        "by era (no model, straight from status):"
    )
    print(dnf_rate_by_era.round(1).to_string())

    best_season = per_season.loc[per_season["precision_at_3"].idxmax()]
    worst_season = per_season.loc[per_season["precision_at_3"].idxmin()]
    print(
        f"\nMost predictable season: {int(best_season['year'])} "
        f"({best_season['precision_at_3'] * 100:.0f}% of podium slots called correctly)"
    )
    print(
        f"Least predictable season: {int(worst_season['year'])} "
        f"({worst_season['precision_at_3'] * 100:.0f}%)"
    )
    print("\nDone! All visualizations created.")
