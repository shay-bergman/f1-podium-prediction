"""Loading the Kaggle CSVs, building the modelling table, and the optional FastF1
enrichment tier (2018-2024).

One row per driver per race. Every feature in here is computed only from
information available before the race starts; the leakage guards are asserted in
f1_qualifying_prediction.py's self-tests rather than merely promised.
"""

import os
import time

import numpy as np
import pandas as pd

from f1_config import DATA_DIR, FASTF1_CACHE

MECHANICAL = {
    "Engine",
    "Gearbox",
    "Transmission",
    "Electrical",
    "Suspension",
    "Brakes",
    "Clutch",
    "Hydraulics",
    "Fuel system",
    "Turbo",
    "Overheating",
    "Driveshaft",
    "Differential",
    "Exhaust",
    "Radiator",
    "Wheel",
    "Throttle",
    "Steering",
    "Tyre",
    "Puncture",
    "Oil leak",
    "Water leak",
    "Fuel pump",
    "Ignition",
    "Halfshaft",
    "Wheel bearing",
    "Alternator",
    "Battery",
    "Power Unit",
    "ERS",
    "Power loss",
    "Mechanical",
    "Vibrations",
    "Oil pressure",
    "Water pressure",
    "Fuel pressure",
    "Spark plugs",
    "Distributor",
    "Pneumatics",
    "Track rod",
    "Rear wing",
    "Front wing",
    "Undertray",
    "Chassis",
    "Fuel",
    "Cooling system",
    "Handling",
    "Drivetrain",
    "Fuel leak",
    "Oil pipe",
    "Fuel pipe",
    "Magneto",
    "Injection",
    "Supercharger",
}
INCIDENT = {
    "Accident",
    "Collision",
    "Spun off",
    "Collision damage",
    "Damage",
    "Fatal accident",
    "Injury",
    "Injured",
}


def finish_class(s: str) -> str:
    """Bucket an Ergast finishing status into one of four outcome classes.

    Args:
        s: The raw status string, e.g. "Finished", "+1 Lap", "Engine", "Accident".

    Returns:
        "classified" (finished or classified a lap down), "incident" (crash or
        collision), "mechanical" (a car failure), or "other".
    """
    if s == "Finished" or s.startswith("+"):
        return "classified"
    if s in INCIDENT:
        return "incident"
    if s in MECHANICAL:
        return "mechanical"
    return "other"


def parse_lap(t: object) -> float:
    """Parse a lap time into seconds.

    Args:
        t: A lap string such as "1:15.962", or an Ergast missing-value sentinel.

    Returns:
        The lap in seconds (``75.962`` for ``"1:15.962"``), or ``nan`` for the
        ``\\N`` sentinel, empty strings and non-strings.
    """
    if not isinstance(t, str) or t in ("\\N", ""):
        return np.nan
    parts = t.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return np.nan


def prior_by_race(
    frame: pd.DataFrame,
    key: str,
    value: str,
    window: int | None = None,
    agg: str = "mean",
) -> np.ndarray:
    """Compute a leak-free prior-race feature.

    ``value`` is aggregated to one row per ``(key, raceId)`` in chronological
    order (via ``race_seq``), then shifted by exactly one race: a prior-``window``
    -race rolling ``agg``, or, if ``window`` is ``None``, a running cumulative sum.

    Race-level aggregation *before* the shift is what keeps this leak-free when
    ``key`` groups more than one row per race, e.g. a constructor fielding two
    drivers: shifting the raw driver-rows by one position shifts by one DRIVER,
    not one RACE, and lets a teammate's result from the very race being predicted
    leak into the feature. Collapsing to one row per race first closes that gap;
    ``key='driverId'`` (one row per race already) is unaffected.

    Args:
        frame: The modelling table, with ``key``, ``raceId``, ``race_seq`` and
            the ``value`` column.
        key: Grouping column (e.g. ``'driverId'`` or ``'constructorId'``).
        value: The column to aggregate.
        window: Rolling window in races, or ``None`` for a cumulative sum.
        agg: Aggregation applied within each race before the shift.

    Returns:
        A float array aligned to ``frame``'s rows.
    """
    race = (
        frame.groupby([key, "raceId"], sort=False)
        .agg(_v=(value, agg), _seq=("race_seq", "first"))
        .reset_index()
        .sort_values([key, "_seq"])
    )
    if window is None:
        race["_out"] = race.groupby(key)["_v"].transform(lambda x: x.shift(1).cumsum())
    else:
        race["_out"] = race.groupby(key)["_v"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    merged = frame[[key, "raceId"]].merge(
        race[[key, "raceId", "_out"]], on=[key, "raceId"], how="left"
    )
    return merged["_out"].values


def load_dataset() -> tuple[pd.DataFrame, list[str], list[int]]:
    """Load the Kaggle CSVs and build the leak-free modelling table.

    Reads the constructor, qualifying, race, result and status tables, assembles
    one row per driver per race, derives the grid, qualifying and pre-race form
    features, and merges the optional FastF1 enrichment tier when its cache file
    is present. Every feature is computed from information available strictly
    before the race it describes.

    Returns:
        A ``(df, fastf1_cols, fastf1_years)`` tuple: the modelling table, the
        names of any merged FastF1 feature columns (empty when the cache is
        absent), and the seasons those features cover.
    """
    constructors = pd.read_csv(os.path.join(DATA_DIR, "constructors.csv"))
    qualifying = pd.read_csv(os.path.join(DATA_DIR, "qualifying.csv"))
    races = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
    results = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    status = pd.read_csv(os.path.join(DATA_DIR, "status.csv"))

    ##### Build the race-entry table (one row per driver per race)

    df = results.merge(
        races[["raceId", "year", "round", "date", "name", "circuitId"]], on="raceId"
    )
    df = df.merge(status, on="statusId")
    df = df.merge(
        constructors[["constructorId", "name"]],
        on="constructorId",
        suffixes=("", "_constructor"),
    )

    # Normalise constructor identity by stripping the
    # engine suffix so Lotus-Climax / Lotus-Ford / Team Lotus all collapse to Lotus.
    df["team"] = (
        df["name_constructor"].str.split("-").str[0].replace({"Team Lotus": "Lotus"})
    )

    # The Indianapolis 500 counted for the championship 1950-1960 but was a separate
    # series with separate entrants; keeping it would corrupt every grid-based feature.
    indy_mask = df["name"].str.contains("Indianapolis", case=False, na=False) & (
        df["year"] <= 1960
    )
    print(f"[prep] dropping {indy_mask.sum()} Indianapolis 500 rows (1950-1960)")
    df = df[~indy_mask].copy()

    df = df.sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True)

    # Ground truth: official classification, regardless of whether the car was running
    # at the flag. A car classified P3 while retired still scores a podium in F1.
    df["podium"] = (df["positionOrder"] <= 3).astype(int)

    # Shared drives: until 1960 two drivers could share one car and both be classified
    # in the same position, so a handful of races have four or more podium rows. The
    # label is still correct (both men were on the podium), but it means "exactly
    # three positives per race" does not hold in the early era, which the permutation
    # test and the precision@3 denominator both account for explicitly.
    shared = df.groupby("raceId")["podium"].transform("sum") > 3
    print(
        f"[prep] {df.loc[shared, 'raceId'].nunique()} races with shared drives "
        f"(>3 podium rows), all in {df.loc[shared, 'year'].min()}-{df.loc[shared, 'year'].max()}"
    )

    # Field size, and pit-lane starts (grid == 0 is Ergast's pit-lane sentinel).
    df["n_starters"] = df.groupby("raceId")["driverId"].transform("size")
    df["pit_lane_start"] = (df["grid"] == 0).astype(int)
    df["grid_pos"] = df["grid"].where(df["grid"] > 0, df["n_starters"] + 1)
    df["grid_norm"] = df["grid_pos"] / df["n_starters"]

    # A few 1960s-80s races list a grid slot beyond the number of classified entries,
    # usually because non-championship or F2 cars shared the grid. Left as-is: the
    # grid slot is real information and the count is negligible.
    over_grid = int(((df["grid"] > df["n_starters"]) & (df["grid"] > 0)).sum())
    print(f"[prep] {over_grid} rows with a grid slot beyond the classified field size")

    df["finish_class"] = df["status"].map(finish_class)

    ##### Qualifying features (2003-2024)

    q = qualifying.copy()
    for col in ("q1", "q2", "q3"):
        q[col + "_s"] = q[col].map(parse_lap)

    q["reached_q2"] = q["q2_s"].notna().astype(int)
    q["reached_q3"] = q["q3_s"].notna().astype(int)
    q["quali_stage"] = np.where(
        q["reached_q3"] == 1, 3, np.where(q["reached_q2"] == 1, 2, 1)
    )

    # The driver's last, most representative time: Q3 if they got there, else Q2, else Q1.
    q["q_time"] = q["q3_s"].fillna(q["q2_s"]).fillna(q["q1_s"])
    q["q_best"] = q[["q1_s", "q2_s", "q3_s"]].min(axis=1)

    #
    session_best = np.select(
        [q["quali_stage"] == 3, q["quali_stage"] == 2],
        [
            q.groupby("raceId")["q3_s"].transform("min"),
            q.groupby("raceId")["q2_s"].transform("min"),
        ],
        default=q.groupby("raceId")["q1_s"].transform("min"),
    )
    q["gap_to_session_best_pct"] = (q["q_time"] - session_best) / session_best * 100

    pole = q.groupby("raceId")["q_best"].transform("min")
    # Scale-invariant: a 0.5 s deficit means something different at Monaco and at Spa.
    q["gap_to_pole_pct"] = (q["q_best"] - pole) / pole * 100

    # Wet or red-flagged sessions produce absurd deficits (max observed 45.6%).
    # Clip so the standardised linear model is not dominated by a handful of laps.
    for col in ("gap_to_pole_pct", "gap_to_session_best_pct"):
        q[col] = q[col].clip(lower=0, upper=20)

    q["quali_pos"] = q["position"]

    qf = q[
        [
            "raceId",
            "driverId",
            "q_best",
            "gap_to_pole_pct",
            "gap_to_session_best_pct",
            "quali_pos",
            "quali_stage",
            "reached_q2",
            "reached_q3",
        ]
    ]
    df = df.merge(qf, on=["raceId", "driverId"], how="left")

    # Places lost to a penalty between qualifying and the grid. 27.9% of 2003-2024 rows
    # have grid_pos != quali_pos, so this is a substantial, purely pre-race signal.
    df["grid_penalty"] = df["grid_pos"] - df["quali_pos"]

    # Grid compression: being P4 0.05 s off P3 is not the same as P4 0.8 s off P3.
    df = df.sort_values(["raceId", "grid_pos"])
    g = df.groupby("raceId")["gap_to_session_best_pct"]
    df["gap_to_car_ahead_pct"] = df["gap_to_session_best_pct"] - g.shift(1)
    df["gap_to_car_behind_pct"] = g.shift(-1) - df["gap_to_session_best_pct"]

    # Teammate comparison isolates the driver from the car.
    tm = df.groupby(["raceId", "constructorId"])["gap_to_session_best_pct"]
    df["teammate_gap_pct"] = df["gap_to_session_best_pct"] - (
        tm.transform("sum") - df["gap_to_session_best_pct"]
    ) / (tm.transform("count") - 1).replace(0, np.nan)
    df["teammate_quali_rank"] = tm.rank(method="min")

    ##### Pre-race context features (leak-free by construction)
    # Everything below is computed from races *strictly before* the current one.
    # Season-to-date shares rather than raw points, because the points system changed
    # 6+ times, so raw points are not comparable across eras.

    df = df.sort_values(["date", "raceId"]).reset_index(drop=True)
    df["race_seq"] = df.groupby("raceId", sort=False).ngroup()

    df["driver_podium_rate_5"] = prior_by_race(df, "driverId", "podium", window=5)
    df["team_podium_rate_5"] = prior_by_race(df, "constructorId", "podium", window=5)
    df["dnf"] = (df["finish_class"].isin(["mechanical", "incident"])).astype(int)
    df["team_dnf_rate_5"] = prior_by_race(df, "constructorId", "dnf", window=5)
    df["driver_starts"] = df.groupby("driverId").cumcount()

    # Season-to-date points share, shifted: standings as they stood before this race.
    # agg='sum' because constructor/season points are additive across drivers, not
    # averaged: a team's race score is both cars' points combined.
    df["pts_cum"] = prior_by_race(df, "constructorId", "points", agg="sum")
    df["season_pts_cum"] = prior_by_race(df, "year", "points", agg="sum")
    df["team_pts_share_ytd"] = df["pts_cum"] / df["season_pts_cum"].replace(0, np.nan)
    df["driver_pts_cum"] = prior_by_race(df, "driverId", "points", agg="sum")
    df["driver_pts_share_ytd"] = df["driver_pts_cum"] / df["season_pts_cum"].replace(
        0, np.nan
    )

    # Optional FastF1 tier
    if os.path.exists(FASTF1_CACHE):
        ff = pd.read_csv(FASTF1_CACHE)
        df = df.merge(ff, on=["raceId", "driverId"], how="left")
        FASTF1_COLS = [c for c in ff.columns if c not in ("raceId", "driverId")]
        # Coverage must be judged against the seasons actually fetched, not the whole
        # 2018-2024 window: a partial pull would otherwise look uniformly sparse.
        FASTF1_YEARS = sorted(
            int(y)
            for y in races.loc[races["raceId"].isin(ff["raceId"]), "year"].unique()
        )
        print(f"[prep] FastF1 features merged: {FASTF1_COLS}")
        print(f"[prep] FastF1 seasons present: {FASTF1_YEARS}")
    else:
        FASTF1_COLS = []
        FASTF1_YEARS = []
        print("[prep] fastf1_features.csv not found, FastF1 tier skipped")

    return df, FASTF1_COLS, FASTF1_YEARS


##### FastF1 enrichment tier: fetch (2018-2024)
#
# Everything produced here is knowable before the race starts:
# qualifying and practice sessions only; the race session is never opened.
#
# F1's API allows 500 calls per hour, which works out at roughly one season per
# hour. Progress is checkpointed after every round so a rate-limited run resumes
# where it stopped rather than starting the season again.


def _ff1_build_key_maps() -> tuple[dict, dict]:
    """Bridge FastF1's driver codes and round numbers back to Kaggle ids.

    FastF1 keys drivers by three-letter code; the Kaggle dataset keys them by
    driverId with a separate `code` column. Joining on (year, round) plus code is
    the only stable link between the two sources.
    """
    races_ = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
    drivers_ = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"))
    race_key = races_.set_index(["year", "round"])["raceId"].to_dict()
    drivers_ = drivers_[drivers_["code"].notna() & (drivers_["code"] != "\\N")]
    driver_key = drivers_.set_index("code")["driverId"].to_dict()
    return race_key, driver_key


def _safe_laps(session: object):
    """Return a session's laps, or None if the timing data is unavailable.

    Some sessions were cancelled or ran without timing (the 2020 Eifel GP FP2 was
    called off in fog). FastF1 raises DataNotLoadedError on attribute access
    rather than returning an empty frame, so this must be guarded or one dead
    session takes the whole season down with it.
    """
    try:
        laps = session.laps
    except Exception:  # noqa: BLE001
        return None
    if laps is None or len(laps) == 0:
        return None
    return laps


def _ff1_quali_features(session: object) -> dict:
    """Sector-level and speed-trap detail from qualifying."""
    laps = _safe_laps(session)
    if laps is None:
        return {}
    out = {}
    for drv in laps["Driver"].dropna().unique():
        d = laps[laps["Driver"] == drv]
        d = d[d["LapTime"].notna()]
        if len(d) == 0:
            continue
        best = d.loc[d["LapTime"].idxmin()]
        actual = best["LapTime"].total_seconds()
        # Theoretical best = the driver's own best sectors summed. The gap to
        # their actual best lap is time left on the table, a proxy for how
        # close to the limit they got when it counted.
        sectors = [
            d[c].min()
            for c in ("Sector1Time", "Sector2Time", "Sector3Time")
            if c in d.columns
        ]
        theoretical = (
            sum(s.total_seconds() for s in sectors)
            if len(sectors) == 3 and all(pd.notna(s) for s in sectors)
            else np.nan
        )
        out[drv] = {
            "ff1_quali_best_s": actual,
            "ff1_lap_vs_theoretical_pct": (
                (actual - theoretical) / theoretical * 100
                if theoretical and np.isfinite(theoretical)
                else np.nan
            ),
            "ff1_speed_trap": best["SpeedST"] if "SpeedST" in best.index else np.nan,
            "ff1_quali_laps_run": int(len(d)),
        }
    return out


def _ff1_practice_features(session: object) -> dict:
    """High-fuel race-simulation pace from FP2, genuinely absent from Kaggle.

    A stint of 5+ laps on one compound is treated as a race run; the median lap
    of the longest such stint is the best available proxy for race pace.
    """
    laps = _safe_laps(session)
    if laps is None or "Stint" not in laps.columns:
        return {}
    out = {}
    for drv in laps["Driver"].dropna().unique():
        d = laps[(laps["Driver"] == drv) & laps["LapTime"].notna()]
        if len(d) < 5:
            continue
        best_stint, best_len = None, 0
        for _, grp in d.groupby(["Stint", "Compound"], dropna=False)["LapTime"]:
            if len(grp) >= 5 and len(grp) > best_len:
                best_stint, best_len = grp, len(grp)
        if best_stint is None:
            continue
        out[drv] = {
            "ff1_fp2_longrun_median_s": best_stint.dt.total_seconds().median(),
            "ff1_fp2_longrun_laps": int(best_len),
        }
    return out


class _RateLimited(Exception):
    """Raised to abort the whole pull the moment the API rate limit is hit."""


def _is_rate_limit(exc: BaseException) -> bool:
    return "ratelimit" in type(exc).__name__.lower() or "rate limit" in str(exc).lower()


def _ff1_process_season(
    fastf1: object,
    year: int,
    race_key: dict,
    driver_key: dict,
    want_fp2: bool = True,
    partial_path: str | None = None,
) -> tuple[list[dict], int, int]:
    """Fetch one FastF1 season. Returns (rows, n_rounds, n_ok); raises on a 429.

    Checkpoints after every round: a season interrupted by the rate limit resumes
    at the next unfetched round rather than starting over. Since one season costs
    roughly one hour of quota, losing half of one to a retry is expensive.
    """
    rows, done_rounds = [], set()
    if partial_path and os.path.exists(partial_path):
        prior = pd.read_csv(partial_path)
        rows = prior.to_dict("records")
        done_rounds = set(prior["_round"].unique())
        print(f"  [resume] {year}: {len(done_rounds)} rounds already fetched")

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    n_rounds, n_ok = 0, len(done_rounds)
    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        race_id = race_key.get((year, rnd))
        if race_id is None:
            print(f"  [skip] {year} R{rnd}: no matching raceId in the Kaggle data")
            continue
        n_rounds += 1
        if rnd in done_rounds:
            continue

        feats = {}
        sessions = [("Q", _ff1_quali_features)]
        if want_fp2:
            sessions.append(("FP2", _ff1_practice_features))
        for kind, fn in sessions:
            try:
                session = fastf1.get_session(year, rnd, kind)
                session.load(telemetry=False, weather=True, messages=False)
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit(exc):
                    if partial_path and rows:
                        pd.DataFrame(rows).to_csv(partial_path, index=False)
                        print(f"  [checkpoint] {year}: saved {n_ok} rounds")
                    raise _RateLimited(str(exc)) from exc
                print(f"  [warn] {year} R{rnd} {kind}: {type(exc).__name__}: {exc}")
                continue
            try:
                extracted = fn(session)
            except Exception as exc:  # noqa: BLE001
                # A malformed session must cost us that session, not the season.
                print(
                    f"  [warn] {year} R{rnd} {kind} features: "
                    f"{type(exc).__name__}: {exc}"
                )
                extracted = {}
            for drv, vals in extracted.items():
                feats.setdefault(drv, {}).update(vals)
            if kind == "Q":
                # A wet Saturday decouples the grid from underlying pace, so the
                # flag is worth carrying even though the pull is per-session.
                try:
                    wet = int(bool(session.weather_data["Rainfall"].any()))
                except Exception:  # noqa: BLE001
                    wet = np.nan
                for drv in feats:
                    feats[drv]["ff1_quali_wet"] = wet

        for drv, vals in feats.items():
            driver_id = driver_key.get(drv)
            if driver_id is None:
                print(f"  [warn] {year} R{rnd}: unknown driver code {drv!r}")
                continue
            rows.append(
                {"raceId": race_id, "driverId": driver_id, "_round": rnd, **vals}
            )
        if feats:
            n_ok += 1
            if partial_path:
                pd.DataFrame(rows).to_csv(partial_path, index=False)
        print(f"  [ok] {year} R{rnd} {event['EventName']}: {len(feats)} drivers")
    return rows, n_rounds, n_ok


def build_fastf1_features(
    seasons: list[int],
    cache_dir: str,
    out_path: str,
    force: bool = False,
    want_fp2: bool = True,
    auto_resume: bool = False,
    wait_minutes: int = 61,
    max_waits: int = 12,
) -> bool:
    """Pull FastF1 features season by season, caching each one to disk.

    Args:
        seasons: Seasons to fetch.
        cache_dir: Directory for FastF1's HTTP cache and the per-season files.
        out_path: Path to write the assembled feature CSV to.
        force: Re-fetch seasons even if a cached file exists.
        want_fp2: Also pull FP2 long-run pace (roughly doubles the API calls).
        auto_resume: Wait out the API rate limit and continue automatically.
        wait_minutes: Minutes to wait after a rate limit.
        max_waits: Give up after this many waits.

    Returns:
        ``True`` if a usable feature file now exists. Never raises: a failed pull
        downgrades to the Kaggle-only analysis rather than killing the run.
    """
    try:
        import fastf1
    except ImportError:
        print('[ff1] fastf1 not installed, run: pip install "fastf1>=3.3"')
        return False

    os.makedirs(cache_dir, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(cache_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[ff1] cache disabled ({exc}); continuing uncached")

    race_key, driver_key = _ff1_build_key_maps()
    waits_used = 0
    pending = list(seasons)
    while pending:
        year = pending[0]
        part = os.path.join(cache_dir, f"features_{year}.csv")
        partial = os.path.join(cache_dir, f"_partial_{year}.csv")
        if os.path.exists(part) and not force:
            print(f"=== {year}: using cached {os.path.basename(part)}")
            pending.pop(0)
            continue
        print(f"=== {year}")
        try:
            rows, n_rounds, n_ok = _ff1_process_season(
                fastf1,
                year,
                race_key,
                driver_key,
                want_fp2=want_fp2,
                partial_path=partial,
            )
        except _RateLimited as exc:
            if not auto_resume:
                print(f"\n[ff1] API rate limit reached ({exc}).")
                print(
                    "[ff1] F1's API allows 500 calls/hour. Progress is checkpointed\n"
                    "      per round, so re-running resumes where this stopped.\n"
                    "      Add --auto-resume to wait and continue automatically."
                )
                break
            if waits_used >= max_waits:
                print(f"\n[ff1] rate limited again after {waits_used} waits, stopping.")
                break
            waits_used += 1
            print(
                f"\n[ff1] rate limit reached. Waiting {wait_minutes} min "
                f"(wait {waits_used}/{max_waits}), then resuming {year}. "
                f"Safe to leave running; Ctrl-C to stop and keep progress."
            )
            try:
                for remaining in range(wait_minutes, 0, -1):
                    print(f"  [wait] {remaining} min remaining...", flush=True)
                    time.sleep(60)
            except KeyboardInterrupt:
                print("\n[ff1] interrupted, checkpointed progress is kept.")
                break
            continue  # retry the same season, resuming mid-way
        except Exception as exc:  # noqa: BLE001
            print(f"[ff1] season {year} failed: {type(exc).__name__}: {exc}")
            pending.pop(0)
            continue

        # A partially fetched season must NOT be cached as if it were complete,
        # or the next run will skip it and silently keep the gap forever.
        if rows and n_rounds and n_ok / n_rounds >= 0.9:
            frame = pd.DataFrame(rows).drop(columns=["_round"], errors="ignore")
            frame.to_csv(part, index=False)
            if os.path.exists(partial):
                os.remove(partial)
            print(f"  [cached] {year}: {n_ok}/{n_rounds} rounds")
        elif rows:
            print(
                f"  [partial] {year}: only {n_ok}/{n_rounds} rounds retrieved, "
                f"not cached, will be retried on the next run"
            )
        pending.pop(0)

    # Rebuild from EVERY cached season, not just the ones requested on this run.
    # Otherwise `--seasons 2020` would overwrite a six-season file with one season.
    cached = sorted(
        f
        for f in os.listdir(cache_dir)
        if f.startswith("features_") and f.endswith(".csv")
    )
    frames = [pd.read_csv(os.path.join(cache_dir, f)) for f in cached]
    if frames:
        print(
            f"[ff1] assembling output from {len(cached)} cached season(s): "
            + ", ".join(f[9:13] for f in cached)
        )

    if not frames:
        print("[ff1] no data retrieved, falling back to the Kaggle-only analysis")
        return False

    out = pd.concat(frames, ignore_index=True)

    # Normalise pace within each race so the features are comparable across
    # circuits, exactly as gap_to_session_best_pct is for the Kaggle tier.
    for src, dst in (
        ("ff1_fp2_longrun_median_s", "ff1_fp2_longrun_gap_pct"),
        ("ff1_quali_best_s", "ff1_quali_gap_pct"),
    ):
        if src in out.columns:
            base = out.groupby("raceId")[src].transform("min")
            out[dst] = (out[src] - base) / base * 100
    if "ff1_speed_trap" in out.columns:
        top = out.groupby("raceId")["ff1_speed_trap"].transform("max")
        out["ff1_speed_trap_pct"] = out["ff1_speed_trap"] / top * 100

    keep = [
        "raceId",
        "driverId",
        "ff1_quali_gap_pct",
        "ff1_lap_vs_theoretical_pct",
        "ff1_speed_trap_pct",
        "ff1_quali_laps_run",
        "ff1_fp2_longrun_gap_pct",
        "ff1_fp2_longrun_laps",
        "ff1_quali_wet",
    ]
    out = out[[c for c in keep if c in out.columns]]
    out = out.drop_duplicates(subset=["raceId", "driverId"])
    out.to_csv(out_path, index=False)
    print(f"\n[ff1] wrote {out_path}: {len(out)} rows, {out['raceId'].nunique()} races")
    print("[ff1] non-null coverage by column (%):")
    print((out.notna().mean() * 100).round(1).to_string())
    return True


##### FastF1 validity audit
# The load log is not evidence of validity,
#  These checks test the data itself.


def audit_fastf1_features(
    df: pd.DataFrame, races: pd.DataFrame, cache_path: str | None = None
) -> int:
    """Independent quality audit of the fetched FastF1 data.

    Checks completeness, per-column coverage, structural integrity, and that the
    FastF1 pace joins soundly onto the Kaggle grid order.

    Args:
        df: The modelling table, for the join-integrity check.
        races: The races table, for mapping ``raceId`` to season and round.
        cache_path: FastF1 feature CSV to audit; defaults to ``FASTF1_CACHE``.

    Returns:
        The number of problems found (0 = clean), so it can drive an exit code.
    """
    cache_path = cache_path or FASTF1_CACHE
    if not os.path.exists(cache_path):
        print(f"[audit] {cache_path} not found, nothing to audit")
        return 1

    ff = pd.read_csv(cache_path)
    ff = ff.merge(races[["raceId", "year", "round"]], on="raceId", how="left")
    problems = []

    print(f"[audit] file: {os.path.basename(cache_path)}")
    print(
        f"[audit] {len(ff)} rows, {ff['raceId'].nunique()} races, seasons "
        f"{sorted(int(y) for y in ff['year'].dropna().unique())}"
    )

    ##### 1. Completeness: rounds fetched versus rounds that exist
    print("\n[audit] rounds fetched per season")
    expected = races.groupby("year")["raceId"].nunique()
    present = {int(y) for y in ff["year"].dropna().unique()}
    for year in sorted(set(present) | set(range(min(present), max(present) + 1))):
        want = int(expected.get(year, 0))
        if year not in present:
            print(f"  {year}: 0/{want} rounds  <-- SEASON MISSING ENTIRELY")
            problems.append(f"{year} not fetched at all")
            continue
        grp = ff[ff["year"] == year]
        got = grp["raceId"].nunique()
        drivers = grp.groupby("raceId").size()
        flag = "" if got >= want else f"  <-- {want - got} MISSING"
        print(
            f"  {year}: {got}/{want} rounds, {drivers.mean():.1f} "
            f"drivers/race (min {drivers.min()}){flag}"
        )
        if got < want:
            problems.append(f"{year} missing {want - got} rounds")
        if drivers.min() < 10:
            problems.append(f"{year} has a race with only {drivers.min()} drivers")

    ##### 2. Coverage per column
    print("\n[audit] non-null coverage per column (%)")
    for col in [
        c for c in ff.columns if c not in ("raceId", "driverId", "year", "round")
    ]:
        cov = ff[col].notna().mean() * 100
        mark = "" if cov >= 30 else "  <-- below the 30% ablation threshold"
        print(f"  {col:32s} {cov:5.1f}{mark}")

    ##### 3. Structural integrity
    print("\n[audit] structural checks")
    dupes = int(ff.duplicated(subset=["raceId", "driverId"]).sum())
    print(f"  duplicate (raceId, driverId) rows: {dupes}")
    if dupes:
        problems.append(f"{dupes} duplicate driver-race rows")

    if "ff1_quali_gap_pct" in ff.columns:
        # Exactly one driver per race set the reference lap, so the minimum gap
        # must be ~0. A non-zero minimum means the normalisation is broken.
        bad = int((ff.groupby("raceId")["ff1_quali_gap_pct"].min() > 0.01).sum())
        neg = int((ff["ff1_quali_gap_pct"] < -0.01).sum())
        print(f"  races whose fastest driver is not at gap 0: {bad}")
        print(f"  negative qualifying gaps (impossible): {neg}")
        if bad:
            problems.append(f"{bad} races with a broken pole reference")
        if neg:
            problems.append(f"{neg} negative gaps")

    ##### 4. Join integrity:
    print("\n[audit] join integrity vs Kaggle grid position")
    if "ff1_quali_gap_pct" in ff.columns:
        merged = ff.merge(
            df[["raceId", "driverId", "grid_pos", "podium"]],
            on=["raceId", "driverId"],
            how="inner",
        )
        share = len(merged) / len(ff) if len(ff) else 0
        print(
            f"  rows matching a Kaggle race entry: {len(merged)}/{len(ff)} "
            f"({share * 100:.1f}%)"
        )
        if share < 0.90:
            problems.append("under 90% of FastF1 rows match a Kaggle entry")

        rhos = []
        for _, grp in merged.groupby("raceId"):
            g = grp[["ff1_quali_gap_pct", "grid_pos"]].dropna()
            if len(g) >= 8:
                rhos.append(
                    g["ff1_quali_gap_pct"].corr(g["grid_pos"], method="spearman")
                )
        if rhos:
            median_rho = float(np.median(rhos))
            print(
                f"  median within-race Spearman(gap, grid) = {median_rho:+.3f} "
                f"over {len(rhos)} races"
            )
            if median_rho >= 0.75:
                print("  => join is sound: FastF1 pace tracks the Kaggle grid closely")
            elif median_rho >= 0.40:
                print(
                    "  => join looks plausible but weaker than expected; check for "
                    "grid penalties and wet sessions"
                )
            else:
                print("  => JOIN IS SUSPECT: pace does not track grid order")
                problems.append(f"weak gap-vs-grid correlation ({median_rho:+.3f})")

        if "podium" in merged.columns and len(merged) > 20:
            top = merged[merged["ff1_quali_gap_pct"] <= 0.3]
            if len(top) > 20:
                print(
                    f"  podium rate within 0.3% of the best lap: "
                    f"{top['podium'].mean() * 100:.1f}% "
                    f"(overall base rate {merged['podium'].mean() * 100:.1f}%)"
                )

    print()
    if problems:
        print(f"[audit] {len(problems)} issue(s) found:")
        for p in problems:
            print(f"  - {p}")
        print("[audit] re-run the fetch to fill gaps; complete seasons are skipped.")
    else:
        print("[audit] no issues found, the FastF1 tier looks sound.")
    return len(problems)
