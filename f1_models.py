"""Models, metrics, baselines, and the forward-chained evaluation protocol."""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1_config import RNG_SEED

##### Models


def make_logreg() -> Pipeline:
    """Build the logistic-regression pipeline: median impute, scale, then fit.

    Returns:
        An unfitted scikit-learn pipeline.

    Note:
        ``C=1.0`` with L2 is deliberate. The feature count is small (<=18) and
        highly collinear (grid_pos, quali_pos and gap_to_pole_pct all encode
        qualifying order), so some shrinkage is needed but heavy regularisation
        loses the pace signal. No ``class_weight``: at 14% prevalence re-weighting
        buys little ranking power and destroys calibration, and every headline
        metric here is rank-based.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=RNG_SEED)),
        ]
    )


def make_hgb(seed: int = RNG_SEED) -> HistGradientBoostingClassifier:
    """Build the gradient-boosting classifier.

    Deliberately shallow: ~9k rows with a strong driver-identity signal would
    otherwise memorise individual drivers instead of learning grid structure.

    Args:
        seed: Random state, so the FastF1 ablation can refit under several seeds.

    Returns:
        An unfitted ``HistGradientBoostingClassifier``.
    """
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=seed,
    )


MODELS: dict[str, Callable[[], object]] = {
    "Logistic regression": make_logreg,
    "Gradient boosting": make_hgb,
}

##### Evaluation helpers


def top3_mask(frame: pd.DataFrame, prob_col: str = "prob") -> np.ndarray:
    """Boolean mask of the three highest-probability entries in each race.

    Args:
        frame: Scored entries, one row per driver per race, with a ``raceId``
            column and the probability column.
        prob_col: Name of the probability column to rank by.

    Returns:
        A boolean array aligned to ``frame``'s rows. ``method='first'`` breaks
        ties deterministically so results are reproducible.
    """
    rank = frame.groupby("raceId")[prob_col].rank(method="first", ascending=False)
    return (rank <= 3).values


def per_race_hits(
    frame: pd.DataFrame, mask: np.ndarray, label_col: str = "podium"
) -> tuple[np.ndarray, np.ndarray]:
    """Count correct picks per race.

    Args:
        frame: Scored entries with a ``raceId`` and the label column.
        mask: Boolean mask of the picked entries (e.g. from :func:`top3_mask`).
        label_col: Name of the 0/1 ground-truth column.

    Returns:
        A ``(hits, k)`` pair of arrays, one element per race: how many of the
        picks were truly on the podium, and how many picks were made.
    """
    picked = frame.loc[mask, ["raceId", label_col]]
    hits = picked.groupby("raceId")[label_col].sum()
    k = picked.groupby("raceId").size()
    return hits.reindex(k.index).fillna(0).values, k.values


def precision_at_3(
    frame: pd.DataFrame, prob_col: str = "prob", label_col: str = "podium"
) -> tuple[float, float]:
    """Precision@3 and the exact-podium rate.

    Per race, take the three highest-probability drivers. Precision@3 is the
    fraction of those picks that were on the real podium. It is field-size
    robust, unlike PR-AUC, so it is comparable across eras.

    Args:
        frame: Scored entries with ``raceId``, the probability and the label.
        prob_col: Name of the probability column.
        label_col: Name of the 0/1 ground-truth column.

    Returns:
        ``(precision_at_3, exact_podium_rate)``, where the second value is the
        share of races whose three picks were all correct.
    """
    mask = top3_mask(frame, prob_col)
    hits, k = per_race_hits(frame, mask, label_col)
    return hits.sum() / k.sum(), float((hits == k).mean())


def evaluate(frame: pd.DataFrame) -> dict[str, float]:
    """Score a frame of predictions on every headline metric.

    Args:
        frame: Scored entries with ``podium`` and ``prob`` columns.

    Returns:
        A dict of PR-AUC, ROC-AUC, Brier score, precision@3, exact-podium rate
        and the row count.
    """
    y, p = frame["podium"].values, frame["prob"].values
    p3, exact = precision_at_3(frame)
    return {
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "Brier": brier_score_loss(y, p),
        "Precision@3": p3,
        "Exact podium": exact,
        "n": len(frame),
    }


def forward_chain(
    data: pd.DataFrame,
    features: list[str],
    model_factory: Callable[[], object],
    burn_in: int,
) -> pd.DataFrame:
    """Train on past seasons, predict the next, one season at a time.

    Random splits would leak: laps and drivers inside one race are correlated,
    and a random split also lets the model see the future of an era it is being
    asked to predict.

    Args:
        data: The modelling table with a ``year`` column and the features.
        features: Feature column names to train on.
        model_factory: Zero-argument callable returning a fresh estimator.
        burn_in: Number of earliest seasons used only for training, never tested.

    Returns:
        The concatenation of every test season with a ``prob`` column added.
    """
    seasons = sorted(data["year"].unique())
    out = []
    for i, yr in enumerate(seasons):
        if i < burn_in:
            continue
        train = data[data["year"] < yr]
        test = data[data["year"] == yr]
        if train["podium"].nunique() < 2 or len(test) == 0:
            continue
        m = model_factory()
        m.fit(train[features], train["podium"])
        t = test.copy()
        t["prob"] = m.predict_proba(test[features])[:, 1]
        out.append(t)
    return pd.concat(out, ignore_index=True)


def baseline_grid_rule(frame: pd.DataFrame) -> tuple[float, float, float]:
    """B1: the three fastest qualifiers finish on the podium.

    The domain heuristic any F1 fan would use, and the bar the model actually
    has to clear.

    Args:
        frame: Entries with ``grid_pos`` and ``podium`` columns.

    Returns:
        ``(precision, recall, f1)`` of the rule against the true podium.
    """
    pred = (frame["grid_pos"] <= 3).astype(int)
    y = frame["podium"]
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def baseline_grid_rate(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """B2: empirical P(podium | grid slot), fitted on the training seasons only.

    The honest test of whether anything beyond grid *rank* carries information.

    Args:
        train: Past seasons used to estimate the per-slot podium rate.
        test: Season to score; unseen grid slots fall back to the base rate.

    Returns:
        A copy of ``test`` with a ``prob`` column.
    """
    rate = train.groupby("grid_pos")["podium"].mean()
    prior = train["podium"].mean()
    t = test.copy()
    t["prob"] = t["grid_pos"].map(rate).fillna(prior)
    return t


def bootstrap_precision_ci(
    frame: pd.DataFrame,
    prob_col: str = "prob",
    mask: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    """95% confidence interval for precision@3, bootstrapped over races.

    Races are the independent unit here, not rows: the 20 rows of one race are
    correlated with each other, so resampling rows would understate the true
    uncertainty.

    Args:
        frame: Scored entries with ``raceId``, the probability and ``podium``.
        prob_col: Name of the probability column.
        mask: Optional pre-computed pick mask; defaults to :func:`top3_mask`.
        n_boot: Number of bootstrap resamples.
        seed: Seed for the bootstrap RNG, so the interval is reproducible.

    Returns:
        The ``(2.5th, 97.5th)`` percentile bounds, or ``(nan, nan)`` when there
        are no races to resample.
    """
    if mask is None:
        mask = top3_mask(frame, prob_col)
    hits, k = per_race_hits(frame, mask)
    n_races = len(hits)
    if n_races == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_races, size=(n_boot, n_races))
    boot = hits[idx].sum(axis=1) / k[idx].sum(axis=1)
    return tuple(np.percentile(boot, [2.5, 97.5]))
