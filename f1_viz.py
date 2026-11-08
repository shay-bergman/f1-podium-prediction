"""The three report figures (H1 model showdown, H2 decade predictability, H3
error-cause shift).
"""

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from f1_config import (
    EXTREME_RED,
    GOOD_GREEN,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MARK_BLUE,
    OUT_DIR,
    SURFACE,
    style_axes,
)

H1_SHOWDOWN_ROWS = [
    ("B0 majority class", "Baseline 0: Random Guessing", "baseline"),
    ("B2 grid base rate", "Baseline 2: Learned Grid Odds", "baseline"),
    ("B1 grid <= 3 rule", "Baseline 1: Top 3 on Grid (Heuristic)", "baseline"),
    ("Gradient boosting | Grid only", "Model (Grid Only)", "model"),
    ("Gradient boosting | Qualifying only", "Model (+ Qualifying Pace)", "model"),
    (
        "Gradient boosting | Qualifying + form",
        "Full Model (+ Qualifying & Form)",
        "model",
    ),
]


def viz_h1_model_showdown(summary: pd.DataFrame) -> None:
    """Figure 1 (H1): precision@3 with 95% CIs, horizontal bars. Every value
    comes from `summary` (res['summary'] from run_h1), nothing hardcoded."""
    n = len(H1_SHOWDOWN_ROWS)
    rows = []
    for (key, label, kind), y in zip(H1_SHOWDOWN_ROWS, range(n - 1, -1, -1)):
        r = summary.loc[key]
        lo, hi = r["Precision@3 CI"]
        rows.append(
            {
                "key": key,
                "label": label,
                "kind": kind,
                "y": y,
                "value": r["Precision@3"] * 100,
                "lo": np.nan if np.isnan(lo) else lo * 100,
                "hi": np.nan if np.isnan(hi) else hi * 100,
            }
        )

    print("\n[viz_h1] Model showdown values used (verify against the results table):")
    for r in rows:
        ci = "-" if np.isnan(r["lo"]) else f"[{r['lo']:.1f}%, {r['hi']:.1f}%]"
        print(f"  {r['label']:38s} {r['value']:5.1f}%  CI={ci}")

    b1 = next(r for r in rows if r["key"] == "B1 grid <= 3 rule")

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for r in rows:
        color = INK_MUTED if r["kind"] == "baseline" else MARK_BLUE
        ax.barh(r["y"], r["value"], color=color, height=0.6, zorder=3, edgecolor="none")
        if np.isnan(r["lo"]):
            label_txt = f"{r['value']:.1f}%"
        else:
            ax.errorbar(
                r["value"],
                r["y"],
                xerr=[[r["value"] - r["lo"]], [r["hi"] - r["value"]]],
                fmt="none",
                ecolor=INK,
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                zorder=4,
            )
            label_txt = f"{r['value']:.1f}%  [{r['lo']:.1f}–{r['hi']:.1f}%]"
        ax.text(
            r["value"] + (r["hi"] - r["value"] if not np.isnan(r["hi"]) else 0) + 1.5,
            r["y"],
            label_txt,
            va="center",
            ha="left",
            fontsize=9.5,
            color=INK,
        )

    ax.axvline(
        b1["value"], color=INK_SECONDARY, linewidth=1.4, linestyle="--", zorder=2
    )
    ax.text(
        b1["value"],
        n - 0.35,
        f"Baseline 1: Top 3 on Grid Heuristic ({b1['value']:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK_SECONDARY,
        fontstyle="italic",
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
        zorder=5,
    )

    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10.5)
    ax.set_ylim(-0.6, n + 0.45)
    ax.set_xlim(0, 92)
    ax.set_xlabel("Precision@3 (top-3 podium accuracy, %)")
    style_axes(ax, "x")
    ax.set_title(
        "Top-3 Podium Accuracy: Models vs. Common-Sense Baseline",
        fontweight="bold",
        pad=18,
        color=INK,
        fontsize=16,
    )

    fig.subplots_adjust(left=0.30, right=0.97, top=0.88, bottom=0.12)
    fig.savefig(
        os.path.join(OUT_DIR, "viz_results_h1_models.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print("[OK] viz_results_h1_models.png saved")


H2_DECADES = [
    (1960, 1969, "1960s"),
    (1970, 1979, "1970s"),
    (1980, 1989, "1980s"),
    (1990, 1999, "1990s"),
    (2000, 2009, "2000s"),
    (2010, 2019, "2010s"),
    (2020, 2024, "2020s"),
]


def viz_h2_decade_predictability(per_season: pd.DataFrame) -> None:
    """Figure 2 (H2): decade-level raw accuracy vs. lift-over-chance."""
    rows = []
    for lo, hi, label in H2_DECADES:
        d = per_season[(per_season["year"] >= lo) & (per_season["year"] <= hi)]
        if len(d):
            rows.append(
                {
                    "decade": label,
                    "accuracy": d["precision_at_3"].mean() * 100,
                    "lift": d["lift_over_chance"].mean() * 100,
                }
            )

    print(
        "\n[viz_h2] Decade predictability values used (verify against the results table):"
    )
    for r in rows:
        print(f"  {r['decade']}: accuracy={r['accuracy']:.1f}%  lift={r['lift']:.1f}pp")

    xs = list(range(len(rows)))
    acc = [r["accuracy"] for r in rows]
    lift = [r["lift"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(
        xs,
        acc,
        color=MARK_BLUE,
        linewidth=2.5,
        marker="o",
        markersize=7.5,
        zorder=4,
        label="Raw Top-3 Accuracy",
    )
    ax.plot(
        xs,
        lift,
        color=GOOD_GREEN,
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=6.5,
        zorder=4,
        label="Lift Over Chance",
    )

    ax.text(
        xs[-1] + 0.12,
        acc[-1],
        f"{acc[-1]:.1f}%",
        color=MARK_BLUE,
        fontsize=10,
        fontweight="bold",
        va="center",
        ha="left",
    )
    ax.text(
        xs[-1] + 0.12,
        lift[-1],
        f"{lift[-1]:.1f}pp",
        color=GOOD_GREEN,
        fontsize=10,
        fontweight="bold",
        va="center",
        ha="left",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([r["decade"] for r in rows])
    ax.set_xlim(-0.3, len(xs) - 1 + 1.0)
    ax.set_xlabel("Decade / Era", fontsize=12, labelpad=10)
    ax.set_ylim(0, 82)
    ax.set_ylabel("Top-3 Accuracy / Lift (%)")
    style_axes(ax, "y")
    ax.set_title(
        "Formula 1 Predictability by Decade (1960s–2020s)",
        fontweight="bold",
        pad=38,
        color=INK,
        fontsize=16,
    )
    ax.text(
        0.5,
        1.16,
        "Accuracy gains persist even when controlling for historical "
        "field-size drift.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color=INK_SECONDARY,
        fontstyle="italic",
        clip_on=False,
    )
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT_DIR, "viz_results_h2_predictability.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("[OK] viz_results_h2_predictability.png saved")


def viz_h3_error_shift(decade_anatomy: pd.DataFrame) -> None:
    """Figure 3 (H3): 100%-stacked mechanical-vs-everything-else by decade.
    `decade_anatomy` comes from run_h3_by_decade() in f1_experiments.py:
    two categories, collapsed from the four raw finish-status categories,
    so they sum cleanly to 100%."""
    labels = list(decade_anatomy.index)
    xs = np.arange(len(labels))
    mech = decade_anatomy["mechanical"].values
    other = decade_anatomy["other_combined"].values

    print(
        "\n[viz_h3] Decade error-anatomy values used (verify against the results table):"
    )
    for lbl, m, o in zip(labels, mech, other):
        print(f"  {lbl}: mechanical={m:.1f}%  other={o:.1f}%")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        xs,
        mech,
        color=EXTREME_RED,
        width=0.62,
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1,
        label="Mechanical DNF / Breakdown",
    )
    ax.bar(
        xs,
        other,
        bottom=mech,
        color=MARK_BLUE,
        width=0.62,
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1,
        label="Beaten on Track / Other",
    )

    for x, m in zip(xs, mech):
        if m >= 6:
            ax.text(
                x,
                m / 2,
                f"{m:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
                color=SURFACE,
                fontweight="bold",
            )
    for x, m, o in zip(xs, mech, other):
        if o >= 6:
            ax.text(
                x,
                m + o / 2,
                f"{o:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
                color=SURFACE,
                fontweight="bold",
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of False-Positive Podium Predictions (%)")
    style_axes(ax, "y")
    ax.set_title(
        "Why Did Predictability Rise? Shift in Prediction Errors Over Time",
        fontweight="bold",
        pad=12,
        color=INK,
        fontsize=15,
    )
    ax.legend(
        fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT_DIR, "viz_results_h3_error_anatomy.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("[OK] viz_results_h3_error_anatomy.png saved")
