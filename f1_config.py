"""Shared constants, era definitions, plot styling, and feature blocks."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "kaggle_f1_dataset")
OUT_DIR = PROJECT_DIR
FASTF1_CACHE = os.path.join(OUT_DIR, "fastf1_features.csv")

RNG_SEED = 42
QUALI_START = 2003  # first season with complete qualifying coverage (see audit)
FASTF1_START = 2018  # first season with FastF1 timing data
BURN_IN_QUALI = 5  # training seasons before the first forward-chained test
BURN_IN_FULL = 10
BURN_IN_FASTF1 = 3

# Multiple seeds so the FastF1 ablation reports a spread, not a single lucky fit.
ABLATION_SEEDS = [42, 7, 13, 21, 99, 123, 2024]

# Eight regulatory eras, each (start_year, end_year, label, band_color). Boundaries
# follow established historical divisions so seasons with similar rules group together.
ERAS = [
    (1958, 1969, "The Golden Age", "#c6e2e9"),
    (1970, 1982, "Aero & Ground Effect", "#f5e6cc"),
    (1983, 1993, "Turbo & Electronics", "#e6d5f0"),
    (1994, 2004, "The Schumacher Era", "#f9d5d5"),
    (2005, 2008, "Shifting Sands", "#d5f5e3"),
    (2009, 2013, "Red Bull Era", "#d5e8f5"),
    (2014, 2021, "Mercedes Hybrid Era", "#fce4d6"),
    (2022, 2024, "Ground Effect Era", "#e8e0d5"),
]

# Chart palette, shared by every figure so the three reports read as one set.
SURFACE = "#fcfcfb"  # chart surface
INK = "#0b0b0b"  # primary ink (titles)
INK_SECONDARY = "#52514e"  # labels, annotations
INK_MUTED = "#898781"  # ticks, reference lines
GRID = "#e1e0d9"  # hairline gridlines
AXIS = "#c3c2b7"  # baseline / remaining spines
MARK_BLUE = "#2a78d6"  # primary mark hue
TREND_ORANGE = "#eb6834"  # trend-line hue
EXTREME_RED = "#d03b3b"  # extreme / failure
GOOD_GREEN = "#006300"  # competitive health

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 11,
        "axes.edgecolor": AXIS,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    }
)


def style_axes(ax: Axes, grid_axis: str = "y") -> None:
    """Apply the shared recessive chrome to an axis.

    Drops the top and right spines and draws a solid hairline grid so the data
    marks carry the figure rather than the frame.

    Args:
        ax: The axis to style.
        grid_axis: Which axis to grid, passed straight to ``Axes.grid`` ("x",
            "y" or "both").
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)


# Feature blocks, ordered from least to most information. Each experiment selects
# one of the named sets below; keeping the blocks here keeps the sets in sync.
BLOCK_G = ["grid_pos", "grid_norm", "pit_lane_start", "n_starters"]
BLOCK_Q = BLOCK_G + [
    "gap_to_session_best_pct",
    "gap_to_pole_pct",
    "gap_to_car_ahead_pct",
    "gap_to_car_behind_pct",
    "teammate_gap_pct",
    "teammate_quali_rank",
    "quali_stage",
    "reached_q2",
    "reached_q3",
    "quali_pos",
    "grid_penalty",
]
BLOCK_C = [
    "driver_podium_rate_5",
    "team_podium_rate_5",
    "team_dnf_rate_5",
    "driver_starts",
    "team_pts_share_ytd",
    "driver_pts_share_ytd",
]

FEATURE_SETS = {
    "Grid only": BLOCK_G,
    "Qualifying only": BLOCK_Q,
    "Qualifying + form": BLOCK_Q + BLOCK_C,
}
