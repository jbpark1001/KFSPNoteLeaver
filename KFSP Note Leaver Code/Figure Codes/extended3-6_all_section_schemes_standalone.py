
# -*- coding: utf-8 -*-
"""
Generate revised age and gender panels for every sectioning scheme.

Age figures:
    x-axis = age group
    lines = section
    y-axis = log-odds

Gender figures:
    x-axis = Female / Male
    grouped bars = section
    y-axis = odds ratio
    capped error bars = 95% confidence intervals
    dotted guides connect the same section across gender

The script reads the model-result CSV files saved by the original analysis.
It does not depend on variables from a notebook or rerun any regression.

Default schemes:
    Section-level:
        third, quart, third5, shuf_third

    Sentence-position:
        sent_real_third, sent_real_quart,
        sent_shuf_third, sent_shuf_quart

Default output:
    Figure Codes/multi_scheme_revised_panels/

Required packages:
    pandas, numpy, matplotlib

Python compatibility:
    Python 3.6 or newer

Run:
    python figure7_all_section_schemes_standalone.py

Examples:
    python figure7_all_section_schemes_standalone.py --dpi 600

    python figure7_all_section_schemes_standalone.py ^
        --schemes third quart third5 sent_real_third sent_real_quart
"""

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_SECTION_RESULTS_DIR = Path(
    r"C:\Users\Jae Bin Park\multi_scheme_outputs"
)
DEFAULT_SENTENCE_RESULTS_DIR = Path(
    r"C:\Users\Jae Bin Park\sentence_df_outputs"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "multi_scheme_revised_panels"
)


# ---------------------------------------------------------------------------
# Categories shown in the publication panels
# ---------------------------------------------------------------------------

THEMES = [
    "Sorry and Shame",
    "Love and Gratitude",
    "Burden",
    "Despair",
    "Post-mortem Affairs",
]

AGE_SENTIMENTS = [
    "Defeat",
    "Exhaustion",
    "Neutral",
    "Sadness",
    "Happiness",
    "Disappointment",
]

GENDER_SENTIMENTS = [
    "Defeat",
    "Hatred",
    "Neutral",
    "Happiness",
    "Anxious",
    "Disappointment",
]

AGE_GROUPS = [1, 2, 3, 4, 5]
AGE_LABELS = ["≤18", "19–34", "35–49", "50–64", "65+"]


# ---------------------------------------------------------------------------
# Scheme definitions
# ---------------------------------------------------------------------------

SCHEME_SPECS: Dict[str, Dict[str, Any]] = {
    "third": {
        "family": "section_level",
        "title": "Third sections",
        "n_sections": 3,
        "section_display": ["Introduction", "Body", "Conclusion"],
    },
    "quart": {
        "family": "section_level",
        "title": "Quartile sections",
        "n_sections": 4,
        "section_display": ["Q1", "Q2", "Q3", "Q4"],
    },
    "third5": {
        "family": "section_level",
        "title": "Third sections (≥5 sentences)",
        "n_sections": 3,
        "section_display": ["Introduction", "Body", "Conclusion"],
    },
    "shuf_third": {
        "family": "section_level",
        "title": "Shuffled third sections",
        "n_sections": 3,
        "section_display": ["Shuffled 1", "Shuffled 2", "Shuffled 3"],
    },
    "sent_real_third": {
        "family": "sentence_position",
        "title": "Sentence-position thirds",
        "n_sections": 3,
        "section_display": ["Introduction", "Body", "Conclusion"],
    },
    "sent_real_quart": {
        "family": "sentence_position",
        "title": "Sentence-position quartiles",
        "n_sections": 4,
        "section_display": ["Q1", "Q2", "Q3", "Q4"],
    },
    "sent_shuf_third": {
        "family": "sentence_position",
        "title": "Shuffled sentence-position thirds",
        "n_sections": 3,
        "section_display": ["Shuffled 1", "Shuffled 2", "Shuffled 3"],
    },
    "sent_shuf_quart": {
        "family": "sentence_position",
        "title": "Shuffled sentence-position quartiles",
        "n_sections": 4,
        "section_display": [
            "Shuffled Q1",
            "Shuffled Q2",
            "Shuffled Q3",
            "Shuffled Q4",
        ],
    },
}

DEFAULT_SCHEMES = list(SCHEME_SPECS)


# ---------------------------------------------------------------------------
# Figure styling
# ---------------------------------------------------------------------------

# Fixed section identity across every figure. The fourth color is used by the
# quartile schemes only.
SECTION_COLORS = {
    "Section 1": "#C7C7C7",
    "Section 2": "#8FB4DD",
    "Section 3": "#3E75AE",
    "Section 4": "#1B3E6D",
}

GRID_COLOR = "#D0D0D0"
TEXT_COLOR = "#111111"
ERROR_COLOR = "#737373"
CONNECTOR_COLOR = "#4F4F4F"
REFERENCE_COLOR = "#2F6FB3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create revised age-line and gender-bar panels for all section "
            "and sentence-position schemes."
        )
    )
    parser.add_argument(
        "--section-results-dir",
        type=Path,
        default=DEFAULT_SECTION_RESULTS_DIR,
        help=(
            "Directory containing third/quart/third5/shuf_third result CSVs "
            f"(default: {DEFAULT_SECTION_RESULTS_DIR})"
        ),
    )
    parser.add_argument(
        "--sentence-results-dir",
        type=Path,
        default=DEFAULT_SENTENCE_RESULTS_DIR,
        help=(
            "Directory containing sentence-position result CSVs "
            f"(default: {DEFAULT_SENTENCE_RESULTS_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=list(SCHEME_SPECS),
        default=DEFAULT_SCHEMES,
        help="Schemes to draw. The default includes every scheme.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "svg", "pdf"],
        default=["png", "svg"],
        help="Output formats (default: png svg).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster resolution for PNG output (default: 300).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Keep the generated figures open after saving.",
    )
    return parser.parse_args()


def section_order(spec: Dict[str, Any]) -> List[str]:
    return [
        f"Section {number}"
        for number in range(1, int(spec["n_sections"]) + 1)
    ]


def section_display_map(spec: Dict[str, Any]) -> Dict[str, str]:
    return dict(zip(section_order(spec), spec["section_display"]))


def result_paths(
    scheme: str,
    kind: str,
    section_results_dir: Path,
    sentence_results_dir: Path,
) -> Tuple[Path, Path]:
    spec = SCHEME_SPECS[scheme]
    if spec["family"] == "section_level":
        return (
            section_results_dir / f"{scheme}_{kind}_age_ovr_results.csv",
            section_results_dir / f"{scheme}_{kind}_gender_results.csv",
        )

    return (
        sentence_results_dir / f"{scheme}_sentence_{kind}_age_results.csv",
        sentence_results_dir / f"{scheme}_sentence_{kind}_gender_results.csv",
    )


def clean_filename(text: str) -> str:
    cleaned = re.sub(r"[^\w-]+", "_", str(text))
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    source_path: Path,
) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise KeyError(f"{source_path} is missing columns: {missing}")


def load_age_results(
    path: Path,
    kind: str,
    scheme: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Age result CSV not found: {path}")

    label_col = "theme" if kind == "theme" else "sentiment"
    data = pd.read_csv(path)
    require_columns(
        data,
        [
            label_col,
            "section",
            "Age Group",
            "p-value",
            "log_odds_ratio",
        ],
        path,
    )

    tidy = data.copy()
    tidy["category"] = tidy[label_col].astype(str)
    tidy["section"] = tidy["section"].astype(str)
    tidy["Age Group"] = pd.to_numeric(
        tidy["Age Group"], errors="coerce"
    )
    tidy["p_value"] = pd.to_numeric(tidy["p-value"], errors="coerce")
    tidy["log_odds"] = pd.to_numeric(
        tidy["log_odds_ratio"], errors="coerce"
    )
    tidy["scheme"] = scheme
    return tidy


def load_gender_results(
    path: Path,
    kind: str,
    scheme: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Gender result CSV not found: {path}")

    label_col = "theme" if kind == "theme" else "sentiment"
    data = pd.read_csv(path)
    require_columns(
        data,
        [
            label_col,
            "section",
            "Odds Ratio",
            "CI Lower",
            "CI Upper",
            "p-value",
        ],
        path,
    )

    tidy = data.copy()
    tidy["category"] = tidy[label_col].astype(str)
    tidy["section"] = tidy["section"].astype(str)
    tidy["odds_ratio"] = pd.to_numeric(
        tidy["Odds Ratio"], errors="coerce"
    )
    tidy["ci_low"] = pd.to_numeric(tidy["CI Lower"], errors="coerce")
    tidy["ci_high"] = pd.to_numeric(tidy["CI Upper"], errors="coerce")
    tidy["p_value"] = pd.to_numeric(tidy["p-value"], errors="coerce")
    tidy["scheme"] = scheme
    return tidy


def significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def choose_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Arial", "Roboto", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "sans-serif"


def nice_axis_step(raw_step: float) -> float:
    if not np.isfinite(raw_step) or raw_step <= 0:
        return 0.25

    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def age_y_scale(
    values: np.ndarray,
    symmetric: bool,
) -> Tuple[float, float, np.ndarray]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        finite = np.array([0.0])

    data_min = min(float(finite.min()), 0.0)
    data_max = max(float(finite.max()), 0.0)

    if symmetric:
        largest = max(abs(data_min), abs(data_max), 0.50)
        step = 0.25 if 2.0 * largest <= 3.0 else nice_axis_step(
            2.0 * largest / 8.0
        )
        tick_limit = math.ceil((largest + 0.08) / step) * step
        ticks = np.arange(-tick_limit, tick_limit + step / 2.0, step)
        return -tick_limit - step, tick_limit + step, ticks

    span = max(data_max - data_min, 0.50)
    step = 0.25 if span <= 3.0 else nice_axis_step(span / 8.0)
    lower_tick = math.floor((data_min - 0.08) / step) * step
    upper_tick = math.ceil((data_max + 0.08) / step) * step
    ticks = np.arange(lower_tick, upper_tick + step / 2.0, step)
    return lower_tick - step, upper_tick + step, ticks


def gender_or_scale(
    results: pd.DataFrame,
) -> Tuple[float, float, np.ndarray]:
    odds = pd.to_numeric(results["odds_ratio"], errors="coerce").to_numpy()
    ci_low = pd.to_numeric(results["ci_low"], errors="coerce").to_numpy()
    ci_high = pd.to_numeric(results["ci_high"], errors="coerce").to_numpy()

    reciprocal_odds = np.divide(
        1.0,
        odds,
        out=np.full_like(odds, np.nan, dtype=float),
        where=odds > 0,
    )
    reciprocal_low = np.divide(
        1.0,
        ci_high,
        out=np.full_like(ci_high, np.nan, dtype=float),
        where=ci_high > 0,
    )
    reciprocal_high = np.divide(
        1.0,
        ci_low,
        out=np.full_like(ci_low, np.nan, dtype=float),
        where=ci_low > 0,
    )

    all_values = np.concatenate(
        [
            odds,
            ci_low,
            ci_high,
            reciprocal_odds,
            reciprocal_low,
            reciprocal_high,
            np.array([1.0]),
        ]
    )
    finite = all_values[np.isfinite(all_values) & (all_values > 0)]
    if not finite.size:
        finite = np.array([1.0])

    data_min = min(float(finite.min()), 1.0)
    data_max = max(float(finite.max()), 1.0)
    span = max(data_max - data_min, 0.50)
    step = 0.25 if span <= 2.5 else nice_axis_step(span / 6.0)
    padding = 0.08 * span

    tick_lower = max(
        0.0,
        math.floor((data_min - padding) / step) * step,
    )
    tick_upper = math.ceil((data_max + padding) / step) * step
    if tick_upper <= tick_lower:
        tick_upper = tick_lower + step

    ticks = np.arange(tick_lower, tick_upper + step / 2.0, step)
    return (
        max(0.0, tick_lower - 0.30 * step),
        tick_upper + 0.30 * step,
        ticks,
    )


def format_tick(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 2:
        return f"{value:.1f}"
    return f"{value:.2f}"


def validate_categories(
    results: pd.DataFrame,
    categories: Sequence[str],
    scheme_sections: Sequence[str],
    age_plot: bool,
) -> pd.DataFrame:
    selected = results[
        results["category"].isin(categories)
        & results["section"].isin(scheme_sections)
    ].copy()

    missing_categories = [
        category
        for category in categories
        if category not in set(selected["category"])
    ]
    if missing_categories:
        raise ValueError(f"Missing result categories: {missing_categories}")

    expected_per_cell = len(AGE_GROUPS) if age_plot else 1
    for category in categories:
        for section in scheme_sections:
            count = len(
                selected[
                    (selected["category"] == category)
                    & (selected["section"] == section)
                ]
            )
            if count != expected_per_cell:
                raise ValueError(
                    f"{category}, {section}: found {count} rows; "
                    f"expected {expected_per_cell}."
                )
    return selected


def style_context() -> Dict[str, Any]:
    return {
        "font.family": choose_font(),
        "font.size": 10.5,
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.linewidth": 1.05,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }


def configure_axis(ax: Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.62)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_COLOR)
        ax.spines[side].set_linewidth(1.05)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=0.95,
        color=TEXT_COLOR,
        pad=5,
    )


def title_fontsize(title: str) -> float:
    if len(title) > 62:
        return 15.5
    if len(title) > 48:
        return 16.5
    return 18.0


def place_legend_and_title(
    fig: Figure,
    axes: np.ndarray,
    number_of_panels: int,
    handles: Sequence[Any],
    overall_title: str,
    n_sections: int,
) -> None:
    unused_axes = list(axes[number_of_panels:])
    for unused_ax in unused_axes:
        unused_ax.set_axis_off()

    if unused_axes:
        fig.suptitle(
            overall_title,
            x=0.5,
            y=0.968,
            ha="center",
            fontsize=title_fontsize(overall_title),
            fontweight="bold",
        )
        unused_axes[0].legend(
            handles=handles,
            title="Section",
            loc="center",
            frameon=True,
            fancybox=False,
            edgecolor="#777777",
            borderpad=1.2,
            labelspacing=1.0,
            handlelength=1.8,
            fontsize=10.5,
            title_fontsize=11,
        )
        return

    fig.suptitle(
        overall_title,
        x=0.5,
        y=0.982,
        ha="center",
        fontsize=title_fontsize(overall_title),
        fontweight="bold",
    )
    fig.legend(
        handles=handles,
        title="Section",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.944),
        ncol=n_sections,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.45,
        columnspacing=1.35,
        fontsize=10,
        title_fontsize=10.5,
    )


def save_figure(
    fig: plt.Figure,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for output_format in dict.fromkeys(formats):
        path = output_stem.with_suffix(f".{output_format}")
        options: Dict[str, Any] = {
            "facecolor": "white",
            "bbox_inches": "tight",
            "pad_inches": 0.12,
        }
        if output_format == "png":
            options["dpi"] = dpi
        fig.savefig(path, **options)
        saved.append(path.resolve())
        print(f"Saved: {path.resolve()}")

    return saved


def draw_age_figure(
    results: pd.DataFrame,
    categories: Sequence[str],
    scheme: str,
    kind: str,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[Figure, List[Path]]:
    spec = SCHEME_SPECS[scheme]
    scheme_sections = section_order(spec)
    display = section_display_map(spec)
    selected = validate_categories(
        results,
        categories,
        scheme_sections,
        age_plot=True,
    )

    n_columns = min(3, len(categories))
    n_rows = math.ceil(len(categories) / n_columns)
    width = 4.7 * n_columns
    height = 4.25 * n_rows + 0.70
    y_lower, y_upper, y_ticks = age_y_scale(
        selected["log_odds"].to_numpy(dtype=float),
        symmetric=(kind == "sentiment"),
    )

    with plt.rc_context(style_context()):
        fig, axes_array = plt.subplots(
            n_rows,
            n_columns,
            figsize=(width, height),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes = axes_array.ravel()
        fig.subplots_adjust(
            left=0.075,
            right=0.975,
            bottom=0.105,
            top=0.845 if len(categories) == len(axes) else 0.875,
            wspace=0.25,
            hspace=0.43,
        )

        for panel_index, category in enumerate(categories):
            ax = axes[panel_index]
            panel = selected[selected["category"] == category]
            configure_axis(ax)

            for section in scheme_sections:
                line = (
                    panel[panel["section"] == section]
                    .sort_values("Age Group")
                    .reset_index(drop=True)
                )
                color = SECTION_COLORS[section]
                ax.plot(
                    line["Age Group"],
                    line["log_odds"],
                    color=color,
                    linewidth=1.4,
                    marker="o",
                    markersize=6.5,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    zorder=3,
                )

                for age_value, log_odds, p_value in line[
                    ["Age Group", "log_odds", "p_value"]
                ].itertuples(index=False, name=None):
                    stars = significance_stars(float(p_value))
                    if stars:
                        ax.annotate(
                            stars,
                            xy=(age_value, log_odds),
                            xytext=(4, 4),
                            textcoords="offset points",
                            ha="left",
                            va="bottom",
                            fontsize=9.5,
                            fontweight="bold",
                            color=TEXT_COLOR,
                            annotation_clip=False,
                            zorder=5,
                        )

            ax.axhline(
                0,
                color=REFERENCE_COLOR,
                linewidth=1.3,
                linestyle=(0, (2.2, 2.2)),
                zorder=2,
            )
            ax.set_xlim(0.6, 5.4)
            ax.set_ylim(y_lower, y_upper)
            ax.set_xticks(AGE_GROUPS)
            ax.set_xticklabels(AGE_LABELS)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([format_tick(tick) for tick in y_ticks])
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_title(category, pad=11)
            ax.set_xlabel("Age Group", labelpad=6)

            if panel_index % n_columns == 0:
                ax.set_ylabel("Log-Odds", labelpad=7)

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=SECTION_COLORS[section],
                marker="o",
                linewidth=1.4,
                markersize=5.5,
                label=display[section],
            )
            for section in scheme_sections
        ]
        overall_title = (
            f"{spec['title']}: age-specific {kind} patterns"
        )
        place_legend_and_title(
            fig,
            axes,
            len(categories),
            legend_handles,
            overall_title,
            len(scheme_sections),
        )
        saved = save_figure(fig, output_stem, formats, dpi)
        return fig, saved


def draw_gender_figure(
    results: pd.DataFrame,
    categories: Sequence[str],
    scheme: str,
    kind: str,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[Figure, List[Path]]:
    spec = SCHEME_SPECS[scheme]
    scheme_sections = section_order(spec)
    display = section_display_map(spec)
    selected = validate_categories(
        results,
        categories,
        scheme_sections,
        age_plot=False,
    )

    n_columns = min(3, len(categories))
    n_rows = math.ceil(len(categories) / n_columns)
    width = 4.7 * n_columns
    height = 4.25 * n_rows + 0.70
    y_lower, y_upper, y_ticks = gender_or_scale(selected)
    y_span = y_upper - y_lower
    section_slot_width = 0.60 / len(scheme_sections)
    bar_width = 0.88 * section_slot_width

    with plt.rc_context(style_context()):
        fig, axes_array = plt.subplots(
            n_rows,
            n_columns,
            figsize=(width, height),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes = axes_array.ravel()
        fig.subplots_adjust(
            left=0.075,
            right=0.975,
            bottom=0.105,
            top=0.845 if len(categories) == len(axes) else 0.875,
            wspace=0.25,
            hspace=0.43,
        )

        for panel_index, category in enumerate(categories):
            ax = axes[panel_index]
            panel = selected[selected["category"] == category]
            configure_axis(ax)

            for section_index, section in enumerate(scheme_sections):
                row = panel[panel["section"] == section].iloc[0]
                odds_ratio = float(row["odds_ratio"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                p_value = float(row["p_value"])

                female_or = 1.0 / odds_ratio
                female_ci_low = 1.0 / ci_high
                female_ci_high = 1.0 / ci_low
                plotted_or = np.array([female_or, odds_ratio])
                plotted_ci_low = np.array([female_ci_low, ci_low])
                plotted_ci_high = np.array([female_ci_high, ci_high])
                y_errors = np.vstack(
                    [
                        np.maximum(plotted_or - plotted_ci_low, 0.0),
                        np.maximum(plotted_ci_high - plotted_or, 0.0),
                    ]
                )

                offset = (
                    section_index - (len(scheme_sections) - 1) / 2.0
                ) * section_slot_width
                x_positions = np.array([0.0, 1.0]) + offset
                color = SECTION_COLORS[section]

                ax.bar(
                    x_positions,
                    plotted_or,
                    width=bar_width,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                ax.errorbar(
                    x_positions,
                    plotted_or,
                    yerr=y_errors,
                    fmt="none",
                    ecolor=ERROR_COLOR,
                    elinewidth=1.05,
                    capsize=3.2,
                    capthick=1.05,
                    zorder=5,
                )
                ax.plot(
                    x_positions,
                    plotted_or,
                    color=CONNECTOR_COLOR,
                    linewidth=0.82,
                    linestyle=(0, (3.2, 3.2)),
                    alpha=0.92,
                    zorder=4,
                )

                stars = significance_stars(p_value)
                if stars:
                    ax.annotate(
                        stars,
                        xy=(
                            x_positions[1],
                            plotted_ci_high[1] + 0.012 * y_span,
                        ),
                        ha="center",
                        va="bottom",
                        fontsize=9.5,
                        fontweight="bold",
                        color=TEXT_COLOR,
                        annotation_clip=False,
                        zorder=6,
                    )

            ax.axhline(
                1,
                color=REFERENCE_COLOR,
                linewidth=1.3,
                linestyle=(0, (2.2, 2.2)),
                zorder=2,
            )
            ax.set_xlim(-0.44, 1.44)
            ax.set_ylim(y_lower, y_upper)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Female", "Male"])
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([format_tick(tick) for tick in y_ticks])
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_title(category, pad=11)

            if panel_index % n_columns == 0:
                ax.set_ylabel("Odds Ratio", labelpad=7)

        legend_handles = [
            Patch(
                facecolor=SECTION_COLORS[section],
                edgecolor="white",
                label=display[section],
            )
            for section in scheme_sections
        ]
        overall_title = (
            f"{spec['title']}: gender-specific {kind} patterns"
        )
        place_legend_and_title(
            fig,
            axes,
            len(categories),
            legend_handles,
            overall_title,
            len(scheme_sections),
        )
        saved = save_figure(fig, output_stem, formats, dpi)
        return fig, saved


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    open_figures: List[Figure] = []

    for scheme in args.schemes:
        spec = SCHEME_SPECS[scheme]
        family_output_dir = args.output_dir / str(spec["family"]) / scheme

        print("\n" + "=" * 80)
        # Keep console text ASCII-compatible for Windows cp949 terminals.
        print(f"SCHEME: {scheme}")
        print("=" * 80)

        for kind in ("theme", "sentiment"):
            age_path, gender_path = result_paths(
                scheme,
                kind,
                args.section_results_dir,
                args.sentence_results_dir,
            )
            age_results = load_age_results(age_path, kind, scheme)
            gender_results = load_gender_results(gender_path, kind, scheme)

            age_categories = THEMES if kind == "theme" else AGE_SENTIMENTS
            gender_categories = (
                THEMES if kind == "theme" else GENDER_SENTIMENTS
            )

            age_stem = (
                family_output_dir
                / f"{clean_filename(scheme)}_{kind}_age_lines"
            )
            age_figure, age_outputs = draw_age_figure(
                results=age_results,
                categories=age_categories,
                scheme=scheme,
                kind=kind,
                output_stem=age_stem,
                formats=args.formats,
                dpi=args.dpi,
            )
            open_figures.append(age_figure)

            gender_stem = (
                family_output_dir
                / f"{clean_filename(scheme)}_{kind}_gender_bars"
            )
            gender_figure, gender_outputs = draw_gender_figure(
                results=gender_results,
                categories=gender_categories,
                scheme=scheme,
                kind=kind,
                output_stem=gender_stem,
                formats=args.formats,
                dpi=args.dpi,
            )
            open_figures.append(gender_figure)

            for plot_type, input_path, output_paths in (
                ("age_lines", age_path, age_outputs),
                ("gender_bars", gender_path, gender_outputs),
            ):
                manifest_rows.append(
                    {
                        "family": spec["family"],
                        "scheme": scheme,
                        "scheme_title": spec["title"],
                        "feature_kind": kind,
                        "plot_type": plot_type,
                        "input_csv": str(input_path.resolve()),
                        "output_files": " | ".join(
                            str(path) for path in output_paths
                        ),
                    }
                )

            if not args.show:
                plt.close(age_figure)
                plt.close(gender_figure)
                open_figures.clear()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"DONE: generated {len(manifest_rows)} panel figures")
    print(f"Manifest: {manifest_path.resolve()}")
    print("=" * 80)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
