# -*- coding: utf-8 -*-
"""
Create the Figure 7 sentiment/theme panels for age and gender.

One run:
  1. loads the source pickle;
  2. rebuilds all 45 sentiment-by-section predictors;
  3. reads the 15 precomputed theme-by-section predictors;
  4. fits age and gender models for both feature families; and
  5. draws age estimates as line plots;
  6. draws gender estimates as grouped odds-ratio bars with 95% confidence
     intervals and dotted Female-to-Male section guides; and
  7. exports four panel figures (sentiment/theme × age/gender) as PNG and SVG.

The code does not depend on variables or results from earlier notebook cells.

Required packages:
    pandas, numpy, statsmodels, matplotlib

Run:
    python figure7_age_group_sentiment_theme_panels_standalone.py
"""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Defaults: edit these if you prefer not to use command-line arguments.
# ---------------------------------------------------------------------------
DEFAULT_INPUT = Path(r"C:\Users\Jae Bin Park\morethan3rawsentiment.pkl")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

DEFAULT_SENTIMENTS = [
    "Defeat",
    "Exhaustion",
    "Neutral",
    "Sadness",
    "Happiness",
    "Disappointment",
]
DEFAULT_GENDER_SENTIMENTS = [
    "Defeat",
    "Hatred",
    "Neutral",
    "Happiness",
    "Anxious",
    "Disappointment",
]
DEFAULT_THEMES = [
    "Sorry and Shame",
    "Love and Gratitude",
    "Burden",
    "Despair",
    "Post-mortem Affairs",
]

SENTIMENT_OUTPUT_STEM = "figure7_sentiment_age_group_panels"
THEME_OUTPUT_STEM = "figure7_theme_age_group_panels"
GENDER_SENTIMENT_OUTPUT_STEM = "figure7_sentiment_gender_panels"
GENDER_THEME_OUTPUT_STEM = "figure7_theme_gender_panels"

SECTION_COLUMNS = [
    "1st_section_sentiment",
    "2nd_section_sentiment",
    "3rd_section_sentiment",
]

# English display name -> label stored in the section sentiment dictionaries.
SENTIMENT_LABELS = {
    "Despair": "절망",
    "Defeat": "패배/자기혐오",
    "Exhaustion": "힘듦/지침",
    "Anxious": "불안/걱정",
    "Disappointment": "안타까움/실망",
    "Anger": "화남/분노",
    "Hatred": "증오/혐오",
    "Resentment": "어이없음",
    "Sadness": "슬픔",
    "Sorrow": "서러움",
    "Gratitude": "고마움",
    "Affection": "흐뭇함(귀여움/예쁨)",
    "Happiness": "행복",
    "Relief": "안심/신뢰",
    "Neutral": "없음",
}

# English display name -> three-element [Opening, Middle, Ending] column.
THEME_COLUMNS = {
    "Sorry and Shame": "sorrysentence_presence",
    "Love and Gratitude": "loveandgratitudesentence_presence",
    "Burden": "burdensomesentence_presence",
    "Despair": "despairsentence_presence",
    "Post-mortem Affairs": "pmaffairssentence_presence",
}

SECTION_ORDER = ["Section 1", "Section 2", "Section 3"]
SECTION_DISPLAY = {
    "Section 1": "Opening",
    "Section 2": "Middle",
    "Section 3": "Ending",
}
AGE_GROUPS = [1, 2, 3, 4, 5]
AGE_LABELS = ["≤18", "19–34", "35–49", "50–64", "65+"]

# One consistent light-to-dark section palette for both the age and gender
# figures. This follows the designer's grouped-bar reference.
SECTION_COLORS = {
    "Section 1": "#C7C7C7",
    "Section 2": "#8FB4DD",
    "Section 3": "#376DA5",
}
GRID_COLOR = "#D0D0D0"
TEXT_COLOR = "#111111"
ERROR_COLOR = "#737373"
CONNECTOR_COLOR = "#4F4F4F"
REFERENCE_COLOR = "#2F6FB3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and draw the Figure 7 sentiment and theme panels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source pickle (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sentiments",
        nargs="+",
        choices=list(SENTIMENT_LABELS),
        default=DEFAULT_SENTIMENTS,
        help="Sentiment panels to include.",
    )
    parser.add_argument(
        "--themes",
        nargs="+",
        choices=list(THEME_COLUMNS),
        default=DEFAULT_THEMES,
        help="Theme panels to include.",
    )
    parser.add_argument(
        "--gender-sentiments",
        nargs="+",
        choices=list(SENTIMENT_LABELS),
        default=DEFAULT_GENDER_SENTIMENTS,
        help="Sentiment panels to include in the gender figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open all four figures interactively after saving.",
    )
    return parser.parse_args()


def labels_from_entry(entry: Any) -> frozenset[str]:
    """Return the sentiment labels stored in one section dictionary."""
    if isinstance(entry, str):
        try:
            entry = ast.literal_eval(entry)
        except (SyntaxError, ValueError):
            return frozenset()

    if not isinstance(entry, dict):
        return frozenset()

    labels = entry.get("labels", [])
    if not isinstance(labels, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(label) for label in labels)


def presence_at_section(value: Any, section_index: int) -> int:
    """Read one flag from a three-element theme-presence vector."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return 0

    if not isinstance(value, (list, tuple, np.ndarray)):
        return 0
    if len(value) <= section_index:
        return 0

    try:
        return int(bool(int(value[section_index])))
    except (TypeError, ValueError):
        return 0


def filter_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "AGE2",
        "SEX",
        *SECTION_COLUMNS,
        *THEME_COLUMNS.values(),
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise KeyError(f"Input data are missing required columns: {missing}")

    valid = source["AGE2"].isin(AGE_GROUPS) & source["SEX"].isin([1, 2])
    return source.loc[valid, list(required)].copy()


def build_sentiment_predictors(data: pd.DataFrame) -> pd.DataFrame:
    """Recreate the original 45 sentiment-by-section predictors."""
    section_label_sets = {
        section_number: data[column].map(labels_from_entry)
        for section_number, column in enumerate(SECTION_COLUMNS, start=1)
    }

    predictors: dict[str, pd.Series] = {}
    for section_number, label_sets in section_label_sets.items():
        for sentiment_name, stored_label in SENTIMENT_LABELS.items():
            feature = f"Section {section_number} {sentiment_name}"
            predictors[feature] = label_sets.map(
                lambda labels, target=stored_label: int(target in labels)
            )

    model_x = pd.DataFrame(predictors, index=data.index, dtype=float)
    model_x["Gender"] = (data["SEX"] == 1).astype(float)
    return model_x


def build_theme_predictors(data: pd.DataFrame) -> pd.DataFrame:
    """Create the original 15 theme-by-section predictors."""
    predictors: dict[str, pd.Series] = {}
    for section_number in range(1, 4):
        section_index = section_number - 1
        for theme_name, source_column in THEME_COLUMNS.items():
            feature = f"Section {section_number} {theme_name}"
            predictors[feature] = data[source_column].map(
                lambda value, index=section_index: presence_at_section(value, index)
            )

    model_x = pd.DataFrame(predictors, index=data.index, dtype=float)
    model_x["Gender"] = (data["SEX"] == 1).astype(float)
    return model_x


def validate_model_data(model_x: pd.DataFrame, age_group: pd.Series) -> None:
    if model_x.isna().any().any() or age_group.isna().any():
        raise ValueError("Missing values remain after building the model matrix.")
    if not model_x.index.equals(age_group.index):
        raise ValueError("Predictor and outcome indices do not match.")


def fit_age_models(model_x: pd.DataFrame, age_group: pd.Series) -> pd.DataFrame:
    """Fit five one-vs-rest logistic models and return section results."""
    validate_model_data(model_x, age_group)
    design = sm.add_constant(model_x, has_constant="add")
    result_rows: list[pd.DataFrame] = []

    for group in AGE_GROUPS:
        outcome = (age_group == group).astype(int)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            fitted = sm.Logit(outcome, design).fit(disp=False, maxiter=200)

        if not bool(fitted.mle_retvals.get("converged", True)):
            warnings.warn(
                f"The AGE2={group} model did not fully converge.",
                RuntimeWarning,
                stacklevel=2,
            )

        result_rows.append(
            pd.DataFrame(
                {
                    "Feature": fitted.params.index,
                    "log_odds": fitted.params.to_numpy(),
                    "p_value": fitted.pvalues.to_numpy(),
                    "Age Group": group,
                }
            )
        )

    tidy = pd.concat(result_rows, ignore_index=True)
    tidy = tidy[tidy["Feature"].str.startswith("Section ")].copy()
    tidy["section"] = tidy["Feature"].str.extract(r"^(Section \d)")
    tidy["category"] = tidy["Feature"].str.replace(
        r"^Section \d\s+", "", regex=True
    )
    return tidy


def fit_gender_model(
    model_x: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Fit Male=1 versus Female=0, controlling for categorical age."""
    if "Gender" not in model_x.columns:
        raise KeyError("The model matrix is missing its Gender column.")

    age_group = data["AGE2"].astype(int)
    validate_model_data(model_x, age_group)

    section_predictors = model_x.drop(columns="Gender")
    age_controls = pd.DataFrame(
        {
            f"AGE2_{group}": (age_group == group).astype(float)
            for group in AGE_GROUPS[1:]
        },
        index=data.index,
    )
    design = sm.add_constant(
        pd.concat([section_predictors, age_controls], axis=1),
        has_constant="add",
    )
    outcome = (data["SEX"] == 1).astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        fitted = sm.Logit(outcome, design).fit(disp=False, maxiter=200)

    if not bool(fitted.mle_retvals.get("converged", True)):
        warnings.warn(
            "The gender model did not fully converge.",
            RuntimeWarning,
            stacklevel=2,
        )

    confidence_intervals = fitted.conf_int()
    tidy = pd.DataFrame(
        {
            "Feature": fitted.params.index,
            "log_odds": fitted.params.to_numpy(),
            "log_ci_low": confidence_intervals.iloc[:, 0].to_numpy(),
            "log_ci_high": confidence_intervals.iloc[:, 1].to_numpy(),
            "p_value": fitted.pvalues.to_numpy(),
        }
    )
    tidy = tidy[tidy["Feature"].str.startswith("Section ")].copy()
    tidy["section"] = tidy["Feature"].str.extract(r"^(Section \d)")
    tidy["category"] = tidy["Feature"].str.replace(
        r"^Section \d\s+", "", regex=True
    )
    return tidy


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def choose_font() -> str:
    """Use Arial when available and otherwise fall back to DejaVu Sans."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    return "Arial" if "Arial" in available else "DejaVu Sans"


def calculate_y_scale(
    values: np.ndarray,
    symmetric: bool,
) -> tuple[float, float, np.ndarray]:
    """Return quarter-unit ticks and padded lower/upper axis limits."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        finite = np.array([0.0])

    if symmetric:
        largest = float(np.abs(finite).max())
        tick_limit = max(0.75, math.ceil((largest + 0.10) / 0.25) * 0.25)
        ticks = np.arange(-tick_limit, tick_limit + 0.125, 0.25)
        return -(tick_limit + 0.25), tick_limit + 0.25, ticks

    # Theme estimates extend much farther below zero than above it. Calculate
    # the two ends independently so the negative ticks can reach -1.75
    # without forcing the positive ticks to reach +1.75.
    lower_tick = min(
        -0.75,
        math.floor((float(finite.min()) - 0.10) / 0.25) * 0.25,
    )
    upper_tick = max(
        0.75,
        math.ceil(float(finite.max()) / 0.25) * 0.25,
    )
    ticks = np.arange(lower_tick, upper_tick + 0.125, 0.25)
    return lower_tick - 0.25, upper_tick + 0.25, ticks


def nice_axis_step(raw_step: float) -> float:
    """Round an arbitrary interval to a readable 1/2/2.5/5 × 10^n step."""
    if not np.isfinite(raw_step) or raw_step <= 0:
        return 0.25

    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def calculate_gender_or_scale(
    results: pd.DataFrame,
) -> tuple[float, float, np.ndarray]:
    """Return a readable linear odds-ratio scale covering both reciprocal CIs."""
    coefficients = pd.to_numeric(results["log_odds"], errors="coerce").to_numpy()
    ci_low = pd.to_numeric(results["log_ci_low"], errors="coerce").to_numpy()
    ci_high = pd.to_numeric(results["log_ci_high"], errors="coerce").to_numpy()

    # Male is exp(beta); Female is its reciprocal exp(-beta). Reversing the
    # signed CI endpoints gives the correctly ordered reciprocal Female CI.
    all_bounds = np.exp(
        np.concatenate([coefficients, -coefficients, ci_low, ci_high, -ci_high, -ci_low])
    )
    finite = all_bounds[np.isfinite(all_bounds) & (all_bounds > 0)]
    if not finite.size:
        finite = np.array([1.0])

    data_min = min(float(finite.min()), 1.0)
    data_max = max(float(finite.max()), 1.0)
    span = max(data_max - data_min, 0.25)
    padding = 0.08 * span
    step = nice_axis_step((span + 2.0 * padding) / 5.0)

    tick_lower = max(0.0, math.floor((data_min - padding) / step) * step)
    tick_upper = math.ceil((data_max + padding) / step) * step
    if tick_upper <= tick_lower:
        tick_upper = tick_lower + step

    ticks = np.arange(tick_lower, tick_upper + step / 2.0, step)
    axis_lower = max(0.0, tick_lower - 0.30 * step)
    axis_upper = tick_upper + 0.30 * step
    return axis_lower, axis_upper, ticks


def format_tick(tick: float) -> str:
    if abs(tick) >= 10:
        return f"{tick:.0f}"
    if abs(tick) >= 2:
        return f"{tick:.1f}"
    return f"{tick:.2f}"


def draw_category_figure(
    results: pd.DataFrame,
    categories: Sequence[str],
    overall_title: str,
    output_stem: Path,
    dpi: int,
    share_y_axis: bool,
    symmetric_y_axis: bool,
) -> plt.Figure:
    """Draw one multi-panel figure with a single shared section legend."""
    selected = results[results["category"].isin(categories)].copy()
    observed = set(selected["category"])
    missing_categories = [category for category in categories if category not in observed]
    if missing_categories:
        raise ValueError(f"Missing model results for: {missing_categories}")

    font_family = choose_font()
    style = {
        "font.family": font_family,
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

    n_columns = min(3, len(categories))
    n_rows = math.ceil(len(categories) / n_columns)
    width = 4.7 * n_columns
    height = 4.25 * n_rows + 0.55
    shared_y_lower: float | None = None
    shared_y_upper: float | None = None
    shared_y_ticks: np.ndarray | None = None
    if share_y_axis:
        shared_y_lower, shared_y_upper, shared_y_ticks = calculate_y_scale(
            selected["log_odds"].to_numpy(dtype=float),
            symmetric=symmetric_y_axis,
        )

    with plt.rc_context(style):
        fig, axes_array = plt.subplots(
            n_rows,
            n_columns,
            figsize=(width, height),
            squeeze=False,
            sharex=True,
            sharey=share_y_axis,
        )
        axes = axes_array.ravel()
        fig.subplots_adjust(
            left=0.075,
            right=0.975,
            bottom=0.105,
            top=0.855,
            wspace=0.25,
            hspace=0.43,
        )

        for panel_index, category in enumerate(categories):
            ax = axes[panel_index]
            panel = selected[selected["category"] == category]

            ax.set_axisbelow(True)
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.62)
            ax.grid(axis="x", visible=False)

            for section in SECTION_ORDER:
                line = (
                    panel[panel["section"] == section]
                    .sort_values("Age Group")
                    .reset_index(drop=True)
                )
                if len(line) != len(AGE_GROUPS):
                    raise ValueError(
                        f"{category}, {section} has {len(line)} estimates; "
                        f"expected {len(AGE_GROUPS)}."
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

                # Place every significance label immediately above-right of
                # its own marker, matching the designer reference.
                for age_value, log_odds, p_value in line[
                    ["Age Group", "log_odds", "p_value"]
                ].itertuples(index=False, name=None):
                    stars = significance_stars(p_value)
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
            if share_y_axis:
                assert shared_y_lower is not None
                assert shared_y_upper is not None
                assert shared_y_ticks is not None
                panel_y_lower = shared_y_lower
                panel_y_upper = shared_y_upper
                panel_y_ticks = shared_y_ticks
            else:
                panel_y_lower, panel_y_upper, panel_y_ticks = calculate_y_scale(
                    panel["log_odds"].to_numpy(dtype=float),
                    symmetric=symmetric_y_axis,
                )

            ax.set_ylim(panel_y_lower, panel_y_upper)
            ax.set_xticks(AGE_GROUPS)
            ax.set_xticklabels(AGE_LABELS)
            ax.set_yticks(panel_y_ticks)
            ax.set_yticklabels([format_tick(tick) for tick in panel_y_ticks])
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_title(category, pad=11)
            ax.set_xlabel("Age-Group", labelpad=6)

            if panel_index % n_columns == 0:
                ax.set_ylabel("Log-Odds", labelpad=7)

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

        unused_axes = list(axes[len(categories):])
        for unused_ax in unused_axes:
            unused_ax.set_axis_off()

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=SECTION_COLORS[section],
                marker="o",
                linewidth=1.4,
                markersize=5.5,
                label=SECTION_DISPLAY[section],
            )
            for section in SECTION_ORDER
        ]
        fig.suptitle(
            overall_title,
            x=0.5 if unused_axes else 0.30,
            y=0.956,
            ha="center",
            fontsize=18,
            fontweight="bold",
        )
        if unused_axes:
            # Five-panel theme figures use the sixth grid cell for the legend,
            # matching the designer's requested section placement.
            unused_axes[0].legend(
                handles=legend_handles,
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
        else:
            fig.legend(
                handles=legend_handles,
                title="Section",
                loc="upper center",
                bbox_to_anchor=(0.73, 0.963),
                ncol=3,
                frameon=False,
                handlelength=1.45,
                handletextpad=0.45,
                columnspacing=1.35,
                fontsize=10,
                title_fontsize=10.5,
            )

        png_path = output_stem.with_suffix(".png")
        svg_path = output_stem.with_suffix(".svg")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=dpi, facecolor="white")
        fig.savefig(svg_path, facecolor="white")
        print(f"Saved PNG: {png_path.resolve()}")
        print(f"Saved SVG: {svg_path.resolve()}")
        return fig


def draw_gender_figure(
    results: pd.DataFrame,
    categories: Sequence[str],
    overall_title: str,
    output_stem: Path,
    dpi: int,
) -> plt.Figure:
    """Draw grouped Female/Male odds-ratio bars with 95% confidence intervals."""
    selected = results[results["category"].isin(categories)].copy()
    observed = set(selected["category"])
    missing_categories = [category for category in categories if category not in observed]
    if missing_categories:
        raise ValueError(f"Missing gender-model results for: {missing_categories}")

    required_columns = {"log_odds", "log_ci_low", "log_ci_high", "p_value"}
    missing_columns = sorted(required_columns.difference(selected.columns))
    if missing_columns:
        raise KeyError(f"Gender results are missing columns: {missing_columns}")

    font_family = choose_font()
    style = {
        "font.family": font_family,
        "font.size": 10.5,
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.linewidth": 1.05,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }

    n_columns = min(3, len(categories))
    n_rows = math.ceil(len(categories) / n_columns)
    width = 4.7 * n_columns
    height = 4.25 * n_rows + 0.55
    y_lower, y_upper, y_ticks = calculate_gender_or_scale(selected)
    y_span = y_upper - y_lower
    section_slot_width = 0.56 / len(SECTION_ORDER)
    bar_width = 0.88 * section_slot_width

    with plt.rc_context(style):
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
            top=0.855,
            wspace=0.25,
            hspace=0.43,
        )

        for panel_index, category in enumerate(categories):
            ax = axes[panel_index]
            panel = selected[selected["category"] == category]

            ax.set_axisbelow(True)
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.62)
            ax.grid(axis="x", visible=False)

            for section_index, section in enumerate(SECTION_ORDER):
                row = panel[panel["section"] == section]
                if len(row) != 1:
                    raise ValueError(
                        f"{category}, {section} has {len(row)} gender estimates; "
                        "expected one."
                    )

                coefficient = float(row.iloc[0]["log_odds"])
                log_ci_low = float(row.iloc[0]["log_ci_low"])
                log_ci_high = float(row.iloc[0]["log_ci_high"])
                p_value = float(row.iloc[0]["p_value"])
                color = SECTION_COLORS[section]

                # The fitted coefficient is Male versus Female. Female is the
                # reciprocal contrast, with reversed reciprocal CI endpoints.
                odds_ratios = np.array(
                    [math.exp(-coefficient), math.exp(coefficient)],
                    dtype=float,
                )
                ci_lows = np.array(
                    [math.exp(-log_ci_high), math.exp(log_ci_low)],
                    dtype=float,
                )
                ci_highs = np.array(
                    [math.exp(-log_ci_low), math.exp(log_ci_high)],
                    dtype=float,
                )
                y_errors = np.vstack(
                    [
                        np.maximum(odds_ratios - ci_lows, 0.0),
                        np.maximum(ci_highs - odds_ratios, 0.0),
                    ]
                )
                section_offset = (
                    section_index - (len(SECTION_ORDER) - 1) / 2.0
                ) * section_slot_width
                x_positions = np.array([0.0, 1.0]) + section_offset

                ax.bar(
                    x_positions,
                    odds_ratios,
                    width=bar_width,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                ax.errorbar(
                    x_positions,
                    odds_ratios,
                    yerr=y_errors,
                    fmt="none",
                    ecolor=ERROR_COLOR,
                    elinewidth=1.05,
                    capsize=3.2,
                    capthick=1.05,
                    zorder=5,
                )

                # The dotted guide connects the same section across genders
                # without turning the categorical contrast into a solid trend.
                ax.plot(
                    x_positions,
                    odds_ratios,
                    color=CONNECTOR_COLOR,
                    linewidth=0.82,
                    linestyle=(0, (3.2, 3.2)),
                    alpha=0.92,
                    zorder=4,
                )

                # The same coefficient test underlies both reciprocal bars, so
                # display the significance mark once over the Male CI.
                stars = significance_stars(p_value)
                if stars:
                    ax.annotate(
                        stars,
                        xy=(x_positions[1], ci_highs[1] + 0.012 * y_span),
                        ha="center",
                        va="bottom",
                        fontsize=9.5,
                        fontweight="bold",
                        color=TEXT_COLOR,
                        annotation_clip=False,
                        zorder=5,
                    )

            ax.axhline(
                1,
                color=REFERENCE_COLOR,
                linewidth=1.3,
                linestyle=(0, (2.2, 2.2)),
                zorder=2,
            )
            ax.set_xlim(-0.42, 1.42)
            ax.set_ylim(y_lower, y_upper)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Female", "Male"])
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([format_tick(tick) for tick in y_ticks])
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_title(category, pad=11)

            if panel_index % n_columns == 0:
                ax.set_ylabel("Odds Ratio", labelpad=7)

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

        unused_axes = list(axes[len(categories):])
        for unused_ax in unused_axes:
            unused_ax.set_axis_off()

        legend_handles = [
            Patch(
                facecolor=SECTION_COLORS[section],
                edgecolor="white",
                label=SECTION_DISPLAY[section],
            )
            for section in SECTION_ORDER
        ]
        fig.suptitle(
            overall_title,
            x=0.5 if unused_axes else 0.30,
            y=0.956,
            ha="center",
            fontsize=18,
            fontweight="bold",
        )
        if unused_axes:
            unused_axes[0].legend(
                handles=legend_handles,
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
        else:
            fig.legend(
                handles=legend_handles,
                title="Section",
                loc="upper center",
                bbox_to_anchor=(0.73, 0.963),
                ncol=3,
                frameon=False,
                handlelength=1.45,
                handletextpad=0.45,
                columnspacing=1.35,
                fontsize=10,
                title_fontsize=10.5,
            )

        png_path = output_stem.with_suffix(".png")
        svg_path = output_stem.with_suffix(".svg")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=dpi, facecolor="white")
        fig.savefig(svg_path, facecolor="white")
        print(f"Saved PNG: {png_path.resolve()}")
        print(f"Saved SVG: {svg_path.resolve()}")
        return fig


def print_selected_results(
    label: str,
    results: pd.DataFrame,
    categories: Sequence[str],
) -> None:
    table = results[results["category"].isin(categories)].copy()
    table["significance"] = table["p_value"].map(significance_stars)
    print(f"\n{label} model estimates:")
    print(
        table[
            [
                "category",
                "section",
                "Age Group",
                "log_odds",
                "p_value",
                "significance",
            ]
        ].to_string(index=False)
    )


def print_gender_results(
    label: str,
    results: pd.DataFrame,
    categories: Sequence[str],
) -> None:
    table = results[results["category"].isin(categories)].copy()
    table["significance"] = table["p_value"].map(significance_stars)
    print(f"\n{label} gender-model estimates (Male versus Female):")
    print(
        table[
            [
                "category",
                "section",
                "log_odds",
                "p_value",
                "significance",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input pickle not found: {args.input}")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    print(f"Loading: {args.input}")
    source = pd.read_pickle(args.input)
    data = filter_source(source)
    age_group = data["AGE2"].astype(int)
    print(f"Analysis sample: {len(data):,} observations")

    print("Building sentiment and theme predictors...")
    sentiment_x = build_sentiment_predictors(data)
    theme_x = build_theme_predictors(data)

    print(
        "Fitting sentiment age models "
        f"({sentiment_x.shape[1] - 1} section terms + gender)..."
    )
    sentiment_results = fit_age_models(sentiment_x, age_group)

    print(
        "Fitting theme age models "
        f"({theme_x.shape[1] - 1} section terms + gender)..."
    )
    theme_results = fit_age_models(theme_x, age_group)

    print("Fitting sentiment gender model (controlling for categorical age)...")
    sentiment_gender_results = fit_gender_model(sentiment_x, data)
    print("Fitting theme gender model (controlling for categorical age)...")
    theme_gender_results = fit_gender_model(theme_x, data)

    print_selected_results("Sentiment", sentiment_results, args.sentiments)
    print_selected_results("Theme", theme_results, args.themes)
    print_gender_results(
        "Sentiment",
        sentiment_gender_results,
        args.gender_sentiments,
    )
    print_gender_results("Theme", theme_gender_results, args.themes)

    sentiment_stem = args.output_dir / SENTIMENT_OUTPUT_STEM
    theme_stem = args.output_dir / THEME_OUTPUT_STEM
    gender_sentiment_stem = args.output_dir / GENDER_SENTIMENT_OUTPUT_STEM
    gender_theme_stem = args.output_dir / GENDER_THEME_OUTPUT_STEM
    sentiment_figure = draw_category_figure(
        results=sentiment_results,
        categories=args.sentiments,
        overall_title="Age-specific sentiment patterns",
        output_stem=sentiment_stem,
        dpi=args.dpi,
        share_y_axis=True,
        symmetric_y_axis=True,
    )
    theme_figure = draw_category_figure(
        results=theme_results,
        categories=args.themes,
        overall_title="Age-specific theme patterns",
        output_stem=theme_stem,
        dpi=args.dpi,
        share_y_axis=True,
        symmetric_y_axis=False,
    )
    gender_sentiment_figure = draw_gender_figure(
        results=sentiment_gender_results,
        categories=args.gender_sentiments,
        overall_title="Gender-specific sentiment patterns",
        output_stem=gender_sentiment_stem,
        dpi=args.dpi,
    )
    gender_theme_figure = draw_gender_figure(
        results=theme_gender_results,
        categories=args.themes,
        overall_title="Gender-specific theme patterns",
        output_stem=gender_theme_stem,
        dpi=args.dpi,
    )

    if args.show:
        plt.show()
    else:
        plt.close(sentiment_figure)
        plt.close(theme_figure)
        plt.close(gender_sentiment_figure)
        plt.close(gender_theme_figure)


if __name__ == "__main__":
    main()
