# -*- coding: utf-8 -*-
"""
Compare real section order with shuffled-order null estimates.

The default figure reproduces the reference design:
    - section-level thirds
    - theme panels
    - x-axis = age group
    - y-axis = log-odds
    - solid, filled-marker lines = real section order
    - dashed, hollow-marker lines = shuffled-order null

The same section uses the same color in both orders so that the real and
shuffled estimates can be compared directly.

The optional --all-comparisons run also creates:
    1. section-level thirds: theme and sentiment
    2. sentence-position thirds: theme and sentiment
    3. sentence-position quartiles: theme and sentiment

The script reads saved model-result CSVs. It does not depend on notebook
variables and does not rerun the regressions.

Required packages:
    pandas, numpy, matplotlib

Python compatibility:
    Python 3.6 or newer

Default run:
    python figure7_real_vs_shuffled_comparison_standalone.py

Create every available comparison:
    python figure7_real_vs_shuffled_comparison_standalone.py --all-comparisons
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
    Path(__file__).resolve().parent / "real_vs_shuffled_comparisons"
)


# ---------------------------------------------------------------------------
# Categories and demographic order
# ---------------------------------------------------------------------------

THEMES = [
    "Sorry and Shame",
    "Love and Gratitude",
    "Burden",
    "Despair",
    "Post-mortem Affairs",
]

SENTIMENTS = [
    "Defeat",
    "Exhaustion",
    "Neutral",
    "Sadness",
    "Happiness",
    "Disappointment",
]

AGE_GROUPS = [1, 2, 3, 4, 5]
AGE_LABELS = ["≤18", "19–34", "35–49", "50–64", "65+"]


# ---------------------------------------------------------------------------
# Comparison definitions
# ---------------------------------------------------------------------------

COMPARISON_SPECS = {
    "section_thirds": {
        "title": "Third-section order",
        "family": "section_level",
        "real_scheme": "third",
        "shuffled_scheme": "shuf_third",
        "n_sections": 3,
        "section_display": ["Introduction", "Body", "Conclusion"],
    },
    "sentence_thirds": {
        "title": "Sentence-position thirds",
        "family": "sentence_position",
        "real_scheme": "sent_real_third",
        "shuffled_scheme": "sent_shuf_third",
        "n_sections": 3,
        "section_display": ["Introduction", "Body", "Conclusion"],
    },
    "sentence_quartiles": {
        "title": "Sentence-position quartiles",
        "family": "sentence_position",
        "real_scheme": "sent_real_quart",
        "shuffled_scheme": "sent_shuf_quart",
        "n_sections": 4,
        "section_display": ["Q1", "Q2", "Q3", "Q4"],
    },
}


# Fixed section identity, shared with the other revised figures.
SECTION_COLORS = {
    "Section 1": "#C7C7C7",
    "Section 2": "#8FB4DD",
    "Section 3": "#3E75AE",
    "Section 4": "#1B3E6D",
}

GRID_COLOR = "#D0D0D0"
TEXT_COLOR = "#111111"
REFERENCE_COLOR = "#6F6F6F"

REAL_LINESTYLE = "-"
SHUFFLED_LINESTYLE = (0, (3.0, 2.2))
REAL_MARKER = "o"
SHUFFLED_MARKER = "o"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay real-order and shuffled-order age estimates using "
            "matched section colors and different line styles."
        )
    )
    parser.add_argument(
        "--comparison",
        choices=list(COMPARISON_SPECS),
        default="section_thirds",
        help=(
            "Comparison to draw when --all-comparisons is not used "
            "(default: section_thirds)."
        ),
    )
    parser.add_argument(
        "--kind",
        choices=["theme", "sentiment"],
        default="theme",
        help=(
            "Feature family to draw when --all-comparisons is not used "
            "(default: theme)."
        ),
    )
    parser.add_argument(
        "--all-comparisons",
        action="store_true",
        help=(
            "Create theme and sentiment figures for section thirds, "
            "sentence-position thirds, and sentence-position quartiles."
        ),
    )
    parser.add_argument(
        "--section-results-dir",
        type=Path,
        default=DEFAULT_SECTION_RESULTS_DIR,
        help=(
            "Directory containing third and shuf_third result CSVs "
            "(default: {0})".format(DEFAULT_SECTION_RESULTS_DIR)
        ),
    )
    parser.add_argument(
        "--sentence-results-dir",
        type=Path,
        default=DEFAULT_SENTENCE_RESULTS_DIR,
        help=(
            "Directory containing real/shuffled sentence-position CSVs "
            "(default: {0})".format(DEFAULT_SENTENCE_RESULTS_DIR)
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: {0})".format(DEFAULT_OUTPUT_DIR),
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
        help="PNG resolution (default: 300).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Keep the generated figure windows open after saving.",
    )
    return parser.parse_args()


def section_order(spec):
    return [
        "Section {0}".format(number)
        for number in range(1, int(spec["n_sections"]) + 1)
    ]


def section_display_map(spec):
    return dict(zip(section_order(spec), spec["section_display"]))


def clean_filename(text):
    cleaned = re.sub(r"[^\w-]+", "_", str(text))
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def result_path(
    scheme,
    kind,
    family,
    section_results_dir,
    sentence_results_dir,
):
    if family == "section_level":
        return section_results_dir / (
            "{0}_{1}_age_ovr_results.csv".format(scheme, kind)
        )
    return sentence_results_dir / (
        "{0}_sentence_{1}_age_results.csv".format(scheme, kind)
    )


def require_columns(data, columns, source_path):
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise KeyError(
            "{0} is missing columns: {1}".format(source_path, missing)
        )


def load_age_results(path, kind, order_name):
    if not path.exists():
        raise FileNotFoundError("Result CSV not found: {0}".format(path))

    label_col = "theme" if kind == "theme" else "sentiment"
    data = pd.read_csv(path)
    require_columns(
        data,
        [
            label_col,
            "section",
            "Age Group",
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
    tidy["log_odds"] = pd.to_numeric(
        tidy["log_odds_ratio"], errors="coerce"
    )
    tidy["order"] = order_name
    return tidy


def prepare_comparison(
    real_results,
    shuffled_results,
    categories,
    sections,
):
    real_selected = real_results[
        real_results["category"].isin(categories)
        & real_results["section"].isin(sections)
    ][["category", "section", "Age Group", "log_odds"]].copy()
    shuffled_selected = shuffled_results[
        shuffled_results["category"].isin(categories)
        & shuffled_results["section"].isin(sections)
    ][["category", "section", "Age Group", "log_odds"]].copy()

    real_selected = real_selected.rename(
        columns={"log_odds": "real_log_odds"}
    )
    shuffled_selected = shuffled_selected.rename(
        columns={"log_odds": "shuffled_log_odds"}
    )

    merged = real_selected.merge(
        shuffled_selected,
        on=["category", "section", "Age Group"],
        how="inner",
        validate="one_to_one",
    )
    merged["delta_real_minus_shuffled"] = (
        merged["real_log_odds"] - merged["shuffled_log_odds"]
    )

    expected_rows = len(categories) * len(sections) * len(AGE_GROUPS)
    if len(merged) != expected_rows:
        raise ValueError(
            "Comparison has {0} matched rows; expected {1}.".format(
                len(merged),
                expected_rows,
            )
        )

    for category in categories:
        for section in sections:
            cell = merged[
                (merged["category"] == category)
                & (merged["section"] == section)
            ]
            observed_ages = sorted(
                int(value)
                for value in cell["Age Group"].dropna().unique()
            )
            if observed_ages != AGE_GROUPS:
                raise ValueError(
                    "{0}, {1} has age groups {2}; expected {3}.".format(
                        category,
                        section,
                        observed_ages,
                        AGE_GROUPS,
                    )
                )
    return merged


def choose_font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Arial", "Roboto", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "sans-serif"


def nice_axis_step(raw_step):
    if not np.isfinite(raw_step) or raw_step <= 0:
        return 0.25

    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def comparison_y_scale(values):
    finite = values[np.isfinite(values)]
    if not finite.size:
        finite = np.array([0.0])

    data_min = min(float(finite.min()), 0.0)
    data_max = max(float(finite.max()), 0.0)
    span = max(data_max - data_min, 0.50)
    step = 0.25 if span <= 3.0 else nice_axis_step(span / 8.0)
    lower_tick = math.floor((data_min - 0.08) / step) * step
    upper_tick = math.ceil((data_max + 0.08) / step) * step
    ticks = np.arange(lower_tick, upper_tick + step / 2.0, step)
    return lower_tick - step, upper_tick + step, ticks


def format_tick(value):
    if abs(value) >= 10:
        return "{0:.0f}".format(value)
    if abs(value) >= 2:
        return "{0:.1f}".format(value)
    return "{0:.2f}".format(value)


def style_context():
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


def configure_axis(ax):
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


def draw_legend_entry(
    ax,
    line_start,
    label_start,
    y_position,
    label,
    color,
    linestyle,
    markerfacecolor,
    linewidth,
):
    """Draw one precisely positioned legend sample and label."""
    line_end = label_start - 0.025
    marker_x = (line_start + line_end) / 2.0

    ax.plot(
        [line_start, line_end],
        [y_position, y_position],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.plot(
        [marker_x],
        [y_position],
        color=color,
        marker="o",
        markersize=5.8,
        markerfacecolor=markerfacecolor,
        markeredgecolor=color,
        markeredgewidth=1.1,
        linestyle="none",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.text(
        label_start,
        y_position,
        label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        color=TEXT_COLOR,
    )


def draw_vertical_legend_panel(legend_ax, spec):
    """Draw Section and Order blocks with identical left alignment."""
    legend_ax.set_axis_off()
    sections = section_order(spec)
    display = section_display_map(spec)
    heading_x = 0.18
    line_start = 0.18
    label_start = 0.36

    legend_ax.text(
        heading_x,
        0.92,
        "Section",
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    section_start_y = 0.80
    section_step = 0.105 if len(sections) == 4 else 0.12
    for section_index, section in enumerate(sections):
        y_position = section_start_y - section_index * section_step
        draw_legend_entry(
            legend_ax,
            line_start,
            label_start,
            y_position,
            display[section],
            SECTION_COLORS[section],
            REAL_LINESTYLE,
            SECTION_COLORS[section],
            1.7,
        )

    last_section_y = section_start_y - (len(sections) - 1) * section_step
    order_title_y = last_section_y - 0.14
    real_y = order_title_y - 0.12
    shuffled_y = real_y - 0.12

    legend_ax.text(
        heading_x,
        order_title_y,
        "Order",
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    draw_legend_entry(
        legend_ax,
        line_start,
        label_start,
        real_y,
        "Real order",
        TEXT_COLOR,
        REAL_LINESTYLE,
        TEXT_COLOR,
        1.7,
    )
    draw_legend_entry(
        legend_ax,
        line_start,
        label_start,
        shuffled_y,
        "Shuffled-order null",
        TEXT_COLOR,
        SHUFFLED_LINESTYLE,
        "white",
        1.35,
    )


def draw_horizontal_legend_panel(fig, spec):
    """Draw two aligned legend rows above a full six-panel figure."""
    legend_ax = fig.add_axes([0.12, 0.79, 0.76, 0.105])
    legend_ax.set_axis_off()
    sections = section_order(spec)
    display = section_display_map(spec)
    heading_x = 0.00
    entry_start_x = 0.14

    legend_ax.text(
        heading_x,
        0.70,
        "Section",
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    section_spacing = 0.82 / len(sections)
    for section_index, section in enumerate(sections):
        line_start = entry_start_x + section_index * section_spacing
        draw_legend_entry(
            legend_ax,
            line_start,
            line_start + 0.075,
            0.70,
            display[section],
            SECTION_COLORS[section],
            REAL_LINESTYLE,
            SECTION_COLORS[section],
            1.7,
        )

    legend_ax.text(
        heading_x,
        0.20,
        "Order",
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    draw_legend_entry(
        legend_ax,
        entry_start_x,
        entry_start_x + 0.090,
        0.20,
        "Real order",
        TEXT_COLOR,
        REAL_LINESTYLE,
        TEXT_COLOR,
        1.7,
    )
    draw_legend_entry(
        legend_ax,
        0.48,
        0.57,
        0.20,
        "Shuffled-order null",
        TEXT_COLOR,
        SHUFFLED_LINESTYLE,
        "white",
        1.35,
    )


def place_legends(
    fig,
    axes,
    number_of_panels,
    spec,
    overall_title,
):
    unused_axes = list(axes[number_of_panels:])

    for unused_ax in unused_axes:
        unused_ax.set_axis_off()

    if unused_axes:
        legend_ax = unused_axes[0]
        fig.suptitle(
            overall_title,
            x=0.5,
            y=0.968,
            ha="center",
            fontsize=17,
            fontweight="bold",
        )
        draw_vertical_legend_panel(legend_ax, spec)
        return

    fig.suptitle(
        overall_title,
        x=0.5,
        y=0.987,
        ha="center",
        fontsize=17,
        fontweight="bold",
    )
    draw_horizontal_legend_panel(fig, spec)


def save_figure(fig, output_stem, formats, dpi):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for output_format in dict.fromkeys(formats):
        path = output_stem.with_suffix(".{0}".format(output_format))
        options = {
            "facecolor": "white",
            "bbox_inches": "tight",
            "pad_inches": 0.12,
        }
        if output_format == "png":
            options["dpi"] = dpi
        fig.savefig(path, **options)
        saved.append(path.resolve())
        print("Saved: {0}".format(path.resolve()))
    return saved


def draw_comparison_figure(
    comparison_data,
    comparison_key,
    kind,
    output_stem,
    formats,
    dpi,
):
    spec = COMPARISON_SPECS[comparison_key]
    categories = THEMES if kind == "theme" else SENTIMENTS
    sections = section_order(spec)
    n_columns = min(3, len(categories))
    n_rows = int(math.ceil(float(len(categories)) / n_columns))
    width = 4.7 * n_columns
    height = 4.25 * n_rows + 0.75

    all_values = np.concatenate(
        [
            comparison_data["real_log_odds"].to_numpy(dtype=float),
            comparison_data["shuffled_log_odds"].to_numpy(dtype=float),
        ]
    )
    y_lower, y_upper, y_ticks = comparison_y_scale(all_values)

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
        has_unused_axis = len(categories) < len(axes)
        fig.subplots_adjust(
            left=0.075,
            right=0.975,
            bottom=0.105,
            # Six-panel figures reserve a dedicated band for the two aligned
            # horizontal legend rows above the subplot titles.
            top=0.875 if has_unused_axis else 0.73,
            wspace=0.25,
            hspace=0.43,
        )

        for panel_index, category in enumerate(categories):
            ax = axes[panel_index]
            panel = comparison_data[
                comparison_data["category"] == category
            ]
            configure_axis(ax)

            for section in sections:
                line = (
                    panel[panel["section"] == section]
                    .sort_values("Age Group")
                    .reset_index(drop=True)
                )
                color = SECTION_COLORS[section]
                x_values = line["Age Group"].to_numpy(dtype=float)

                ax.plot(
                    x_values,
                    line["real_log_odds"],
                    color=color,
                    linestyle=REAL_LINESTYLE,
                    linewidth=1.7,
                    marker=REAL_MARKER,
                    markersize=6.4,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    zorder=4,
                )
                ax.plot(
                    x_values,
                    line["shuffled_log_odds"],
                    color=color,
                    linestyle=SHUFFLED_LINESTYLE,
                    linewidth=1.35,
                    marker=SHUFFLED_MARKER,
                    markersize=5.8,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.15,
                    zorder=3,
                )

            ax.axhline(
                0,
                color=REFERENCE_COLOR,
                linewidth=1.0,
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

        title_kind = "Theme" if kind == "theme" else "Sentiment"
        overall_title = (
            "{0} effects in real versus shuffled {1}\n"
            "Solid / filled = real order; dashed / hollow = shuffled-order null"
        ).format(title_kind, spec["title"].lower())
        place_legends(
            fig,
            axes,
            len(categories),
            spec,
            overall_title,
        )
        saved_paths = save_figure(fig, output_stem, formats, dpi)
        return fig, saved_paths


def generate_one(
    comparison_key,
    kind,
    args,
):
    spec = COMPARISON_SPECS[comparison_key]
    categories = THEMES if kind == "theme" else SENTIMENTS
    sections = section_order(spec)

    real_path = result_path(
        spec["real_scheme"],
        kind,
        spec["family"],
        args.section_results_dir,
        args.sentence_results_dir,
    )
    shuffled_path = result_path(
        spec["shuffled_scheme"],
        kind,
        spec["family"],
        args.section_results_dir,
        args.sentence_results_dir,
    )

    real_results = load_age_results(real_path, kind, "real")
    shuffled_results = load_age_results(
        shuffled_path,
        kind,
        "shuffled",
    )
    comparison_data = prepare_comparison(
        real_results,
        shuffled_results,
        categories,
        sections,
    )

    output_stem = args.output_dir / (
        "{0}_{1}_real_vs_shuffled".format(
            clean_filename(comparison_key),
            kind,
        )
    )
    comparison_csv = output_stem.with_name(
        output_stem.name + "_values"
    ).with_suffix(".csv")
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_data.to_csv(
        comparison_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print("Saved: {0}".format(comparison_csv.resolve()))

    figure, output_paths = draw_comparison_figure(
        comparison_data,
        comparison_key,
        kind,
        output_stem,
        args.formats,
        args.dpi,
    )
    return {
        "comparison": comparison_key,
        "feature_kind": kind,
        "real_input_csv": str(real_path.resolve()),
        "shuffled_input_csv": str(shuffled_path.resolve()),
        "comparison_values_csv": str(comparison_csv.resolve()),
        "figure_files": " | ".join(
            str(path) for path in output_paths
        ),
    }, figure


def main():
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    if args.all_comparisons:
        jobs = [
            (comparison_key, kind)
            for comparison_key in COMPARISON_SPECS
            for kind in ("theme", "sentiment")
        ]
    else:
        jobs = [(args.comparison, args.kind)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    open_figures = []

    for comparison_key, kind in jobs:
        print("\n" + "=" * 80)
        print(
            "COMPARISON: {0} / {1}".format(
                comparison_key,
                kind,
            )
        )
        print("=" * 80)
        manifest_row, figure = generate_one(
            comparison_key,
            kind,
            args,
        )
        manifest_rows.append(manifest_row)
        open_figures.append(figure)

        if not args.show:
            plt.close(figure)
            open_figures = []

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "comparison_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(
        "DONE: generated {0} real-versus-shuffled figure(s)".format(
            len(manifest_rows)
        )
    )
    print("Manifest: {0}".format(manifest_path.resolve()))
    print("=" * 80)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
