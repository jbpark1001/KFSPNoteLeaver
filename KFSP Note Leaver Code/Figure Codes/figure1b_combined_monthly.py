# -*- coding: utf-8 -*-
"""Figure 2b: monthly note/non-note counts and note-leaver proportion.

This replaces the two separate figures in the original plotting code with one
designer-style figure:

* non-leaver and note-leaver bars share the same zero baseline;
* the bars are placed side by side so neither category is hidden;
* the note-leaver proportion is shown on a secondary right axis;
* only odd-numbered months are labelled;
* year separators, year labels, a legend, and horizontal guides are included.

NOTE_EX codes follow the original analysis:
    1 = note-leaver
    2 = non-leaver
Other codes (including 88) are excluded from both counts and the proportion.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = Path(r"G:\KFSP\Raw Data\KFSPdatacopy.xlsx")
START_YEAR = 2013
END_YEAR = 2020

# Keep the codebook mapping explicit. These constants are used for the table,
# bar heights, colors, legend, and proportion calculation.
NOTE_LEAVER_CODE = 1
NON_LEAVER_CODE = 2

SCRIPT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)
OUTPUT_PREFIX = (
    SCRIPT_DIR.parent
    / "Figure Images"
    / "Final Ver"
    / "figure2b_combined_monthly"
)

PANEL_LABEL = "c"
FIGSIZE = (19.8, 5.4)
DPI = 600
SHOW_FIGURE = True

NON_NOTE_COLOR = "#C7C7C7"
NOTE_COLOR = "#7EA9D6"
LINE_COLOR = "#1F4E7D"
GRID_COLOR = "#D9D9D9"
BAR_WIDTH = 0.38

COUNT_TICK_INTERVAL = 200
PREFERRED_PERCENT_LIMITS = (30, 50)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def build_monthly_table(kfsp, start_year=START_YEAR, end_year=END_YEAR):
    """Return one row per month with counts and the note-leaver proportion."""
    required = {"SUICIDE_DATE", "NOTE_EX"}
    missing = required.difference(kfsp.columns)
    if missing:
        raise ValueError(
            "KFSP data are missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = kfsp.loc[:, ["SUICIDE_DATE", "NOTE_EX"]].copy()

    # Converting through nullable integers safely handles Excel numeric dates
    # such as 20130101.0 without creating a trailing ".0" in the text.
    date_numbers = pd.to_numeric(data["SUICIDE_DATE"], errors="coerce")
    date_strings = date_numbers.round().astype("Int64").astype(str)
    data["date"] = pd.to_datetime(
        date_strings,
        format="%Y%m%d",
        errors="coerce",
    )
    data["NOTE_EX"] = pd.to_numeric(data["NOTE_EX"], errors="coerce")

    invalid_dates = int(data["date"].isna().sum())
    if invalid_dates:
        raise ValueError(
            f"{invalid_dates:,} SUICIDE_DATE values could not be parsed."
        )

    start = pd.Timestamp(start_year, 1, 1)
    end = pd.Timestamp(end_year, 12, 31)
    in_period = data["date"].between(start, end)
    valid_note_code = data["NOTE_EX"].isin(
        [NOTE_LEAVER_CODE, NON_LEAVER_CODE]
    )

    excluded_codes = (
        data.loc[in_period & ~valid_note_code, "NOTE_EX"]
        .value_counts(dropna=False)
        .sort_index()
    )
    if not excluded_codes.empty:
        print("Excluded NOTE_EX codes (not used in counts or denominator):")
        print(excluded_codes.to_string())

    data = data.loc[in_period & valid_note_code].copy()
    data["year_month"] = data["date"].dt.to_period("M")

    full_months = pd.period_range(
        start=f"{start_year}-01",
        end=f"{end_year}-12",
        freq="M",
    )
    counts = (
        data.groupby(["year_month", "NOTE_EX"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=full_months, fill_value=0)
        .reindex(
            columns=[NOTE_LEAVER_CODE, NON_LEAVER_CODE],
            fill_value=0,
        )
    )

    monthly = pd.DataFrame(
        {
            "year_month": full_months.astype(str),
            "year": full_months.year,
            "month": full_months.month,
            "note_leavers": counts[NOTE_LEAVER_CODE].to_numpy(
                dtype=int
            ),
            "non_leavers": counts[NON_LEAVER_CODE].to_numpy(
                dtype=int
            ),
        }
    )
    monthly["eligible_total"] = (
        monthly["note_leavers"] + monthly["non_leavers"]
    )
    monthly["note_leaver_proportion"] = np.where(
        monthly["eligible_total"] > 0,
        monthly["note_leavers"] / monthly["eligible_total"],
        np.nan,
    )

    print(
        "\nVerified NOTE_EX mapping:"
        f"\n  Blue note-leavers (code {NOTE_LEAVER_CODE}): "
        f"{monthly['note_leavers'].sum():,}"
        f"\n  Gray non-leavers (code {NON_LEAVER_CODE}): "
        f"{monthly['non_leavers'].sum():,}"
    )
    return monthly


# =============================================================================
# PLOTTING
# =============================================================================

def _count_axis_max(values, interval=COUNT_TICK_INTERVAL):
    """Add headroom while retaining 200-count horizontal guide spacing."""
    maximum = float(np.nanmax(values))
    half_interval = interval / 2
    return max(interval, int(np.ceil(maximum / half_interval) * half_interval))


def _percent_axis_limits(percent_values):
    """Keep the designer's 30-50% range unless the data require more room."""
    finite = np.asarray(percent_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return PREFERRED_PERCENT_LIMITS

    lower = min(
        PREFERRED_PERCENT_LIMITS[0],
        5 * np.floor((finite.min() - 1) / 5),
    )
    upper = max(
        PREFERRED_PERCENT_LIMITS[1],
        5 * np.ceil((finite.max() + 1) / 5),
    )
    return float(lower), float(upper)


def plot_combined_monthly(
    monthly,
    output_prefix=OUTPUT_PREFIX,
    panel_label=PANEL_LABEL,
    show=SHOW_FIGURE,
):
    """Draw and save the combined count/proportion figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    required = {
        "year_month",
        "year",
        "month",
        "note_leavers",
        "non_leavers",
        "note_leaver_proportion",
    }
    missing = required.difference(monthly.columns)
    if missing:
        raise ValueError(
            "Monthly table is missing columns: "
            + ", ".join(sorted(missing))
        )

    x = np.arange(len(monthly))
    note = monthly["note_leavers"].to_numpy(dtype=float)
    non_note = monthly["non_leavers"].to_numpy(dtype=float)
    proportion = monthly["note_leaver_proportion"].to_numpy(dtype=float)

    with plt.rc_context(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.subplots_adjust(
            left=0.055,
            right=0.940,
            bottom=0.250,
            top=0.680,
        )

        # Side-by-side placement keeps both full bar heights visible. Gray is
        # the larger non-leaver group; blue is the smaller note-leaver group.
        ax.bar(
            x - BAR_WIDTH / 2,
            non_note,
            width=BAR_WIDTH,
            color=NON_NOTE_COLOR,
            edgecolor="none",
            label="Non-leavers",
            zorder=2,
        )
        ax.bar(
            x + BAR_WIDTH / 2,
            note,
            width=BAR_WIDTH,
            color=NOTE_COLOR,
            edgecolor="none",
            label="Note-leavers",
            zorder=3,
        )

        count_max = _count_axis_max(np.concatenate([note, non_note]))
        ax.set_ylim(0, count_max)
        ax.yaxis.set_major_locator(MultipleLocator(COUNT_TICK_INTERVAL))
        ax.grid(
            axis="y",
            color=GRID_COLOR,
            linewidth=0.8,
            alpha=0.85,
            zorder=0,
        )
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=12, length=0, pad=8)

        # Show 1, 3, 5, 7, 9, and 11 only, as in the designer's revision.
        odd_month_mask = monthly["month"].mod(2).eq(1).to_numpy()
        tick_positions = x[odd_month_mask]
        tick_labels = monthly.loc[odd_month_mask, "month"].astype(str)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=12)
        ax.tick_params(axis="x", length=0, pad=10)
        ax.set_xlim(-1.6, len(monthly) - 0.2)

        # Year dividers extend below the axis to separate the year labels.
        year_starts = np.flatnonzero(monthly["month"].eq(1).to_numpy())
        for start_position in year_starts[1:]:
            ax.axvline(
                start_position - 0.5,
                ymin=-0.29,
                ymax=1.03,
                color="#555555",
                linewidth=0.8,
                clip_on=False,
                zorder=1,
            )

        for year, group in monthly.groupby("year", sort=True):
            center = (group.index.min() + group.index.max()) / 2
            ax.text(
                center,
                -0.215,
                str(int(year)),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=14,
                clip_on=False,
            )

        # The right axis carries the proportion line.
        ax_right = ax.twinx()
        ax_right.set_zorder(ax.get_zorder() + 1)
        ax_right.patch.set_visible(False)
        ax_right.plot(
            x,
            proportion,
            color=LINE_COLOR,
            marker="o",
            markersize=4.6,
            markerfacecolor=LINE_COLOR,
            markeredgecolor=LINE_COLOR,
            linewidth=1.8,
            label="Proportion of Note Leavers",
            zorder=5,
        )

        percent_values = proportion * 100
        percent_lower, percent_upper = _percent_axis_limits(percent_values)
        ax_right.set_ylim(percent_lower / 100, percent_upper / 100)
        ax_right.yaxis.set_major_locator(MultipleLocator(0.05))
        ax_right.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=0)
        )
        ax_right.tick_params(axis="y", labelsize=12, length=0, pad=6)

        # Keep only the axes lines visible in the reference.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["left"].set_visible(False)
        ax_right.spines["bottom"].set_visible(False)

        # Horizontal headings make both scales easy to identify.
        ax.text(
            -0.028,
            1.105,
            "Monthly Suicide Counts (per Suicide)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
        )
        ax_right.text(
            1.015,
            1.105,
            "Proportion of Note Leavers (%)",
            transform=ax_right.transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
        )

        fig.text(
            0.027,
            0.915,
            panel_label,
            ha="left",
            va="top",
            fontsize=28,
            fontweight="bold",
        )

        legend_handles = [
            Patch(
                facecolor=NON_NOTE_COLOR,
                edgecolor="none",
                label="Non-leavers",
            ),
            Patch(
                facecolor=NOTE_COLOR,
                edgecolor="none",
                label="Note-leavers",
            ),
            Line2D(
                [0],
                [0],
                color=LINE_COLOR,
                marker="o",
                markersize=5,
                linewidth=1.8,
                label="Proportion of Note Leavers",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.016, 1.725),
            frameon=True,
            fancybox=False,
            framealpha=1,
            edgecolor="black",
            borderpad=0.9,
            labelspacing=0.7,
            handlelength=2.2,
            fontsize=10,
        )

        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        png_path = output_prefix.with_suffix(".png")
        svg_path = output_prefix.with_suffix(".svg")
        csv_path = output_prefix.parent / (
            output_prefix.name + "_values.csv"
        )

        fig.savefig(
            png_path,
            dpi=DPI,
            facecolor="white",
        )
        fig.savefig(
            svg_path,
            facecolor="white",
        )
        monthly.to_csv(csv_path, index=False, encoding="utf-8-sig")

        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved values: {csv_path}")
    return png_path, svg_path, csv_path


def main():
    print("Reading only SUICIDE_DATE and NOTE_EX from the KFSP workbook...")
    kfsp = pd.read_excel(
        DATA_PATH,
        usecols=["SUICIDE_DATE", "NOTE_EX"],
    )
    monthly = build_monthly_table(kfsp)

    print("\nMonthly values used in the figure:")
    display = monthly.copy()
    display["note_leaver_percent"] = (
        100 * display["note_leaver_proportion"]
    ).round(1)
    print(
        display[
            [
                "year_month",
                "note_leavers",
                "non_leavers",
                "note_leaver_percent",
            ]
        ].to_string(index=False)
    )

    plot_combined_monthly(monthly)


if __name__ == "__main__":
    main()
