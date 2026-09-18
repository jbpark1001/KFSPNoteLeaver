# -*- coding: utf-8 -*-
"""Figure 1: pooled annual suicide rates and raw counts side by side.

Left panel:
    Pooled annual suicide rate per 100,000, 2013–2020.

Right panel:
    Cumulative raw suicide count, 2013–2020.

Both panels use the same map geometry, muted-blue palette, label layout,
leader-line style, typography, and bottom-caption design.
"""

from pathlib import Path
import re
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

YEAR_START = 2013
YEAR_END = 2020

KFSP_PATH = Path(r"G:\KFSP\Raw Data\KFSPdatacopy.xlsx")
POPULATION_PATH = Path(
    r"C:\Users\Jae Bin Park\Desktop"
    r"\행정구역_시군구_별__성별_인구수_20250828123758.xlsx"
)
SHP_PATH = Path(r"C:\Users\Jae Bin Park\kr_shp\kr.shp")
SHP_NAME_FIELD = "name"

SCRIPT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)
OUT_PREFIX = SCRIPT_DIR / "figure1_rate_and_raw_counts_side_by_side"

# The reduced canvas width is paired with tighter map-axis padding below.
# Together these remove white space while retaining approximately the same
# physical geographic-map size and unchanged font sizes.
FIGSIZE = (21, 10)
DPI = 600

LEFT_PANEL_LABEL = "a"
RIGHT_PANEL_LABEL = "b"

LABEL_FONTSIZE = 14
COLORBAR_FONTSIZE = 13
CAPTION_FONTSIZE = 17
PANEL_LABEL_FONTSIZE = 28

# Protected space between the left colorbar and the right map panel.
CENTER_GUTTER_RATIO = 1.5

# Relative GridSpec widths. MAP_COLUMN_RATIO is the main map-size control:
# increasing it makes each map larger without changing the text size.
MAP_COLUMN_RATIO = 40
COLORBAR_COLUMN_RATIO = 1

# Horizontal positions of the external-label columns, expressed relative to
# the full geographic width. Larger positive values on the left and more
# negative values on the right move labels inward and shorten leader lines.
LEFT_CALLOUT_X_OFFSET = 0.080
RIGHT_CALLOUT_X_OFFSET = -0.060

# Empty data-coordinate space around each map. These are deliberately tighter
# than the original 0.37/0.28 settings because the callout columns now sit
# closer to the map. Reducing these values zooms the geography within its axis.
MAP_PAD_LEFT = 0.250
MAP_PAD_RIGHT = 0.180
MAP_PAD_BOTTOM = 0.025
MAP_PAD_TOP = 0.045


# =============================================================================
# REGION NAME HARMONIZATION
# =============================================================================

CANON_SIDO = [
    "서울특별시",
    "부산광역시",
    "대구광역시",
    "인천광역시",
    "광주광역시",
    "대전광역시",
    "울산광역시",
    "세종특별자치시",
    "경기도",
    "강원도",
    "충청북도",
    "충청남도",
    "전라북도",
    "전라남도",
    "경상북도",
    "경상남도",
    "제주특별자치도",
]

REGION_MAP = {
    "강원특별자치도": "강원도",
    "전북특별자치도": "전라북도",
    "서울시": "서울특별시",
    "부산시": "부산광역시",
    "대구시": "대구광역시",
    "인천시": "인천광역시",
    "광주시": "광주광역시",
    "대전시": "대전광역시",
    "울산시": "울산광역시",
    "세종시": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전라북도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
    "제주도": "제주특별자치도",
}

HANGUL_TO_EN = {
    "서울특별시": "Seoul",
    "부산광역시": "Busan",
    "대구광역시": "Daegu",
    "인천광역시": "Incheon",
    "광주광역시": "Gwangju",
    "대전광역시": "Daejeon",
    "울산광역시": "Ulsan",
    "세종특별자치시": "Sejong",
    "경기도": "Gyeonggi",
    "강원도": "Gangwon",
    "충청북도": "North Chungcheong",
    "충청남도": "South Chungcheong",
    "전라북도": "North Jeolla",
    "전라남도": "South Jeolla",
    "경상북도": "North Gyeongsang",
    "경상남도": "South Gyeongsang",
    "제주특별자치도": "Jeju",
}


def standardize_region(raw):
    """Convert a raw Korean region label to one of the 17 canonical names."""
    if pd.isna(raw):
        return np.nan

    region = str(raw).strip()
    region = re.sub(r"\s*\(.*?\)\s*$", "", region)
    region = re.sub(
        r"(?:\s*계|\s*합계|\s*총계)$",
        "",
        region,
    ).strip()
    region = REGION_MAP.get(region, region)

    if region not in CANON_SIDO:
        for canonical_region in CANON_SIDO:
            if region.startswith(canonical_region):
                return canonical_region

    return region


# =============================================================================
# PREPARE THE COMMON REGION-YEAR DATA
# =============================================================================

def build_annual_counts(
    kfsp,
    region_column="FIND_SIDO",
    year_column="YEAR",
):
    """Return one raw suicide count for each region and study year."""
    required = {region_column, year_column}
    missing = required.difference(kfsp.columns)
    if missing:
        raise ValueError(
            "KFSP data are missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = kfsp[[region_column, year_column]].rename(
        columns={
            region_column: "region",
            year_column: "year",
        }
    )
    data["region"] = data["region"].map(standardize_region)
    data["year"] = pd.to_numeric(
        data["year"],
        errors="coerce",
    )
    data = data.dropna(
        subset=["region", "year"],
    ).copy()
    data["year"] = data["year"].astype(int)
    data = data.loc[
        data["region"].isin(CANON_SIDO)
        & data["year"].between(YEAR_START, YEAR_END)
    ]

    annual_counts = (
        data.groupby(
            ["region", "year"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "deaths"})
    )

    # Make absent region-year combinations explicit rather than dropping them.
    complete_index = pd.MultiIndex.from_product(
        [
            CANON_SIDO,
            range(YEAR_START, YEAR_END + 1),
        ],
        names=["region", "year"],
    )
    annual_counts = (
        annual_counts.set_index(["region", "year"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    annual_counts["deaths"] = annual_counts["deaths"].astype(int)
    return annual_counts


def find_population_region_column(numbers):
    target = "행정구역(시군구)별"
    for column in numbers.columns:
        if str(column).strip() == target:
            return column
    raise ValueError(
        f"Population workbook must contain a '{target}' column."
    )


def population_year_columns(numbers):
    """Return {Excel column: integer year} for total-population columns."""
    result = {}
    for column in numbers.columns:
        text = str(column).strip()
        try:
            year = int(text)
        except ValueError:
            continue

        if YEAR_START <= year <= YEAR_END:
            # Columns named 2013, 2014, etc. are total populations.
            # Excel's 2013.1/2013.2 columns contain male/female populations.
            result[column] = year

    if not result:
        raise ValueError(
            f"No population columns from {YEAR_START} to "
            f"{YEAR_END} were found."
        )
    return result


def build_annual_populations(numbers):
    """Convert the wide population workbook to region-year totals."""
    region_source_column = find_population_region_column(numbers)
    year_column_map = population_year_columns(numbers)

    population = numbers.rename(
        columns={region_source_column: "region"}
    ).copy()
    population["region"] = population["region"].map(
        standardize_region
    )
    population = population.melt(
        id_vars=["region"],
        value_vars=list(year_column_map),
        var_name="source_year",
        value_name="population",
    )
    population["year"] = population["source_year"].map(
        year_column_map
    )
    population["population"] = pd.to_numeric(
        population["population"],
        errors="coerce",
    )
    population = population.dropna(
        subset=["region", "year", "population"],
    )
    population = population.loc[
        population["region"].isin(CANON_SIDO)
    ]
    population = (
        population.groupby(
            ["region", "year"],
            as_index=False,
        )["population"]
        .sum()
    )
    return population


def calculate_combined_summary(annual_counts, annual_populations):
    """Calculate both plotted metrics from the same annual count table."""
    # Raw totals must remain independent of population-data availability.
    raw_summary = (
        annual_counts.groupby("region", as_index=False)
        .agg(total_deaths=("deaths", "sum"))
    )

    merged = annual_populations.merge(
        annual_counts,
        on=["region", "year"],
        how="left",
        validate="one_to_one",
    )
    merged["deaths"] = merged["deaths"].fillna(0).astype(int)
    merged = merged.loc[merged["population"] > 0].copy()
    merged["annual_rate_per_100k"] = (
        merged["deaths"]
        / merged["population"]
        * 100_000
    )

    rate_summary = (
        merged.groupby("region", as_index=False)
        .agg(
            rate_period_deaths=("deaths", "sum"),
            sum_population=("population", "sum"),
            mean_rate_per_100k_2013_2020=(
                "annual_rate_per_100k",
                "mean",
            ),
            number_of_years=("year", "nunique"),
        )
    )
    rate_summary["pooled_rate_per_100k_2013_2020"] = (
        rate_summary["rate_period_deaths"]
        / rate_summary["sum_population"]
        * 100_000
    )
    summary = raw_summary.merge(
        rate_summary.drop(columns="rate_period_deaths"),
        on="region",
        how="left",
        validate="one_to_one",
    )
    summary["region_en"] = summary["region"].map(HANGUL_TO_EN)

    expected_years = YEAR_END - YEAR_START + 1
    incomplete = summary.loc[
        summary["number_of_years"] != expected_years,
        ["region", "number_of_years"],
    ]
    if not incomplete.empty:
        warnings.warn(
            "Some regions do not contain all study years:\n"
            + incomplete.to_string(index=False),
            stacklevel=2,
        )

    if len(summary) != len(CANON_SIDO):
        warnings.warn(
            f"Expected {len(CANON_SIDO)} regions but calculated "
            f"{len(summary)}.",
            stacklevel=2,
        )

    missing_names = summary.loc[
        summary["region_en"].isna(),
        "region",
    ].tolist()
    if missing_names:
        raise ValueError(
            "Missing English shapefile names for: "
            + ", ".join(map(str, missing_names))
        )

    return merged, summary


def load_combined_map(summary):
    """Merge both metrics into one projected GeoDataFrame."""
    map_data = gpd.read_file(SHP_PATH)

    if SHP_NAME_FIELD not in map_data.columns:
        raise ValueError(
            f"Shapefile does not contain '{SHP_NAME_FIELD}'. "
            f"Available columns: {list(map_data.columns)}"
        )

    if map_data.crs is None:
        warnings.warn(
            "Shapefile has no CRS; assuming WGS84 (EPSG:4326).",
            stacklevel=2,
        )
        map_data = map_data.set_crs(epsg=4326)

    map_data = map_data.to_crs(epsg=5179)
    metric_columns = [
        "region_en",
        "total_deaths",
        "pooled_rate_per_100k_2013_2020",
        "mean_rate_per_100k_2013_2020",
    ]
    map_data = map_data.merge(
        summary[metric_columns],
        left_on=SHP_NAME_FIELD,
        right_on="region_en",
        how="left",
        validate="one_to_one",
    )

    unmatched = map_data.loc[
        map_data["total_deaths"].isna(),
        SHP_NAME_FIELD,
    ].dropna().tolist()
    if unmatched:
        warnings.warn(
            "Shapefile regions without calculated values: "
            + ", ".join(map(str, unmatched)),
            stacklevel=2,
        )

    return map_data


# =============================================================================
# SHARED DESIGN AND LABEL POSITIONS
# =============================================================================

INSIDE_REGIONS = {
    "Seoul",
    "Gyeonggi",
    "Gangwon",
    "North Chungcheong",
    "Daejeon",
    "North Jeolla",
    "South Jeolla",
    "North Gyeongsang",
    "South Gyeongsang",
    "Jeju",
}

# (side, vertical displacement as a fraction of the map height)
CALLOUT_REGIONS = {
    "Incheon": ("left", 0.025),
    "South Chungcheong": ("left", 0.000),
    # Keep Sejong above South Chungcheong so their two-line labels and
    # leader lines remain separated and follow the anchors' geographic order.
    "Sejong": ("left", 0.075),
    "Gwangju": ("left", 0.000),
    "Daegu": ("right", 0.030),
    "Ulsan": ("right", 0.000),
    "Busan": ("right", -0.015),
}

# (horizontal, vertical) fractions of the unpadded map bounds
INSIDE_OFFSETS = {
    "Seoul": (-0.008, 0.028),
    "Gyeonggi": (0.015, -0.055),
    "Gangwon": (-0.010, 0.040),
    "North Chungcheong": (-0.005, 0.020),
    "Daejeon": (0.022, -0.018),
    "North Jeolla": (-0.015, 0.015),
    "South Jeolla": (0.005, -0.010),
    "North Gyeongsang": (-0.010, 0.010),
    "South Gyeongsang": (0.015, -0.005),
    "Jeju": (0.000, -0.002),
}

DISPLAY_NAMES = {
    "North Chungcheong": "North\nChungcheong",
    "South Chungcheong": "South Chungcheong",
    "North Gyeongsang": "North\nGyeongsang",
    "South Gyeongsang": "South Gyeongsang",
    "North Jeolla": "North Jeolla",
    "South Jeolla": "South Jeolla",
}

EXPECTED_REGIONS = INSIDE_REGIONS | set(CALLOUT_REGIONS)

QUIET_BLUES = mpl.colors.LinearSegmentedColormap.from_list(
    "quiet_blues",
    [
        "#dce8f8",
        "#a8c7e8",
        "#6e9fd0",
        "#376da5",
        "#183d68",
    ],
)


def contrasting_text_color(fill_rgba):
    red, green, blue = fill_rgba[:3]
    luminance = (
        0.2126 * red
        + 0.7152 * green
        + 0.0722 * blue
    )
    return "white" if luminance < 0.57 else "#0b0b0b"


def finite_limits(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("The selected panel metric has no finite values.")

    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.05, 0.5)
        vmin -= padding
        vmax += padding
    return vmin, vmax


def draw_map_panel(
    ax,
    colorbar_ax,
    map_data,
    metric_column,
    value_formatter,
    colorbar_label,
    panel_label,
    colorbar_formatter=None,
):
    """Draw one map into existing axes using the shared layout."""
    available_regions = set(
        map_data.loc[
            map_data[metric_column].notna(),
            "region_en",
        ]
    )
    missing_regions = sorted(
        EXPECTED_REGIONS.difference(available_regions)
    )
    if missing_regions:
        warnings.warn(
            f"Panel {panel_label} has no value for: "
            + ", ".join(missing_regions),
            stacklevel=2,
        )

    vmin, vmax = finite_limits(map_data[metric_column])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    map_data.plot(
        column=metric_column,
        cmap=QUIET_BLUES,
        norm=norm,
        linewidth=0.55,
        edgecolor="white",
        legend=False,
        missing_kwds={
            "color": "#efefef",
            "edgecolor": "white",
        },
        ax=ax,
        zorder=1,
    )
    ax.set_axis_off()

    xmin, ymin, xmax, ymax = map_data.total_bounds
    map_width = xmax - xmin
    map_height = ymax - ymin

    # The large horizontal padding contains the external labels inside each
    # panel. The central GridSpec gutter then separates the two padded panels.
    ax.set_xlim(
        xmin - MAP_PAD_LEFT * map_width,
        xmax + MAP_PAD_RIGHT * map_width,
    )
    ax.set_ylim(
        ymin - MAP_PAD_BOTTOM * map_height,
        ymax + MAP_PAD_TOP * map_height,
    )
    ax.set_aspect("equal")

    plotted_rows = map_data.loc[
        map_data[metric_column].notna()
    ]
    for _, row in plotted_rows.iterrows():
        region = row["region_en"]
        value = float(row[metric_column])
        display_name = DISPLAY_NAMES.get(region, region)
        label = f"{display_name}\n{value_formatter(value)}"

        anchor = row.geometry.centroid
        anchor_x = anchor.x
        anchor_y = anchor.y

        if region in CALLOUT_REGIONS:
            side, vertical_shift = CALLOUT_REGIONS[region]
            if side == "left":
                text_x = (
                    xmin
                    + LEFT_CALLOUT_X_OFFSET * map_width
                )
                horizontal_alignment = "right"
            else:
                text_x = (
                    xmax
                    + RIGHT_CALLOUT_X_OFFSET * map_width
                )
                horizontal_alignment = "left"

            text_y = anchor_y + vertical_shift * map_height
            ax.scatter(
                anchor_x,
                anchor_y,
                s=34,
                color="black",
                linewidth=0,
                zorder=5,
            )
            ax.annotate(
                label,
                xy=(anchor_x, anchor_y),
                xytext=(text_x, text_y),
                xycoords="data",
                textcoords="data",
                ha=horizontal_alignment,
                va="center",
                fontsize=LABEL_FONTSIZE,
                linespacing=1.15,
                color="black",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "black",
                    "linewidth": 1.2,
                    "shrinkA": 8,
                    "shrinkB": 2,
                },
                annotation_clip=False,
                zorder=6,
            )
        else:
            offset_x, offset_y = INSIDE_OFFSETS.get(
                region,
                (0.0, 0.0),
            )
            fill_color = QUIET_BLUES(norm(value))
            ax.text(
                anchor_x + offset_x * map_width,
                anchor_y + offset_y * map_height,
                label,
                ha="center",
                va="center",
                fontsize=LABEL_FONTSIZE,
                linespacing=1.15,
                color=contrasting_text_color(fill_color),
                zorder=5,
            )

    scalar_mappable = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=QUIET_BLUES,
    )
    scalar_mappable.set_array([])
    colorbar = ax.figure.colorbar(
        scalar_mappable,
        cax=colorbar_ax,
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.yaxis.set_major_locator(
        MaxNLocator(nbins=8, integer=True)
    )
    if colorbar_formatter is not None:
        colorbar.ax.yaxis.set_major_formatter(colorbar_formatter)
    colorbar.ax.tick_params(
        labelsize=COLORBAR_FONTSIZE,
        length=5,
        width=1,
        color="black",
    )
    colorbar.set_label(
        colorbar_label,
        fontsize=COLORBAR_FONTSIZE,
        labelpad=11,
    )

    ax.text(
        0.015,
        0.985,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        clip_on=False,
        zorder=10,
    )


def plot_combined_figure(
    map_data,
    output_prefix=OUT_PREFIX,
    show=True,
):
    """Draw the pooled-rate map left and raw-count map right."""
    with mpl.rc_context(
        {
            "font.family": "Arial",
            "font.size": LABEL_FONTSIZE,
            "axes.linewidth": 0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

        # map | colorbar | protected center gutter | map | colorbar
        grid = fig.add_gridspec(
            nrows=1,
            ncols=5,
            width_ratios=(
                MAP_COLUMN_RATIO,
                COLORBAR_COLUMN_RATIO,
                CENTER_GUTTER_RATIO,
                MAP_COLUMN_RATIO,
                COLORBAR_COLUMN_RATIO,
            ),
            left=0.020,
            right=0.980,
            bottom=0.135,
            top=0.965,
            wspace=0.045,
        )

        rate_ax = fig.add_subplot(grid[0, 0])
        rate_colorbar_ax = fig.add_subplot(grid[0, 1])
        gutter_ax = fig.add_subplot(grid[0, 2])
        count_ax = fig.add_subplot(grid[0, 3])
        count_colorbar_ax = fig.add_subplot(grid[0, 4])
        gutter_ax.set_axis_off()

        draw_map_panel(
            ax=rate_ax,
            colorbar_ax=rate_colorbar_ax,
            map_data=map_data,
            metric_column="pooled_rate_per_100k_2013_2020",
            value_formatter=lambda value: f"{value:.2f}",
            colorbar_label="Rate (per 100k/yr)",
            panel_label=LEFT_PANEL_LABEL,
            colorbar_formatter=StrMethodFormatter("{x:,.0f}"),
        )
        draw_map_panel(
            ax=count_ax,
            colorbar_ax=count_colorbar_ax,
            map_data=map_data,
            metric_column="total_deaths",
            value_formatter=lambda value: f"{int(round(value)):,}",
            colorbar_label=(
                f"Number of suicides, {YEAR_START}–{YEAR_END}"
            ),
            panel_label=RIGHT_PANEL_LABEL,
            colorbar_formatter=StrMethodFormatter("{x:,.0f}"),
        )

        # Equal-aspect map axes can become shorter than their GridSpec cells.
        # Realign each colorbar vertically to the map it describes.
        fig.canvas.draw()
        for map_ax, colorbar_ax in (
            (rate_ax, rate_colorbar_ax),
            (count_ax, count_colorbar_ax),
        ):
            map_box = map_ax.get_position()
            colorbar_box = colorbar_ax.get_position()
            colorbar_ax.set_position(
                [
                    colorbar_box.x0,
                    map_box.y0,
                    colorbar_box.width,
                    map_box.height,
                ]
            )

        # Recalculate positions, then center each caption beneath its complete
        # block (map plus its own aligned colorbar).
        fig.canvas.draw()
        rate_box = rate_ax.get_position()
        rate_colorbar_box = rate_colorbar_ax.get_position()
        count_box = count_ax.get_position()
        count_colorbar_box = count_colorbar_ax.get_position()

        rate_center = (
            rate_box.x0 + rate_colorbar_box.x1
        ) / 2
        count_center = (
            count_box.x0 + count_colorbar_box.x1
        ) / 2

        fig.text(
            rate_center,
            0.058,
            (
                f"{LEFT_PANEL_LABEL}. "
                "Pooled Annual Suicide Rate per 100,000 "
                f"({YEAR_START}–{YEAR_END})"
            ),
            ha="center",
            va="center",
            fontsize=CAPTION_FONTSIZE,
            fontweight="bold",
        )
        fig.text(
            count_center,
            0.058,
            (
                f"{RIGHT_PANEL_LABEL}. "
                "Total Number of Suicides by Region "
                f"({YEAR_START}–{YEAR_END})"
            ),
            ha="center",
            va="center",
            fontsize=CAPTION_FONTSIZE,
            fontweight="bold",
        )

        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        png_path = (
            output_prefix.parent
            / f"{output_prefix.name}.png"
        )
        svg_path = (
            output_prefix.parent
            / f"{output_prefix.name}.svg"
        )
        pdf_path = (
            output_prefix.parent
            / f"{output_prefix.name}.pdf"
        )

        fig.savefig(
            png_path,
            dpi=DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            svg_path,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )

        if show:
            plt.show()
        plt.close(fig)

    print("\nCombined figure files saved:")
    print(f"  PNG: {png_path}")
    print(f"  SVG: {svg_path}")
    print(f"  PDF: {pdf_path}")
    return png_path, svg_path, pdf_path


# =============================================================================
# COMPLETE WORKFLOW
# =============================================================================

def main():
    print("Reading KFSP case-level data...")
    kfsp = pd.read_excel(
        KFSP_PATH,
        usecols=["FIND_SIDO", "YEAR"],
    )

    print("Reading population data...")
    numbers = pd.read_excel(POPULATION_PATH)

    print("Calculating both regional metrics...")
    annual_counts = build_annual_counts(kfsp)
    annual_populations = build_annual_populations(numbers)
    merged_annual, summary = calculate_combined_summary(
        annual_counts,
        annual_populations,
    )

    verification_columns = [
        "region_en",
        "total_deaths",
        "pooled_rate_per_100k_2013_2020",
    ]
    print("\nValues used in the two panels:")
    print(
        summary[verification_columns]
        .sort_values(
            "pooled_rate_per_100k_2013_2020",
            ascending=False,
        )
        .to_string(index=False)
    )

    # Save the shared source values for figure verification.
    annual_csv = (
        OUT_PREFIX.parent
        / f"{OUT_PREFIX.name}_annual_data.csv"
    )
    summary_csv = (
        OUT_PREFIX.parent
        / f"{OUT_PREFIX.name}_regional_summary.csv"
    )
    merged_annual.to_csv(
        annual_csv,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nLoading the shapefile once for both panels...")
    map_data = load_combined_map(summary)

    print("Drawing the side-by-side figure...")
    plot_combined_figure(map_data)


if __name__ == "__main__":
    main()
