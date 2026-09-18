# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:48:32 2026

@author: Jae Bin Park
"""


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as fm
from pathlib import Path

kfsp=pd.read_excel(r'G:\KFSP\Raw Data\KFSPdatacopy.xlsx')
regioncountsforeachyear=kfsp.groupby("FIND_SIDO")["YEAR"].value_counts()
numbers=pd.read_excel(r"C:\Users\Jae Bin Park\Desktop\행정구역_시군구_별__성별_인구수_20250828123758.xlsx")


#%%


"""Figure 1a. This is the final code for the Geographic Map in the KFSP paper""" 

# =========================================================
# CONFIG
# =========================================================
YEAR_START, YEAR_END = 2013, 2020
YEARS = list(range(YEAR_START, YEAR_END + 1))

# Choose metric for the map:
#   "pooled_rate_per_100k_2013_2020"  (Σ deaths / Σ population × 100k)  ← recommended for multi-year panel
#   "mean_rate_per_100k_2013_2020"    (mean of annual crude rates)
CHOROPLETH_METRIC = "pooled_rate_per_100k_2013_2020"

# Shapefile path & name field (English names that match HANGUL_TO_EN mapping)
SHP_PATH = r"C:\Users\Jae Bin Park\kr_shp\kr.shp"
SHP_NAME_FIELD = "name"   # change if your shapefile uses a different column (e.g., "NAME_1")

# Optional font (or set to None)
FONT_PATH = r"C:\Windows\Fonts\Arial.ttf"

OUT_PREFIX = f"kr_suicide_{CHOROPLETH_METRIC}_2013_2020"

# =========================================================
# Canonical regions & name harmonization
# =========================================================
CANON_SIDO = [
    "서울특별시","부산광역시","대구광역시","인천광역시","광주광역시","대전광역시","울산광역시",
    "세종특별자치시","경기도","강원도","충청북도","충청남도","전라북도","전라남도",
    "경상북도","경상남도","제주특별자치도"
]

REGION_MAP = {
    # upgrades / official renames
    "강원특별자치도": "강원도",
    "전북특별자치도": "전라북도",
    # common alternates
    "서울시":"서울특별시","부산시":"부산광역시","대구시":"대구광역시","인천시":"인천광역시",
    "광주시":"광주광역시","대전시":"대전광역시","울산시":"울산광역시","세종시":"세종특별자치시",
    "경기":"경기도","강원":"강원도","충남":"충청남도","충북":"충청북도","전남":"전라남도",
    "전북":"전라북도","경남":"경상남도","경북":"경상북도","제주도":"제주특별자치도","제주":"제주특별자치도"
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

def std_region(raw) -> str:
    """Standardize region label to a canonical Si/Do. Also collapses '서울특별시 종로구' → '서울특별시'."""
    if pd.isna(raw):
        return raw
    s = str(raw).strip()
    # Remove trailing parenthetical or counts like "(계)", "합계", "총계"
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    s = re.sub(r"(?:\s*계|\s*합계|\s*총계)$", "", s).strip()
    # Direct map first
    if s in REGION_MAP:
        s = REGION_MAP[s]
    # If still not canonical, try prefix collapse: pick the first canonical Si/Do that matches the start of the string
    if s not in CANON_SIDO:
        for sido in CANON_SIDO:
            if s.startswith(sido):
                s = sido
                break
    return s

def detect_year_and_region_cols(df: pd.DataFrame, deaths_col: str = "deaths"):
    """Given a DataFrame from Series.reset_index(), robustly identify which columns are YEAR and REGION."""
    # candidates = all non-deaths columns
    candidates = [c for c in df.columns if c != deaths_col]
    if len(candidates) < 2:
        raise ValueError("Not enough index columns found to detect (region, year).")

    # Score columns by how much they look like years in [YEAR_START, YEAR_END]
    def year_score(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce")
        mask = vals.between(YEAR_START, YEAR_END)
        return float(mask.mean())

    scores = {c: year_score(df[c]) for c in candidates}
    # pick the best "year-like" column
    year_col = max(scores, key=scores.get)
    if scores[year_col] == 0.0:
        # Try name-based heuristics if values aren't numeric-looking
        for c in candidates:
            name = str(c).lower()
            if any(tok in name for tok in ["year", "연도", "년도"]):
                year_col = c
                break
        else:
            raise ValueError(f"Could not detect a year column among {candidates}.")
    # region column is the other one (prefer one with more unique strings)
    region_candidates = [c for c in candidates if c != year_col]
    region_col = max(region_candidates, key=lambda c: df[c].astype(str).nunique())
    return region_col, year_col

# =========================================================
# 1) Build deaths DataFrame from your Series `regioncountsforeachyear`
# =========================================================
s = regioncountsforeachyear.copy()

# If index holds tuples but isn't a MultiIndex, convert
if not isinstance(s.index, pd.MultiIndex) and s.index.dtype == 'object':
    try:
        tuples = list(s.index)
        if tuples and isinstance(tuples[0], tuple) and len(tuples[0]) == 2:
            s.index = pd.MultiIndex.from_tuples(tuples)
    except Exception:
        pass

deaths = s.to_frame("deaths").reset_index()
# If the reset_index() produced columns named like ['level_0','level_1','deaths'] or original names,
# detect which is year vs region:
reg_col, yr_col = detect_year_and_region_cols(deaths, deaths_col="deaths")
deaths = deaths.rename(columns={reg_col: "region", yr_col: "year"})
deaths["region"] = deaths["region"].map(std_region)
deaths["year"]   = pd.to_numeric(deaths["year"], errors="coerce").astype("Int64")
deaths["deaths"] = pd.to_numeric(deaths["deaths"], errors="coerce").fillna(0).astype(int)
deaths = (
    deaths[deaths["year"].between(YEAR_START, YEAR_END)]
    .groupby(["region","year"], as_index=False)["deaths"].sum()
)

# =========================================================
# 2) Populations from `numbers` (wide 2013..2020), region in "행정구역(시군구)별"
# =========================================================
num = numbers.copy()
if "행정구역(시군구)별" not in num.columns:
    raise ValueError("`numbers` must contain a '행정구역(시군구)별' column.")

num = num.rename(columns={"행정구역(시군구)별": "region"})
num["region"] = num["region"].map(std_region)

# Detect year columns present
year_cols = [c for c in num.columns if str(c).isdigit() and YEAR_START <= int(str(c)) <= YEAR_END]
year_cols = sorted(year_cols, key=lambda x: int(str(x)))
if not year_cols:
    raise ValueError("No population year columns (2013..2020) found in `numbers`.")

# Wide → long + aggregate to Si/Do in case lower units are present
pop_long = num.melt(id_vars=["region"], value_vars=year_cols,
                    var_name="year", value_name="population")
pop_long["year"] = pop_long["year"].astype(int)
pop_long["population"] = pd.to_numeric(pop_long["population"], errors="coerce")
pop_long = pop_long.dropna(subset=["population"])
pop_long = pop_long.groupby(["region","year"], as_index=False)["population"].sum()

# Keep canonical regions only
pop_long = pop_long[pop_long["region"].isin(CANON_SIDO)].copy()
deaths   = deaths[deaths["region"].isin(CANON_SIDO)].copy()

# =========================================================
# 3) Merge & compute rates
# =========================================================
merged = pop_long.merge(deaths, on=["region","year"], how="left")
merged["deaths"] = merged["deaths"].fillna(0).astype(int)
merged = merged[merged["population"] > 0].copy()
merged["rate_per_100k"] = (merged["deaths"] / merged["population"]) * 100_000

# Summaries per region across 2013–2020
summary = (
    merged.groupby("region", as_index=False)
          .agg(total_deaths=("deaths","sum"),
               sum_population=("population","sum"),
               mean_rate_per_100k_2013_2020=("rate_per_100k","mean"))
)
summary["pooled_rate_per_100k_2013_2020"] = summary["total_deaths"] / summary["sum_population"] * 100_000
summary["region_en"] = summary["region"].map(HANGUL_TO_EN)

# Diagnostics: check mapping completeness
if summary["region_en"].isna().any():
    missing = summary.loc[summary["region_en"].isna(), "region"].unique().tolist()
    print("WARNING: Missing English names for shapefile merge:", missing)

# =========================================================
# 4) Choropleth map (publication-grade)
# =========================================================
# Load & project shapefile
gdf = gpd.read_file(SHP_PATH)
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)
gdf = gdf.to_crs(epsg=5179)

# Merge metrics into shapes
gdfm = gdf.merge(
    summary[["region_en","pooled_rate_per_100k_2013_2020","mean_rate_per_100k_2013_2020","total_deaths"]],
    left_on=SHP_NAME_FIELD, right_on="region_en", how="left"
)

# Labels
if CHOROPLETH_METRIC == "pooled_rate_per_100k_2013_2020":
    metric_label = "Suicides per 100,000 per year (pooled 2013–2020)"
    title = "Pooled Annual Suicide Rate per 100,000 (2013–2020)"
elif CHOROPLETH_METRIC == "mean_rate_per_100k_2013_2020":
    metric_label = "Mean of annual suicide rates per 100,000 (2013–2020)"
    title = "Mean of Annual Suicide Rates per 100,000 (2013–2020)"
else:
    raise ValueError("Invalid CHOROPLETH_METRIC.")

# Font
fontprop = fm.FontProperties(fname=FONT_PATH) if FONT_PATH and Path(FONT_PATH).exists() else None


# --- DROP-IN: layout-safe plotting helpers with wrapping & column widths ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import textwrap

def _build_table_rows(summary, metric_col, wrap_chars=18):
    """Return table rows [Region, Deaths, Rate] with wrapped region names."""
    tab = summary.assign(metric=summary[metric_col]).sort_values("metric", ascending=False).copy()
    tab["metric_disp"] = tab["metric"].map(lambda x: f"{x:,.2f}")
    tab["deaths_disp"] = tab["total_deaths"].map(lambda x: f"{x:,}")
    # Wrap only the region column
    def wrap_str(s):
        s = "" if s is None else str(s)
        return "\n".join(textwrap.wrap(s, width=wrap_chars, break_long_words=False, break_on_hyphens=True)) or s
    regions_wrapped = [wrap_str(r) for r in tab["region_en"]]
    rows = [[r, d, m] for r, d, m in zip(regions_wrapped, tab["deaths_disp"], tab["metric_disp"])]
    return rows

def plot_map_with_table_right(
    gdfm, summary, metric_col, title, metric_label,
    fontprop=None, out_prefix="figure_right",
    figsize=(11.5, 8.5),         # wider canvas for the table
    width_ratios=(3.2, 1.7),     # give table more room
    col_widths=(0.62, 0.2, 0.18),# table col widths (sum ~= 1.0)
    wrap_chars=18,               # wrap region names at ~18 chars
    top_n=None                   # optionally show only top N rows
):
    fig = plt.figure(figsize=figsize, dpi=600, constrained_layout=False)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=width_ratios, wspace=0.02)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1]); ax_tbl.axis("off")

    # --- Map
    vals = gdfm[metric_col]
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    gdfm.plot(column=metric_col, cmap="Reds", linewidth=0.4, edgecolor="black",
              legend=False, ax=ax_map, norm=norm)
    ax_map.set_axis_off()
    if fontprop is not None:
        ax_map.set_title(title, fontsize=12, fontproperties=fontprop, pad=12)
    else:
        ax_map.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Colorbar
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="2.6%", pad=0.04)  # a bit slimmer
    sm = mpl.cm.ScalarMappable(cmap="Reds", norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    if fontprop is not None:
        cb.set_label(metric_label, fontsize=10, fontproperties=fontprop)
        for t in cb.ax.get_yticklabels():
            t.set_fontproperties(fontprop); t.set_fontsize(8)
    else:
        cb.set_label(metric_label, fontsize=10); cb.ax.tick_params(labelsize=8)

    # --- Table (right)
    rows = _build_table_rows(summary, metric_col, wrap_chars=wrap_chars)
    if top_n is not None:
        rows = rows[:top_n]
    col_labels = ["Region", "Deaths (Σ 2013–2020)", "Rate"]

    # Create table with explicit column widths (prevents cutting)
    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=list(col_widths)  # width fraction of the axes
    )
    tbl.auto_set_font_size(False)
    for (r, c), cell in tbl.get_celld().items():
        fs = 9 if r == 0 else 9          # header 7, body 6
        cell.get_text().set_fontsize(fs)
    
    # Determine dynamic sizes
    nrows = len(rows) + 1  # header row + data
    base_fs = 8 if nrows <= 22 else 7 if nrows <= 30 else 6

    # Compute per-row heights based on wrapped line count in the first column
    # More lines -> taller row
    header_height = 0.09
    base_row_h   = 0.05 if nrows <= 22 else 0.045 if nrows <= 30 else 0.04

    # Collect line counts for each data row
    line_counts = [r[0].count("\n") + 1 for r in rows]
    max_lines = max(line_counts) if line_counts else 1

    # Apply sizes & wrapping; left-align text
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("none")
        txt = cell.get_text()
        if fontprop is not None:
            txt.set_fontproperties(fontprop)
        txt.set_fontsize(base_fs)
        txt.set_ha("left"); txt.set_va("center")
        if r == 0:
            cell.set_height(header_height)
        else:
            # Increase row height proportional to number of wrapped lines in first column
            lc = line_counts[r-1]  # r-1 because r=0 is header
            cell.set_height(base_row_h * (0.9 + 0.45*(lc-1)/max(1, max_lines-1)))

    # Save
    png = f"{out_prefix}.png"; svg = f"{out_prefix}.svg"
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.show()
    print(f"Saved: {png}\nSaved: {svg}")

def plot_map_with_table_bottom(
    gdfm, summary, metric_col, title, metric_label,
    fontprop=None, out_prefix="figure_bottom",
    figsize=(8.5, 11.0),
    height_ratios=(3.0, 1.7),
    col_widths=(0.35, 0.25, 0.20),
    wrap_chars=20,
    top_n=None
):
    fig = plt.figure(figsize=figsize, dpi=600, constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=height_ratios, hspace=0.04)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0]); ax_tbl.axis("off")

    # --- Map
    vals = gdfm[metric_col]
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    gdfm.plot(column=metric_col, cmap="Reds", linewidth=0.4, edgecolor="black",
              legend=False, ax=ax_map, norm=norm)
    ax_map.set_axis_off()
    if fontprop is not None:
        ax_map.set_title(title, fontsize=12, fontproperties=fontprop, pad=12)
    else:
        ax_map.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Colorbar
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="2.8%", pad=0.04)
    sm = mpl.cm.ScalarMappable(cmap="Reds", norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    if fontprop is not None:
        cb.set_label(metric_label, fontsize=10, fontproperties=fontprop)
        for t in cb.ax.get_yticklabels():
            t.set_fontproperties(fontprop); t.set_fontsize(8)
    else:
        cb.set_label(metric_label, fontsize=10); cb.ax.tick_params(labelsize=8)

    # --- Table (bottom)
    rows = _build_table_rows(summary, metric_col, wrap_chars=wrap_chars)
    if top_n is not None:
        rows = rows[:top_n]
    col_labels = ["Region", "Deaths (Σ 2013–2020)", "Rate (per 100k/yr)"]

    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       cellLoc="left", colLoc="left", loc="upper left",
                       colWidths=list(col_widths))
    tbl.auto_set_font_size(False)

    nrows = len(rows) + 1
    base_fs = 8 if nrows <= 26 else 7 if nrows <= 34 else 6
    header_height = 0.10
    base_row_h   = 0.06 if nrows <= 26 else 0.052 if nrows <= 34 else 0.045

    line_counts = [r[0].count("\n") + 1 for r in rows]
    max_lines = max(line_counts) if line_counts else 1

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("none")
        txt = cell.get_text()
        if fontprop is not None:
            txt.set_fontproperties(fontprop)
        txt.set_fontsize(base_fs)
        txt.set_ha("left"); txt.set_va("center")
        if r == 0:
            cell.set_height(header_height)
        else:
            lc = line_counts[r-1]
            cell.set_height(base_row_h * (0.9 + 0.45*(lc-1)/max(1, max_lines-1)))

    png = f"{out_prefix}.png"; svg = f"{out_prefix}.svg"
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.show()
    print(f"Saved: {png}\nSaved: {svg}")



plot_map_with_table_bottom(
    gdfm=gdfm,
    summary=summary,
    metric_col=CHOROPLETH_METRIC,
    title=title,
    metric_label=metric_label,
    fontprop=fontprop,
    out_prefix=f"{OUT_PREFIX}_bottom",
    col_widths=(0.24, 0.24, 0.18),
    wrap_chars=18
)

#%%
# ----------------------
# Map with LEFT inset table (rate-only), identical layout to your first script
# ----------------------

font_path = r'C:\Windows\Fonts\Arialbd.ttf'
fontprop = fm.FontProperties(fname=font_path)

def plot_map_with_left_inset_rate(
    gdfm, summary, metric_col, title, metric_label,
    fontprop=None,
    out_prefix="kr_suicide_map_left_inset",
    # match your earlier figure geometry
    figsize=(8, 10), dpi=600,
    # inset table position [x, y, w, h] in figure fractions
    inset_rect=(0.03, 0.40, 0.30, 0.28),
    # colorbar styling identical to the reference
    cbar_size="3%", cbar_pad=0.4,
    cmap="Reds"
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # --- Map
    vals = gdfm[metric_col]
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    gdfm.plot(column=metric_col, cmap=cmap, linewidth=0.4, edgecolor="black",
              legend=False, ax=ax, norm=norm)

    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontproperties=fontprop, pad=12) if fontprop \
        else ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # --- Colorbar (right), same look as reference
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    if fontprop is not None:
        cb.set_label(metric_label, fontsize=10, fontproperties=fontprop)
        for ticklab in cb.ax.get_yticklabels():
            ticklab.set_fontproperties(fontprop)
            ticklab.set_fontsize(8)
    else:
        cb.set_label(metric_label, fontsize=10)
        cb.ax.tick_params(labelsize=8)

    # --- LEFT inset table (Region + Rate only)
    # Sort by metric (descending) and format as requested
    table_data = (summary
                  .assign(metric=summary[metric_col])
                  .sort_values("metric", ascending=False)
                  [["region_en", "metric"]]
                  .copy())
    table_values = [[r, f"{m:,.2f}"] for r, m in zip(table_data["region_en"], table_data["metric"])]
    table_col_labels = ["Region", "Rate (per 100k/yr)"]

    table_ax = fig.add_axes(inset_rect)
    table_ax.axis("off")

    tbl = table_ax.table(
        cellText=table_values,
        colLabels=table_col_labels,
        cellLoc='left',
        colLoc='left',
        loc='center'
    )

    # Styling identical to your reference
    tbl.auto_set_font_size(False)
    table_font_size = 8  # tweak if needed

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('none')
        # tighter rows, slightly taller header
        cell.set_height(0.05 if row != 0 else 0.10)
        if fontprop is not None:
            cell.get_text().set_fontproperties(fontprop)
        cell.get_text().set_fontsize(table_font_size)

    # Save like before
    png = f"{out_prefix}.png"
    svg = f"{out_prefix}.svg"
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.show()
    print(f"Saved: {png}\nSaved: {svg}")


plot_map_with_left_inset_rate(
    gdfm=gdfm,
    summary=summary,
    metric_col=CHOROPLETH_METRIC,   # e.g., "pooled_rate_per_100k_2013_2020"
    title=title,
    metric_label=metric_label,
    fontprop=fontprop,
    out_prefix=f"{OUT_PREFIX}_left_inset"
)
#%%

# -*- coding: utf-8 -*-
"""
Figure 1a2. Geographic distribution of raw suicide counts in South Korea, 2013–2020.

The choropleth shading represents the cumulative number of suicides
recorded in each region across 2013–2020. These values are raw counts,
not population-adjusted rates.
"""

# =========================================================
# IMPORTS
# =========================================================
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =========================================================
# CONFIGURATION
# =========================================================
YEAR_START = 2013
YEAR_END = 2020

DATA_PATH = r"G:\KFSP\Raw Data\KFSPdatacopy.xlsx"

# Administrative-region shapefile
SHP_PATH = r"C:\Users\Jae Bin Park\kr_shp\kr.shp"

# Column in the shapefile containing English region names
SHP_NAME_FIELD = "name"

# Font used in the figure
FONT_PATH = r"C:\Windows\Fonts\Arialbd.ttf"

# Figure output
OUT_PREFIX = "kr_suicide_raw_counts_2013_2020"

# Map appearance
CMAP = "Reds"
FIGSIZE = (8, 10)
DPI = 600

# Left inset table position:
# [left, bottom, width, height] in figure coordinates
INSET_RECT = (0.005, 0.37, 0.27, 0.34)


# =========================================================
# CANONICAL REGION NAMES
# =========================================================
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
    "제주특별자치도"
]


REGION_MAP = {
    # Official renamings
    "강원특별자치도": "강원도",
    "전북특별자치도": "전라북도",

    # Common shortened labels
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
    "제주도": "제주특별자치도"
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
    "제주특별자치도": "Jeju"
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def standardize_region(raw):
    """
    Convert a raw regional label into one of the 17 canonical
    South Korean province/metropolitan-city labels.

    Examples
    --------
    서울시 -> 서울특별시
    강원특별자치도 -> 강원도
    서울특별시 종로구 -> 서울특별시
    """
    if pd.isna(raw):
        return np.nan

    region = str(raw).strip()

    # Remove trailing parenthetical labels
    region = re.sub(r"\s*\(.*?\)\s*$", "", region)

    # Remove total-like suffixes
    region = re.sub(
        r"(?:\s*계|\s*합계|\s*총계)$",
        "",
        region
    ).strip()

    # Direct alternate-name conversion
    if region in REGION_MAP:
        region = REGION_MAP[region]

    # Collapse district-level names to the parent province/city
    if region not in CANON_SIDO:
        for canonical_region in CANON_SIDO:
            if region.startswith(canonical_region):
                region = canonical_region
                break

    return region


def check_required_columns(df, required_columns):
    """Raise an informative error if required variables are missing."""
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The KFSP dataset is missing the following required columns: "
            + ", ".join(missing_columns)
        )


def build_raw_count_summary(
    df,
    region_column="FIND_SIDO",
    year_column="YEAR",
    year_start=2013,
    year_end=2020
):
    """
    Calculate annual and cumulative suicide counts by region.

    Returns
    -------
    annual_counts : DataFrame
        One row per region-year.

    summary : DataFrame
        One row per region, containing cumulative counts for
        the complete study period.
    """
    check_required_columns(
        df,
        [region_column, year_column]
    )

    data = df[[region_column, year_column]].copy()

    data = data.rename(
        columns={
            region_column: "region",
            year_column: "year"
        }
    )

    data["region"] = data["region"].map(standardize_region)
    data["year"] = pd.to_numeric(
        data["year"],
        errors="coerce"
    )

    # Keep only valid region and year values
    data = data.dropna(
        subset=["region", "year"]
    ).copy()

    data["year"] = data["year"].astype(int)

    data = data[
        data["year"].between(year_start, year_end)
        & data["region"].isin(CANON_SIDO)
    ].copy()

    # Each row of the original dataset represents one suicide case.
    annual_counts = (
        data.groupby(
            ["region", "year"],
            as_index=False
        )
        .size()
        .rename(columns={"size": "deaths"})
    )

    # Create all possible region-year combinations so that missing
    # combinations are represented explicitly by zero.
    complete_index = pd.MultiIndex.from_product(
        [
            CANON_SIDO,
            range(year_start, year_end + 1)
        ],
        names=["region", "year"]
    )

    annual_counts = (
        annual_counts
        .set_index(["region", "year"])
        .reindex(
            complete_index,
            fill_value=0
        )
        .reset_index()
    )

    annual_counts["deaths"] = (
        annual_counts["deaths"]
        .astype(int)
    )

    summary = (
        annual_counts
        .groupby(
            "region",
            as_index=False
        )
        .agg(
            total_deaths=("deaths", "sum")
        )
    )

    summary["region_en"] = summary["region"].map(
        HANGUL_TO_EN
    )

    summary = summary.sort_values(
        "total_deaths",
        ascending=False
    ).reset_index(drop=True)

    return annual_counts, summary


def load_and_merge_shapefile(
    shapefile_path,
    shapefile_name_field,
    summary
):
    """
    Load the shapefile and attach cumulative regional counts.
    """
    gdf = gpd.read_file(shapefile_path)

    if shapefile_name_field not in gdf.columns:
        raise ValueError(
            f"The shapefile does not contain the field "
            f"'{shapefile_name_field}'.\n"
            f"Available fields are:\n{list(gdf.columns)}"
        )

    # Assign WGS84 only when the source file has no CRS metadata
    if gdf.crs is None:
        print(
            "WARNING: The shapefile has no CRS metadata. "
            "EPSG:4326 will be assigned."
        )
        gdf = gdf.set_crs(epsg=4326)

    # Korean 2000 / Unified CS
    gdf = gdf.to_crs(epsg=5179)

    map_data = gdf.merge(
        summary[
            [
                "region_en",
                "total_deaths"
            ]
        ],
        left_on=shapefile_name_field,
        right_on="region_en",
        how="left",
        validate="many_to_one"
    )

    # Show shapefile names that failed to match
    unmatched_shapes = map_data.loc[
        map_data["total_deaths"].isna(),
        shapefile_name_field
    ].dropna().unique().tolist()

    if unmatched_shapes:
        print(
            "\nWARNING: These shapefile region names did not match "
            "the KFSP summary:"
        )
        for name in unmatched_shapes:
            print(f"  - {name}")

    # Show summary regions that failed to match a polygon
    shapefile_names = set(
        map_data[shapefile_name_field]
        .dropna()
        .astype(str)
    )

    unmatched_summary = summary.loc[
        ~summary["region_en"].isin(shapefile_names),
        "region_en"
    ].dropna().tolist()

    if unmatched_summary:
        print(
            "\nWARNING: These KFSP regions did not match any "
            "shapefile polygon:"
        )
        for name in unmatched_summary:
            print(f"  - {name}")

    return map_data


def plot_raw_count_map(
    map_data,
    summary,
    shapefile_name_field,
    font_properties=None,
    output_prefix="kr_suicide_raw_counts_2013_2020",
    figsize=(8, 10),
    dpi=600,
    inset_rect=(0.02, 0.37, 0.33, 0.34),
    cmap="Reds"
):
    """
    Plot a publication-quality choropleth of cumulative raw counts
    with a ranked inset table.
    """
    metric_column = "total_deaths"

    title = "Total Number of Suicides by Region (2013–2020)"
    colorbar_label = "Number of suicides, 2013–2020"

    plot_values = map_data[metric_column].dropna()

    if plot_values.empty:
        raise ValueError(
            "No regional counts were successfully merged into the "
            "shapefile. Check SHP_NAME_FIELD and HANGUL_TO_EN."
        )

    vmin = float(plot_values.min())
    vmax = float(plot_values.max())

    if vmin == vmax:
        vmax = vmin + 1

    norm = mpl.colors.Normalize(
        vmin=vmin,
        vmax=vmax
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    # -----------------------------------------------------
    # Base map
    # -----------------------------------------------------
    map_data.plot(
        column=metric_column,
        cmap=cmap,
        linewidth=0.45,
        edgecolor="black",
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "black",
            "hatch": "///",
            "label": "Missing data"
        },
        legend=False,
        ax=ax,
        norm=norm
    )

    ax.set_axis_off()

    if font_properties is not None:
        ax.set_title(
            title,
            fontsize=13,
            fontproperties=font_properties,
            pad=14
        )
    else:
        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
            pad=14
        )

    # -----------------------------------------------------
    # Colorbar
    # -----------------------------------------------------
    divider = make_axes_locatable(ax)

    colorbar_axis = divider.append_axes(
        "right",
        size="3%",
        pad=0.35
    )

    scalar_mappable = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )
    scalar_mappable.set_array([])

    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_axis
    )

    colorbar.ax.yaxis.set_major_formatter(
        StrMethodFormatter("{x:,.0f}")
    )

    if font_properties is not None:
        colorbar.set_label(
            colorbar_label,
            fontsize=10,
            fontproperties=font_properties
        )

        for tick_label in colorbar.ax.get_yticklabels():
            tick_label.set_fontproperties(font_properties)
            tick_label.set_fontsize(8)
    else:
        colorbar.set_label(
            colorbar_label,
            fontsize=10
        )
        colorbar.ax.tick_params(
            labelsize=8
        )

    # -----------------------------------------------------
    # Ranked inset table
    # -----------------------------------------------------
    table_data = (
        summary[
            [
                "region_en",
                "total_deaths"
            ]
        ]
        .sort_values(
            "total_deaths",
            ascending=False
        )
        .copy()
    )

    table_values = [
        [
            region,
            f"{int(count):,}"
        ]
        for region, count in zip(
            table_data["region_en"],
            table_data["total_deaths"]
        )
    ]

    table_axis = fig.add_axes(inset_rect)
    table_axis.axis("off")

    table = table_axis.table(
        cellText=table_values,
        colLabels=[
            "Region",
            "Count"
        ],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.64, 0.36]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("none")
        cell.set_facecolor("none")

        text = cell.get_text()

        if font_properties is not None:
            text.set_fontproperties(font_properties)

        text.set_ha("left")
        text.set_va("center")

        if row == 0:
            cell.set_height(0.070)
            text.set_fontsize(7.5)
            text.set_weight("bold")
        else:
            cell.set_height(0.050)
            text.set_fontsize(7.5)

        # Right-align count values
        if column == 1:
            text.set_ha("right")

    # -----------------------------------------------------
    # Save figure
    # -----------------------------------------------------
    png_path = f"{output_prefix}.png"
    svg_path = f"{output_prefix}.svg"
    pdf_path = f"{output_prefix}.pdf"

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    plt.close(fig)

    print("\nFigure files saved:")
    print(f"  PNG: {png_path}")
    print(f"  SVG: {svg_path}")
    print(f"  PDF: {pdf_path}")


# =========================================================
# MAIN ANALYSIS
# =========================================================
def main():
    # -----------------------------------------------------
    # 1. Load KFSP case-level data
    # -----------------------------------------------------


    print(f"Total rows loaded: {len(kfsp):,}")

    # -----------------------------------------------------
    # 2. Calculate raw regional counts
    # -----------------------------------------------------
    annual_counts, summary = build_raw_count_summary(
        df=kfsp,
        region_column="FIND_SIDO",
        year_column="YEAR",
        year_start=YEAR_START,
        year_end=YEAR_END
    )

    print(
        f"\nCases included from {YEAR_START} through {YEAR_END}: "
        f"{summary['total_deaths'].sum():,}"
    )

    print("\nRaw counts by region:")
    print(
        summary[
            [
                "region_en",
                "total_deaths"
            ]
        ].to_string(
            index=False
        )
    )

    # Save the regional count tables for verification
    annual_counts.to_csv(
        f"{OUT_PREFIX}_annual_counts.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        f"{OUT_PREFIX}_regional_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # 3. Load and merge geographic polygons
    # -----------------------------------------------------
    print("\nLoading shapefile...")

    map_data = load_and_merge_shapefile(
        shapefile_path=SHP_PATH,
        shapefile_name_field=SHP_NAME_FIELD,
        summary=summary
    )

    # -----------------------------------------------------
    # 4. Load font
    # -----------------------------------------------------
    if (
        FONT_PATH is not None
        and Path(FONT_PATH).exists()
    ):
        font_properties = fm.FontProperties(
            fname=FONT_PATH
        )
    else:
        font_properties = None
        print(
            "\nWARNING: The specified font file was not found. "
            "Matplotlib's default font will be used."
        )

    # -----------------------------------------------------
    # 5. Plot cumulative raw-count map
    # -----------------------------------------------------
    plot_raw_count_map(
        map_data=map_data,
        summary=summary,
        shapefile_name_field=SHP_NAME_FIELD,
        font_properties=font_properties,
        output_prefix=OUT_PREFIX,
        figsize=FIGSIZE,
        dpi=DPI,
        inset_rect=INSET_RECT,
        cmap=CMAP
    )


# =========================================================
# RUN SCRIPT
# =========================================================
if __name__ == "__main__":
    main()
#%%
"""Figure 1bc. Final Code for Plotting Note Leaver and No Note Leaver Count Distribution across Years of Months
   + a second figure with the monthly proportion of note leavers as a line plot
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.ticker import PercentFormatter

import datetime

def season(x):
    if x in ['03','04','05']:
        return '1'
    elif x in ['06','07','08']:
        return '2'
    elif x in ['09','10','11']:
        return '3'
    elif x in ['12','01','02']:
        return '4'
    
    
kfsp['Suicide_Day'] = kfsp['SUICIDE_DATE'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d').weekday())
kfsp['Suicide_Year'] = kfsp['SUICIDE_DATE'].apply(lambda x: str(x)[0:4])
kfsp['Suicide_Month'] = kfsp['SUICIDE_DATE'].apply(lambda x: str(x)[4:6])
kfsp['Suicide_Season'] = kfsp['Suicide_Month'].apply(lambda x: season(x))
# Create 'Suicide_YearMonth' column in "YYYY-MM" format
kfsp['Suicide_YearMonth'] = kfsp['SUICIDE_DATE'].apply(lambda x: str(x)[0:4] + '-' + str(x)[4:6])
kfsp['Suicide_YearSeason'] = kfsp['Suicide_Year'] + '-' + kfsp['Suicide_Season']

# --- Helper: ensure a full border around an axes ---
def add_axes_border(ax, linewidth=1, color="black"):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(linewidth)
        ax.spines[side].set_edgecolor(color)
    ax.set_frame_on(True)

# Load font
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
font_path = r'C:\Windows\Fonts\Arialbd.ttf'

font_prop = fm.FontProperties(fname=font_path)

# Group by YearMonth and NOTE_EX only (no gender)
grouped = kfsp.groupby(['Suicide_YearMonth', 'NOTE_EX']).size()

# Sorted time periods
years = sorted(kfsp['Suicide_YearMonth'].unique())
x = np.arange(len(years))
width = 0.5

# Extract counts
note = [grouped.get((year, 1), 0) for year in years]      # NOTE_EX = 1 (left a note)
non_note = [grouped.get((year, 2), 0) for year in years]  # NOTE_EX = 2 (did not leave a note)

# Identify positions between Dec -> Jan for vertical lines
transition_indices = []
for i in range(len(years) - 1):
    if years[i].endswith('-12') and years[i + 1].endswith('-01'):
        transition_indices.append(i + 0.5)  # line between Dec and Jan

# Numeric month labels
month_labels = [ym[-2:] for ym in years]  # '01', '02', ...

# -------------------------
# FIGURE 1: Diverging bars (counts)
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# Ensure axes border is on for Figure 1
add_axes_border(ax, linewidth=1, color="black")
# (Optional) add a border around the whole figure canvas:
# fig.set_frameon(True)
# fig.patch.set_edgecolor("black")
# fig.patch.set_linewidth(1)

# Bars
ax.bar(x, note, width=width, label='Note Leaver', color='#2166AC', edgecolor='black')
ax.bar(x, [-v for v in non_note], width=width, label='Non Note Leaver', color='#BDBDBD', edgecolor='black')

# Vertical year divider lines
for xpos in transition_indices:
    ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1)

# Format x-axis with month ticks
ax.set_xticks(x)
ax.set_xticklabels(month_labels, fontproperties=font_prop, rotation=0, ha='center', fontsize=13)
ax.set_ylabel('Monthly Suicide Counts (per Suicide)', fontproperties=font_prop, fontsize=13)

# Add year labels manually under x-axis for Januarys
max_val = max(max(note) if note else 0, max(non_note) if non_note else 0)
min_y = -max_val * 1.25 if max_val > 0 else -1  # safe default if all zeros

for i, ym in enumerate(years):
    if ym.endswith('-01'):
        year = ym[:4]
        ax.text(i + 4, min_y, year, ha='left', va='center', fontproperties=font_prop, fontsize=14)

# Format y-axis as absolute values
yticks = ax.get_yticks()
ax.set_yticklabels([abs(int(t)) for t in yticks], fontproperties=font_prop, fontsize=13)

#ax.legend(prop=font_prop)
plt.tight_layout()
plt.show()

# -------------------------
# FIGURE 2: Line plot (proportion of note leavers each month)
# -------------------------

# Compute monthly proportion = note / (note + non_note)
note_arr = np.array(note, dtype=float)
non_note_arr = np.array(non_note, dtype=float)
total_arr = note_arr + non_note_arr

with np.errstate(divide='ignore', invalid='ignore'):
    note_ratio = np.where(total_arr > 0, note_arr / total_arr, np.nan)

fig2, ax2 = plt.subplots(figsize=(12, 4.5))

# Ensure axes border is on for Figure 2
add_axes_border(ax2, linewidth=1, color="black")
# (Optional) add a border around the whole figure canvas:
# fig2.set_frameon(True)
# fig2.patch.set_edgecolor("black")
# fig2.patch.set_linewidth(1)

# Line plot of proportions
ax2.plot(x, note_ratio, marker='o', linewidth=2, label='Note Leaver / Total')

# Vertical year divider lines (same as above)
for xpos in transition_indices:
    ax2.axvline(x=xpos, color='black', linestyle='--', linewidth=1)

# X-axis formatting
ax2.set_xticks(x)
ax2.set_xticklabels(month_labels, fontproperties=font_prop, rotation=0, ha='center', fontsize=13)

# Y-axis as percentage (0–100%)
ax2.set_ylim(0.3, 0.5)
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
for lab in ax2.get_yticklabels():
    lab.set_fontproperties(font_prop)
    lab.set_fontsize(12)

ax2.set_ylabel('Proportion of Note Leavers (%)', fontproperties=font_prop, fontsize=13)

# Optional: year labels below axis for Januarys (using axis coords for clean placement)
for i, ym in enumerate(years):
    if ym.endswith('-01'):
        year = ym[:4]
        ax2.text(i + 4, -0.1, year, ha='left', va='center',
                 fontproperties=font_prop, fontsize=13,
                 transform=ax2.get_xaxis_transform())

#ax2.legend(prop=font_prop)

plt.tight_layout()
plt.show()


"per Month"


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.ticker import PercentFormatter

# --- Helper: ensure a full border around an axes ---
def add_axes_border(ax, linewidth=1, color="black"):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(linewidth)
        ax.spines[side].set_edgecolor(color)
    ax.set_frame_on(True)

# Load font
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
font_prop = fm.FontProperties(fname=font_path)

# Group by YearMonth and NOTE_EX only (no gender)
grouped = kfsp.groupby(['Suicide_Month', 'NOTE_EX']).size()

# Sorted time periods
years = sorted(kfsp['Suicide_Month'].unique())
x = np.arange(len(years))
width = 0.5

# Extract counts
note = [grouped.get((year, 1), 0) for year in years]      # NOTE_EX = 1 (left a note)
non_note = [grouped.get((year, 2), 0) for year in years]  # NOTE_EX = 2 (did not leave a note)

# Identify positions between Dec -> Jan for vertical lines
transition_indices = []
for i in range(len(years) - 1):
    if years[i].endswith('-12') and years[i + 1].endswith('-01'):
        transition_indices.append(i + 0.5)  # line between Dec and Jan

# Numeric month labels
month_labels = [ym[-2:] for ym in years]  # '01', '02', ...

# -------------------------
# FIGURE 1: Diverging bars (counts)
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# Ensure axes border is on for Figure 1
add_axes_border(ax, linewidth=1, color="black")
# (Optional) add a border around the whole figure canvas:
# fig.set_frameon(True)
# fig.patch.set_edgecolor("black")
# fig.patch.set_linewidth(1)

# Bars
ax.bar(x, note, width=width, label='Note Leaver', color='#2166AC', edgecolor='black')
ax.bar(x, [-v for v in non_note], width=width, label='Non Note Leaver', color='#BDBDBD', edgecolor='black')

# Vertical year divider lines
for xpos in transition_indices:
    ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1)

# Format x-axis with month ticks
ax.set_xticks(x)
ax.set_xticklabels(month_labels, fontproperties=font_prop, rotation=0, ha='center', fontsize=10)
ax.set_ylabel('Monthly Suicide Counts (per Suicide)', fontproperties=font_prop, fontsize=10)

# Add year labels manually under x-axis for Januarys
max_val = max(max(note) if note else 0, max(non_note) if non_note else 0)
min_y = -max_val * 1.18 if max_val > 0 else -1  # safe default if all zeros

for i, ym in enumerate(years):
    if ym.endswith('-01'):
        year = ym[:4]
        ax.text(i + 4, min_y, year, ha='left', va='center', fontproperties=font_prop, fontsize=12)

# Format y-axis as absolute values
yticks = ax.get_yticks()
ax.set_yticklabels([abs(int(t)) for t in yticks], fontproperties=font_prop, fontsize=10)

#ax.legend(prop=font_prop)
plt.tight_layout()
plt.show()

# -------------------------
# FIGURE 2: Line plot (proportion of note leavers each month)
# -------------------------

# Compute monthly proportion = note / (note + non_note)
note_arr = np.array(note, dtype=float)
non_note_arr = np.array(non_note, dtype=float)
total_arr = note_arr + non_note_arr

with np.errstate(divide='ignore', invalid='ignore'):
    note_ratio = np.where(total_arr > 0, note_arr / total_arr, np.nan)

fig2, ax2 = plt.subplots(figsize=(12, 4.5))

# Ensure axes border is on for Figure 2
add_axes_border(ax2, linewidth=1, color="black")
# (Optional) add a border around the whole figure canvas:
# fig2.set_frameon(True)
# fig2.patch.set_edgecolor("black")
# fig2.patch.set_linewidth(1)

# Line plot of proportions
ax2.plot(x, note_ratio, marker='o', linewidth=2, label='Note Leaver / Total')

# Vertical year divider lines (same as above)
for xpos in transition_indices:
    ax2.axvline(x=xpos, color='black', linestyle='--', linewidth=1)

# X-axis formatting
ax2.set_xticks(x)
ax2.set_xticklabels(month_labels, fontproperties=font_prop, rotation=0, ha='center', fontsize=10)

# Y-axis as percentage (0–100%)
ax2.set_ylim(0.3, 0.5)
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
for lab in ax2.get_yticklabels():
    lab.set_fontproperties(font_prop)
    lab.set_fontsize(10)

ax2.set_ylabel('Proportion of Note Leavers', fontproperties=font_prop, fontsize=10)

# Optional: year labels below axis for Januarys (using axis coords for clean placement)
for i, ym in enumerate(years):
    if ym.endswith('-01'):
        year = ym[:4]
        ax2.text(i + 4, -0.08, year, ha='left', va='center',
                 fontproperties=font_prop, fontsize=12,
                 transform=ax2.get_xaxis_transform())

ax2.grid(True, axis='y', alpha=0.3)
#ax2.legend(prop=font_prop)

plt.tight_layout()
plt.show()

"""Correlation Test"""
note = [grouped.get((year, 1), 0) for year in years]
non_note = [grouped.get((year, 2), 0) for year in years]

from scipy.stats import pearsonr

r, p = pearsonr(note, non_note)
print(f"Pearson correlation: r = {r:.3f}, p = {p:.4f}")

from scipy.stats import spearmanr

rho, pval = spearmanr(note, non_note)
print(f"Spearman correlation: rho = {rho:.3f}, p = {pval:.4f}")

#%%
"""Figure 3 ab"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

# -----------------------------
# Font helpers
# -----------------------------
def _fontprop(font_path=None, size=11):
    return fm.FontProperties(fname=font_path, size=size) if font_path else None

def _maybe_set_global_font(font_path):
    """
    Registers the font and sets it as the default family so seaborn/mpl inherit it.
    Safe no-op if font_path is None or registration fails.
    """
    if not font_path:
        return
    try:
        fm.fontManager.addfont(font_path)
        name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = name
    except Exception:
        # If anything goes wrong, we simply don't override the global font.
        pass

def _apply_axis_font(ax, fp, size=11, xtick_size=None, ytick_size=None):
    """Apply FontProperties to labels, ticks, and legend on a given axis."""
    if fp is None:
        return
    ax.set_xlabel(ax.get_xlabel(), fontproperties=fp, fontsize=size-2)
    ax.set_ylabel(ax.get_ylabel(), fontproperties=fp, fontsize=size-2)

    xtick_size = size if xtick_size is None else xtick_size
    ytick_size = size if ytick_size is None else ytick_size

    for lab in ax.get_xticklabels():
        lab.set_fontproperties(fp)
        lab.set_fontsize(xtick_size)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(fp)
        lab.set_fontsize(ytick_size)

    leg = ax.get_legend()
    if leg is not None:
        leg.get_title().set_fontproperties(fp)
        leg.get_title().set_fontsize(size)
        for txt in leg.get_texts():
            txt.set_fontproperties(fp)
            txt.set_fontsize(size)


# -----------------------------
# Palettes
# -----------------------------
def build_palette(categories, scheme="npg20", include_other_gray=True):
    """
    Returns {category -> color} using a Nature/NHB-friendly discrete palette.
    - scheme="npg20": 20 curated colors (extends NPG tones).
    - include_other_gray=True: any category starting with 'Other' gets a neutral gray.
    """
    NPG20 = [
        # Core NPG tones
        "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
        "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
        # Extended but harmonized (distinct hues / luminance)
        "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
        "#CC79A7", "#EE6677", "#228833", "#9467BD", "#8C564B",
    ]

    PALETTES = {
        "npg20": NPG20,
    }

    base = PALETTES.get(scheme, NPG20)
    mapping = {}
    i = 0
    for cat in categories:
        if include_other_gray and str(cat).lower().startswith("other"):
            mapping[cat] = "#8C8C8C"  # neutral gray for "Other"
        else:
            mapping[cat] = base[i % len(base)]
            i += 1
    return mapping


# -----------------------------
# 2) Prep function: add labels & categories; choose metric
# -----------------------------
def prepare_word_table(df,
                       word_col="Word",
                       count_col="Count",
                       translation_dict=None,
                       category_map=None,
                       top_n=50,
                       proportion=True):
    d = df[[word_col, count_col]].copy()
    d[count_col] = pd.to_numeric(d[count_col], errors="coerce").fillna(0)

    if proportion:
        total = d[count_col].sum()
        d["value"] = (d[count_col] / total) if total > 0 else 0.0
        xlab = "Share of tokens"
        xfmt = PercentFormatter(1.0)
    else:
        d["value"] = d[count_col].astype(float)
        xlab = "Frequency"
        xfmt = None

    # --- English-only labels (fallback to the original token if no translation) ---
    if translation_dict:
        eng = d[word_col].map(translation_dict).fillna("")
        d["Label"] = np.where(eng != "", eng, d[word_col])
    else:
        d["Label"] = d[word_col]

    # Categories
    if category_map:
        d["Category"] = d[word_col].map(category_map).fillna("Other")
    else:
        d["Category"] = "Other"

    # Rank & take top_n overall
    d = d.sort_values("value", ascending=False).head(top_n).reset_index(drop=True)
    return d, xlab, xfmt


# -----------------------------
# 3) Panel A: Category composition (horizontal bar)
# -----------------------------
def plot_category_composition(df_plot, ax, palette, font_path=r'C:\Windows\Fonts\Arialbd.ttf', font_size=14):
    fp = _fontprop(font_path, font_size)

    comp = df_plot.groupby("Category", as_index=False)["value"].sum()
    comp = comp.sort_values("value", ascending=True)

    colors = [palette.get(c, (0.6,0.6,0.6)) for c in comp["Category"]]
    ax.barh(comp["Category"], comp["value"], color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Semantic category", fontproperties=fp, fontsize=12)

    for side in ("top","right","bottom","left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)

    # If values are proportions, format as %
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of tokens (%)", fontproperties=fp, fontsize=12)

    for lab in ax.get_xticklabels():
        lab.set_fontproperties(fp)
        lab.set_fontsize(10)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(fp)
        lab.set_fontsize(10)



    # Add value labels
    for y, v in enumerate(comp["value"]):
        ax.text(v, y, f"  {v:.2%}", va="center", ha="left",
                fontproperties=fp, fontsize=max(font_size, 10))


# -----------------------------
# 4) Panel B: Ranked lollipop plot (words colored by category)
# -----------------------------
def plot_ranked_lollipop(df_plot, ax, xlab, xfmt, palette,
                         font_path=r'C:\Windows\Fonts\Arialbd.ttf', font_size=12,
                         ytick_font_size=10, value_decimals=2):
    fp = _fontprop(font_path, font_size)

    d = df_plot.copy()
    d["y"] = np.arange(len(d))[::-1]

    # stems + markers
    ax.hlines(y=d["y"], xmin=0, xmax=d["value"], color="black", linewidth=1.0, zorder=2)
    ax.scatter(d["value"], d["y"],
               s=40,
               c=[palette.get(c, (0.6,0.6,0.6)) for c in d["Category"]],
               edgecolors="black", linewidths=0.6, zorder=2)

    # y-ticks: smaller than the rest
    if ytick_font_size is None:
        ytick_font_size = max(font_size - 2, 6)
    ax.set_yticks(d["y"])
    ax.set_yticklabels(d["Label"], fontproperties=fp, fontsize=12)

    ax.set_ylabel("", fontproperties=fp, fontsize=font_size)
    ax.set_xlabel("Share of tokens (%)", fontproperties=fp, fontsize=12)


    for lab in ax.get_xticklabels():
        lab.set_fontproperties(fp)
        lab.set_fontsize(10)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(fp)
        lab.set_fontsize(10)


    if xfmt is not None:
        ax.xaxis.set_major_formatter(xfmt)

    for side in ("top","right","bottom","left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)

    # value labels: match x-axis percent formatting
    use_percent = isinstance(xfmt, PercentFormatter)
    for _, row in d.iterrows():
        label = f"  {row['value']:.{value_decimals}%}" if use_percent else f"  {row['value']:.3f}"
        ax.text(row["value"], row["y"], label,
                va="center", ha="left", fontproperties=fp, fontsize=10)


# -----------------------------
# 5) Orchestrator: two-panel figure (legend shown in a SECOND figure)
# -----------------------------
def figure_words_with_categories(df,
                                 word_col="Word",
                                 count_col="Count",
                                 translation_dict=None,
                                 category_map=None,
                                 top_n=50,
                                 proportion=True,
                                 suptitle="Word usage by semantic category",
                                 font_path=r'C:\Windows\Fonts\Arialbd.ttf',
                                 font_size=14,
                                 set_global_font=True):
    """
    Same API/behavior as before EXCEPT the legend is drawn in a separate figure.
    Call pattern unchanged: fig = figure_words_with_categories(...); plt.show()
    """
    if set_global_font:
        _maybe_set_global_font(font_path)

    df_plot, xlab, xfmt = prepare_word_table(df,
                                             word_col=word_col,
                                             count_col=count_col,
                                             translation_dict=translation_dict,
                                             category_map=category_map,
                                             top_n=top_n,
                                             proportion=proportion)

    cats = list(pd.unique(df_plot["Category"]))
    palette = build_palette(cats)

    # Main figure (no legend on axes)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 4.6),
                                   gridspec_kw={"width_ratios":[1, 2]}, dpi=150)
    plot_category_composition(df_plot, axA, palette, font_path=font_path, font_size=font_size)
    plot_ranked_lollipop(df_plot, axB, xlab, xfmt, palette, font_path=font_path, font_size=font_size)

    # --- SECOND FIGURE: legend only (uses same categories & colors) ---
    handles = [Line2D([0],[0], marker="o", linestyle="",
                      markerfacecolor=palette.get(c, (0.6,0.6,0.6)),
                      markeredgecolor="black", label=c) for c in cats]
    fp_leg = _fontprop(font_path, max(font_size-1, 8))

    # Size adjusts with number of categories (single column to mirror previous look)
    legend_height = max(0.6, 0.35 * len(cats))  # tweak if you want tighter/looser legend fig
    fig_leg = plt.figure(figsize=(3.2, legend_height), dpi=150)
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")
    leg = ax_leg.legend(handles=handles,
                        title="Category",
                        loc="center left",
                        frameon=True,
                        borderaxespad=0.4,
                        prop=fp_leg)
    if leg and fp_leg:
        leg.get_title().set_fontproperties(fp_leg)

    # Main title (same as before)
    fig.suptitle(suptitle, fontproperties=_fontprop(font_path, font_size), fontsize=font_size)
    fig.tight_layout()
    return fig

whole_nng_counts_results=pd.read_pickle("C:/Users/Jae Bin Park/whole_nng_counts_results")
whole_vv_counts_results=pd.read_pickle("C:/Users/Jae Bin Park/whole_vv_counts_results")
# =========================
# Category definitions (brief, for Methods/Supplement)
# =========================
noun_category_defs = {
    "Family & Kinship": "Terms denoting relatives and kinship roles.",
    "Finance & Assets": "Money, debt, banking, and material assets.",
    "Health & Clinical": "Illness, treatment, health states, and substances.",
    "Affect & Values": "Positive affect/values (love, hope, courage, etc.).",
    "Distress/Suffering": "Negative affect, hardship, pain, tears, harm.",
    "Mortality & Funerary": "Death and funerary references.",
    "Moral/Accountability": "Sin, forgiveness, truth/lying, choice as agency.",
    "Communication & Contact": "Messages, contact, phone, suicide note as object.",
    "Time & Duration": "Day, time, moment, lifetime, end/last, age.",
    "Self & Identity": "Self-reference and personal appearance.",
    "Body & Physical": "Body parts and somatic references.",
    "Social Relations & People": "People, social ties, gender/role labels, isolation.",
    "Work & Roles": "Work, workplace, positions/roles.",
    "Place & Environment": "Home, landscape, sky, path.",
    "Objects & Technology": "Physical artifacts/devices/vehicles.",
    "Security & Access": "Credentials and access keys.",
    "Situation & Circumstance": "General problems/situations/methods.",
    "Life & Existence": "Life as a condition or life-course concept.",
    "Other/Ambiguous": "Polysemous or context-dependent items."
}

verb_category_defs = {
    "Apology & Forgiveness": "Apologizing, forgiving, gratitude/repair acts.",
    "Love & Attachment": "Expressions of love/attachment.",
    "Life & Death": "Living, dying, birth, suicide, survival.",
    "Requests & Wants": "Requests, wishes, wants, pleading.",
    "Communication": "Saying, informing, calling, contacting, relaying.",
    "Cognition & Memory": "Thinking, knowing, understanding, remembering/forgetting, belief.",
    "Perception & Discovery": "Seeing, hearing, showing, finding, being found.",
    "Giving & Transfer": "Giving, receiving, sending, leaving behind, offering.",
    "Caretaking & Support": "Helping, caring, protecting, accompanying, raising.",
    "Organization/Documentation/Resolution": "Writing, composing, handling, resolving, arranging, placing/disposal.",
    "Motion & Transition": "Going/coming/leaving/starting; movement and change.",
    "Persistence & Effort": "Enduring, trying, persevering.",
    "Decision & Choice": "Choosing, giving up.",
    "Work & Economy": "Working, buying, selling, repaying.",
    "Existence/State & Outcomes": "Existing, remaining, ability/inability, things going wrong, reduction.",
    "Emotion Expression": "Crying, grieving, feeling, resentment, exhaustion.",
    "Creation/Construction": "Making, building.",
    "Harm/Impact": "Hitting, causing (impact on others).",
    "Ritual/Funeral": "Cremation, burial, scattering ashes.",
    "Self-care & Daily Actions": "Eating, drinking, resting.",
    "Control/Causation": "Causing others to act.",
    "Social Interaction": "Meeting/relating directly.",
    "Generic Action": "Generic honorific 'do'.",
    "Other/Ambiguous": "Polysemous or context-dependent items."
}

# =========================
# Noun -> Category (primary assignment; one per lemma)
# =========================
noun_category_map = {
    # Family & Kinship
    "엄마":"Family & Kinship","아빠":"Family & Kinship","아들":"Family & Kinship","가족":"Family & Kinship",
    "형":"Family & Kinship","아버지":"Family & Kinship","딸":"Family & Kinship","누나":"Family & Kinship",
    "어머니":"Family & Kinship","동생":"Family & Kinship","부모":"Family & Kinship","오빠":"Family & Kinship",
    "언니":"Family & Kinship","남편":"Family & Kinship","아내":"Family & Kinship","형제":"Family & Kinship",
    "할머니":"Family & Kinship","식구":"Family & Kinship","아이":"Family & Kinship","어머님":"Family & Kinship",

    # Finance & Assets
    "돈":"Finance & Assets","빚":"Finance & Assets","통장":"Finance & Assets","재산":"Finance & Assets","카드":"Finance & Assets",

    # Health & Clinical
    "병원":"Health & Clinical","병":"Health & Clinical","우울증":"Health & Clinical","약":"Health & Clinical",
    "건강":"Health & Clinical","술":"Health & Clinical",

    # Affect & Values (positive)
    "마음":"Affect & Values","사랑":"Affect & Values","희망":"Affect & Values","행복":"Affect & Values",
    "용기":"Affect & Values","힘":"Affect & Values","맘":"Affect & Values",

    # Distress/Suffering (negative affect)
    "고생":"Distress/Suffering","고통":"Distress/Suffering","눈물":"Distress/Suffering","걱정":"Distress/Suffering",
    "피해":"Distress/Suffering","짐":"Distress/Suffering","상처":"Distress/Suffering",

    # Mortality & Funerary
    "죽음":"Mortality & Funerary","자살":"Mortality & Funerary","사망":"Mortality & Funerary",
    "장례":"Mortality & Funerary","장례식":"Mortality & Funerary","시신":"Mortality & Funerary",

    # Moral/Accountability
    "죄":"Moral/Accountability","용서":"Moral/Accountability","거짓말":"Moral/Accountability","선택":"Moral/Accountability","진심":"Moral/Accountability",

    # Communication & Contact
    "메시지":"Communication & Contact","연락":"Communication & Contact","전화":"Communication & Contact",
    "연락처":"Communication & Contact","유서":"Communication & Contact",

    # Time & Duration
    "시간":"Time & Duration","하루":"Time & Duration","날":"Time & Duration","순간":"Time & Duration",
    "평생":"Time & Duration","마지막":"Time & Duration","끝":"Time & Duration","나이":"Time & Duration",

    # Self & Identity
    "자신":"Self & Identity","모습":"Self & Identity",

    # Body & Physical
    "몸":"Body & Physical","가슴":"Body & Physical",

    # Social Relations & People
    "사람":"Social Relations & People","친구":"Social Relations & People","지인":"Social Relations & People",
    "남":"Social Relations & People","여자":"Social Relations & People","인간":"Social Relations & People",
    "혼자":"Social Relations & People","도움":"Social Relations & People",

    # Work & Roles
    "일":"Work & Roles","회사":"Work & Roles","자리":"Work & Roles",

    # Place & Environment
    "집":"Place & Environment","길":"Place & Environment","산":"Place & Environment","하늘":"Place & Environment",

    # Objects & Technology
    "차":"Objects & Technology","핸드폰":"Objects & Technology",

    # Security & Access
    "비밀번호":"Security & Access",

    # Situation & Circumstance
    "상황":"Situation & Circumstance","문제":"Situation & Circumstance","방법":"Situation & Circumstance",

    # Life & Existence
    "삶":"Life & Existence","인생":"Life & Existence",
}

# =========================
# Verb -> Category (primary assignment; one per lemma)
# =========================
verb_category_map = {
    # Apology & Forgiveness
    "미안하":"Apology & Forgiveness","죄송하":"Apology & Forgiveness","용서하":"Apology & Forgiveness","감사하":"Apology & Forgiveness",

    # Love & Attachment
    "사랑하":"Love & Attachment",

    # Life & Death
    "살":"Life & Death","살았":"Life & Death","살아가":"Life & Death","죽":"Life & Death","죽었":"Life & Death",
    "자살하":"Life & Death","태어나":"Life & Death","먼저가":"Life & Death",

    # Requests & Wants
    "부탁하":"Requests & Wants","부탁드리":"Requests & Wants","원하":"Requests & Wants","바라":"Requests & Wants","빌":"Requests & Wants",

    # Communication
    "말하":"Communication","연락하":"Communication","전화하":"Communication","전하":"Communication","알리":"Communication",

    # Cognition & Memory
    "생각하":"Cognition & Memory","알":"Cognition & Memory","알았":"Cognition & Memory","이해하":"Cognition & Memory",
    "믿":"Cognition & Memory","모르":"Cognition & Memory","모르겠":"Cognition & Memory","잊":"Cognition & Memory",

    # Perception & Discovery
    "보":"Perception & Discovery","보이":"Perception & Discovery","듣":"Perception & Discovery",
    "찾":"Perception & Discovery","발견되":"Perception & Discovery",

    # Giving & Transfer
    "주":"Giving & Transfer","드리":"Giving & Transfer","받":"Giving & Transfer","보내":"Giving & Transfer","남기":"Giving & Transfer",

    # Caretaking & Support
    "돕":"Caretaking & Support","보살피":"Caretaking & Support","챙기":"Caretaking & Support",
    "모시":"Caretaking & Support","지키":"Caretaking & Support","키우":"Caretaking & Support","위하":"Caretaking & Support",

    # Organization/Documentation/Resolution
    "쓰":"Organization/Documentation/Resolution","작성하":"Organization/Documentation/Resolution","적":"Organization/Documentation/Resolution",
    "정리하":"Organization/Documentation/Resolution","처리하":"Organization/Documentation/Resolution","해결하":"Organization/Documentation/Resolution",
    "두":"Organization/Documentation/Resolution","버리":"Organization/Documentation/Resolution",

    # Motion & Transition
    "가":"Motion & Transition","오":"Motion & Transition","떠나":"Motion & Transition",
    "나":"Motion & Transition","나오":"Motion & Transition","시작하":"Motion & Transition",
    "가었":"Motion & Transition","오았":"Motion & Transition",  # tolerate past-stem variants if present
    "가았":"Motion & Transition",  # as provided in your list

    # Persistence & Effort
    "버티":"Persistence & Effort","노력하":"Persistence & Effort",

    # Decision & Choice
    "선택하":"Decision & Choice","포기하":"Decision & Choice",

    # Work & Economy
    "일하":"Work & Economy","사":"Work & Economy","팔":"Work & Economy","갚":"Work & Economy",

    # Existence/State & Outcomes
    "있었":"Existence/State & Outcomes","남":"Existence/State & Outcomes","지내":"Existence/State & Outcomes",
    "안되":"Existence/State & Outcomes","잘못되":"Existence/State & Outcomes","인하":"Existence/State & Outcomes","가지":"Existence/State & Outcomes",
    "못하":"Existence/State & Outcomes","잘하":"Existence/State & Outcomes",

    # Emotion Expression
    "울":"Emotion Expression","느끼":"Emotion Expression","슬퍼하":"Emotion Expression","원망하":"Emotion Expression","지치":"Emotion Expression",

    # Creation/Construction
    "만들":"Creation/Construction","짓":"Creation/Construction",

    # Harm/Impact
    "치":"Harm/Impact","끼치":"Harm/Impact",

    # Ritual/Funeral
    "화장하":"Ritual/Funeral","묻":"Ritual/Funeral","뿌리":"Ritual/Funeral",

    # Self-care & Daily Actions
    "먹":"Self-care & Daily Actions","마시":"Self-care & Daily Actions","쉬":"Self-care & Daily Actions",

    # Control/Causation
    "시키":"Control/Causation",

    # Social Interaction
    "만나":"Social Interaction",

    # Generic Action
    "하시":"Generic Action",
}


korean_wholenouns_to_english = {
    "엄마": "Mom",
    "아빠": "Dad",
    "사람": "Person",
    "아들": "Son",
    "가족": "Family",
    "돈": "Money",
    "마음": "Heart",
    "일": "Work",
    "형": "Older brother",
    "아버지": "Father",
    "마지막": "Last",
    "딸": "Daughter",
    "집": "Home",
    "유서": "Will",
    "친구": "Friend",
    "누나": "Older sister (by male)",
    "어머니": "Mother",
    "생각": "Thought",
    "동생": "Younger sibling",
    "자신": "Oneself",
    "몸": "Body",
    "삶": "Life",
    "부탁": "Request",
    "인생": "Life (whole)",
    "부모": "Parents",
    "날": "Day",
    "끝": "End",
    "오빠": "Older brother (by female)",
    "길": "Path",
    "시간": "Time",
    "짐": "Burden",
    "자식": "Children",
    "고생": "Hardship",
    "언니": "Older sister (by female)",
    "빚": "Debt",
    "선택": "Choice",
    "고통": "Pain",
    "힘": "Strength",
    "모습": "Appearance",
    "통장": "Bank account",
    "죽음": "Death",
    "아이": "Child",
    "사랑": "Love",
    "남편": "Husband",
    "죄": "Sin",
    "용서": "Forgiveness",
    "아내": "Wife",
    "병원": "Hospital",
    "가슴": "Chest / Heart",
    "상처": "Wound / Hurt",
    "진심": "Sincerity",
    "장례": "Funeral",
    "형제": "Siblings",
    "메시지": "Message",
    "술": "Alcohol",
    "방법": "Method",
    "남": "Others / Male",
    "연락": "Contact",
    "장례식": "Funeral Ceremony",
    "자살": "Suicide",
    "전화": "Phone call",
    "하루": "Day",
    "할머니": "Grandmother",
    "병": "Illness",
    "사망": "Death (formal)",
    "도움": "Help",
    "지인": "Acquaintance",
    "피해": "Harm",
    "혼자": "Alone",
    "우울증": "Depression",
    "여자": "Woman",
    "순간": "Moment",
    "평생": "Lifetime",
    "어머님": "Mother (honorific)",
    "눈물": "Tears",
    "걱정": "Worry",
    "정신": "Mind",
    "희망": "Hope",
    "인간": "Human",
    "회사": "Company",
    "상황": "Situation",
    "재산": "Assets",
    "카드": "Card",
    "비밀번호": "Password",
    "문제": "Problem",
    "자리": "Position",
    "약": "Medicine",
    "연락처": "Contact info",
    "산": "Mountain",
    "행복": "Happiness",
    "차": "Car",
    "건강": "Health",
    "맘": "Heart (colloquial)",
    "거짓말": "Lie",
    "용기": "Courage",
    "핸드폰": "Cell phone",
    "식구": "Family member",
    "나이": "Age",
    "시신": "Corpse",
    "하늘": "Sky"
}

korean_verbs_to_english = {
    "미안하": "Be sorry",
    "사랑하": "Love",
    "살": "Live",
    "죄송하": "Apologize",
    "가": "Go",
    "죽": "Die",
    "부탁하": "Ask",
    "바라": "Hope",
    "보": "See",
    "생각하": "Think",
    "용서하": "Forgive",
    "받": "Receive",
    "알": "Know",
    "만나": "Meet",
    "못하": "Not do",
    "화장하": "Cremate",
    "안되": "Not work",
    "오": "Come",
    "뿌리": "Scatter",
    "쓰": "Write",
    "떠나": "Leave",
    "잘하": "Do well",
    "남기": "Leave behind",
    "주": "Give",
    "감사하": "Thank",
    "드리": "Offer",
    "먹": "Eat",
    "살았": "Have lived",
    "보내": "Send",
    "말하": "Say",
    "지키": "Keep",
    "보이": "Show",
    "지내": "Spend time",
    "믿": "Believe",
    "모르": "Not know",
    "모르겠": "Not know",
    "위하": "Care for",
    "잊": "Forget",
    "돕": "Help",
    "챙기": "Take care of",
    "연락하": "Contact",
    "사": "Buy",
    "전하": "Deliver",
    "알리": "Inform",
    "이해하": "Understand",
    "키우": "Raise",
    "태어나": "Be born",
    "갚": "Repay",
    "마시": "Drink",
    "처리하": "Handle",
    "찾": "Find",
    "슬퍼하": "Grieve",
    "나": "Come out",
    "정리하": "Organize",
    "인하": "Reduce",
    "가지": "Have",
    "선택하": "Choose",
    "나오": "Come out",
    "남": "Remain",
    "원망하": "Resent",
    "가았": "Have gone",
    "듣": "Hear",
    "대하": "Face",
    "만들": "Make",
    "먼저가": "Go first",
    "살아가": "Live on",
    "노력하": "Try",
    "시키": "Make someone do",
    "전화하": "Call",
    "알았": "Have known",
    "있었": "Have existed",
    "버리": "Throw away",
    "쉬": "Rest",
    "울": "Cry",
    "오았": "Have come",
    "포기하": "Give up",
    "모시": "Accompany",
    "잘못되": "Go wrong",
    "적": "Write down",
    "작성하": "Write",
    "죽었": "Have died",
    "끼치": "Cause",
    "두": "Put",
    "해결하": "Resolve",
    "치": "Hit",
    "보살피": "Take care of",
    "하시": "Do (honorific)",
    "자살하": "Commit suicide",
    "일하": "Work",
    "팔": "Sell",
    "느끼": "Feel",
    "부탁드리": "Humbly ask",
    "묻": "Bury",
    "짓": "Build",
    "원하": "Want",
    "지치": "Be exhausted",
    "시작하": "Start",
    "버티": "Endure",
    "빌": "Beg",
    "발견되": "Be found"
}


# List of words you want to remove
stopwords  = ['말', '이상', '애', '내용', '앞', '동안', '이제', '중략','밖','곳', '다음', '옆', '글', '이유', 
              '속', '모두', '문자', '이름', '눈', '전', '그동안', '변사자', '장', '후', "지금", "미안", "조금", "곁",
              "뒤", "이번", "처음", "잘못", "위", "나중", "그때", "층", '테', '당']
whole_cleanednng_counts_results = whole_nng_counts_results[~whole_nng_counts_results["Word"].isin(stopwords)].reset_index(drop=True)



# List of words you want to remove
stopverbs  = ["하","되","있","그러", "하았", "안하", "들", "되었", "그러었", "말", "잘살", "다하"]

mask = whole_vv_counts_results['Word'].isin(['모르', '모르겠'])
merged_count = whole_vv_counts_results.loc[mask, 'Count'].sum()

# Remove the old rows and add the merged one
whole_vv_counts_results = whole_vv_counts_results.loc[~mask]
whole_vv_counts_results.loc[len(whole_vv_counts_results)] = ['모르', merged_count]

# Sort if desired
whole_vv_counts_results = whole_vv_counts_results.sort_values('Count', ascending=False).reset_index(drop=True)

wholenotevv = whole_vv_counts_results[~whole_vv_counts_results["Word"].isin(stopverbs)].reset_index(drop=True)

# Nouns
fig_nouns = figure_words_with_categories(
    whole_cleanednng_counts_results,
    word_col="Word",
    count_col="Count",
    translation_dict=korean_wholenouns_to_english,
    category_map=noun_category_map,     # expand this map as you finalize the codebook
    top_n=50,
    proportion=True,
    suptitle="",
    font_path = r'C:\Windows\Fonts\Arialbd.ttf',
    font_size=10
)
fig_nouns.savefig("nouns_categories.jpg", bbox_inches="tight", dpi=600) # vector for journal
plt.show()

# Verbs (you may want a separate verb-specific category_map)
fig_verbs = figure_words_with_categories(
    wholenotevv,
    word_col="Word",
    count_col="Count",
    translation_dict=korean_verbs_to_english,
    category_map=verb_category_map,     # or a verb-tailored map
    top_n=50,
    proportion=True,
    suptitle="",
    font_path = r'C:\Windows\Fonts\Arialbd.ttf',
    font_size=10
)
fig_verbs.savefig("verbs_categories.jpg", bbox_inches="tight", dpi=600)  # ← add this
plt.show()

#%%
"""Figure 4b."""


"""This is the final code I used for PCA - K-Means Clustering (Use sentiment_environment)"""
# -*- coding: utf-8 -*-
"""
Theme PCA -> KMeans clustering (Sections x Themes)
- Uses your existing `morethan3["full_theme_vector"]` (length = 3 sections * 5 themes = 15)
- Prints explained variance, silhouette score
- 2D scatter with distinct colors + markers (no Set2 duplication)
- Heatmap of reconstructed cluster centers in original theme space
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from itertools import cycle
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

morethan3=pd.read_pickle(r"C:\Users\Jae Bin Park\morethan3rawformainfigures")
# =========================
# Config & Pretty printing
# =========================
# Optional: Use your custom font


font_path = r'C:\Windows\Fonts\Arialbd.ttf'
try:
    font_prop = fm.FontProperties(fname=font_path)
    custom_font = font_prop.get_name()
    plt.rcParams['font.family'] = custom_font
except Exception:
    font_prop = fm.FontProperties()  # fallback to default

# Themes and Sections
themes = ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]
sections = ["Section 1", "Section 2", "Section 3"]

# Columns in section-major order (Section 1 all themes, then Section 2, then Section 3)
theme_cols = [f"{sec} {theme}" for sec in sections for theme in themes]  # len=15

# ======================================================
# Step 1: Build theme_matrix from your vectorized column
# ======================================================
# Assumes: morethan3["full_theme_vector"] is a list-like of length 15 per row
theme_matrix = pd.DataFrame(
    morethan3["full_theme_vector"].tolist(),
    index=morethan3.index,
    columns=theme_cols
)

# =======================================
# Step 2: PCA (reduce dimensionality)
# =======================================
pca = PCA(n_components=5, random_state=0)
X_pca = pca.fit_transform(theme_matrix)

# Explained variance reporting
evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)
evr_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(evr))],
    "Explained_Variance_Ratio": evr,
    "Cumulative": cum_evr
})
print("\nExplained variance by component (fractions):")
print(evr_df.to_string(index=False))
print(f"\nTotal variance explained by {pca.n_components_} PCs: {cum_evr[-1]*100:.2f}%")
print(f"Variance explained by first 2 PCs: {cum_evr[1]*100:.2f}%")

for thr in [0.80, 0.90, 0.95]:
    if cum_evr[-1] >= thr:
        k_needed = int(np.argmax(cum_evr >= thr) + 1)
        print(f"{int(thr*100)}% variance reached with {k_needed} PCs.")
    else:
        print(f"{int(thr*100)}% variance not reached with {pca.n_components_} PCs (max = {cum_evr[-1]*100:.2f}%).")

# Scree plots
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(evr)+1), evr, marker='o')
plt.title("PCA Explained Variance per Component", fontproperties=font_prop)
plt.xlabel("Principal Component", fontproperties=font_prop)
plt.ylabel("Explained Variance Ratio", fontproperties=font_prop)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cum_evr)+1), cum_evr, marker='o')
plt.axhline(0.90, linestyle='--')
plt.title("PCA Cumulative Explained Variance", fontproperties=font_prop)
plt.xlabel("Number of Components", fontproperties=font_prop)
plt.ylabel("Cumulative Explained Variance", fontproperties=font_prop)
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================================
# Step 3: KMeans clustering on PCA space
# =======================================
n_clusters = 15
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
clusters = kmeans.fit_predict(X_pca)
morethan3["theme_cluster_pca"] = clusters

# Optional: map numeric clusters to descriptive names
cluster_name_map = {
  0: "1. Love and Gratitude",
  1: "2. Sorry and Shame (First & Last Section)",
  2: "3. Sorry and Shame (First & Middle Section)",
  3: "4. Sorry and Shame + Love and Gratitude",
  4: "5. Theme-absent",
  5: "6. Sorry and Shame (Middle & Last Section)",
  6: "7. Sorry and Shame (First Section)",
  7: "8. Pervasive Sorry and Shame",
  8: "9. Sorry and Shame (Last Section)",
  9: "10. Pervasive Love and Gratitude + Sorry and Shame (Last Section)",
  10: "11. Pervasive Love and Gratitude + Sorry and Shame (First & Middle Section)",
  11: "12. Sorry and Shame (Middle Section)",
  12: "13. Sorry and Shame + Love and Gratitude (First & Last Section)",
  13: "14. Pervasive Sorry and Shame + Post-Mortem Affairs (Middle & Last Section)",
  14: "15. Sorry and Shame + Love and Gratitude (First Section) + Post-Mortem Affairs (Last Section)"
}

morethan3["theme_cluster_pca_named"] = morethan3["theme_cluster_pca"].map(cluster_name_map)

# Evaluate clustering quality
sil_score = silhouette_score(X_pca, clusters)
print(f"Silhouette Score: {sil_score:.3f}")

# ============================================================
# Step 4: 2D Visualization with distinct colors + markers
# ============================================================
# Use the first two PCs directly
X_vis = X_pca[:, :2]
centroids_2d = kmeans.cluster_centers_[:, :2]

def distinct_palette(n):
    """
    Returns n visually distinct colors.
    - Start with reordered tab20 (20 categorical colors) to maximize separation.
    - If n > 20, extend with evenly spaced HSV hues.
    """
    base = list(plt.get_cmap('tab20').colors)  # 20 RGBA tuples
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))  # separate similar pairs
    palette = [base[i] for i in order]
    if n <= len(palette):
        return palette[:n]
    extra = [cm.get_cmap('hsv')(x) for x in np.linspace(0, 1, n - len(palette), endpoint=False)]
    return palette + extra

colors = distinct_palette(n_clusters)
markers = ['o','s','^','v','<','>','P','X','D','*','h','H','8','p','+','x']

plt.figure(figsize=(12, 6))
for i in range(n_clusters):
    pts = X_vis[clusters == i]
    marker_i = markers[i % len(markers)]
    plt.scatter(
        pts[:, 0], pts[:, 1],
        c=[colors[i]], marker=marker_i, alpha=0.85,
        edgecolor='k', linewidths=0.5
    )

# Legend with both color & marker
legend_handles = []
for i in range(n_clusters):
    marker_i = markers[i % len(markers)]
    label_i = cluster_name_map.get(i, f"Cluster {i}")
    handle = Line2D(
        [0], [0],
        marker=marker_i, color='w', label=label_i,
        markerfacecolor=colors[i], markeredgecolor='k',
        markersize=8, linewidth=0
    )
    legend_handles.append(handle)

plt.title("Theme-based PCA - KMeans Clustering", fontproperties=font_prop)
plt.xlabel("PCA1", fontproperties=font_prop)
plt.ylabel("PCA2", fontproperties=font_prop)
plt.legend(handles=legend_handles, title="Clusters",
           bbox_to_anchor=(1.05, 1), loc="upper left", prop=font_prop)
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================================================
# Step 5: Heatmap of cluster centers (original space)
# ======================================================
# Transform cluster centers back to original theme space
centers_original_space = pca.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(centers_original_space, columns=theme_cols)

# Reindex rows by descriptive names (optional)
row_labels = [cluster_name_map.get(i, f"Cluster {i}") for i in range(n_clusters)]
cluster_centers_df.index = row_labels

# For display, keep section-major ordering for columns
ordered_labels = [f"{sec} {theme}" for sec in sections for theme in themes]
cluster_centers_df = cluster_centers_df[ordered_labels]

# Plot heatmap
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(cluster_centers_df.values, cmap="YlGnBu")

# X tick labels: show only theme (strip "Section i ")
xtick_labels = [" ".join(col.split(" ")[2:]) for col in ordered_labels]
ax.set_xticks(np.arange(len(ordered_labels)))
ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9, fontproperties=font_prop)

# Add overarching section labels below the heatmap
# =============================================================================
# themes_per_section = len(themes)
# x_section_positions = [i * themes_per_section + (themes_per_section / 2 - 0.5) for i in range(len(sections))]
# for i, section in enumerate(sections):
#     ax.text(
#         x_section_positions[i],
#         cluster_centers_df.shape[0] + 1.5,   # just below the last row
#         section,
#         ha="center",
#         va="top",
#         fontsize=11,
#         fontproperties=font_prop,
#         weight="bold",
#         transform=ax.transData,
#         clip_on=False
#     )
# 
# =============================================================================
# Y ticks (cluster names)
ax.set_yticks(np.arange(cluster_centers_df.shape[0]))
ax.set_yticklabels(cluster_centers_df.index, fontproperties=font_prop)

# Annotate cells with dynamic text color
vals = cluster_centers_df.values
vmax = float(vals.max()) if vals.size else 0.0
threshold = vmax * 0.6 if vmax > 0 else 0.0
for i in range(cluster_centers_df.shape[0]):
    for j in range(cluster_centers_df.shape[1]):
        val = cluster_centers_df.iloc[i, j]
        text_color = "white" if val > threshold else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontproperties=font_prop)

ax.set_title("Reconstructed Cluster Centers in Theme Space (Ordered by Section)", fontproperties=font_prop)
fig.tight_layout()
plt.show()

# Optional: reset rcParams
plt.rcdefaults()

#%%

"""Figure 4c"""

"""This is the final code I used for Sentiment PCA - K-Means Clustering (Use sentiment_environment)"""

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from itertools import cycle
import matplotlib.cm as cm


morethan3rawsentiment=pd.read_pickle(r"C:\Users\Jae Bin Park\morethan3rawsentimentformainfigures")

# -----------------------------
# Build sentiment presence cols
# -----------------------------
def create_sentiment_presence_vector(df, sentiment_labels, section_cols, label_prefix):
    tqdm.pandas(desc=f"Processing sentiment presence for: {label_prefix}")

    def check_row(row):
        presence = []
        for col in section_cols:
            entry = row[col]
            if isinstance(entry, dict):
                section_labels = entry.get("labels", [])
                has_any = int(any(lbl in sentiment_labels for lbl in section_labels))
            else:
                has_any = 0
            presence.append(has_any)
        return presence

    df[f"{label_prefix}_presence"] = df.progress_apply(check_row, axis=1)
    return df

section_cols = [
    "1st_section_sentiment",
    "2nd_section_sentiment",
    "3rd_section_sentiment"
]

sentiment_targets = [
    # Hopelessness / Despair / Helplessness
    "절망",           # Despair
    "패배/자기혐오",   # Defeat / Self-loathing
    "힘듦/지침",       # Exhaustion / Fatigue
    "불안/걱정",       # Anxiety / Worry

    # Shame / Guilt / Self-directed Negativity
    "죄책감",         # Guilt
    "부끄러움",        # Shame
    "한심함",         # Worthlessness
    "불쌍함/연민",
    "안타까움/실망",

    # Anger / Resentment / Rejection
    "화남/분노",       # Anger / Rage
    "증오/혐오",       # Hatred / Disgust
    "어이없음",        # Resentment / Disbelief

    # Sadness / Isolation / Cry for Help
    "슬픔",           # Sadness
    "서러움",         # Deep sadness / Feeling wronged

    # Mixed / Ambivalent States
    "고마움",         # Gratitude
    "흐뭇함(귀여움/예쁨)",  # Affection
    "행복",           # Happiness
    "안심/신뢰",        # Relief / Trust
    "없음"            # Neutral / None
]

for label in sentiment_targets:
    morethan3rawsentiment = create_sentiment_presence_vector(
        morethan3rawsentiment,
        sentiment_labels=[label],  # one sentiment per run
        section_cols=section_cols,
        label_prefix=label
    )

# -----------------------------
# Map presence vectors (3-long)
# -----------------------------
sentiment_presence_columns = {
    "Despair": "절망_presence",
    "Defeat": "패배/자기혐오_presence",
    "Exhaustion": "힘듦/지침_presence",
    "Anxious": "불안/걱정_presence",
    "Disappointment": "안타까움/실망_presence",
    "Anger": "화남/분노_presence",
    "Hatred": "증오/혐오_presence",
    "Resentment": "어이없음_presence",
    "Sadness": "슬픔_presence",
    "Sorrow": "서러움_presence",
    "Gratitude": "고마움_presence",
    "Affection": "흐뭇함(귀여움/예쁨)_presence",
    "Happiness": "행복_presence",
    "Relief": "안심/신뢰_presence",
    "Neutral": "없음_presence"
}

def get_sentimentsection_vector(row, section_idx):
    return [
        row[sentiment_presence_columns[sentiment]][section_idx]
        for sentiment in sentiment_presence_columns
    ]

morethan3rawsentiment["section1_sentiment_vector"] = morethan3rawsentiment.apply(lambda row: get_sentimentsection_vector(row, 0), axis=1)
morethan3rawsentiment["section2_sentiment_vector"] = morethan3rawsentiment.apply(lambda row: get_sentimentsection_vector(row, 1), axis=1)
morethan3rawsentiment["section3_sentiment_vector"] = morethan3rawsentiment.apply(lambda row: get_sentimentsection_vector(row, 2), axis=1)

def concatenate_sentiment_vectors(row):
    return (
        row["section1_sentiment_vector"] +
        row["section2_sentiment_vector"] +
        row["section3_sentiment_vector"]
    )

morethan3rawsentiment["full_sentiment_vector"] = morethan3rawsentiment.apply(concatenate_sentiment_vectors, axis=1)

# -----------------------------
# PCA + KMeans
# -----------------------------
# Custom Font
font_path = r'C:\Windows\Fonts\Arialbd.ttf'
font_prop = fm.FontProperties(fname=font_path)
custom_font = font_prop.get_name()

# Build sentiment_matrix (15 sentiments × 3 sections), sentiment-major order
sections = ["Section 1", "Section 2", "Section 3"]
sentiment_labels = [
    "Despair", "Defeat", "Exhaustion", "Anxious", "Disappointment",
    "Anger", "Hatred", "Resentment", "Sadness", "Sorrow",
    "Gratitude", "Affection", "Happiness", "Relief", "Neutral"
]
# PCA-friendly column order: all 3 sections per sentiment
pca_cols = [f"Section {i+1} {label}" for label in sentiment_labels for i in range(3)]
sentiment_matrix = pd.DataFrame(
    morethan3rawsentiment["full_sentiment_vector"].tolist(),
    columns=pca_cols,
    index=morethan3rawsentiment.index
)

# PCA
pca = PCA(n_components=6, random_state=42)
X_pca = pca.fit_transform(sentiment_matrix)

# Explained variance
evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)
evr_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(evr))],
    "Explained_Variance_Ratio": evr,
    "Cumulative": cum_evr
})
print("\nExplained variance by component (fractions):")
print(evr_df.to_string(index=False))
print(f"\nTotal variance explained by {pca.n_components_} PCs: {cum_evr[-1]*100:.2f}%")
print(f"Variance explained by first 2 PCs: {cum_evr[1]*100:.2f}%")

for thr in [0.80, 0.90, 0.95]:
    if cum_evr[-1] >= thr:
        k_needed = np.argmax(cum_evr >= thr) + 1
        print(f"{thr*100:.0f}% variance reached with {k_needed} PCs.")
    else:
        print(f"{thr*100:.0f}% variance not reached with {pca.n_components_} PCs (max = {cum_evr[-1]*100:.2f}%).")

# Scree plots
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(evr)+1), evr, marker='o')
plt.title("PCA Explained Variance per Component", fontproperties=font_prop)
plt.xlabel("Principal Component", fontproperties=font_prop)
plt.ylabel("Explained Variance Ratio", fontproperties=font_prop)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cum_evr)+1), cum_evr, marker='o')
plt.axhline(0.90, linestyle='--')
plt.title("PCA Cumulative Explained Variance", fontproperties=font_prop)
plt.xlabel("Number of Components", fontproperties=font_prop)
plt.ylabel("Cumulative Explained Variance", fontproperties=font_prop)
plt.grid(True)
plt.tight_layout()
plt.show()

# KMeans
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)
morethan3rawsentiment["sentiment_cluster"] = cluster_labels

cluster_name_map = {
  0: "1. Pervasive Hostility",
  1: "2. Defeat + Exhaustion",
  2: "3. Hatred + Resentment",
  3: "4. Emotionally Flat",
  4: "5. Affection + Happiness",
  5: "6. Hopelessness + Affiliative Remarks",
  6: "7. Hopelessness + Hostility + Affiliative Remarks",
  7: "8. Emotional Turmoil",
  8: "9. Defeat + Exhaustion + Happiness",
  9: "10. Despair + Defeat + Exhaustion"
}

morethan3rawsentiment["sentiment_cluster_pca_named"] = morethan3rawsentiment["sentiment_cluster"].map(cluster_name_map)

sil_score = silhouette_score(X_pca, cluster_labels)
print(f"Silhouette Score: {sil_score:.3f}")

# -----------------------------
# Step 5: 2D Visualization (fixed colors + markers)
# -----------------------------
# Use the first two PCs directly for visualization
X_vis = X_pca[:, :2]
centroids_2d = kmeans.cluster_centers_[:, :2]

def distinct_palette(n):
    """
    Return n visually distinct colors.
    - Start with a reordered tab20 (20 categorical colors) to maximize separation.
    - If n > 20, extend with evenly spaced HSV hues.
    """
    base = list(plt.get_cmap('tab20').colors)  # 20 RGBA tuples
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))  # separate similar pairs
    palette = [base[i] for i in order]
    if n <= len(palette):
        return palette[:n]
    extra = [cm.get_cmap('hsv')(x) for x in np.linspace(0, 1, n - len(palette), endpoint=False)]
    return palette + extra

colors = distinct_palette(n_clusters)

# Distinct markers help even more when colors are similar
markers = ['o','s','^','v','<','>','P','X','D','*','h','H','8','p','+','x']
plt.figure(figsize=(12, 6))
for i in range(n_clusters):
    pts = X_vis[cluster_labels == i]
    marker_i = markers[i % len(markers)]
    plt.scatter(
        pts[:, 0], pts[:, 1],
        c=[colors[i]], marker=marker_i, alpha=0.85,
        edgecolor='k', linewidths=0.5
    )

# Legend reflecting both color & marker
legend_handles = []
for i in range(n_clusters):
    marker_i = markers[i % len(markers)]
    handle = Line2D(
        [0], [0],
        marker=marker_i, color='w', label=cluster_name_map[i],
        markerfacecolor=colors[i], markeredgecolor='k',
        markersize=8, linewidth=0
    )
    legend_handles.append(handle)

plt.legend(handles=legend_handles, title="Clusters",
           bbox_to_anchor=(1.05, 1), loc="upper left", prop=font_prop)
plt.xlabel("PCA1", fontproperties=font_prop)
plt.ylabel("PCA2", fontproperties=font_prop)
plt.title("Sentiment-Based PCA - KMeans Clustering", fontproperties=font_prop)
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 6: Cluster Center Heatmap
# -----------------------------
# Inverse transform cluster centers back to original sentiment-major space (pca_cols)
cluster_centers_original = pca.inverse_transform(kmeans.cluster_centers_)
rounded_centers = pd.DataFrame(cluster_centers_original, columns=pca_cols)

# Reorder to section-major for visualization:
# e.g., Section 1 Despair → Section 1 Defeat → ...; then Section 2 ...; then Section 3 ...
sections = ["Section 1", "Section 2", "Section 3"]
section_major_cols = [f"{sec} {label}" for sec in sections for label in sentiment_labels]
rounded_centers = rounded_centers[section_major_cols]

fig, ax = plt.subplots(figsize=(18, 6))
im = ax.imshow(rounded_centers.values, cmap="YlGnBu")

# Clean x-tick labels (only emotion name)
xtick_labels = [" ".join(col.split(" ")[2:]) for col in section_major_cols]
ax.set_xticks(np.arange(len(section_major_cols)))
ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9, fontproperties=font_prop)

# Y-tick labels (cluster names)
ax.set_yticks(np.arange(n_clusters))
ax.set_yticklabels([cluster_name_map[i] for i in range(n_clusters)], fontproperties=font_prop)

# Annotate cells
vmax = rounded_centers.values.max()
threshold = vmax * 0.6 if vmax > 0 else 0.0
for i in range(n_clusters):
    for j in range(len(section_major_cols)):
        val = rounded_centers.iloc[i, j]
        color = "white" if val > threshold else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontproperties=font_prop)

# Section labels below x-ticks
# =============================================================================
# sentiments_per_section = len(sentiment_labels)
# x_section_positions = [i * sentiments_per_section + sentiments_per_section / 2 - 0.5 for i in range(len(sections))]
# for i, section in enumerate(sections):
#     ax.text(
#         x_section_positions[i],
#         n_clusters + 2.5,   # just below x-tick labels
#         section,
#         ha="center",
#         va="bottom",
#         fontsize=10,
#         fontproperties=font_prop,
#         weight="bold",
#         transform=ax.transData,
#         clip_on=False
#     )
# =============================================================================

ax.set_title("Sentiment Prevalence per Cluster (All Sentiments)", fontproperties=font_prop)
fig.tight_layout()
plt.show()

# Optional: reset default font settings
plt.rcdefaults()