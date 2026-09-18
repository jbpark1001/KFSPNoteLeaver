# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 12:28:56 2025

@author: Jae Bin Park
"""


#%%

# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as fm

# ----------------------
# Font: apply this to every text element manually (no rcParams)
# ----------------------
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'

font_path = r'C:\Windows\Fonts\Arialbd.ttf'

fontprop = fm.FontProperties(fname=font_path)

# ----------------------
# 1) Data
# ----------------------
data = {
    "region_kr": ["경기도", "서울특별시", "부산광역시", "경상남도", "경상북도", "인천광역시",
                  "충청남도", "대구광역시", "강원도", "전라북도", "전라남도", "충청북도",
                  "대전광역시", "광주광역시", "울산광역시", "제주특별자치도", "세종특별자치시"],
    "count": [24233, 15658, 7185, 6984, 6283, 6000,
              5692, 4678, 4551, 4200, 4135, 3882,
              2916, 2493, 2214, 1419, 399]
}
df = pd.DataFrame(data)

# Korean → English (matches shapefile's 'name')
hangul_to_en = {
    "강원도": "Gangwon",
    "경기도": "Gyeonggi",
    "충청남도": "South Chungcheong",
    "인천광역시": "Incheon",
    "전라북도": "North Jeolla",
    "전라남도": "South Jeolla",
    "경상남도": "South Gyeongsang",
    "부산광역시": "Busan",
    "울산광역시": "Ulsan",
    "경상북도": "North Gyeongsang",
    "제주특별자치도": "Jeju",
    "서울특별시": "Seoul",
    "대전광역시": "Daejeon",
    "세종특별자치시": "Sejong",
    "충청북도": "North Chungcheong",
    "광주광역시": "Gwangju",
    "대구광역시": "Daegu"
}
df["region_en"] = df["region_kr"].map(hangul_to_en)

# ----------------------
# 2) Shapefile & merge
# ----------------------
shp_path = r"C:\Users\Jae Bin Park\kr_shp\kr.shp"
gdf = gpd.read_file(shp_path)
gdf = gdf.merge(df[["region_en", "count"]], left_on="name", right_on="region_en", how="left")

# Projection (equal-area for KR)
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)
gdf = gdf.to_crs(epsg=5179)

# ----------------------
# 3) Figure with map + separate colorbar
# ----------------------
fig, ax = plt.subplots(figsize=(8, 10), dpi=300)

gdf.plot(column="count", cmap="Reds", linewidth=0.4, edgecolor="black",
         legend=False, ax=ax)

ax.set_axis_off()
ax.set_title("Distribution of Suicide Cases by Region in South Korea",
             fontsize=12, fontproperties=fontprop, pad=12)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.4)

norm = mpl.colors.Normalize(vmin=gdf["count"].min(), vmax=gdf["count"].max())
sm = mpl.cm.ScalarMappable(cmap="Reds", norm=norm)
sm.set_array([])

cb = fig.colorbar(sm, cax=cax)
# Apply font to colorbar label and ticks
cb.set_label("Number of Suicides", fontsize=10, fontproperties=fontprop)
for ticklab in cb.ax.get_yticklabels():
    ticklab.set_fontproperties(fontprop)
    ticklab.set_fontsize(8)

# ----------------------
# 4) Table inset (smaller, middle-left)
# ----------------------
table_data = df.sort_values("count", ascending=False)
table_values = [[r, f"{c:,}"] for r, c in zip(table_data["region_en"], table_data["count"])]
table_col_labels = ["Region", "Count"]

# Create a dedicated inset axis for the table at middle-left of the figure
# [x, y, width, height] in figure fractions — tweak if needed
table_ax = fig.add_axes([0.03, 0.40, 0.30, 0.28])
table_ax.axis("off")

tbl = table_ax.table(cellText=table_values,
                     colLabels=table_col_labels,
                     cellLoc='left',
                     colLoc='left',
                     loc='center')

# Styling: smaller font, tight rows, no cell borders, bold header
tbl.auto_set_font_size(False)
table_font_size = 8  # <-- adjust here

# Apply font to every cell explicitly
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('none')
    cell.set_height(0.05 if row != 0 else 0.10)  # tighter rows, slightly taller header
    cell.get_text().set_fontproperties(fontprop)
    cell.get_text().set_fontsize(table_font_size)  # manual font size

    if row == 0:
        cell.get_text().set_fontproperties(fontprop)  # header uses same Roboto-Bold
        cell.get_text().set_fontsize(table_font_size)  # manual font size

plt.savefig("suicides_by_region_with_table_middle_left.png", dpi=600, bbox_inches="tight")
plt.savefig("suicides_by_region_with_table_middle_left.svg", bbox_inches="tight")
plt.show()
#%%

# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as fm

# ----------------------
# Font (apply explicitly; no rcParams)
# ----------------------
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
fontprop = fm.FontProperties(fname=font_path)

# ----------------------
# 1) Data (counts)
# ----------------------
data = {
    "region_kr": ["경기도", "서울특별시", "부산광역시", "경상남도", "경상북도", "인천광역시",
                  "충청남도", "대구광역시", "강원도", "전라북도", "전라남도", "충청북도",
                  "대전광역시", "광주광역시", "울산광역시", "제주특별자치도", "세종특별자치시"],
    "count": [24233, 15658, 7185, 6984, 6283, 6000,
              5692, 4678, 4551, 4200, 4135, 3882,
              2916, 2493, 2214, 1419, 399]
}
df = pd.DataFrame(data)

# Korean → English (matches shapefile 'name')
hangul_to_en = {
    "강원도": "Gangwon",
    "경기도": "Gyeonggi",
    "충청남도": "South Chungcheong",
    "인천광역시": "Incheon",
    "전라북도": "North Jeolla",
    "전라남도": "South Jeolla",
    "경상남도": "South Gyeongsang",
    "부산광역시": "Busan",
    "울산광역시": "Ulsan",
    "경상북도": "North Gyeongsang",
    "제주특별자치도": "Jeju",
    "서울특별시": "Seoul",
    "대전광역시": "Daejeon",
    "세종특별자치시": "Sejong",
    "충청북도": "North Chungcheong",
    "광주광역시": "Gwangju",
    "대구광역시": "Daegu"
}
df["region_en"] = df["region_kr"].map(hangul_to_en)

# ----------------------
# 1.5) Population by region (FILL THESE WITH YOUR ACTUAL POPULATIONS)
#       Units: persons (not thousands). Year should match your counts.
# ----------------------
POPULATION = {
    "Gyeonggi": 12806196.75,
    "Seoul": 9902537.00,
    "Busan": 3472154.38 ,
    "South Gyeongsang": 3359976.38 
,
    "North Gyeongsang": 2684656.63 
,
    "Incheon": 2931789.00 
,
    "South Chungcheong": 2096508.75 
,
    "Daegu": 2470076.88 
,
    "Gangwon": 1545569.25 
,
    "North Jeolla": 2684656.63 
,
    "South Jeolla": 1890693.75 
,
    "North Chungcheong": 1590221.25 
,
    "Daejeon": 1503585.00 
,
    "Gwangju": 1464980.38 
,
    "Ulsan": 1159185.75 
,
    "Jeju": 642130.25 
,
    "Sejong": 252855.25 
,
}

pop_df = (
    pd.Series(POPULATION, name="population")
      .rename_axis("region_en")
      .reset_index()
)

# Sanity check: ensure no missing population values
if pop_df["population"].isna().any():
    missing = pop_df.loc[pop_df["population"].isna(), "region_en"].tolist()
    raise ValueError(
        f"Missing population values for: {missing}. "
        f"Please fill POPULATION dict with actual counts (persons)."
    )

# Merge counts + population; compute rate per 100k
df = df.merge(pop_df, on="region_en", how="left")
df["rate_per_100k"] = (df["count"] / df["population"]) * 100_000

# ----------------------
# 2) Shapefile & merge
# ----------------------
shp_path = r"C:\Users\Jae Bin Park\kr_shp\kr.shp"
gdf = gpd.read_file(shp_path)
gdf = gdf.merge(df[["region_en", "count", "rate_per_100k"]], left_on="name", right_on="region_en", how="left")

# Projection (equal-area for KR)
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)
gdf = gdf.to_crs(epsg=5179)

# ----------------------
# 3) Plot settings
#    Choose which metric to visualize: "rate_per_100k" (default) or "count"
# ----------------------
METRIC = "rate_per_100k"   # change to "count" if you want raw counts
TITLE = "Suicide Rate per 100,000 by Region in South Korea" if METRIC == "rate_per_100k" \
        else "Distribution of Suicide Cases by Region in South Korea"
CBAR_LABEL = "Suicides per 100,000" if METRIC == "rate_per_100k" else "Number of Suicides"

fig, ax = plt.subplots(figsize=(8, 10), dpi=300)

gdf.plot(
    column=METRIC,
    cmap="Reds",
    linewidth=0.4,
    edgecolor="black",
    legend=False,
    ax=ax
)

ax.set_axis_off()
ax.set_title(TITLE, fontsize=12, fontproperties=fontprop, pad=12)

# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.4)

vals = gdf[METRIC]
norm = mpl.colors.Normalize(vmin=vals.min(), vmax=vals.max())
sm = mpl.cm.ScalarMappable(cmap="Reds", norm=norm)
sm.set_array([])

cb = fig.colorbar(sm, cax=cax)
cb.set_label(CBAR_LABEL, fontsize=10, fontproperties=fontprop)
for ticklab in cb.ax.get_yticklabels():
    ticklab.set_fontproperties(fontprop)
    ticklab.set_fontsize(8)

# ----------------------
# 4) Table inset (sorted by chosen metric)
# ----------------------
table_df = df.sort_values(METRIC, ascending=False).copy()
table_df["rate_display"] = table_df["rate_per_100k"].map(lambda x: f"{x:,.1f}")
table_df["count_display"] = table_df["count"].map(lambda x: f"{x:,}")

table_values = [
    [r, c, rt]
    for r, c, rt in zip(table_df["region_en"], table_df["count_display"], table_df["rate_display"])
]
table_col_labels = ["Region", "Count", "Rate per 100k"]

# Inset axis for table (middle-left)
table_ax = fig.add_axes([0.03, 0.38, 0.33, 0.32])  # widened to fit 3 cols
table_ax.axis("off")

tbl = table_ax.table(
    cellText=table_values,
    colLabels=table_col_labels,
    cellLoc='left',
    colLoc='left',
    loc='center'
)

# Styling: set font and row heights
tbl.auto_set_font_size(False)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('none')
    cell.set_height(0.08 if row != 0 else 0.10)
    cell.get_text().set_fontproperties(fontprop)
    cell.get_text().set_fontsize(8)

# ----------------------
# 5) Caption & save
# ----------------------
plt.annotate(
    "Data: KFSP Nationwide Suicide Dataset (rates computed as count / population × 100,000)",
    xy=(0.02, 0.05), xycoords="figure fraction",
    fontsize=6, fontproperties=fontprop, color="gray"
)

out_png = "suicides_by_region_RATE_per100k.png" if METRIC == "rate_per_100k" else "suicides_by_region_COUNT.png"
out_svg = out_png.replace(".png", ".svg")
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.savefig(out_svg, bbox_inches="tight")
plt.show()

#%%




#%% Final Code for Geographic Map

"""This is the final code for the Geographic Map in the KFSP paper""" 


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
