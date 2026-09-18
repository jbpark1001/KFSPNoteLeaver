# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


# =========================================================
# Configuration
# =========================================================
DATA_PATH = Path(r"G:\KFSP\Raw Data\KFSPdatacopy.xlsx")
OUTPUT_DIR = Path(
    r"C:\Users\Jae Bin Park\OneDrive\KAIST Research"
    r"\Suicide Research\KFSP Journal\Note Leavers"
    r"\Nature Ver\KFSP Final Figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "figure_2_combined_months.png"
JPG_PATH = OUTPUT_DIR / "figure_2_combined_months.jpg"

START_DATE = "2013-01-01"
END_DATE = "2020-12-31"

NOTE_LEAVER_CODE = 1
NON_LEAVER_CODE = 2


# =========================================================
# Arial font
# =========================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =========================================================
# Load and prepare data
# =========================================================
print("Loading data...")

kfsp = pd.read_excel(DATA_PATH)

required_columns = {"SUICIDE_DATE", "NOTE_EX"}
missing_columns = required_columns.difference(kfsp.columns)

if missing_columns:
    raise KeyError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

# Convert dates such as 20130101 or 20130101.0
date_text = (
    kfsp["SUICIDE_DATE"]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

kfsp["Suicide_Date"] = pd.to_datetime(
    date_text,
    format="%Y%m%d",
    errors="coerce"
)

# Fallback for actual Excel datetime values
unparsed = kfsp["Suicide_Date"].isna()

if unparsed.any():
    kfsp.loc[unparsed, "Suicide_Date"] = pd.to_datetime(
        kfsp.loc[unparsed, "SUICIDE_DATE"],
        errors="coerce"
    )

kfsp["NOTE_EX"] = pd.to_numeric(
    kfsp["NOTE_EX"],
    errors="coerce"
)

# Keep 2013–2020 and NOTE_EX values 1 or 2
plot_data = kfsp.loc[
    kfsp["Suicide_Date"].between(START_DATE, END_DATE)
    & kfsp["NOTE_EX"].isin(
        [NOTE_LEAVER_CODE, NON_LEAVER_CODE]
    )
].copy()

if plot_data.empty:
    raise ValueError("No observations remained after filtering.")

print(f"Observations used: {len(plot_data):,}")

# Extract month number only; years are pooled
plot_data["Month"] = plot_data["Suicide_Date"].dt.month


# =========================================================
# Pool counts across all years by calendar month
# =========================================================
monthly = (
    plot_data
    .groupby(["Month", "NOTE_EX"])
    .size()
    .unstack(fill_value=0)
    .reindex(range(1, 13), fill_value=0)
)

note = monthly.get(
    NOTE_LEAVER_CODE,
    pd.Series(0, index=monthly.index)
).to_numpy(dtype=float)

non_note = monthly.get(
    NON_LEAVER_CODE,
    pd.Series(0, index=monthly.index)
).to_numpy(dtype=float)

total = note + non_note

with np.errstate(divide="ignore", invalid="ignore"):
    note_percentage = np.where(
        total > 0,
        100 * note / total,
        np.nan
    )

# Display the pooled results in the console
summary = pd.DataFrame({
    "Month": [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ],
    "Note_leavers": note.astype(int),
    "Non_leavers": non_note.astype(int),
    "Total": total.astype(int),
    "Note_leaver_percentage": note_percentage
})

print("\nPooled monthly results:")
print(summary.to_string(index=False))


# =========================================================
# Plot
# =========================================================
month_labels = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

x = np.arange(12)

fig, ax = plt.subplots(
    figsize=(12, 5.1),
    dpi=100,
    facecolor="white"
)

fig.subplots_adjust(
    left=0.08,
    right=0.92,
    bottom=0.16,
    top=0.67
)

non_note_color = "#C9C9C9"
note_color = "#7CA9D6"
line_color = "#204F7F"

# Draw non-leaver counts first
ax.bar(
    x,
    non_note,
    width=0.72,
    color=non_note_color,
    edgecolor="white",
    linewidth=0.5,
    zorder=2
)

# Overlay note-leaver counts
ax.bar(
    x,
    note,
    width=0.72,
    color=note_color,
    edgecolor="white",
    linewidth=0.5,
    zorder=3
)


# =========================================================
# Left count axis
# =========================================================
ax.set_xlim(-0.7, 11.7)

maximum_count = max(note.max(), non_note.max())
ax.set_ylim(0, maximum_count * 1.08)

ax.yaxis.set_major_locator(
    MaxNLocator(nbins=5, integer=True)
)

ax.set_xticks(x)
ax.set_xticklabels(month_labels, fontsize=11)

ax.tick_params(
    axis="x",
    length=0,
    pad=9
)

ax.tick_params(
    axis="y",
    length=0,
    pad=8,
    labelsize=11
)

ax.set_axisbelow(True)
ax.grid(
    axis="y",
    color="#E2E2E2",
    linewidth=0.8
)


# =========================================================
# Right proportion axis
# =========================================================
ax_right = ax.twinx()
ax_right.patch.set_visible(False)

ax_right.plot(
    x,
    note_percentage,
    color=line_color,
    linewidth=1.7,
    marker="o",
    markersize=5,
    markerfacecolor=line_color,
    markeredgecolor=line_color,
    zorder=5
)

ax_right.set_ylim(30, 50)
ax_right.set_yticks([30, 35, 40, 45, 50])

ax_right.tick_params(
    axis="y",
    length=0,
    pad=6,
    labelsize=11
)


# =========================================================
# Panel borders
# =========================================================
for side in ("left", "bottom", "top"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color("black")
    ax.spines[side].set_linewidth(1.2)

ax.spines["right"].set_visible(False)

for side in ("left", "bottom", "top"):
    ax_right.spines[side].set_visible(False)

ax_right.spines["right"].set_visible(True)
ax_right.spines["right"].set_color("black")
ax_right.spines["right"].set_linewidth(1.2)


# =========================================================
# Titles
# =========================================================
ax.text(
    -0.03,
    1.10,
    "Pooled Suicide Counts by Calendar Month",
    transform=ax.transAxes,
    ha="left",
    va="center",
    fontsize=14
)

ax_right.text(
    1.03,
    1.10,
    "Proportion of Note Leavers (%)",
    transform=ax_right.transAxes,
    ha="right",
    va="center",
    fontsize=14
)

fig.text(
    0.025,
    0.93,
    "b",
    ha="left",
    va="top",
    family="Arial",
    weight="bold",
    fontsize=28
)


# =========================================================
# Legend
# =========================================================
legend_handles = [
    Patch(
        facecolor=non_note_color,
        edgecolor="none",
        label="Non-leavers"
    ),
    Patch(
        facecolor=note_color,
        edgecolor="none",
        label="Note-leavers"
    ),
    Line2D(
        [0],
        [0],
        color=line_color,
        linewidth=1.7,
        marker="o",
        markersize=6,
        label="Proportion of Note Leavers"
    )
]

fig.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(0.92, 0.94),
    prop={"family": "Arial", "size": 9},
    frameon=True,
    fancybox=False,
    framealpha=1,
    edgecolor="black",
    borderpad=1
)



# =========================================================
# Save as PNG and JPEG
# =========================================================
common_save_options = {
    "dpi": 600,
    "facecolor": "white",
    "bbox_inches": "tight",
    "pad_inches": 0.05
}

# Lossless PNG
fig.savefig(
    PNG_PATH,
    format="png",
    **common_save_options
)

# High-quality JPEG
fig.savefig(
    JPG_PATH,
    format="jpg",
    pil_kwargs={
        "quality": 95,
        "subsampling": 0
    },
    **common_save_options
)

print("\nFigures saved successfully:")
print(f"PNG: {PNG_PATH}")
print(f"JPG: {JPG_PATH}")


plt.show()