# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:18:41 2025

@author: Jae Bin Park
"""


#%%
"""Final Code for Plotting Note Leaver and No Note Leaver Count Distribution across Years of Months
   + a second figure with the monthly proportion of note leavers as a line plot
"""

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
#%%

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

#%% Correlation Test
note = [grouped.get((year, 1), 0) for year in years]
non_note = [grouped.get((year, 2), 0) for year in years]

from scipy.stats import pearsonr

r, p = pearsonr(note, non_note)
print(f"Pearson correlation: r = {r:.3f}, p = {p:.4f}")

from scipy.stats import spearmanr

rho, pval = spearmanr(note, non_note)
print(f"Spearman correlation: rho = {rho:.3f}, p = {pval:.4f}")

#%% Finding the lowest correlation score
corr_df = pd.DataFrame({
    'YearMonth': years,
    'note_leaver': note,
    'non_note_leaver': non_note
})

# Use rolling window (e.g., 12 months) to compute correlation between the two columns
corr_df['pearson_corr'] = corr_df[['note_leaver', 'non_note_leaver']].rolling(window=12).corr().iloc[0::2]['non_note_leaver'].reset_index(drop=True)

# Drop NaNs from initial windows
lowest_corr = corr_df.dropna().sort_values(by='pearson_corr')

print("Months with the lowest correlation between note and non-note leavers:")
print(lowest_corr[['YearMonth', 'pearson_corr']].head(5))

#%%

import pandas as pd
from scipy.stats import pearsonr

# Store rolling Pearson r and p-value
def rolling_pearson_r_p(df):
    r_vals = []
    p_vals = []
    for i in range(len(df) - 11):  # 12-month rolling window
        window = df.iloc[i:i+12]
        r, p = pearsonr(window['note_leaver'], window['non_note_leaver'])
        r_vals.append(r)
        p_vals.append(p)
    
    # Pad the beginning with NaNs to match original index
    nan_pad = [float('nan')] * 11
    return pd.Series(nan_pad + r_vals), pd.Series(nan_pad + p_vals)

corr_df['pearson_corr'], corr_df['pearson_pval'] = rolling_pearson_r_p(corr_df)
lowest_corr = corr_df.dropna().sort_values(by='pearson_corr')
print(lowest_corr[['YearMonth', 'pearson_corr', 'pearson_pval']].head(5))

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Load custom font
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Regular.ttf'
font_prop = fm.FontProperties(fname=font_path)

# Group data by YEAR, SEX, and NOTE_EX
grouped = kfsp.groupby(['Suicide_YearSeason', 'SEX', 'NOTE_EX']).size()

# Prepare index values
years = sorted(kfsp['Suicide_YearSeason'].unique())
x = np.arange(len(years))
width = 0.2

# Initialize lists for each category
male_note = []
female_note = []
male_non_note = []
female_non_note = []

# Fill counts for each category
for year in years:
    male_note.append(grouped.get((year, 1, 1), 0))         # Male - Note
    female_note.append(grouped.get((year, 2, 1), 0))       # Female - Note
    male_non_note.append(grouped.get((year, 1, 2), 0))     # Male - No Note
    female_non_note.append(grouped.get((year, 2, 2), 0))   # Female - No Note

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plot positive bars for note leavers
ax.bar(x - width/2, male_note, width=width, label='Male - Note', color='red', edgecolor='black', zorder=2)
ax.bar(x + width/2, female_note, width=width, label='Female - Note', color='#FF9999', edgecolor='black', zorder=2)

# Plot negative bars for non-note leavers
ax.bar(x - width/2, [-v for v in male_non_note], width=width, label='Male - No Note', color='lightgrey', edgecolor='black', zorder=2)
ax.bar(x + width/2, [-v for v in female_non_note], width=width, label='Female - No Note', color='darkgrey', edgecolor='black', zorder=2)

# Format axis
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, fontproperties=font_prop)
ax.set_ylabel('Count', fontproperties=font_prop)

# Show positive labels for negative bars
yticks = ax.get_yticks()
ax.set_yticklabels([abs(int(t)) for t in yticks], fontproperties=font_prop)



# Legend and grid
# Modify the font_prop to include size
font_prop_legend = fm.FontProperties(fname=font_path, size=9)

# Then use it in the legend
#ax.legend(prop=font_prop_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)


plt.grid(True, axis='both', linestyle='-', alpha=0.7 , zorder=0)
plt.tight_layout()
plt.show()


#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Load custom font
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Regular.ttf'
font_prop = fm.FontProperties(fname=font_path)

# Group data by YEAR, SEX, and NOTE_EX
grouped = kfsp.groupby(['Suicide_YearSeason', 'AGE2', 'NOTE_EX']).size()

# Prepare index values
years = sorted(kfsp['Suicide_YearSeason'].unique())
x = np.arange(len(years))
width = 0.2

# Initialize lists for each age group category
age1 = []  # ≤18
age2 = []  # 19–34
age3 = []  # 35–49
age4 = []  # 50–64
age5 = []  # 65+

age1_nonote = []
age2_nonote = []
age3_nonote = []
age4_nonote = []
age5_nonote = []

# Fill counts for each category
for year in years:
    age1.append(grouped.get((year, 1, 1), 0))         
    age2.append(grouped.get((year, 2, 1), 0))       
    age3.append(grouped.get((year, 3, 1), 0))        
    age4.append(grouped.get((year, 4, 1), 0))      
    age5.append(grouped.get((year, 5, 1), 0))        

    age1_nonote.append(grouped.get((year, 1, 2), 0))
    age2_nonote.append(grouped.get((year, 2, 2), 0))
    age3_nonote.append(grouped.get((year, 3, 2), 0))
    age4_nonote.append(grouped.get((year, 4, 2), 0))
    age5_nonote.append(grouped.get((year, 5, 2), 0))

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))

# Define parameters
width = 0.12
x = np.arange(len(years))
offsets = [-2*width, -width, 0, width, 2*width]

# Plot positive bars for note leavers
ax.bar(x + offsets[0], age1, width=width, label='≤18 - Note', color='crimson', edgecolor='black', zorder=2)
ax.bar(x + offsets[1], age2, width=width, label='19–34 - Note', color='salmon', edgecolor='black', zorder=2)
ax.bar(x + offsets[2], age3, width=width, label='35–49 - Note', color='orange', edgecolor='black', zorder=2)
ax.bar(x + offsets[3], age4, width=width, label='50–64 - Note', color='gold', edgecolor='black', zorder=2)
ax.bar(x + offsets[4], age5, width=width, label='65+ - Note', color='tomato', edgecolor='black', zorder=2)

# Plot negative bars for non-note leavers
ax.bar(x + offsets[0], [-v for v in age1_nonote], width=width, label='≤18 - No Note', color='lightgrey', edgecolor='black', zorder=2)
ax.bar(x + offsets[1], [-v for v in age2_nonote], width=width, label='19–34 - No Note', color='darkgrey', edgecolor='black', zorder=2)
ax.bar(x + offsets[2], [-v for v in age3_nonote], width=width, label='35–49 - No Note', color='grey', edgecolor='black', zorder=2)
ax.bar(x + offsets[3], [-v for v in age4_nonote], width=width, label='50–64 - No Note', color='dimgray', edgecolor='black', zorder=2)
ax.bar(x + offsets[4], [-v for v in age5_nonote], width=width, label='65+ - No Note', color='black', edgecolor='black', zorder=2)

# Format axis
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, fontproperties=font_prop)
ax.set_ylabel('Count', fontproperties=font_prop)

# Show positive labels for negative bars
yticks = ax.get_yticks()
ax.set_yticklabels([abs(int(t)) for t in yticks], fontproperties=font_prop)


# Legend and grid
# Modify the font_prop to include size
font_prop_legend = fm.FontProperties(fname=font_path, size=9)

# Then use it in the legend
#ax.legend(prop=font_prop_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

plt.grid(True, axis='both', linestyle='-', alpha=0.7 , zorder=0)
plt.tight_layout()
plt.show()
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Load custom font
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Regular.ttf'
font_prop = fm.FontProperties(fname=font_path)

# Group data by YEAR, CAUSE, and NOTE_EX
grouped = kfsp.groupby(['Suicide_YearMonth', 'MAIN_CAUSE', 'NOTE_EX']).size()

# Prepare index values
years = sorted(kfsp['Suicide_YearMonth'].unique())
x = np.arange(len(years))
width = 0.12
offsets = [-2.5*width, -1.5*width, -0.5*width, 0.5*width, 1.5*width, 2.5*width]

# Initialize lists for each cause group
cause1, cause2, cause3, cause4, cause5, cause6 = [], [], [], [], [], []
cause1_nonote, cause2_nonote, cause3_nonote, cause4_nonote, cause5_nonote, cause6_nonote = [], [], [], [], [], []

# Fill counts for each category
for year in years:
    cause1.append(grouped.get((year, 1, 1), 0))  # Job Problems - Note
    cause2.append(grouped.get((year, 2, 1), 0))  # Economical Problems - Note
    cause3.append(grouped.get((year, 3, 1), 0))  # Family Problems - Note
    cause4.append(grouped.get((year, 4, 1), 0))  # Social Problems - Note
    cause5.append(grouped.get((year, 5, 1), 0))  # Physical Health - Note
    cause6.append(grouped.get((year, 6, 1), 0))  # Mental Health - Note

    cause1_nonote.append(grouped.get((year, 1, 2), 0))
    cause2_nonote.append(grouped.get((year, 2, 2), 0))
    cause3_nonote.append(grouped.get((year, 3, 2), 0))
    cause4_nonote.append(grouped.get((year, 4, 2), 0))
    cause5_nonote.append(grouped.get((year, 5, 2), 0))
    cause6_nonote.append(grouped.get((year, 6, 2), 0))

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))

# Plot positive bars (Note leavers)
ax.bar(x + offsets[0], cause1, width=width, label='Job Problems - Note', color='#D32F2F', edgecolor='black', zorder=2)
ax.bar(x + offsets[1], cause2, width=width, label='Economical Problems - Note', color='#F57C00', edgecolor='black', zorder=2)
ax.bar(x + offsets[2], cause3, width=width, label='Family Problems - Note', color='#FBC02D', edgecolor='black', zorder=2)
ax.bar(x + offsets[3], cause4, width=width, label='Social Problems - Note', color='#388E3C', edgecolor='black', zorder=2)
ax.bar(x + offsets[4], cause5, width=width, label='Physical Health Problems - Note', color='#1976D2', edgecolor='black', zorder=2)
ax.bar(x + offsets[5], cause6, width=width, label='Mental Health Problems - Note', color='#7B1FA2', edgecolor='black', zorder=2)

# Plot negative bars (No Note leavers)
ax.bar(x + offsets[0], [-v for v in cause1_nonote], width=width, label='Job Problems - No Note', color='lightgrey', edgecolor='black', zorder=2)
ax.bar(x + offsets[1], [-v for v in cause2_nonote], width=width, label='Economical Problems - No Note', color='darkgrey', edgecolor='black', zorder=2)
ax.bar(x + offsets[2], [-v for v in cause3_nonote], width=width, label='Family Problems - No Note', color='grey', edgecolor='black', zorder=2)
ax.bar(x + offsets[3], [-v for v in cause4_nonote], width=width, label='Social Problems - No Note', color='dimgray', edgecolor='black', zorder=2)
ax.bar(x + offsets[4], [-v for v in cause5_nonote], width=width, label='Physical Health Problems - No Note', color='black', edgecolor='black', zorder=2)
ax.bar(x + offsets[5], [-v for v in cause6_nonote], width=width, label='Mental Health Problems - No Note', color='#444444', edgecolor='black', zorder=2)

# Format axis
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, fontproperties=font_prop)
ax.set_ylabel('Count', fontproperties=font_prop)

# Show positive labels for negative bars
yticks = ax.get_yticks()
ax.set_yticklabels([abs(int(t)) for t in yticks], fontproperties=font_prop)

# Legend and grid
# Modify the font_prop to include size
font_prop_legend = fm.FontProperties(fname=font_path, size=9)

# Then use it in the legend
#ax.legend(prop=font_prop_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

plt.grid(True, axis='both', linestyle='-', alpha=0.7 , zorder=0)
plt.tight_layout()
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Load custom font
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Regular.ttf'
font_prop = fm.FontProperties(fname=font_path)

# ASSUMING you have already created the METHOD_GROUP column:
# method_mapping = {1: 1, 2: 1, 3: 1, 16: 1, 4: 2, 5: 2, ..., 20: 6, 999: 6}
# kfsp['METHOD_GROUP'] = kfsp['MAIN_METHOD'].map(method_mapping)

# Group data by YEAR, METHOD_GROUP, and NOTE_EX
grouped = kfsp.groupby(['Suicide_YearMonth', 'METHOD_GROUP', 'NOTE_EX']).size()

# Prepare index values
years = sorted(kfsp['Suicide_YearMonth'].unique())
x = np.arange(len(years))
width = 0.12
offsets = [-2.5*width, -1.5*width, -0.5*width, 0.5*width, 1.5*width, 2.5*width]

# Initialize lists for each method group
method1, method2, method3, method4, method5, method6 = [], [], [], [], [], []
method1_nonote, method2_nonote, method3_nonote, method4_nonote, method5_nonote, method6_nonote = [], [], [], [], [], []

# Fill counts for each category
for year in years:
    method1.append(grouped.get((year, 1, 1), 0))  # Medicine - Note
    method2.append(grouped.get((year, 2, 1), 0))  # Chemical - Note
    method3.append(grouped.get((year, 3, 1), 0))  # Asphyxiation - Note
    method4.append(grouped.get((year, 4, 1), 0))  # Jumping - Note
    method5.append(grouped.get((year, 5, 1), 0))  # Injury - Note
    method6.append(grouped.get((year, 6, 1), 0))  # Other - Note

    method1_nonote.append(grouped.get((year, 1, 2), 0))  # Medicine - No Note
    method2_nonote.append(grouped.get((year, 2, 2), 0))  # Chemical - No Note
    method3_nonote.append(grouped.get((year, 3, 2), 0))  # Asphyxiation - No Note
    method4_nonote.append(grouped.get((year, 4, 2), 0))  # Jumping - No Note
    method5_nonote.append(grouped.get((year, 5, 2), 0))  # Injury - No Note
    method6_nonote.append(grouped.get((year, 6, 2), 0))  # Other - No Note

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))


# Positive bars (Note leavers)
ax.bar(x + offsets[0], method1, width=width, label='Medicine - Note', color='#D32F2F', edgecolor='black', zorder=2)
ax.bar(x + offsets[1], method2, width=width, label='Chemical - Note', color='#F57C00', edgecolor='black', zorder=2)
ax.bar(x + offsets[2], method3, width=width, label='Asphyxiation - Note', color='#FBC02D', edgecolor='black', zorder=2)
ax.bar(x + offsets[3], method4, width=width, label='Jumping - Note', color='#388E3C', edgecolor='black', zorder=2)
ax.bar(x + offsets[4], method5, width=width, label='Self-Injury - Note', color='#1976D2', edgecolor='black', zorder=2)
ax.bar(x + offsets[5], method6, width=width, label='Other - Note', color='#7B1FA2', edgecolor='black', zorder=2)

# Negative bars (No Note leavers)
ax.bar(x + offsets[0], [-v for v in method1_nonote], width=width, label='Medicine - No Note', color='lightgrey', edgecolor='black', zorder=2)
ax.bar(x + offsets[1], [-v for v in method2_nonote], width=width, label='Chemical - No Note', color='darkgrey', edgecolor='black', zorder=2)
ax.bar(x + offsets[2], [-v for v in method3_nonote], width=width, label='Asphyxiation - No Note', color='grey', edgecolor='black', zorder=2)
ax.bar(x + offsets[3], [-v for v in method4_nonote], width=width, label='Jumping - No Note', color='dimgray', edgecolor='black', zorder=2)
ax.bar(x + offsets[4], [-v for v in method5_nonote], width=width, label='Self-Injury - No Note', color='black', edgecolor='black', zorder=2)
ax.bar(x + offsets[5], [-v for v in method6_nonote], width=width, label='Other - No Note', color='#444444', edgecolor='black', zorder=2)

# Format axis
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, fontproperties=font_prop)
ax.set_ylabel('Count', fontproperties=font_prop)

# Show positive labels for negative bars
yticks = ax.get_yticks()
ax.set_yticklabels([abs(int(t)) for t in yticks], fontproperties=font_prop)

# Legend and grid
font_prop_legend = fm.FontProperties(fname=font_path, size=9)
#ax.legend(prop=font_prop_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

plt.grid(True, axis='both', linestyle='-', alpha=0.7 , zorder=0)
plt.tight_layout()
plt.show()
#%% For proportions

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Load custom font
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Regular.ttf'
font_prop = fm.FontProperties(fname=font_path)

# Assume METHOD_GROUP is already created
# Group data by YEAR, METHOD_GROUP, and NOTE_EX
grouped = kfsp.groupby(['Suicide_YearSeason', 'METHOD_GROUP', 'NOTE_EX']).size()

# Prepare index values
years = sorted(kfsp['Suicide_YearSeason'].unique())
x = np.arange(len(years))
width = 0.12
offsets = [-2.5*width, -1.5*width, -0.5*width, 0.5*width, 1.5*width, 2.5*width]

# Initialize lists
method1, method2, method3, method4, method5, method6 = [], [], [], [], [], []
method1_nonote, method2_nonote, method3_nonote, method4_nonote, method5_nonote, method6_nonote = [], [], [], [], [], []

# Fill counts proportionally
for year in years:
    # Calculate total suicides that year (both with and without notes)
    total = sum([grouped.get((year, method, note), 0) for method in range(1, 7) for note in [1, 2]])

    if total == 0:
        total = 1  # To avoid division by zero

    method1.append(grouped.get((year, 1, 1), 0) / total)
    method2.append(grouped.get((year, 2, 1), 0) / total)
    method3.append(grouped.get((year, 3, 1), 0) / total)
    method4.append(grouped.get((year, 4, 1), 0) / total)
    method5.append(grouped.get((year, 5, 1), 0) / total)
    method6.append(grouped.get((year, 6, 1), 0) / total)

    method1_nonote.append(grouped.get((year, 1, 2), 0) / total)
    method2_nonote.append(grouped.get((year, 2, 2), 0) / total)
    method3_nonote.append(grouped.get((year, 3, 2), 0) / total)
    method4_nonote.append(grouped.get((year, 4, 2), 0) / total)
    method5_nonote.append(grouped.get((year, 5, 2), 0) / total)
    method6_nonote.append(grouped.get((year, 6, 2), 0) / total)

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))

# Positive bars (Note leavers)
ax.bar(x + offsets[0], method1, width=width, label='Medicine - Note', color='#D32F2F', edgecolor='black', zorder=2)
ax.bar(x + offsets[1], method2, width=width, label='Chemical - Note', color='#F57C00', edgecolor='black', zorder=2)
ax.bar(x + offsets[2], method3, width=width, label='Asphyxiation - Note', color='#FBC02D', edgecolor='black', zorder=2)
ax.bar(x + offsets[3], method4, width=width, label='Jumping - Note', color='#388E3C', edgecolor='black', zorder=2)
ax.bar(x + offsets[4], method5, width=width, label='Injury - Note', color='#1976D2', edgecolor='black', zorder=2)
ax.bar(x + offsets[5], method6, width=width, label='Other - Note', color='#7B1FA2', edgecolor='black', zorder=2)

# Negative bars (No Note leavers)
ax.bar(x + offsets[0], [-v for v in method1_nonote], width=width, label='Medicine - No Note', color='lightgrey', edgecolor='black', zorder=2)
ax.bar(x + offsets[1], [-v for v in method2_nonote], width=width, label='Chemical - No Note', color='darkgrey', edgecolor='black', zorder=2)
ax.bar(x + offsets[2], [-v for v in method3_nonote], width=width, label='Asphyxiation - No Note', color='grey', edgecolor='black', zorder=2)
ax.bar(x + offsets[3], [-v for v in method4_nonote], width=width, label='Jumping - No Note', color='dimgray', edgecolor='black', zorder=2)
ax.bar(x + offsets[4], [-v for v in method5_nonote], width=width, label='Injury - No Note', color='black', edgecolor='black', zorder=2)
ax.bar(x + offsets[5], [-v for v in method6_nonote], width=width, label='Other - No Note', color='#444444', edgecolor='black', zorder=2)

# Format axis
#ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, fontproperties=font_prop)
ax.set_ylabel('Proportion', fontproperties=font_prop)

# Update y-axis ticks to percentage format
yticks = ax.get_yticks()
ax.set_yticklabels([f'{abs(t*100):.0f}%' for t in yticks], fontproperties=font_prop)

# Legend and grid
font_prop_legend = fm.FontProperties(fname=font_path, size=9)
# ax.legend(prop=font_prop_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

plt.grid(True, axis='both', linestyle='-', alpha=0.7 , zorder=0)
plt.tight_layout()
plt.show()
#%%

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

def plot_word_frequencies_df(df,
                             word_col="word",
                             freq_col="count",
                             title="Word Frequency Plot",
                             font_path=None,
                             font_size=12,
                             num_words=None,
                             start=1,
                             end=None,
                             horizontal=True,
                             show_values=True,
                             sort=True,
                             palette=None,
                             proportion=False,
                             color="darkred",
                             ax=None,
                             word_translation_dict=None):  # NEW PARAMETER

    """
    Plots word frequency data from a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - word_col: column name for the word labels.
    - freq_col: column name for frequencies or proportions.
    - title: title of the plot.
    - font_path: path to the font file (.ttf).
    - font_size: size of the font.
    - num_words: number of top words to display.
    - start, end: custom row range (1-based).
    - horizontal: plot horizontal bars if True.
    - show_values: show value labels on bars.
    - sort: sort by frequency before selecting top rows.
    - palette: optional seaborn color palette.
    - proportion: if True, normalize using total of entire dataset.
    - color: single bar color if palette is not used.
    - ax: optional matplotlib axis object for subplotting.
    """

    # Compute proportions before any filtering
    if proportion:
        total = df[freq_col].sum()
        df_plot = df[[word_col, freq_col]].copy()
        df_plot[freq_col] = df_plot[freq_col] / total
    else:
        df_plot = df[[word_col, freq_col]].copy()

    # Sort and slice
    if sort:
        df_plot = df_plot.sort_values(by=freq_col, ascending=False)

    if num_words is not None:
        df_plot = df_plot.head(num_words)
    else:
        end = end if end is not None else len(df_plot)
        df_plot = df_plot.iloc[start - 1:end]

    # Font setup
    fontprop = fm.FontProperties(fname=font_path, size=font_size) if font_path else None

    # Create figure if not passed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    # Plot
    if horizontal:
        sns.barplot(data=df_plot, x=freq_col, y=word_col,
                    palette=palette, color=color, edgecolor='black', ax=ax, zorder=2)
        ax.set_xlabel("Proportion" if proportion else "Frequency", fontproperties=fontprop)
        ax.set_ylabel("", fontproperties=fontprop)
    else:
        sns.barplot(data=df_plot, x=word_col, y=freq_col,
                    palette=palette, color=color, edgecolor='black', ax=ax, zorder=2)
        ax.set_xlabel("", fontproperties=fontprop)
        ax.set_ylabel("Proportion" if proportion else "Frequency", fontproperties=fontprop)
    if word_translation_dict:
        if horizontal:
            labels = [t.get_text() for t in ax.get_yticklabels()]
            new_labels = [word_translation_dict.get(label, label) for label in labels]
            ax.set_yticklabels(new_labels, fontproperties=fontprop)
        else:
            labels = [t.get_text() for t in ax.get_xticklabels()]
            new_labels = [word_translation_dict.get(label, label) for label in labels]
            ax.set_xticklabels(new_labels, rotation=90, fontproperties=fontprop)

    ax.set_title(title, fontproperties=fontprop, fontsize=font_size + 2)

    if show_values:
        for container in ax.containers:
            if proportion:
                ax.bar_label(container, fmt='%.3f', fontproperties=fontprop)
            else:
                ax.bar_label(container, fmt='%.0f', fontproperties=fontprop)

    if created_fig:
        plt.tight_layout()
        plt.show()


#%%Noun Filter

# List of words you want to remove
stopwords  = ['말', '이상', '애', '내용', '앞', '동안', '이제', '중략','밖','곳', '다음', '옆', '글', '이유', 
              '속', '모두', '문자', '이름', '눈', '전', '그동안', '변사자', '장', '후', "지금", "미안", "조금", "곁",
              "뒤", "이번", "처음", "잘못", "위", "나중", "그때", "층", '테', '당']


#%% To run this code, run the necessary sections in KFSP Note Analysis (Selectable Main GOIs)
gendernoun={}
for name, df in gender_nng_counts_results.items():
    df_cleaned = df[~df['Word'].isin(stopwords)].reset_index(drop=True)
    gendernoun[name] = df_cleaned


agenoun={}
for name, df in age_nng_counts_results.items():
    df_cleaned = df[~df['Word'].isin(stopwords)].reset_index(drop=True)
    agenoun[name] = df_cleaned
    
psmmurdernoun={}
for name, df in psmmurder_nng_counts_results.items():
    df_cleaned = df[~df['Word'].isin(stopwords)].reset_index(drop=True)
    psmmurdernoun[name] = df_cleaned

whole_cleanednng_counts_results = whole_nng_counts_results[~whole_nng_counts_results["Word"].isin(stopwords)].reset_index(drop=True)

#%%

wholenote100nng_cleaned=whole_cleanednng_counts_results[:100]

male100noun=gendernoun["kfspmale"][:100]
female100noun=gendernoun["kfspfemale"][:100]

murder100noun=psmmurdernoun["psmmurder"][:100]
other100noun=psmmurdernoun["psmnomurder"][:100]



age1_100noun = agenoun["kfspage1"][:100]
age2_100noun = agenoun["kfspage2"][:100]
age3_100noun = agenoun["kfspage3"][:100]
age4_100noun = agenoun["kfspage4"][:100]
age5_100noun = agenoun["kfspage5"][:100]

#%%

wholenote150nng_cleaned=whole_cleanednng_counts_results[:150]
male150noun=gendernoun["kfspmale"][:150]
female150noun=gendernoun["kfspfemale"][:150]

murder150noun=psmmurdernoun["psmmurder"][:150]
other150noun=psmmurdernoun["psmnomurder"][:150]



age1_150noun = agenoun["kfspage1"][:150]
age2_150noun = agenoun["kfspage2"][:150]
age3_150noun = agenoun["kfspage3"][:150]
age4_150noun = agenoun["kfspage4"][:150]
age5_150noun = agenoun["kfspage5"][:150]
#%%

# List of words you want to remove
stopverbs  = ["하","되","있","그러", "하았", "안하", "들", "되었", "그러었", "말", "잘살", "다하"]


#%%
genderverb={}
for name, df in gender_vv_counts_results.items():
    df_cleaned = df[~df['Word'].isin(stopverbs)].reset_index(drop=True)
    genderverb[name] = df_cleaned



ageverb={}
for name, df in age_vv_counts_results.items():
    df_cleaned = df[~df['Word'].isin(stopverbs)].reset_index(drop=True)
    ageverb[name] = df_cleaned

psmmurderverb={}
for name, df in psmmurder_vv_counts_results.items():
    df_cleaned = df[~df['Word'].isin(stopverbs)].reset_index(drop=True)
    psmmurderverb[name] = df_cleaned


wholenotevv = whole_vv_counts_results[~whole_vv_counts_results["Word"].isin(stopverbs)].reset_index(drop=True)


#%%
wholenote100vv_cleaned=wholenotevv[:100]


male100verb = genderverb["kfspmale"][:100]
female100verb = genderverb["kfspfemale"][:100]

murder100verb=psmmurderverb["psmmurder"][:100]
other100verb=psmmurderverb["psmnomurder"][:100]


age1_100verb = ageverb["kfspage1"][:100]
age2_100verb = ageverb["kfspage2"][:100]
age3_100verb = ageverb["kfspage3"][:100]
age4_100verb = ageverb["kfspage4"][:100]
age5_100verb = ageverb["kfspage5"][:100]

#%%

wholenote150vv_cleaned=wholenotevv[:150]


male150verb = genderverb["kfspmale"][:150]
female150verb = genderverb["kfspfemale"][:150]

murder150verb=psmmurderverb["psmmurder"][:150]
other150verb=psmmurderverb["psmnomurder"][:150]


age1_150verb = ageverb["kfspage1"][:150]
age2_150verb = ageverb["kfspage2"][:150]
age3_150verb = ageverb["kfspage3"][:150]
age4_150verb = ageverb["kfspage4"][:150]
age5_150verb = ageverb["kfspage5"][:150]

#%%

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

#%%

fig, axs = plt.subplots(2, 1, figsize=(16, 6))

plot_word_frequencies_df(wholenote100nng_cleaned,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=50,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=True,
                         show_values=False,
                         proportion=True,
                         ax=axs[0],
                         word_translation_dict=korean_wholenouns_to_english,
                         color="#2166AC")

plot_word_frequencies_df(wholenote100vv_cleaned,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=50,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=True,
                         show_values=False,
                         proportion=True,
                         ax=axs[1],
                         word_translation_dict=korean_verbs_to_english,
                         color="#2166AC")
                         

#%%

fig, axs = plt.subplots(2, 1, figsize=(16, 6))

plot_word_frequencies_df(male100noun,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=50,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=True,
                         show_values=False,
                         proportion=True,
                         ax=axs[0],
                         word_translation_dict=korean_wholenouns_to_english)

plot_word_frequencies_df(male100verb,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=50,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=True,
                         show_values=False,
                         proportion=True,
                         ax=axs[1],
                         word_translation_dict=korean_verbs_to_english)
                         

#%%

fig, axs = plt.subplots(2, 1, figsize=(16, 6))

plot_word_frequencies_df(female100noun,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=50,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[0],
                         word_translation_dict=korean_wholenouns_to_english)

plot_word_frequencies_df(female100verb,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=50,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[1],
                         word_translation_dict=korean_verbs_to_english)
                         

#%% Age 1 Frequency Graph

age1_100noun["Word"]["엄마아빠"]


korean_nouns_to_english = {
    "엄마": "Mom",
    "아빠": "Dad",
    "사람": "Person",
    "친구": "Friend",
    "가족": "Family",
    "일": "Work",
    "마지막": "Last",
    "생각": "Thought",
    "날": "Day",
    "딸": "Daughter",
    "학교": "School",
    "언니": "Older Sister (by female)",
    "누나": "Older Sister (by male)",
    "자신": "Oneself",
    "부모": "Parents",
    "공부": "Study",
    "오빠": "Older Brother (by female)",
    "마음": "Heart",
    "유서": "Will",
    "동생": "Younger Sibling",
    "집": "Home",
    "돈": "Money",
    "시간": "Time",
    "끝": "End",
    "인생": "Life",
    "형": "Older Brother (by male)",
    "죽음": "Death",
    "자살": "Suicide",
    "선생": "Teacher",
    "할머니": "Grandmother",
    "엄마아빠": "Mom and Dad",
    "삶": "Life",
    "꿈": "Dream",
    "아버지": "Father",
    "편지": "Letter",
    "선택": "Choice",
    "사랑": "Love",
    "어머니": "Mother",
    "장례식": "Funeral",
    "탓": "Blame",
    "아들": "Son",
    "학년": "Grade",
    "모습": "Appearance",
    "부탁": "Request",
    "메시지": "Message",
    "상처": "Wound",
    "술": "Alcohol",
    "진심": "Sincerity",
    "미래": "Future",
    "남": "Others",
    "핸드폰": "Cell Phone",
    "눈물": "Tears",
    "혼자": "Alone",
    "이야기": "Story",
    "문제": "Problem",
    "행동": "Action",
    "하늘": "Sky",
    "인간": "Human",
    "스트레스": "Stress",
    "기억": "Memory",
    "순간": "Moment",
    "길": "Road",
    "추억": "Reminiscence",
    "성격": "Personality",
    "용기": "Courage",
    "방": "Room",
    "고등학교": "High School",
    "오늘": "Today",
    "기분": "Mood",
    "얘기": "Talk",
    "몸": "Body",
    "카톡": "KakaoTalk",
    "필요": "Need",
    "사망": "Death",
    "반": "Class",
    "아이": "Child",
    "힘": "Strength",
    "화": "Anger",
    "남자친구": "Boyfriend",
    "사이": "Relationship",
    "얼굴": "Face",
    "걱정": "Worry",
    "방법": "Method",
    "자식": "Offspring",
    "머리": "Head",
    "휴대폰": "Mobile Phone",
    "여행": "Trip",
    "현실": "Reality",
    "소리": "Sound",
    "몫": "Share",
    "이모": "Aunt",
    "말씀": "Words",
    "사진": "Photo",
    "후회": "Regret",
    "성적": "Grades"
}
korean_verbs_to_english = {
    "미안하": "Be sorry",
    "사랑하": "Love",
    "죽": "Die",
    "살": "Live",
    "가": "Go",
    "죄송하": "Apologize",
    "보": "See",
    "생각하": "Think",
    "알": "Know",
    "말하": "Say",
    "못하": "Not do",
    "감사하": "Thank",
    "모르겠": "Not know",
    "쓰": "Write",
    "받": "Receive",
    "오": "Come",
    "잘하": "Do well",
    "안되": "Not work",
    "모르": "Not know",
    "만나": "Meet",
    "바라": "Hope",
    "울": "Cry",
    "태어나": "Be born",
    "슬퍼하": "Grieve",
    "키우": "Raise",
    "자살하": "Commit suicide",
    "먹": "Eat",
    "지내": "Spend time",
    "잊": "Forget",
    "전하": "Deliver",
    "챙기": "Take care of",
    "위하": "Care",
    "믿": "Believe",
    "노력하": "Try",
    "보이": "Show",
    "알았": "Have known",
    "사": "Buy",
    "나": "Come out",
    "놀": "Play",
    "보내": "Send",
    "원하": "Want",
    "나오": "Come out",
    "웃": "Laugh",
    "공부하": "Study",
    "맞": "Fit",
    "있었": "Have existed",
    "떠나": "Leave",
    "느끼": "Feel",
    "주": "Give",
    "이해하": "Understand",
    "듣": "Hear",
    "대하": "Face",
    "찾": "Find",
    "죽었": "Have died",
    "가았": "Have gone",
    "용서하": "Forgive",
    "드리": "Offer",
    "행복하": "Be happy",
    "썩이": "Rot",
    "싸우": "Fight",
    "남": "Remain",
    "지키": "Keep",
    "남기": "Leave behind",
    "포기하": "Give up",
    "읽": "Read",
    "버티": "Endure",
    "원망하": "Resent",
    "살았": "Have lived",
    "이러": "Be like this",
    "가지": "Have",
    "기억하": "Remember",
    "다니": "Attend",
    "죽이": "Kill",
    "치": "Hit",
    "고맙": "Be thankful",
    "돕": "Help",
    "부탁하": "Ask",
    "욕하": "Swear",
    "알리": "Inform",
    "작성하": "Write",
    "알겠": "Understand",
    "사라지": "Disappear",
    "성공하": "Succeed",
    "모르았": "Have not known",
    "만들": "Make",
    "낳": "Give birth",
    "쉬": "Rest",
    "하시": "Do",
    "살시": "Be alive",
    "들었": "Heard",
    "올리": "Raise",
    "잡": "Grab",
    "들어가": "Enter",
    "선택하": "Choose",
    "걱정하": "Worry",
    "못되": "Go wrong",
    "내": "Let out",
    "끼치": "Cause",
    "이루": "Achieve",
    "지치": "Be exhausted"
}

fig, axs = plt.subplots(2, 1, figsize=(16, 6))

plot_word_frequencies_df(age1_100noun,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=75,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[0],
                         word_translation_dict=korean_nouns_to_english)

plot_word_frequencies_df(age1_100verb,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=75,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[1],
                         word_translation_dict=korean_verbs_to_english)
#%%

korean_nouns_to_english = {
    "엄마": "Mom",
    "사람": "Person",
    "아빠": "Dad",
    "가족": "Family",
    "친구": "Friend",
    "일": "Work",
    "마지막": "Last moment",
    "돈": "Money",
    "마음": "Heart",
    "생각": "Thought",
    "오빠": "Older brother (by female)",
    "아버지": "Father",
    "자신": "Oneself",
    "형": "Older brother (by male)",
    "누나": "Older sister (by male)",
    "부모": "Parents",
    "아들": "Son",
    "유서": "Suicide note",
    "언니": "Older sister (by female)",
    "어머니": "Mother",
    "동생": "Younger sibling",
    "집": "House",
    "날": "Day",
    "삶": "Life",
    "인생": "Life (whole)",
    "끝": "End",
    "선택": "Choice",
    "시간": "Time",
    "부탁": "Request",
    "딸": "Daughter",
    "빚": "Debt",
    "모습": "Appearance",
    "상처": "Wound",
    "사랑": "Love",
    "힘": "Strength",
    "장례식": "Funeral",
    "몸": "Body",
    "죽음": "Death",
    "자살": "Suicide",
    "짐": "Burden",
    "진심": "Sincerity",
    "순간": "Moment",
    "고통": "Pain",
    "거짓말": "Lie",
    "남": "Others",
    "술": "Alcohol",
    "자식": "Child",
    "할머니": "Grandmother",
    "메시지": "Message",
    "평생": "Lifetime",
    "연락": "Contact",
    "길": "Path",
    "상황": "Situation",
    "아이": "Child",
    "고생": "Hardship",
    "방법": "Method",
    "혼자": "Alone",
    "탓": "Blame",
    "기억": "Memory",
    "여자친구": "Girlfriend",
    "편지": "Letter",
    "우울증": "Depression",
    "걱정": "Worry",
    "정신": "Mind",
    "꿈": "Dream",
    "용기": "Courage",
    "회사": "Company",
    "피해": "Harm",
    "핸드폰": "Cell phone",
    "지인": "Acquaintance",
    "비밀번호": "Password",
    "얼굴": "Face",
    "행복": "Happiness",
    "눈물": "Tears",
    "가슴": "Chest",
    "하루": "Day (whole)",
    "엄마아빠": "Mom and Dad",
    "장례": "Funeral",
    "죄": "Sin",
    "여자": "Woman",
    "죄책감": "Guilt",
    "희망": "Hope",
    "문제": "Problem",
    "행동": "Action",
    "전화": "Phone call",
    "통장": "Bankbook",
    "사진": "Photo",
    "얘기": "Story",
    "톡": "KakaoTalk",
    "남자": "Man",
    "주변": "Surroundings",
    "하늘": "Sky",
    "남편": "Husband",
    "사망": "Death (formal)",
    "불효": "Unfilial behavior",
    "대출": "Loan",
    "지옥": "Hell",
    "미래": "Future",
    "인간": "Human",
    "도움": "Help"
}

korean_verbs_to_english = {
    "미안하": "Sorry",
    "사랑하": "Love",
    "살": "Live",
    "죄송하": "Apologize",
    "죽": "Die",
    "가": "Go",
    "생각하": "Think",
    "보": "See",
    "알": "Know",
    "받": "Receive",
    "만나": "Meet",
    "안되": "Not work",
    "못하": "Not do",
    "부탁하": "Ask",
    "감사하": "Thank",
    "쓰": "Write",
    "용서하": "Forgive",
    "바라": "Hope",
    "말하": "Say",
    "모르겠": "Not know",
    "오": "Come",
    "잘하": "Do well",
    "떠나": "Leave",
    "먹": "Eat",
    "남기": "Leave behind",
    "태어나": "Be born",
    "살았": "Have lived",
    "주": "Give",
    "보내": "Send",
    "잊": "Forget",
    "보이": "Show",
    "키우": "Raise",
    "지내": "Spend time",
    "드리": "Offer",
    "모르": "Not know",
    "지키": "Keep",
    "믿": "Believe",
    "위하": "Care",
    "챙기": "Take care of",
    "사": "Buy",
    "슬퍼하": "Grieve",
    "전하": "Deliver",
    "나": "Come out",
    "화장하": "Cremate",
    "이해하": "Understand",
    "뿌리": "Scatter",
    "있었": "Have existed",
    "갚": "Repay",
    "알았": "Have known",
    "마시": "Drink",
    "쉬": "Rest",
    "듣": "Hear",
    "울": "Cry",
    "알리": "Inform",
    "행복하": "Be happy",
    "대하": "Face",
    "돕": "Help",
    "노력하": "Try",
    "느끼": "Feel",
    "만들": "Make",
    "가지": "Have",
    "인하": "Reduce",
    "나오": "Come out",
    "선택하": "Choose",
    "찾": "Find",
    "가았": "Have gone",
    "원망하": "Resent",
    "남": "Remain",
    "포기하": "Give up",
    "연락하": "Contact",
    "죽었": "Have died",
    "살아가": "Live on",
    "지치": "Be exhausted",
    "자살하": "Commit suicide",
    "원하": "Want",
    "시작하": "Start",
    "맞": "Fit",
    "잘못되": "Go wrong",
    "치": "Hit",
    "일하": "Work",
    "먼저가": "Go first",
    "웃": "Laugh",
    "얘기하": "Talk",
    "이러": "Be like this",
    "받았": "Have received",
    "버티": "Endure",
    "버리": "Throw away",
    "모르았": "Have not known",
    "낳": "Give birth",
    "끼치": "Cause",
    "다니": "Attend",
    "하시": "Do (honorific)",
    "기억하": "Remember",
    "그만하": "Stop",
    "정리하": "Organize",
    "잘지내": "Get along well",
    "결혼하": "Marry",
    "이기": "Win",
    "오았": "Have come",
    "시키": "Make someone do"
}

fig, axs = plt.subplots(2, 1, figsize=(16, 6))


plot_word_frequencies_df(age2_100noun,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=75,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[0],
                         word_translation_dict=korean_nouns_to_english)

plot_word_frequencies_df(age2_100verb,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=75,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[1],
                         word_translation_dict=korean_verbs_to_english)

#%%

korean_nouns_to_english = {
    "엄마": "Mom",
    "사람": "Person",
    "아빠": "Dad",
    "형": "Older brother (by male)",
    "가족": "Family",
    "돈": "Money",
    "아들": "Son",
    "마음": "Heart",
    "일": "Work",
    "마지막": "Last moment",
    "아버지": "Father",
    "누나": "Older sister (by male)",
    "어머니": "Mother",
    "부모": "Parents",
    "동생": "Younger sibling",
    "생각": "Thought",
    "집": "Home",
    "부탁": "Request",
    "딸": "Daughter",
    "유서": "Will",
    "삶": "Life",
    "친구": "Friend",
    "자신": "Oneself",
    "빚": "Debt",
    "날": "Day",
    "인생": "Life (whole)",
    "오빠": "Older brother (by female)",
    "끝": "End",
    "시간": "Time",
    "짐": "Burden",
    "언니": "Older sister (by female)",
    "몸": "Body",
    "선택": "Choice",
    "아이": "Child",
    "모습": "Appearance",
    "힘": "Strength",
    "사랑": "Love",
    "자식": "Children",
    "남편": "Husband",
    "고통": "Pain",
    "고생": "Hardship",
    "길": "Path",
    "상처": "Wound",
    "죄": "Sin",
    "진심": "Sincerity",
    "아내": "Wife",
    "죽음": "Death",
    "메시지": "Message",
    "용서": "Forgiveness",
    "가슴": "Chest (or Heart)",
    "방법": "Method",
    "맘": "Heart (informal)",
    "연락": "Contact",
    "장례": "Funeral",
    "술": "Alcohol",
    "통장": "Bank account",
    "순간": "Moment",
    "피해": "Harm",
    "상황": "Situation",
    "여자": "Woman",
    "지인": "Acquaintance",
    "회사": "Company",
    "남": "Man",
    "장례식": "Funeral ceremony",
    "걱정": "Worry",
    "하루": "Day (one day)",
    "어머님": "Mother (formal)",
    "도움": "Help",
    "전화": "Phone call",
    "자살": "Suicide",
    "희망": "Hope",
    "혼자": "Alone",
    "거짓말": "Lie",
    "눈물": "Tears",
    "차": "Car",
    "정신": "Mind",
    "카드": "Card",
    "용기": "Courage",
    "말씀": "Words (honorific)",
    "사망": "Death",
    "형제": "Siblings",
    "대출": "Loan",
    "인간": "Human",
    "문제": "Problem",
    "핸드폰": "Cell phone",
    "재산": "Property",
    "평생": "Lifetime",
    "병원": "Hospital",
    "우울증": "Depression",
    "기억": "Memory",
    "후회": "Regret",
    "하늘": "Sky",
    "식구": "Family member",
    "꿈": "Dream",
    "자리": "Place",
    "할머니": "Grandmother",
    "새끼": "Brat (colloquial/derogatory)",
    "얼굴": "Face",
    "조카": "Niece/nephew",
    "연락처": "Contact info"
}

korean_verbs_to_english = {
    "미안하": "Apologize",
    "사랑하": "Love",
    "살": "Live",
    "죄송하": "Apologize (Formal)",
    "가": "Go",
    "죽": "Die",
    "부탁하": "Ask",
    "보": "See",
    "생각하": "Think",
    "용서하": "Forgive",
    "알": "Know",
    "바라": "Hope",
    "만나": "Meet",
    "받": "Receive",
    "안되": "Fail",
    "못하": "Cannot do",
    "잘하": "Do well",
    "떠나": "Leave",
    "남기": "Leave behind",
    "화장하": "Cremate",
    "드리": "Offer",
    "뿌리": "Scatter",
    "오": "Come",
    "감사하": "Thank",
    "쓰": "Write",
    "주": "Give",
    "살았": "Lived",
    "지키": "Protect",
    "보내": "Send",
    "믿": "Believe",
    "보이": "Show",
    "말하": "Say",
    "모르겠": "Not know",
    "먹": "Eat",
    "모르": "Not know",
    "챙기": "Take care",
    "잊": "Forget",
    "위하": "Care for",
    "돕": "Help",
    "마시": "Drink",
    "키우": "Raise",
    "태어나": "Be born",
    "갚": "Repay",
    "인하": "Due to",
    "전하": "Deliver",
    "이해하": "Understand",
    "연락하": "Contact",
    "지내": "Get along",
    "사": "Buy",
    "정리하": "Organize",
    "슬퍼하": "Feel sad",
    "남": "Remain",
    "알리": "Inform",
    "나": "Come out",
    "찾": "Find",
    "듣": "Listen",
    "만들": "Make",
    "나오": "Come out",
    "선택하": "Choose",
    "가지": "Have",
    "처리하": "Handle",
    "원망하": "Resent",
    "알았": "Understood",
    "먼저가": "Go first",
    "포기하": "Give up",
    "부탁드리": "Humbly ask",
    "가았": "Went",
    "대하": "Treat",
    "노력하": "Try",
    "있었": "Existed",
    "잘못되": "Go wrong",
    "끼치": "Cause",
    "쉬": "Rest",
    "지치": "Be exhausted",
    "울": "Cry",
    "시작하": "Start",
    "오았": "Came",
    "살아가": "Go on living",
    "일하": "Work",
    "해결하": "Resolve",
    "전화하": "Call",
    "치": "Hit",
    "버티": "Endure",
    "버리": "Throw away",
    "얘기하": "Talk",
    "짓": "Build",
    "보살피": "Take care of",
    "적": "Write down",
    "살시": "Live (dialectal or typo?)",
    "시키": "Make someone do",
    "주시": "Please give",
    "감당하": "Bear",
    "하시": "Do (honorific)",
    "치르": "Pay (or undergo)",
    "느끼": "Feel",
    "맞": "Be right / Fit",
    "빌리": "Borrow",
    "미워하": "Hate",
    "작성하": "Write",
    "두": "Put / Keep"
}


fig, axs = plt.subplots(2, 1, figsize=(16, 6))


plot_word_frequencies_df(age3_100noun,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=75,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[0],
                         word_translation_dict=korean_nouns_to_english)


plot_word_frequencies_df(age3_100verb,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=75,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[1],
                         word_translation_dict=korean_verbs_to_english)

#%%

korean_nouns_to_english = {
    "엄마": "Mom",
    "아빠": "Dad",
    "사람": "Person",
    "아들": "Son",
    "돈": "Money",
    "가족": "Family",
    "마음": "Heart",
    "형": "Older brother (by male)",
    "딸": "Daughter",
    "일": "Work",
    "동생": "Younger sibling",
    "마지막": "Last",
    "집": "Home",
    "유서": "Suicide note",
    "몸": "Body",
    "인생": "Life",
    "아버지": "Father",
    "삶": "Life (existence)",
    "부탁": "Request",
    "길": "Road",
    "누나": "Older sister (by male)",
    "친구": "Friend",
    "어머니": "Mother",
    "짐": "Burden",
    "자신": "Oneself",
    "생각": "Thought",
    "끝": "End",
    "고생": "Hardship",
    "날": "Day",
    "자식": "Children",
    "시간": "Time",
    "용서": "Forgiveness",
    "힘": "Strength",
    "고통": "Pain",
    "죽음": "Death",
    "모습": "Appearance",
    "통장": "Bank account",
    "죄": "Sin",
    "형제": "Siblings",
    "빚": "Debt",
    "남편": "Husband",
    "아내": "Wife",
    "부모": "Parents",
    "아이": "Child",
    "선택": "Choice",
    "가슴": "Chest",
    "병원": "Hospital",
    "언니": "Older sister (by female)",
    "장례": "Funeral",
    "어머님": "Mother (honorific)",
    "사랑": "Love",
    "메시지": "Message",
    "술": "Alcohol",
    "산": "Mountain",
    "오빠": "Older brother (by female)",
    "남": "Man",
    "도움": "Help",
    "사망": "Death (formal)",
    "방법": "Method",
    "전화": "Phone call",
    "지인": "Acquaintance",
    "진심": "Sincerity",
    "하루": "Day (one day)",
    "카드": "Card",
    "연락": "Contact",
    "우울증": "Depression",
    "피해": "Harm",
    "건강": "Health",
    "인간": "Human",
    "시신": "Corpse",
    "병": "Illness",
    "보험": "Insurance",
    "희망": "Hope",
    "식구": "Household member",
    "사장": "Boss",
    "할머니": "Grandmother",
    "상처": "Wound",
    "장례식": "Funeral ceremony",
    "자리": "Place",
    "자살": "Suicide",
    "재산": "Assets",
    "여자": "Woman",
    "눈물": "Tears",
    "혼자": "Alone",
    "연락처": "Contact info",
    "처리": "Handling",
    "세월": "Years",
    "평생": "Lifetime",
    "회사": "Company",
    "정신": "Mind",
    "차": "Car",
    "가정": "Home (household)",
    "행복": "Happiness",
    "누님": "Older sister (honorific)",
    "문제": "Problem",
    "걱정": "Worry",
    "화장": "Cremation",
    "저승": "Afterlife",
    "약": "Medicine",
    "처": "Wife (formal)"
}

korean_verbs_to_english = {
    "미안하": "Apologize",
    "살": "Live",
    "사랑하": "Love",
    "가": "Go",
    "죄송하": "Apologize (formal)",
    "부탁하": "Request",
    "죽": "Die",
    "바라": "Hope",
    "용서하": "Forgive",
    "화장하": "Cremate",
    "보": "See",
    "생각하": "Think",
    "뿌리": "Scatter",
    "받": "Receive",
    "만나": "Meet",
    "알": "Know",
    "오": "Come",
    "못하": "Cannot do",
    "떠나": "Leave",
    "안되": "Fail",
    "드리": "Give (honorific)",
    "주": "Give",
    "남기": "Leave behind",
    "잘하": "Do well",
    "쓰": "Write",
    "먹": "Eat",
    "감사하": "Thank",
    "보내": "Send",
    "살았": "Lived",
    "연락하": "Contact",
    "처리하": "Handle",
    "지키": "Keep",
    "믿": "Believe",
    "보이": "Show",
    "모르": "Not know",
    "알리": "Inform",
    "돕": "Help",
    "위하": "Care for",
    "지내": "Spend (time)",
    "전하": "Deliver",
    "갚": "Repay",
    "정리하": "Organize",
    "이해하": "Understand",
    "사": "Buy",
    "챙기": "Take care of",
    "잊": "Forget",
    "말하": "Say",
    "마시": "Drink",
    "찾": "Find",
    "모르겠": "Don’t know",
    "인하": "Cause",
    "원망하": "Resent",
    "가지": "Have",
    "선택하": "Choose",
    "살아가": "Go on living",
    "시키": "Make (someone do)",
    "나오": "Come out",
    "모시": "Serve",
    "남": "Remain",
    "키우": "Raise",
    "대하": "Treat",
    "먼저가": "Go first",
    "나": "Come out",
    "태어나": "Be born",
    "전화하": "Call",
    "슬퍼하": "Grieve",
    "버리": "Abandon",
    "노력하": "Try",
    "가았": "Went",
    "듣": "Listen",
    "보살피": "Care for",
    "적": "Write down",
    "팔": "Sell",
    "만들": "Make",
    "오았": "Came",
    "작성하": "Fill out",
    "마감하": "Close",
    "해결하": "Resolve",
    "짓": "Build",
    "빌": "Pray",
    "두": "Put",
    "잘못되": "Go wrong",
    "끼치": "Cause (trouble)",
    "치르": "Carry out",
    "빌리": "Borrow",
    "올리": "Raise",
    "포기하": "Give up",
    "묻": "Bury",
    "잘못하": "Do wrong",
    "발견되": "Be found",
    "하시": "Do (honorific)",
    "부탁드리": "Humbly ask",
    "쉬": "Rest",
    "주시": "Give (honorific)",
    "상의하": "Consult",
    "지": "Stop",
    "치": "Treat (medical)",
    "알았": "Knew",
    "죽었": "Died",
    "있었": "Existed"
}


fig, axs = plt.subplots(2, 1, figsize=(16, 6))


plot_word_frequencies_df(age4_100noun,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=75,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[0],
                         word_translation_dict=korean_nouns_to_english)

plot_word_frequencies_df(age4_100verb,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=75,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[1],
                         word_translation_dict=korean_verbs_to_english)


#%%

korean_nouns_to_english = {
    "엄마": "Mother",
    "아들": "Son",
    "아버지": "Father",
    "사람": "Person",
    "돈": "Money",
    "딸": "Daughter",
    "아빠": "Dad",
    "마음": "Heart",
    "집": "Home",
    "몸": "Body",
    "가족": "Family",
    "유서": "Will",
    "자식": "Children",
    "고생": "Hardship",
    "일": "Work",
    "길": "Path",
    "통장": "Bankbook",
    "고통": "Pain",
    "병원": "Hospital",
    "어머니": "Mother (formal)",
    "마지막": "Last",
    "생각": "Thought",
    "병": "Illness",
    "인생": "Life",
    "부탁": "Request",
    "동생": "Younger sibling",
    "며느리": "Daughter-in-law",
    "끝": "End",
    "날": "Day",
    "삶": "Living",
    "형제": "Siblings",
    "짐": "Burden",
    "죽음": "Death",
    "힘": "Strength",
    "형": "Older brother (by male)",
    "자신": "Oneself",
    "아내": "Wife",
    "애비": "Father (colloquial)",
    "전화": "Call",
    "건강": "Health",
    "부모": "Parents",
    "남편": "Husband",
    "용서": "Forgiveness",
    "약": "Medicine",
    "죄": "Sin",
    "산": "Mountain",
    "하루": "Daytime",
    "시간": "Time",
    "재산": "Assets",
    "장례": "Funeral",
    "가슴": "Chest",
    "연락처": "Contact info",
    "아이": "Child",
    "제사": "Rite",
    "세월": "Time passing",
    "할머니": "Grandmother",
    "사망": "Death (formal)",
    "손자": "Grandson",
    "나이": "Age",
    "남매": "Brother and sister",
    "친구": "Friend",
    "비밀번호": "Password",
    "시신": "Remains",
    "가정": "Household",
    "선택": "Choice",
    "화장": "Cremation",
    "자녀": "Offspring",
    "장례식": "Funeral service",
    "모습": "Appearance",
    "남": "Others",
    "손": "Hand",
    "연락": "Contact",
    "동": "Building",
    "자살": "Suicide",
    "자리": "Seat / Position",
    "식구": "Household member",
    "할아버지": "Grandfather",
    "우울증": "Depression",
    "도움": "Help",
    "방법": "Method",
    "통증": "Pain (physical)",
    "본인": "Oneself (formal)",
    "생활": "Living (daily life)",
    "전화번호": "Phone number",
    "어머님": "Mother (honorific)",
    "인간": "Human",
    "처": "Wife (archaic/formal)",
    "사랑": "Love",
    "평생": "Lifetime",
    "누나": "Older sister",
    "방": "Room",
    "눈물": "Tear",
    "우애": "Sibling bond",
    "처리": "Handling",
    "번호": "Number",
    "아파트": "Apartment",
    "여자": "Woman",
    "손녀": "Granddaughter",
    "카드": "Card",
    "정신": "Mind"
}
korean_verbs_to_english = {
    "미안하": "Apologize",
    "살": "Live",
    "가": "Go",
    "사랑하": "Love",
    "바라": "Hope",
    "부탁하": "Ask",
    "죽": "Die",
    "용서하": "Forgive",
    "보": "See",
    "화장하": "Cremate",
    "생각하": "Think",
    "쓰": "Write",
    "오": "Come",
    "받": "Receive",
    "뿌리": "Scatter",
    "떠나": "Leave",
    "알": "Know",
    "먹": "Eat",
    "죄송하": "Apologize (formal)",
    "주": "Give",
    "못하": "Fail",
    "만나": "Meet",
    "지내": "Get along",
    "잘하": "Do well",
    "살았": "Lived",
    "남기": "Leave behind",
    "안되": "Not work",
    "처리하": "Handle",
    "알리": "Inform",
    "보내": "Send",
    "연락하": "Contact",
    "말하": "Say",
    "사": "Buy",
    "찾": "Find",
    "위하": "Care for",
    "모시": "Serve",
    "감사하": "Thank",
    "모르": "Not know",
    "드리": "Offer",
    "돕": "Help",
    "이해하": "Understand",
    "시키": "Make someone do",
    "가지": "Have",
    "보이": "Show",
    "전화하": "Call",
    "가았": "Went",
    "선택하": "Choose",
    "믿": "Believe",
    "전하": "Deliver",
    "키우": "Raise",
    "견디": "Endure",
    "두": "Put",
    "나오": "Come out",
    "잊": "Forget",
    "나": "Emerge",
    "묻": "Bury",
    "챙기": "Take care",
    "팔": "Sell",
    "지키": "Protect",
    "버리": "Throw away",
    "참": "Endure",
    "적": "Write down",
    "마시": "Drink",
    "원망하": "Resent",
    "마감하": "Close",
    "작성하": "Fill out",
    "먼저가": "Go first",
    "정리하": "Organize",
    "슬퍼하": "Be sad",
    "오았": "Came",
    "갚": "Repay",
    "대하": "Treat",
    "택하": "Select",
    "살아가": "Go on living",
    "살아오았": "Have lived",
    "죽었": "Died",
    "나누": "Share",
    "보살피": "Look after",
    "부르": "Call",
    "하시": "Do (honorific)",
    "지": "Lose",
    "하였": "Did",
    "넣": "Insert",
    "남": "Remain",
    "자살하": "Commit suicide",
    "돌보": "Care for",
    "듣": "Hear",
    "모르겠": "Don’t know",
    "사용하": "Use",
    "다니": "Attend",
    "인하": "Lower",
    "빌": "Beg",
    "택하았": "Had chosen",
    "짓": "Build",
    "상의하": "Consult",
    "신고하": "Report",
    "따르": "Follow",
    "해결하": "Resolve",
    "만들": "Make",
    "발견되": "Be discovered"
}

fig, axs = plt.subplots(2, 1, figsize=(16, 6))


plot_word_frequencies_df(age5_100noun,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=75,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[0],
                         word_translation_dict=korean_nouns_to_english)

plot_word_frequencies_df(age5_100verb,
                         word_col="Word",
                         freq_col="Count",
                         title="",
                         num_words=75,
                         font_size=10,
                         font_path=r'C:\Windows\Fonts\malgunbd.ttf',
                         horizontal=False,
                         show_values=False,
                         proportion=False,
                         ax=axs[1],
                         word_translation_dict=korean_verbs_to_english)



#%%

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def makewordcloud_from_freqs(data, word_col, count_col, wordnum, 
                             font_path=r'C:\Windows\Fonts\malgunbd.ttf', 
                             save_path=None,
                             save_format='png',
                             dpi=300,
                             shape='custom',
                             mask_path=None,
                             color='viridis',
                             half_donut_side='top',  # 'top', 'bottom', 'left', or 'right'
                             ):  # 👈 NEW PARAMETER
    """
    Creates a high-resolution word cloud from word frequency data.

    Parameters:
    - shape: 'circle', 'rectangle', or 'custom' (default: circle)
    - mask_path: Path to custom mask image (used if shape='custom')
    """

    # Convert DataFrame to dictionary {word: count}
    word_freqs = dict(zip(data[word_col], data[count_col]))

        # --- Create or load a mask
    mask = None
    if shape == 'circle':
        x, y = np.ogrid[:800, :800]
        center = (400, 400)
        radius = 380
        mask = (x - center[0])**2 + (y - center[1])**2 > radius**2
        mask = 255 * mask.astype(int)
    
    elif shape == 'document':
        width, height = 600, 900  # Portrait rectangle
        mask = np.full((height, width), 255, dtype=np.uint8)
        # Add optional padding by making a border
        border_thickness = 40
        mask[border_thickness:-border_thickness, border_thickness:-border_thickness] = 0
    
    elif shape == 'smallcircle':
        x, y = np.ogrid[:400, :400]  # 👈 smaller canvas
        center = (200, 200)
        radius = 180
        mask = (x - center[0])**2 + (y - center[1])**2 > radius**2
        mask = 255 * mask.astype(int)
        
    elif shape == 'half_donut':
        size = 800
        inner_radius = 200
        outer_radius = 380
        center = (size // 2, size // 2)
    
        y, x = np.ogrid[:size, :size]
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
        # Create full ring mask
        ring = np.logical_and(distance <= outer_radius, distance >= inner_radius)
    
        # Apply directional masking
        if half_donut_side == 'top':
            ring[center[1]:, :] = False  # remove bottom
        elif half_donut_side == 'bottom':
            ring[:center[1], :] = False  # remove top
        elif half_donut_side == 'left':
            ring[:, center[0]:] = False  # remove right
        elif half_donut_side == 'right':
            ring[:, :center[0]] = False  # remove left
    
        # Create final mask (255 = white = background, 0 = text allowed)
        mask = np.ones((size, size), dtype=np.uint8) * 255
        mask[ring] = 0
    
    elif shape == 'donut_slice':
        size = 800
        inner_radius = 200
        outer_radius = 380
        center = (size // 2, size // 2)
    
        y, x = np.ogrid[:size, :size]
        dx = x - center[0]
        dy = y - center[1]
        angle = (np.arctan2(-dy, dx) * 180 / np.pi) % 360
        distance = np.sqrt(dx**2 + dy**2)
    
        ring_mask = np.logical_and(distance <= outer_radius, distance >= inner_radius)
    
        slice_angles = {
            'slice_0': 0,
            'slice_1': 72,
            'slice_2': 144,
            'slice_3': 216,
            'slice_4': 288
        }
    
        if mask_path not in slice_angles:
            raise ValueError("Invalid mask_path for donut_slice. Use one of: slice_0 to slice_4")
    
        angle_start = slice_angles[mask_path]
        angle_end = (angle_start + 72) % 360
    
        if angle_end > angle_start:
            angle_range = np.logical_and(angle >= angle_start, angle < angle_end)
        else:
            angle_range = np.logical_or(angle >= angle_start, angle < angle_end)
    
        mask = np.ones((size, size), dtype=np.uint8) * 255
        mask[np.logical_and(ring_mask, angle_range)] = 0
    
    
    elif shape == 'custom' and mask_path is not None:
        mask_image = np.array(Image.open(mask_path).convert('L'))
        mask = np.where(mask_image > 128, 255, 0)
    # Create the WordCloud object
    wordcloud = WordCloud(
        font_path=font_path,
        background_color="white",
        max_words=wordnum,
        contour_width=2,
        contour_color='black',
        colormap=color,
        collocations=False,
        mask=mask,
        prefer_horizontal=1.0 
    ).generate_from_frequencies(word_freqs)

    # Plot at high resolution
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
    
    plt.show()

    return wordcloud

#%%
"""Final Code for Wordcloud"""

from wordcloud import get_single_color_func
import random

def get_dark_color_func(base_color='darkblue'):
    """Returns a color function that avoids white and very light colors."""
    class DarkColorFunc(object):
        def __init__(self, base_color):
            self.color_func = get_single_color_func(base_color)
        
        def __call__(self, word, font_size, position, orientation, font_path, random_state):
            r, g, b = random.randint(0, 150), random.randint(0, 150), random.randint(0, 150)
            return f"rgb({r}, {g}, {b})"
    
    return DarkColorFunc(base_color)

def nature_color_func():
    colors = [
        (59, 76, 192),    # Navy Blue
        (136, 204, 238),  # Sky Blue
        (68, 170, 153),   # Teal
        (17, 119, 51),    # Olive Green
        (221, 204, 119),  # Gold
        (204, 102, 119),  # Orange
        (136, 34, 85),    # Wine Red
        (153, 153, 51)    # Slate Gray
    ]
    def color_func(word, font_size, position, orientation, font_path, random_state):
        r, g, b = random.choice(colors)
        return f"rgb({r}, {g}, {b})"
    return color_func

def dark_earth_color_func():
    def color_func(word, font_size, position, orientation, font_path, random_state):
        palettes = [
            (34, 85, 34),   # Dark forest green
            (51, 51, 0),    # Olive
            (102, 51, 0),   # Earth brown
            (80, 120, 60),  # Moss green
        ]
        return f"rgb{random.choice(palettes)}"
    return color_func

def dark_cool_color_func():
    def color_func(word, font_size, position, orientation, font_path, random_state):
        r = random.randint(20, 70)
        g = random.randint(20, 70)
        b = random.randint(100, 180)
        return f"rgb({r}, {g}, {b})"
    return color_func

def dark_warm_color_func():
    warm_colors = [
        (139, 0, 0),     # Dark red
        (100, 30, 30),   # Maroon
        (70, 30, 30),    # Dark chocolate
        (50, 50, 50),    # Charcoal gray
    ]
    def color_func(word, font_size, position, orientation, font_path, random_state):
        return f"rgb{random.choice(warm_colors)}"
    return color_func


def eng_makewordcloud_from_freqs(data, word_col, count_col, wordnum, 
                             font_path=r'C:\Windows\Fonts\GARABD.ttf', 
                             save_path=None,
                             save_format='png',
                             dpi=1000,
                             shape='custom',
                             mask_path=None,
                             color='viridis',
                             half_donut_side='top',
                             background_color='white',
                             half_circle_side=None,  # 'top', 'bottom', 'left', 'right' or None
                             pizza_slice=None,  # 'slice_0' to 'slice_4'
                             colorfunction=get_dark_color_func()
):  # 👈 NEW PARAMETER
    """
    Creates a high-resolution word cloud from word frequency data.

    Parameters:
    - shape: 'circle', 'rectangle', or 'custom' (default: circle)
    - mask_path: Path to custom mask image (used if shape='custom')
    """

    # Convert DataFrame to dictionary {word: count}
    word_freqs = dict(zip(data[word_col], data[count_col]))

        # --- Create or load a mask
    mask = None
    if shape == 'circle':
        x, y = np.ogrid[:800, :800]
        center = (400, 400)
        radius = 380
        mask = (x - center[0])**2 + (y - center[1])**2 > radius**2
        mask = 255 * mask.astype(int)
    
    elif shape == 'donut':
        size = 800*3
        inner_radius = 200*3
        outer_radius = 380*3
        center = (size // 2, size // 2)
    
        y, x = np.ogrid[:size, :size]
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
        # Create a donut mask: keep only pixels between inner and outer radius
        ring = np.logical_and(distance <= outer_radius, distance >= inner_radius)
    
        # Mask: white background (255), black where text can go (0)
        mask = np.ones((size, size), dtype=np.uint8) * 255
        mask[ring] = 0

    elif shape == 'document':
        width, height = 600, 900  # Portrait rectangle
        mask = np.full((height, width), 255, dtype=np.uint8)
        # Add optional padding by making a border
        border_thickness = 40
        mask[border_thickness:-border_thickness, border_thickness:-border_thickness] = 0
    
    elif shape == 'smallcircle':
        x, y = np.ogrid[:400, :400]  # 👈 smaller canvas
        center = (200, 200)
        radius = 180
        mask = (x - center[0])**2 + (y - center[1])**2 > radius**2
        mask = 255 * mask.astype(int)
        
    elif shape == 'half_donut':
        size = 800*2
        inner_radius = 200*2
        outer_radius = 380*2
        center = (size // 2, size // 2)
    
        y, x = np.ogrid[:size, :size]
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
        # Create full ring mask
        ring = np.logical_and(distance <= outer_radius, distance >= inner_radius)
    
        # Apply directional masking
        if half_donut_side == 'top':
            ring[center[1]:, :] = False  # remove bottom
        elif half_donut_side == 'bottom':
            ring[:center[1], :] = False  # remove top
        elif half_donut_side == 'left':
            ring[:, center[0]:] = False  # remove right
        elif half_donut_side == 'right':
            ring[:, :center[0]] = False  # remove left
    
        # Create final mask (255 = white = background, 0 = text allowed)
        mask = np.ones((size, size), dtype=np.uint8) * 255
        mask[ring] = 0
    
    elif shape == 'donut_slice':
        size = 800*2
        inner_radius = 200*2
        outer_radius = 380*2
        center = (size // 2, size // 2)
    
        y, x = np.ogrid[:size, :size]
        dx = x - center[0]
        dy = y - center[1]
        angle = (np.arctan2(-dy, dx) * 180 / np.pi) % 360
        distance = np.sqrt(dx**2 + dy**2)
    
        ring_mask = np.logical_and(distance <= outer_radius, distance >= inner_radius)
    
        slice_angles = {
            'slice_0': 0,
            'slice_1': 72,
            'slice_2': 144,
            'slice_3': 216,
            'slice_4': 288
        }
    
        if mask_path not in slice_angles:
            raise ValueError("Invalid mask_path for donut_slice. Use one of: slice_0 to slice_4")
    
        angle_start = slice_angles[mask_path]
        angle_end = (angle_start + 72) % 360
    
        if angle_end > angle_start:
            angle_range = np.logical_and(angle >= angle_start, angle < angle_end)
        else:
            angle_range = np.logical_or(angle >= angle_start, angle < angle_end)
    
        mask = np.ones((size, size), dtype=np.uint8) * 255
        mask[np.logical_and(ring_mask, angle_range)] = 0
    
    elif shape == 'pizza_slice':
        size = 800*2
        radius = 380*2
        center = (size // 2, size // 2)
        y, x = np.ogrid[:size, :size]
        dx = x - center[0]
        dy = y - center[1]
        angle = (np.arctan2(-dy, dx) * 180 / np.pi) % 360
        distance = np.sqrt(dx**2 + dy**2)
    
        # Entire pie (full radius)
        circle_mask = distance <= radius
    
        # Define 5 equally spaced angular sectors
        slice_angles = {
            'slice_0': 0,
            'slice_1': 72,
            'slice_2': 144,
            'slice_3': 216,
            'slice_4': 288
        }
    
        if pizza_slice not in slice_angles:
            raise ValueError("Invalid pizza_slice. Use one of: 'slice_0' to 'slice_4'")
    
        angle_start = slice_angles[pizza_slice]
        angle_end = (angle_start + 72) % 360
    
        if angle_end > angle_start:
            angle_range = np.logical_and(angle >= angle_start, angle < angle_end)
        else:
            angle_range = np.logical_or(angle >= angle_start, angle < angle_end)
    
        # Combine circular region with angle
        mask = np.ones((size, size), dtype=np.uint8) * 255
        mask[np.logical_and(circle_mask, angle_range)] = 0
    
    
    elif shape == 'half_circle':
        size = 800*2
        radius = 380*2
        center = (size // 2, size // 2)
        x, y = np.ogrid[:size, :size]
        circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
        # Directional masking
        if half_circle_side == 'top':
            circle[center[1]:, :] = False
        elif half_circle_side == 'bottom':
            circle[:center[1], :] = False
        elif half_circle_side == 'left':
            circle[:, center[0]:] = False
        elif half_circle_side == 'right':
            circle[:, :center[0]] = False
        else:
            raise ValueError("half_circle_side must be one of 'top', 'bottom', 'left', 'right'")
    
        mask = np.ones((size, size), dtype=np.uint8) * 255
        mask[circle] = 0
    
    
    elif shape == 'custom' and mask_path is not None:
        mask_image = np.array(Image.open(mask_path).convert('L'))
        mask = np.where(mask_image > 128, 255, 0)
        
        
# =============================================================================
#     # Create the WordCloud object
#     wordcloud = WordCloud(
#         font_path=font_path,
#         background_color=background_color,
#         max_words=wordnum,
#         contour_width=0,
#         contour_color='black',
#         colormap=color,
#         collocations=False,
#         mask=mask,
#         prefer_horizontal=0.9,
#     ).generate_from_frequencies(word_freqs)
# 
# =============================================================================
    wordcloud = WordCloud(
        font_path=font_path,
        background_color=background_color,
        max_words=wordnum,
        contour_width=0,
        contour_color='black',
        collocations=False,
        mask=mask,
        prefer_horizontal=0.8,
        color_func=colorfunction # 👈 Ensures no white colors
    ).generate_from_frequencies(word_freqs)

    # Plot at high resolution
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
    
    plt.show()

    return wordcloud, mask

#%% Wordcloud (Frequency)

makewordcloud_from_freqs(male100noun, word_col="Word", count_col='Count', wordnum=len(male100noun), shape="half_donut", color="cividis_r", half_donut_side="left")
makewordcloud_from_freqs(female100noun, word_col="Word", count_col='Count', wordnum=len(female100noun), shape="half_donut", color="cividis_r", half_donut_side="right")


makewordcloud_from_freqs(male100verb, word_col="Word", count_col='Count', wordnum=len(male100verb), shape="half_donut", color="cividis_r", half_donut_side="left")
makewordcloud_from_freqs(female100verb, word_col="Word", count_col='Count', wordnum=len(female100verb), shape="half_donut", color="cividis_r", half_donut_side="right")


#%%

def getsigwords(df):
    df=df[df["significant"].isin([True])]
    df=df[df["odds_ratio"]>=1]
    return df

    
signounmale=getsigwords(gender_chi_square_nounresults_dict["kfspmale_vs_rest"])
signounfemale=getsigwords(gender_chi_square_nounresults_dict["kfspfemale_vs_rest"])
signounage1=getsigwords(age_chi_square_nounresults_dict["kfspage1_vs_rest"])
signounage2=getsigwords(age_chi_square_nounresults_dict["kfspage2_vs_rest"])
signounage3=getsigwords(age_chi_square_nounresults_dict["kfspage3_vs_rest"])
signounage4=getsigwords(age_chi_square_nounresults_dict["kfspage4_vs_rest"])
signounage5=getsigwords(age_chi_square_nounresults_dict["kfspage5_vs_rest"])

signounmale = signounmale[~signounmale['word'].isin(stopwords)].reset_index(drop=True)
signounfemale = signounfemale[~signounfemale['word'].isin(stopwords)].reset_index(drop=True)
signounage1 = signounage1[~signounage1['word'].isin(stopwords)].reset_index(drop=True)
signounage2 = signounage2[~signounage2['word'].isin(stopwords)].reset_index(drop=True)
signounage3 = signounage3[~signounage3['word'].isin(stopwords)].reset_index(drop=True)
signounage4 = signounage4[~signounage4['word'].isin(stopwords)].reset_index(drop=True)
signounage5 = signounage5[~signounage5['word'].isin(stopwords)].reset_index(drop=True)


sigverbmale=getsigwords(gender_chi_square_verbresults_dict["kfspmale_vs_rest"])
sigverbfemale=getsigwords(gender_chi_square_verbresults_dict["kfspfemale_vs_rest"])
sigverbage1=getsigwords(age_chi_square_verbresults_dict["kfspage1_vs_rest"])
sigverbage2=getsigwords(age_chi_square_verbresults_dict["kfspage2_vs_rest"])
sigverbage3=getsigwords(age_chi_square_verbresults_dict["kfspage3_vs_rest"])
sigverbage4=getsigwords(age_chi_square_verbresults_dict["kfspage4_vs_rest"])
sigverbage5=getsigwords(age_chi_square_verbresults_dict["kfspage5_vs_rest"])


sigverbmale = sigverbmale[~sigverbmale['word'].isin(stopverbs)].reset_index(drop=True)
sigverbfemale = sigverbfemale[~sigverbfemale['word'].isin(stopverbs)].reset_index(drop=True)
sigverbage1 = sigverbage1[~sigverbage1['word'].isin(stopverbs)].reset_index(drop=True)
sigverbage2 = sigverbage2[~sigverbage2['word'].isin(stopverbs)].reset_index(drop=True)
sigverbage3 = sigverbage3[~sigverbage3['word'].isin(stopverbs)].reset_index(drop=True)
sigverbage4 = sigverbage4[~sigverbage4['word'].isin(stopverbs)].reset_index(drop=True)
sigverbage5 = sigverbage5[~sigverbage5['word'].isin(stopverbs)].reset_index(drop=True)

#%% Translate sig words

#%% Wordcloud (Sig Words)



makewordcloud_from_freqs(signounmale, word_col="word", count_col='proportion_m2', wordnum=len(signounmale), shape="circle", color="cividis_r")
makewordcloud_from_freqs(sigverbmale, word_col="word", count_col='proportion_m2', wordnum=len(sigverbmale), shape="circle", color="cividis_r")


makewordcloud_from_freqs(signounfemale, word_col="word", count_col='proportion_m2', wordnum=len(signounfemale), shape="circle", color="cividis_r")
makewordcloud_from_freqs(sigverbfemale, word_col="word", count_col='proportion_m2', wordnum=len(sigverbfemale), shape="circle", color="cividis_r")


makewordcloud_from_freqs(signounage1, word_col="word", count_col='proportion_m2', wordnum=len(signounage1), shape="circle", color="cividis_r")
makewordcloud_from_freqs(sigverbage1, word_col="word", count_col='proportion_m2', wordnum=len(sigverbage1), shape="circle", color="cividis_r")


makewordcloud_from_freqs(signounage2, word_col="word", count_col='proportion_m2', wordnum=len(signounage2), shape="circle", color="cividis_r")
makewordcloud_from_freqs(sigverbage2, word_col="word", count_col='proportion_m2', wordnum=len(sigverbage2), shape="circle", color="cividis_r")

makewordcloud_from_freqs(signounage3, word_col="word", count_col='proportion_m2', wordnum=len(signounage3), shape="circle", color="cividis_r")
makewordcloud_from_freqs(sigverbage3, word_col="word", count_col='proportion_m2', wordnum=len(sigverbage3), shape="circle", color="cividis_r")

makewordcloud_from_freqs(signounage4, word_col="word", count_col='proportion_m2', wordnum=len(signounage4), shape="circle", color="cividis_r")
makewordcloud_from_freqs(sigverbage4, word_col="word", count_col='proportion_m2', wordnum=len(sigverbage4), shape="circle", color="cividis_r")

makewordcloud_from_freqs(signounage5, word_col="word", count_col='proportion_m2', wordnum=len(signounage5), shape="circle", color="cividis_r")
makewordcloud_from_freqs(sigverbage5, word_col="word", count_col='proportion_m2', wordnum=len(sigverbage5), shape="circle", color="cividis_r")


#%%

from PIL import Image, ImageChops

def crop_whitespace(image_path, output_path):
    img = Image.open(image_path).convert("RGB")
    bg = Image.new("RGB", img.size, img.getpixel((0, 0)))  # Assumes white or uniform background
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        cropped.save(output_path)

        
crop_whitespace("t1.png", "t11.png")
crop_whitespace("t2.png", "t22.png")
crop_whitespace("t3.png", "t33.png")
crop_whitespace("t4.png", "t44.png")
crop_whitespace("t5.png", "t55.png")

#%%

makewordcloud_from_freqs(signounmale, word_col="word", count_col='proportion_m2', wordnum=len(signounmale), shape="half_donut", color="cividis_r", half_donut_side="left")
makewordcloud_from_freqs(signounfemale, word_col="word", count_col='proportion_m2', wordnum=len(signounfemale), shape="half_donut", color="cividis_r", half_donut_side="right")




makewordcloud_from_freqs(sigverbmale, word_col="word", count_col='proportion_m2', wordnum=len(sigverbmale), shape="half_donut", color="cividis_r", half_donut_side="left")
makewordcloud_from_freqs(sigverbfemale, word_col="word", count_col='proportion_m2', wordnum=len(sigverbfemale), shape="half_donut", color="cividis_r", half_donut_side="right")

#%%
makewordcloud_from_freqs(signounage1, word_col="word", count_col='proportion_m2', wordnum=len(signounage1), shape="donut_slice", color="cividis_r", mask_path="slice_4")
makewordcloud_from_freqs(signounage2, word_col="word", count_col='proportion_m2', wordnum=len(signounage2), shape="circle", color="cividis_r")
makewordcloud_from_freqs(signounage3, word_col="word", count_col='proportion_m2', wordnum=len(signounage3), shape="circle", color="cividis_r")
makewordcloud_from_freqs(signounage4, word_col="word", count_col='proportion_m2', wordnum=len(signounage4), shape="circle", color="cividis_r")
makewordcloud_from_freqs(signounage5, word_col="word", count_col='proportion_m2', wordnum=len(signounage5), shape="circle", color="cividis_r")


#%%

makewordcloud_from_freqs(male100noun, word_col="Word", count_col='Count', wordnum=100, shape="circle")
makewordcloud_from_freqs(male100verb, word_col="Word", count_col='Count', wordnum=100, shape="circle")


#%%

makewordcloud_from_freqs(female100noun, word_col="Word", count_col='Count', wordnum=100, shape="circle")
makewordcloud_from_freqs(female100verb, word_col="Word", count_col='Count', wordnum=100, shape="circle")
#%%


makewordcloud_from_freqs(age1_100noun, word_col="Word", count_col='Count', wordnum=100, shape="circle")
makewordcloud_from_freqs(age1_100verb, word_col="Word", count_col='Count', wordnum=100, shape="circle")


makewordcloud_from_freqs(age2_100noun, word_col="Word", count_col='Count', wordnum=100, shape="circle")
makewordcloud_from_freqs(age2_100verb, word_col="Word", count_col='Count', wordnum=100, shape="circle")


makewordcloud_from_freqs(age3_100noun, word_col="Word", count_col='Count', wordnum=100, shape="circle")
makewordcloud_from_freqs(age3_100verb, word_col="Word", count_col='Count', wordnum=100, shape="circle")


makewordcloud_from_freqs(age4_100noun, word_col="Word", count_col='Count', wordnum=100, shape="circle")
makewordcloud_from_freqs(age4_100verb, word_col="Word", count_col='Count', wordnum=100, shape="circle")



makewordcloud_from_freqs(age5_100noun, word_col="Word", count_col='Count', wordnum=100, shape="circle")
makewordcloud_from_freqs(age5_100verb, word_col="Word", count_col='Count', wordnum=100, shape="circle")
