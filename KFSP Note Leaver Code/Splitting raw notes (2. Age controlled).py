# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 11:47:42 2025

@author: Jae Bin Park
"""

# -*- coding: utf-8 -*-
"""
Gender ~ Themes (15) + AGE2 (categorical) logistic regression
- Builds 3×5 theme×section indicators from presence lists already in morethan3
- Adds AGE2 as categorical (≤18 is the reference)
- Fits statsmodels Logit with intercept
- Plots theme heatmap (controlling for age) without pivot duplicates
- Prints AGE2 dummy effects table
Author: Jae Bin Park
Date: 2025-08-14
"""

# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 0) Inputs & Assumptions
# =========================
# Assumes `morethan3` is already in memory and contains:
# - Section presence arrays: *_presence with lists like [Intro, Body, Conclusion] for each theme
# - Columns: SEX (1=male, 2=female), AGE2 in {1,2,3,4,5}
# If you need to re-map presence columns, set them here:

presence_columns = {
    "Sorry and Shame": "sorrysentence_presence",
    "Love and Gratitude": "loveandgratitudesentence_presence",
    "Burden": "burdensomesentence_presence",
    "Despair": "despairsentence_presence",
    "Post-mortem Affairs": "pmaffairssentence_presence"
}

themes   = ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]
sections = ["Section 1", "Section 2", "Section 3"]  # Intro, Body, Conclusion order

# =========================
# 1) Filter to valid AGE2 & build 3×5 theme vectors
# =========================
df = morethan3[morethan3["AGE2"].isin([1, 2, 3, 4, 5])].copy()

def get_section_vector(row, section_idx):
    return [
        row[presence_columns["Sorry and Shame"]][section_idx],
        row[presence_columns["Love and Gratitude"]][section_idx],
        row[presence_columns["Burden"]][section_idx],
        row[presence_columns["Despair"]][section_idx],
        row[presence_columns["Post-mortem Affairs"]][section_idx],
    ]

df["section1_theme_vector"] = df.apply(lambda r: get_section_vector(r, 0), axis=1)
df["section2_theme_vector"] = df.apply(lambda r: get_section_vector(r, 1), axis=1)
df["section3_theme_vector"] = df.apply(lambda r: get_section_vector(r, 2), axis=1)

# Concatenate 15 binary features + append AGE2 as last element (for inspection only)
def concatenate_theme_vectors(row):
    return (
        row["section1_theme_vector"]
        + row["section2_theme_vector"]
        + row["section3_theme_vector"]
        + [row["AGE2"]]
    )

df["full_theme_vector"] = df.apply(concatenate_theme_vectors, axis=1)

# =========================
# 2) Make a labeled feature matrix
# =========================
theme_feature_names = [f"{sec} {th}" for sec in sections for th in themes]  # 15 names
theme_feature_names_plus_age = theme_feature_names + ["AGE2"]

theme_matrix = pd.DataFrame(
    df["full_theme_vector"].tolist(),
    index=df.index,
    columns=theme_feature_names_plus_age,
)

# Replace any pre-existing columns with fresh ones (defensive)
existing_cols = [c for c in theme_feature_names_plus_age if c in df.columns]
df = df.drop(columns=existing_cols, errors="ignore")
df = pd.concat([df, theme_matrix], axis=1)

# =========================
# 3) Prepare data for gender model
# =========================
# Keep valid SEX, recode Male=1, Female=0
filtered_df = df[df["SEX"].isin([1, 2])].copy()
filtered_df["SEX"] = filtered_df["SEX"].replace({1: 1, 2: 0}).astype(int)

# Theme-only matrix (first 15 elements of full_theme_vector to be safe)
theme_only_matrix = pd.DataFrame(
    filtered_df["full_theme_vector"].apply(lambda v: list(map(int, v[:15]))).tolist(),
    index=filtered_df.index,
    columns=theme_feature_names,
)

# AGE2 as categorical dummies (≤18 = reference when drop_first=True)
age_dummies = pd.get_dummies(filtered_df["AGE2"], prefix="AGE2", drop_first=True)

# Ensure all age dummy columns exist even if some bins are absent in this subset
for col in [f"AGE2_{k}" for k in [2, 3, 4, 5]]:
    if col not in age_dummies.columns:
        age_dummies[col] = 0
age_dummies = age_dummies[[f"AGE2_{k}" for k in [2, 3, 4, 5]]]

# =========================
# 4) Build X, clean, and fit Logit
# =========================
# Align by the same index and concatenate
X = pd.concat([theme_only_matrix, age_dummies], axis=1)
X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = sm.add_constant(X, has_constant="add")

y = filtered_df["SEX"].loc[X.index].astype(int)

# Drop any rows with NaN/Inf in X or missing y
mask = ~(X.isna().any(axis=1) | pd.isna(y))
X_clean = X.loc[mask].copy()
y_clean = y.loc[mask].copy()

# Optional diagnostics
#print("Dropped rows due to NaN/Inf:", (~mask).sum())
#print("Class balance:", y_clean.value_counts().to_dict())

# Fit model
model = sm.Logit(y_clean, X_clean)
result = model.fit(disp=False)

# =========================
# 5) Summarize coefficients (OR, CI, p)
# =========================
odds_ratios = np.exp(result.params)
conf = np.exp(result.conf_int())
conf.columns = ["CI Lower", "CI Upper"]

summary_df = pd.DataFrame({
    "term": result.params.index,
    "Coefficient": result.params.values,
    "Odds Ratio": odds_ratios.values,
    "p-value": result.pvalues.values,
    "CI Lower": conf["CI Lower"].values,
    "CI Upper": conf["CI Upper"].values,
})

# Drop intercept for plotting
summary_df = summary_df[summary_df["term"] != "const"].copy()
summary_df["log_odds_ratio"] = np.log(summary_df["Odds Ratio"])

def get_sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

summary_df["sig_star"] = summary_df["p-value"].apply(get_sig_star)
summary_df["annot"] = summary_df["log_odds_ratio"].round(2).astype(str) + summary_df["sig_star"]

# =========================
# 6) Print AGE2 categorical effects
# =========================
age_terms = summary_df[summary_df["term"].str.startswith("AGE2_")].copy()
if not age_terms.empty:
    print("\nAGE2 categorical effects (reference = ≤18):")
    print(
        age_terms[["term", "Coefficient", "Odds Ratio", "p-value", "CI Lower", "CI Upper"]]
        .sort_values("term")
        .to_string(index=False)
    )

# =========================
# 7) Theme heatmap (exclude age dummies)
# =========================
ordered_terms = [f"{sec} {th}" for sec in sections for th in themes]  # 15 exact names
summary_df_theme = summary_df[summary_df["term"].isin(ordered_terms)].copy()

# Label for plotting
summary_df_theme["feature"] = pd.Categorical(summary_df_theme["term"],
                                             categories=ordered_terms, ordered=True)
summary_df_theme["Demographic"] = "Gender"

# Defensive: drop duplicates if any
summary_df_theme = (summary_df_theme
                    .sort_values(["feature"])
                    .drop_duplicates(subset=["feature", "Demographic"], keep="first"))

# Pivot robustly
heatmap_data = pd.pivot_table(
    summary_df_theme,
    index="feature", columns="Demographic",
    values="log_odds_ratio", aggfunc="first"
)
annot_data = pd.pivot_table(
    summary_df_theme,
    index="feature", columns="Demographic",
    values="annot", aggfunc="first"
)

# Plot
plt.figure(figsize=(5.5, 9.0))
sns.heatmap(
    heatmap_data, annot=annot_data, fmt="",
    cmap="coolwarm", center=0, linewidths=0.5,
    cbar_kws={'label': 'Log-Odds'}
)
plt.title("Log(Odds Ratio) by Thematic Feature for Gender\n(Controlling for Age)")
plt.xlabel("Demographic")
plt.ylabel("Thematic Feature")
plt.tight_layout()
plt.show()

# =========================
# 8) One-liners for interpretation
# =========================
print("\nModel fit:")
print(f"  Log-Likelihood: {result.llf: .3f}   AIC: {result.aic: .2f}")
print("\nReminder: SEX coded Male=1, Female=0. OR>1 ⇒ feature associated with higher odds of Male.")

# =========================
# Gender line plot + Combined (Age + Gender) heatmap
# =========================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.font_manager as fm

# ---------- Helpers & constants ----------
def get_sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

themes   = ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]
sections = ["Section 1", "Section 2", "Section 3"]  # (Intro, Body, Conclusion)
# 15 features, section-first to help grouped row labels later
ordered_terms = [f"{sec} {th}" for sec in sections for th in themes]

# ---------- 1) Tidy Gender (Male) results from the earlier model ----------
# Keep only the 15 theme terms; compute log-OR and annotations
gender_tbl = summary_df[summary_df["term"].isin(ordered_terms)].copy()
gender_tbl["Demographic"] = "Gender (Male)"
gender_tbl["log_odds_ratio"] = np.log(gender_tbl["Odds Ratio"])
gender_tbl["sig_star"] = gender_tbl["p-value"].apply(get_sig_star)
gender_tbl["annot"] = gender_tbl["log_odds_ratio"].round(2).astype(str) + gender_tbl["sig_star"]
gender_tbl = gender_tbl.rename(columns={"term": "Feature"})

# For the line plot, split section & theme for plotting by theme
gender_tbl["section"] = gender_tbl["Feature"].apply(lambda x: " ".join(x.split()[:2]))   # 'Section 1'
gender_tbl["theme"]   = gender_tbl["Feature"].apply(lambda x: " ".join(x.split()[2:]))

# Build Female from Male correctly: log(OR_female) = -log(OR_male); same p-values/stars
male_df = gender_tbl.copy()
male_df["GenderVal"] = 1  # Male

female_df = male_df.copy()
female_df["GenderVal"] = 0
female_df["log_odds_ratio"] = -female_df["log_odds_ratio"]
# p-values test the same coefficient, so keep them; stars recomputed from same p-values
female_df["sig_star"] = female_df["p-value"].apply(get_sig_star)
female_df["annot"] = female_df["log_odds_ratio"].round(2).astype(str) + female_df["sig_star"]

plot_df = pd.concat([female_df, male_df], ignore_index=True)

# ---------- 2) Gender line plot (Male vs Female) ----------
# Optional font
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
font_path = r'C:\Windows\Fonts\Arialbd.ttf'

try:
    custom_font = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family": custom_font})
    font_prop = fm.FontProperties(fname=font_path)
except Exception:
    font_prop = None

sns.set(style="white", context="notebook", font_scale=1.2)
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.spines.right": False,
    "axes.spines.top": False
})

gender_labels = ["Female", "Male"]
n_rows, n_cols = 2, 3  # 5 themes -> 2x3 grid; last empty will be removed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8), sharey=True)
fig.subplots_adjust(right=0.82)

for idx, theme in enumerate(themes):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    theme_data = plot_df[plot_df["theme"] == theme]

    for section in sections:
        line_data = theme_data[theme_data["section"] == section].sort_values("GenderVal")
        ax.plot(
            line_data["GenderVal"],
            line_data["log_odds_ratio"],
            marker="o",
            label=section,
            linewidth=2
        )
        # annotate significance
        for _, r in line_data.iterrows():
            if r["sig_star"]:
                ax.text(r["GenderVal"], r["log_odds_ratio"], r["sig_star"],
                        ha="center", va="bottom", fontsize=11, weight='bold')

    ax.set_title(theme, fontsize=13, fontproperties=font_prop if font_prop else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(gender_labels, fontsize=12, fontproperties=font_prop if font_prop else None)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Gender", fontsize=12, fontproperties=font_prop if font_prop else None)
    ax.tick_params(bottom=True, left=True) 

    if col == 0:
        ax.set_ylabel("Log-Odds", fontsize=11, fontproperties=font_prop if font_prop else None)
    # tick fonts
    if font_prop:
        for lab in ax.get_yticklabels() + ax.get_xticklabels():
            lab.set_fontproperties(font_prop)

    # Set font for y-axis tick labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(12)

    
    # Set font for x-axis tick labels (optional, but already done earlier with set_xticklabels)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(12)


# remove extra subplot
if len(themes) < n_rows * n_cols:
    fig.delaxes(axes[n_rows - 1, n_cols - 1])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, ["Introduction", "Body", "Conclusion"], title="Section",
           loc='center left', bbox_to_anchor=(0.87, 0.5),
           frameon=False, fontsize=11,
           title_fontsize=12 if not font_prop else None)
plt.suptitle("Theme-Section Effects on Gender (Logit Log-Odds, Controlling for Age)", 
             fontsize=14, y=0.95, fontproperties=font_prop if font_prop else None)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])
plt.show()

# ---------- 3) Age one-vs-rest models (Age1..Age5) using same 15 theme features + Gender covariate ----------
# Build 15 theme features anew from df["full_theme_vector"] for full alignment
theme_only_matrix_all = pd.DataFrame(
    df["full_theme_vector"].apply(lambda v: list(map(int, v[:15]))).tolist(),
    index=df.index,
    columns=ordered_terms
)
age_df = df[df["AGE2"].isin([1,2,3,4,5]) & df["SEX"].isin([1,2])].copy()
age_df["Gender"] = age_df["SEX"].replace({1:1, 2:0}).astype(int)

X_age = pd.concat([theme_only_matrix_all.loc[age_df.index], age_df["Gender"]], axis=1)
X_age = X_age.apply(pd.to_numeric, errors="coerce")
X_age.replace([np.inf, -np.inf], np.nan, inplace=True)
X_age = sm.add_constant(X_age, has_constant="add")

group_results = {}
for group in [1, 2, 3, 4, 5]:
    y = (age_df["AGE2"] == group).astype(int)
    mask = ~(X_age.isna().any(axis=1) | pd.isna(y))
    Xi = X_age.loc[mask]
    yi = y.loc[mask]
    model_i = sm.Logit(yi, Xi).fit(disp=False)

    odds_ratios_i = np.exp(model_i.params)
    conf_i = np.exp(model_i.conf_int()); conf_i.columns = ["CI Lower", "CI Upper"]
    res_i = pd.DataFrame({
        "Feature": model_i.params.index,
        "Coefficient": model_i.params.values,
        "Odds Ratio": odds_ratios_i.values,
        "p-value": model_i.pvalues.values,
        "CI Lower": conf_i["CI Lower"].values,
        "CI Upper": conf_i["CI Upper"].values,
        "Age Group": group
    })
    group_results[group] = res_i

# ---------- 4) Tidy Age table (keep only 15 theme terms) ----------
age_tbl = pd.concat(group_results.values(), ignore_index=True)
age_tbl = age_tbl[age_tbl["Feature"].isin(ordered_terms)].copy()
age_tbl["Demographic"] = age_tbl["Age Group"].apply(lambda g: f"Age{g}")
age_tbl["log_odds_ratio"] = np.log(age_tbl["Odds Ratio"])
age_tbl["sig_star"] = age_tbl["p-value"].apply(get_sig_star)
age_tbl["annot"] = age_tbl["log_odds_ratio"].round(2).astype(str) + age_tbl["sig_star"]
#%%Line plot V2

# === Line Plot (Gender): x = Section 1/2/3, lines = Female & Male, one panel per Theme ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 1) Tidy gender_tbl from your model (must already exist)
plot_gender = gender_tbl.copy()  # columns: Feature, log_odds_ratio, p-value, etc.

# Extract section & theme from "Feature" like "Section 1 Sorry and Shame"
plot_gender["section"] = plot_gender["Feature"].str.extract(r'(Section \d)')
plot_gender["theme"]   = plot_gender["Feature"].str.replace(r'^Section \d\s+', '', regex=True)

# Keep ordered sections 1→3
section_order = ["Section 1", "Section 2", "Section 3"]
plot_gender = plot_gender[plot_gender["section"].isin(section_order)].copy()
plot_gender["section"] = pd.Categorical(plot_gender["section"], categories=section_order, ordered=True)

# Build Male and (mirrored) Female log-OR lines
male_only = plot_gender.copy()
male_only["Label"] = "Male"  # log_odds_ratio already for Male

female_only = male_only.copy()
female_only["Label"] = "Female"
female_only["log_odds_ratio"] = -female_only["log_odds_ratio"]  # invert sign for Female

gdf = pd.concat([female_only, male_only], ignore_index=True)

# 2) Style / font (graceful fallback)
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
font_path = r'C:\Windows\Fonts\Arialbd.ttf'

try:
    custom_font = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family": custom_font})
    font_prop = fm.FontProperties(fname=font_path)
except Exception:
    font_prop = fm.FontProperties()

sns.set(style="white", context="notebook", font_scale=1.2)
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.spines.right": False,
    "axes.spines.top": False
})

def star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

themes = ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]

# 3) Plot (5 themes → 2×3 grid; last axis removed)
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
axes = axes.flatten()

for idx, theme in enumerate(themes):
    ax = axes[idx]
    tdf = gdf[gdf["theme"] == theme].copy()

    # Two lines: Female, Male
    for label in ["Female", "Male"]:
        sub = (tdf[tdf["Label"] == label]
               .sort_values("section"))
        x = np.arange(1, len(section_order) + 1)
        y = sub["log_odds_ratio"].to_numpy()

        ax.plot(x, y, marker="o", linewidth=2, label=label)

        # Significance stars (use the p-values from the male coefficient)
        # We can read p-values from either subset since p-values are identical for ± lines
        for i, (_, r_) in enumerate(sub.iterrows(), start=1):
            s = star(r_["p-value"])
            if s:
                ax.text(i, r_["log_odds_ratio"], s, ha="center", va="bottom", fontsize=11, weight="bold")

    # Cosmetics
    ax.set_title(theme, fontsize=13, fontproperties=font_prop)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(section_order, rotation=0, fontsize=10, fontproperties=font_prop)
    ax.set_xlabel("Section", fontsize=11, fontproperties=font_prop)
    if idx % n_cols == 0:
        ax.set_ylabel("Log-Odds", fontsize=11, fontproperties=font_prop)
    if font_prop:
        for lab in ax.get_yticklabels() + ax.get_xticklabels():
            lab.set_fontproperties(font_prop)
            lab.set_fontsize(10)
# Remove unused last axis
if len(themes) < len(axes):
    fig.delaxes(axes[-1])

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Gender", loc='upper right', bbox_to_anchor=(0.98, 0.5),
           frameon=False, fontsize=11, title_fontsize=12)

plt.suptitle("Theme-wise Effects Across Sections (Gender Model)\nlines = Female/Male, x = Sections; controlling for Age",
             fontsize=14, weight='bold', y=0.98, fontproperties=font_prop)

plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.show()
#%%
# === Line Plot (Gender): x = Introduction/Body/Conclusion, lines = Female & Male, one panel per Theme ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 1) Tidy gender_tbl from your model (must already exist)
plot_gender = gender_tbl.copy()  # columns: Feature, log_odds_ratio, p-value, etc.

# Extract section & theme from "Feature" like "Section 1 Sorry and Shame"
plot_gender["section"] = plot_gender["Feature"].str.extract(r'(Section \d)')
plot_gender["theme"]   = plot_gender["Feature"].str.replace(r'^Section \d\s+', '', regex=True)

# Keep ordered sections 1→3 (used internally for sorting/merging)
section_order = ["Section 1", "Section 2", "Section 3"]
plot_gender = plot_gender[plot_gender["section"].isin(section_order)].copy()
plot_gender["section"] = pd.Categorical(plot_gender["section"], categories=section_order, ordered=True)

# >>> NEW: human-friendly labels for the x-axis
display_labels = ["Introduction", "Body", "Conclusion"]

# Build Male and (mirrored) Female log-OR lines
male_only = plot_gender.copy()
male_only["Label"] = "Male"  # log_odds_ratio already for Male

female_only = male_only.copy()
female_only["Label"] = "Female"
female_only["log_odds_ratio"] = -female_only["log_odds_ratio"]  # invert sign for Female

gdf = pd.concat([female_only, male_only], ignore_index=True)

# 2) Style / font (graceful fallback)
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
font_path = r'C:\Windows\Fonts\Arialbd.ttf'

try:
    custom_font = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family": custom_font})
    font_prop = fm.FontProperties(fname=font_path)
except Exception:
    font_prop = fm.FontProperties()

sns.set(style="white", context="notebook", font_scale=1.2)
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.spines.right": False,
    "axes.spines.top": False
})

def star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

themes = ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]

# 3) Plot (5 themes → 2×3 grid; last axis removed)
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
axes = axes.flatten()

for idx, theme in enumerate(themes):
    ax = axes[idx]
    tdf = gdf[gdf["theme"] == theme].copy()

    # Two lines: Female, Male
    for label in ["Female", "Male"]:
        sub = (tdf[tdf["Label"] == label]
               .sort_values("section"))
        x = np.arange(1, len(section_order) + 1)
        y = sub["log_odds_ratio"].to_numpy()

        ax.plot(x, y, marker="o", linewidth=2, label=label)

        # Significance stars
        for i, (_, r_) in enumerate(sub.iterrows(), start=1):
            s = star(r_["p-value"])
            if s:
                ax.text(i, r_["log_odds_ratio"], s, ha="center", va="bottom", fontsize=11, weight="bold")

    # Cosmetics
    ax.set_title(theme, fontsize=13, fontproperties=font_prop)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks([1, 2, 3])
    # >>> UPDATED: use display labels instead of "Section 1/2/3"
    ax.set_xticklabels(display_labels, rotation=0, fontsize=10, fontproperties=font_prop)
    ax.set_xlabel("Section", fontsize=11, fontproperties=font_prop)
    if idx % n_cols == 0:
        ax.set_ylabel("Log-Odds", fontsize=11, fontproperties=font_prop)
    if font_prop:
        for lab in ax.get_yticklabels() + ax.get_xticklabels():
            lab.set_fontproperties(font_prop)
            lab.set_fontsize(10)
            
# Remove unused last axis
if len(themes) < len(axes):
    fig.delaxes(axes[-1])

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Gender", loc='upper right', bbox_to_anchor=(0.98, 0.5),
           frameon=False, fontsize=11, title_fontsize=12)

plt.suptitle("Theme-wise Effects Across Note Segments (Gender Model)\nlines = Female/Male, x = Introduction–Body–Conclusion; controlling for Age",
             fontsize=14, weight='bold', y=0.98, fontproperties=font_prop)
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.show()

#%% Comprehensive heatmap with CI (two-line labels, adaptive text color)

# ---------- 5) Stack Age + Gender into merged_df (adds log-OR CI strings) ----------
common_cols = [
    'Feature', 'Coefficient', 'Odds Ratio', 'p-value', 'CI Lower', 'CI Upper',
    'Demographic', 'log_odds_ratio', 'sig_star'
]

def add_log_ci_and_annot(df_in):
    df = df_in[common_cols].copy()
    # guard against nonpositive values (CI are OR-scale)
    eps = np.finfo(float).tiny
    df["CI Lower"] = pd.to_numeric(df["CI Lower"], errors="coerce").clip(lower=eps)
    df["CI Upper"] = pd.to_numeric(df["CI Upper"], errors="coerce").clip(lower=eps)
    df["log_odds_ratio"] = pd.to_numeric(df["log_odds_ratio"], errors="coerce")
    df["ci_lo_log"] = np.log(df["CI Lower"])
    df["ci_hi_log"] = np.log(df["CI Upper"])
    # split main and CI parts (we'll render them with different font sizes)
    df["annot_main"] = df["log_odds_ratio"].round(2).astype(str) + df["sig_star"].astype(str)
    df["annot_ci_only"] = (
        "[" + df["ci_lo_log"].round(2).astype(str) +
        ", " + df["ci_hi_log"].round(2).astype(str) + "]"
    )
    return df

gender_tbl2 = add_log_ci_and_annot(gender_tbl)
age_tbl2    = add_log_ci_and_annot(age_tbl)

merged_df = pd.concat([age_tbl2, gender_tbl2], ignore_index=True)

# Column order and display labels (edit these if you need different labels)
age_raw_labels     = [f"Age{i}" for i in range(1, 6)]
age_display_labels = ["≤18", "19–34", "35–49", "50–64", "65+"]
demographic_order  = age_raw_labels + ["Gender (Male)"]
xtick_labels       = age_display_labels + ["Gender (Male)"]

# ---------- build a section-first row order (ordered_terms) ----------
# If you already have `ordered_terms`, keep it; else derive from data:
try:
    ordered_terms  # if defined earlier, keep it
except NameError:
    # try to use your predefined `sections`; otherwise infer
    try:
        _sections = sections
    except NameError:
        # infer unique prefixes like "Section 1", "Section 2", ...
        _sections = (merged_df["Feature"].str.extract(r'^(Section \d+)')[0]
                     .dropna().drop_duplicates().tolist())
    # within each section, preserve first-seen order of the remaining part
    ordered_terms = []
    for sec in _sections:
        part = merged_df.loc[
            merged_df["Feature"].str.startswith(sec),
            "Feature"
        ].drop_duplicates()
        # Preserve original order
        ordered_terms.extend(part.tolist())

# ---------- 6) Pivot for heatmap + two-layer annotations ----------
heatmap_data = pd.pivot_table(
    merged_df, index="Feature", columns="Demographic",
    values="log_odds_ratio", aggfunc="first"
).reindex(index=ordered_terms, columns=demographic_order)

annot_main_tbl = pd.pivot_table(
    merged_df, index="Feature", columns="Demographic",
    values="annot_main", aggfunc="first"
).reindex(index=ordered_terms, columns=demographic_order)

annot_ci_tbl = pd.pivot_table(
    merged_df, index="Feature", columns="Demographic",
    values="annot_ci_only", aggfunc="first"
).reindex(index=ordered_terms, columns=demographic_order)

# ---------- 7) Plot: manual two-line text with adaptive color ----------
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    heatmap_data,
    annot=False,              # we'll draw text manually
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    linecolor="gray",
    cbar_kws={'label': 'Log-Odds'}
)

# helper: adaptive text color based on underlying cell color
def pick_text_color_from_value(val, mappable, thresh=0.53):
    # map data value -> RGBA using the mesh's norm and cmap
    rgba = mappable.cmap(mappable.norm(val))
    r, g, b, _ = rgba
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    return "black" if lum > thresh else "white"

mesh = ax.collections[0]
n_rows, n_cols = heatmap_data.shape

# font handle (optional)
try:
    fp = font_prop  # from your earlier code
except NameError:
    fp = None

# draw two lines of text per cell (main + smaller CI)
for i, row_key in enumerate(heatmap_data.index):
    for j, col_key in enumerate(heatmap_data.columns):
        val = heatmap_data.loc[row_key, col_key]
        main_txt = annot_main_tbl.loc[row_key, col_key]
        ci_txt   = annot_ci_tbl.loc[row_key, col_key]

        if pd.isna(val) and (pd.isna(main_txt) or pd.isna(ci_txt)):
            continue

        color = pick_text_color_from_value(val if pd.notna(val) else 0.0, mesh)

        # line 1: log-odds + stars (regular size)
        if pd.notna(main_txt):
            ax.text(
                j + 0.5, i + 0.42, str(main_txt),
                ha="center", va="center",
                fontsize=12, color=color,
                fontproperties=fp
            )
        # line 2: [CI_low, CI_high] (smaller size)
        if pd.notna(ci_txt):
            ax.text(
                j + 0.5, i + 0.68, str(ci_txt),
                ha="center", va="center",
                fontsize=9, color=color,   # smaller font just for CI
                fontproperties=fp
            )

# Colorbar label
cbar = ax.collections[0].colorbar
if fp is not None:
    cbar.ax.set_ylabel("Log-Odds", fontproperties=fp, fontsize=12)
else:
    cbar.ax.set_ylabel("Log-Odds", fontsize=12)

# X ticks
ax.set_xticks(np.arange(len(demographic_order)) + 0.5)
if fp is not None:
    ax.set_xticklabels(xtick_labels, rotation=0, fontproperties=fp, fontsize=14)
else:
    ax.set_xticklabels(xtick_labels, rotation=0, fontsize=12)

# Y ticks: show only the theme/feature name (drop "Section X ")
ytick_labels = [" ".join(lbl.split(" ")[2:]) for lbl in heatmap_data.index]
if fp is not None:
    ax.set_yticklabels(ytick_labels, rotation=0, fontproperties=fp, fontsize=14)
else:
    ax.set_yticklabels(ytick_labels, rotation=0, fontsize=12)

plt.title("Log(Odds Ratio) by Thematic Feature across Demographics (Age & Gender)\n"
          "Top: log(OR) with stars, Bottom: 95% CI on log scale",
          fontproperties=fp, fontsize=12 if fp is not None else 12)
plt.xlabel("Demographic", fontproperties=fp, fontsize=14 if fp is not None else 12)
ax.set_ylabel("Thematic Feature", fontproperties=fp, fontsize=14 if fp is not None else 12)
ax.yaxis.set_label_coords(-0.4, 0.5)

# Optional: add vertical section labels on far left
# ---------- Optional: add vertical section labels on far left (custom names) ----------
import re

# Map the underlying prefixes in your Feature strings to display labels
# e.g., Feature values like "Section 1 — Apology", "Section 2 — Love", etc.
section_display_map = {
    "Section 1": "Opening",
    "Section 2": "Middle",
    "Section 3": "Ending",
}

# Where to draw the labels (to the left of the heatmap)
x_section_label = -2.0  # adjust leftward/rightward as needed

row_labels = list(heatmap_data.index)

# Find the block of rows for each section and place the label at the block midpoint
for section_prefix, display_name in section_display_map.items():
    rows_for_section = [
        i for i, lbl in enumerate(row_labels)
        if re.match(fr"^{re.escape(section_prefix)}\b", str(lbl))
    ]
    if not rows_for_section:
        continue  # skip if this section isn't present in the current data

    y_middle = (rows_for_section[0] + rows_for_section[-1]) / 2.0
    ax.text(
        x=x_section_label, y=y_middle, s=display_name,
        va="center", ha="center", rotation=90,
        fontsize=14, weight="bold", color="black",
        transform=ax.transData, clip_on=False,
        fontproperties=fp
    )

ax.tick_params(bottom=True, left=True) 

plt.tight_layout()
plt.show()


#%% Without confidence intervals
# ---------- 5) Stack Age + Gender into merged_df ----------
common_cols = ['Feature', 'Coefficient', 'Odds Ratio', 'p-value', 'CI Lower', 'CI Upper',
               'Demographic', 'log_odds_ratio', 'sig_star', 'annot']
gender_tbl2 = gender_tbl[common_cols].copy()
age_tbl2    = age_tbl[common_cols].copy()

merged_df = pd.concat([age_tbl2, gender_tbl2], ignore_index=True)
merged_df["feature"] = pd.Categorical(merged_df["Feature"], categories=ordered_terms, ordered=True)

# Column order and display labels
age_raw_labels = [f"Age{i}" for i in range(1,6)]
age_display_labels = ["≤18", "19–34", "35–49", "50–64", "65+"]
demographic_order = age_raw_labels + ["Gender (Male)"]
xtick_labels = age_display_labels + ["Gender (Male)"]

# ---------- 6) Combined heatmap ----------
heatmap_data = pd.pivot_table(
    merged_df, index="feature", columns="Demographic",
    values="log_odds_ratio", aggfunc="first"
)
annot_data = pd.pivot_table(
    merged_df, index="feature", columns="Demographic",
    values="annot", aggfunc="first"
)
heatmap_data = heatmap_data.reindex(index=ordered_terms, columns=demographic_order)
annot_data   = annot_data.reindex(index=ordered_terms, columns=demographic_order)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    heatmap_data,
    annot=annot_data,
    fmt="",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    linecolor="gray",
    cbar_kws={'label': 'Log-Odds'}
)

for text in ax.texts:
    text.set_fontproperties(font_prop)
    text.set_fontsize(10)  # override size if needed


# Colorbar label font
cbar = ax.collections[0].colorbar
if font_prop is not None:
    cbar.ax.set_ylabel("Log-Odds", fontproperties=font_prop, fontsize=12)
else:
    cbar.ax.set_ylabel("Log-Odds", fontsize=10)

# X ticks
ax.set_xticks(np.arange(len(demographic_order)) + 0.5)
if font_prop is not None:
    ax.set_xticklabels(xtick_labels, rotation=0, fontproperties=font_prop, fontsize=12)
else:
    ax.set_xticklabels(xtick_labels, rotation=0)

# Y ticks: show only the theme name (drop "Section X")
ytick_labels = [" ".join(lbl.split(" ")[2:]) for lbl in heatmap_data.index]
if font_prop is not None:
    ax.set_yticklabels(ytick_labels, rotation=0, fontproperties=font_prop, fontsize=12)
else:
    ax.set_yticklabels(ytick_labels, rotation=0)

# Titles & labels
title_str = "Log(Odds Ratio) by Thematic Feature across Demographics (Age & Gender)"
if font_prop is not None:
    plt.title(title_str, fontsize=12, fontproperties=font_prop)
    plt.xlabel("Demographic", fontproperties=font_prop, fontsize=12)
    ax.set_ylabel("Thematic Feature", fontproperties=font_prop, fontsize=12)
    ax.yaxis.set_label_coords(-0.4, 0.5)

else:
    plt.title(title_str, fontsize=12)
    plt.xlabel("Demographic")
    ax.set_ylabel("Thematic Feature")
    ax.yaxis.set_label_coords(-20.5, 0.5)


# Add vertical section labels on the far left
themes_per_section = len(themes)
x_section_label = -2.0
for i, section in enumerate(sections):
    y_middle = i * themes_per_section + themes_per_section / 2 - 0.5
    ax.text(
        x=x_section_label, y=y_middle, s=section,
        va="center", ha="center", rotation=90,
        fontproperties=font_prop if font_prop else None,
        fontsize=12, weight="bold", color="black",
        transform=ax.transData, clip_on=False
    )

plt.tight_layout()
plt.show()


#%% SENTIMENT

# -*- coding: utf-8 -*-
"""
Sentiment → Gender logistic regression controlling for Age (categorical)
+ Gender line plot (Male vs Female, mirrored log-odds)
+ Combined heatmap for Age (OVR) & Gender
"""

# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =========================
# 0) Data prep: build 45 sentiment×section features + AGE2
# =========================
# Filter rows with valid AGE2 bins
df = morethan3rawsentiment[morethan3rawsentiment["AGE2"].isin([1,2,3,4,5])].copy()

# Sentiment presence columns (each is a list [Intro, Body, Conclusion])
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
sentiments = list(sentiment_presence_columns.keys())
sections = ["Section 1", "Section 2", "Section 3"]  # Intro, Body, Conclusion
sentiment_feature_names = [f"{sec} {s}" for sec in sections for s in sentiments]  # 45 names

def get_sentimentsection_vector(row, section_idx):
    return [row[sentiment_presence_columns[s]][section_idx] for s in sentiments]

df["section1_sentiment_vector"] = df.apply(lambda r: get_sentimentsection_vector(r, 0), axis=1)
df["section2_sentiment_vector"] = df.apply(lambda r: get_sentimentsection_vector(r, 1), axis=1)
df["section3_sentiment_vector"] = df.apply(lambda r: get_sentimentsection_vector(r, 2), axis=1)

def concatenate_sentiment_vectors_with_age(row):
    return (
        row["section1_sentiment_vector"]
        + row["section2_sentiment_vector"]
        + row["section3_sentiment_vector"]
        + [row["AGE2"]]
    )

df["full_sentiment_vector"] = df.apply(concatenate_sentiment_vectors_with_age, axis=1)

# Materialize matrix: 45 features + AGE2 (last)
sentiment_feature_names_plus_age = sentiment_feature_names + ["AGE2"]
sentiment_matrix = pd.DataFrame(
    df["full_sentiment_vector"].apply(lambda v: list(map(int, v))).tolist(),
    index=df.index,
    columns=sentiment_feature_names_plus_age
)

# Replace any stale columns and merge
df = df.drop(columns=[c for c in sentiment_feature_names_plus_age if c in df.columns], errors="ignore")
df = pd.concat([df, sentiment_matrix], axis=1)

# =========================
# 1) Gender model: SEX ~ 45 sentiments + AGE2 (categorical dummies)
# =========================
# Keep valid SEX and recode: Male=1, Female=0
gdf = df[df["SEX"].isin([1,2])].copy()
gdf["SEX"] = gdf["SEX"].replace({1:1, 2:0}).astype(int)

# Build X: 45 sentiment features from full_sentiment_vector
X_sent = pd.DataFrame(
    gdf["full_sentiment_vector"].apply(lambda v: list(map(int, v[:45]))).tolist(),
    index=gdf.index, columns=sentiment_feature_names
)
# AGE2 categorical dummies (≤18 as reference)
age_dummies = pd.get_dummies(gdf["AGE2"], prefix="AGE2", drop_first=True)
for col in [f"AGE2_{k}" for k in [2,3,4,5]]:
    if col not in age_dummies.columns:
        age_dummies[col] = 0
age_dummies = age_dummies[[f"AGE2_{k}" for k in [2,3,4,5]]]

# Concatenate, clean, add intercept
X = pd.concat([X_sent, age_dummies], axis=1)
X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = sm.add_constant(X, has_constant="add")
y = gdf["SEX"]

mask = ~(X.isna().any(axis=1) | y.isna())
X_clean = X.loc[mask].copy()
y_clean = y.loc[mask].copy()

gmodel = sm.Logit(y_clean, X_clean).fit(disp=False)

# Summarize coefficients
odds_ratios = np.exp(gmodel.params)
conf = np.exp(gmodel.conf_int()); conf.columns = ["CI Lower", "CI Upper"]
summary_df = pd.DataFrame({
    "term": gmodel.params.index,
    "Coefficient": gmodel.params.values,
    "Odds Ratio": odds_ratios.values,
    "p-value": gmodel.pvalues.values,
    "CI Lower": conf["CI Lower"].values,
    "CI Upper": conf["CI Upper"].values,
})
summary_df = summary_df[summary_df["term"] != "const"].copy()
summary_df["log_odds_ratio"] = np.log(summary_df["Odds Ratio"])

# Print AGE2 categorical effects table (reference = ≤18)
age_terms = summary_df[summary_df["term"].str.startswith("AGE2_")].copy()
if not age_terms.empty:
    print("\nAGE2 categorical effects in Gender model (reference = ≤18):")
    print(
        age_terms[["term","Coefficient","Odds Ratio","p-value","CI Lower","CI Upper"]]
        .sort_values("term").to_string(index=False)
    )

# =========================
# 2) Gender line plot (Male vs Female; mirrored log-odds)
# =========================
def get_sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# Limit to the 45 sentiment terms (drop age dummies)
gplot = summary_df[summary_df["term"].isin(sentiment_feature_names)].copy()
gplot["section"] = gplot["term"].apply(lambda x: " ".join(x.split()[:2]))
gplot["sentiment"] = gplot["term"].apply(lambda x: " ".join(x.split()[2:]))
gplot["sig_star"] = gplot["p-value"].apply(get_sig_star)

# Build Male/Female frames
male_df = gplot.copy()
male_df["GenderVal"] = 1

female_df = gplot.copy()
female_df["GenderVal"] = 0
female_df["log_odds_ratio"] = -female_df["log_odds_ratio"]  # mirror on log-odds
female_df["sig_star"] = female_df["p-value"].apply(get_sig_star)

plot_df = pd.concat([female_df, male_df], ignore_index=True)

# Font (optional)
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
try:
    custom_font = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family": custom_font})
    font_prop = fm.FontProperties(fname=font_path)
except Exception:
    font_prop = None

sns.set(style="white", context="notebook", font_scale=1.2)
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.spines.right": False,
    "axes.spines.top": False
})

# Choose which sentiments to display in the line plot (6 panels)
lineplot_sentiments = ["Defeat", "Hatred", "Neutral", "Happiness", "Anxious", "Disappointment"]
gender_labels = ["Female", "Male"]
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8), sharey=True)
fig.subplots_adjust(right=0.82)

for idx, sent in enumerate(lineplot_sentiments):
    r, c = divmod(idx, n_cols)
    ax = axes[r, c]
    d = plot_df[plot_df["sentiment"] == sent]

    for sec in sections:
        line = d[d["section"] == sec].sort_values("GenderVal")
        ax.plot(line["GenderVal"], line["log_odds_ratio"], marker="o", label=sec, linewidth=2)
        for _, row_ in line.iterrows():
            if row_["sig_star"]:
                ax.text(row_["GenderVal"], row_["log_odds_ratio"], row_["sig_star"],
                        ha="center", va="bottom", fontsize=11, weight='bold')

    ax.set_title(sent, fontsize=13, fontproperties=font_prop if font_prop else None)
    ax.set_xticks([0,1]); ax.set_xticklabels(gender_labels, fontsize=10,
                                             fontproperties=font_prop if font_prop else None)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Gender", fontsize=12, fontproperties=font_prop if font_prop else None)
    ax.tick_params(bottom=True, left=True) 

    if c == 0:
        ax.set_ylabel("Log-Odds", fontsize=11, fontproperties=font_prop if font_prop else None)
    # Set font for y-axis tick labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(12)

    
    # Set font for x-axis tick labels (optional, but already done earlier with set_xticklabels)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(12)
        
# Remove extra axes if any
if len(lineplot_sentiments) < n_rows * n_cols:
    for i in range(len(lineplot_sentiments), n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, title="Section",
           loc='center left', bbox_to_anchor=(0.87, 0.5),
           frameon=False, fontsize=11, title_fontsize=12 if not font_prop else None)

plt.suptitle("Sentiment×Section Effects on Gender (Logit Log-Odds, Controlling for Age)",
             fontsize=14, y=0.95, fontproperties=font_prop if font_prop else None)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])
plt.show()


#%% Version 2


# =========================
# 2) Gender line plot (Female vs Male; x = Sections 1–3; two lines)
# =========================
def get_sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# Limit to the sentiment terms only (drop age dummies)
gplot = summary_df[summary_df["term"].isin(sentiment_feature_names)].copy()
gplot["section"]   = gplot["term"].apply(lambda x: " ".join(x.split()[:2]))       # 'Section 1'
gplot["sentiment"] = gplot["term"].apply(lambda x: " ".join(x.split()[2:]))       # e.g., 'Defeat'
gplot["sig_star"]  = gplot["p-value"].apply(get_sig_star)

# Build Male (as-is) and Female (mirrored) frames + labels
male_df = gplot.copy()
male_df["Label"] = "Male"

female_df = gplot.copy()
female_df["Label"] = "Female"
female_df["log_odds_ratio"] = -female_df["log_odds_ratio"]  # mirror male log-OR

plot_df = pd.concat([female_df, male_df], ignore_index=True)

# Ordered sections for x-axis
section_order = ["Section 1", "Section 2", "Section 3"]
plot_df = plot_df[plot_df["section"].isin(section_order)].copy()
plot_df["section"] = pd.Categorical(plot_df["section"], categories=section_order, ordered=True)

# Font (optional)
import matplotlib.font_manager as fm
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
try:
    custom_font = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family": custom_font})
    font_prop = fm.FontProperties(fname=font_path)
except Exception:
    font_prop = None

# Style
sns.set(style="white", context="notebook", font_scale=1.2)
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.spines.right": False,
    "axes.spines.top": False
})

# Choose which sentiments to display in the line plot (6 panels)
lineplot_sentiments = ["Defeat", "Hatred", "Neutral", "Happiness", "Anxious", "Disappointment"]

# Layout
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
axes = axes.flatten()
fig.subplots_adjust(right=0.82)

# Plot each sentiment: x = sections; lines = Female & Male
import numpy as np

for idx, sent in enumerate(lineplot_sentiments):
    ax = axes[idx]
    sdata = plot_df[plot_df["sentiment"] == sent].copy()

    for label in ["Female", "Male"]:
        sub = (sdata[sdata["Label"] == label]
               .sort_values("section"))
        # x as numeric 1..3; labels set below
        x = np.arange(1, len(section_order) + 1)
        y = sub["log_odds_ratio"].to_numpy()

        ax.plot(x, y, marker="o", linewidth=2, label=label)

        # significance stars (same p-values for ± mirrored lines)
        for i, (_, r_) in enumerate(sub.iterrows(), start=1):
            if r_["sig_star"]:
                ax.text(i, r_["log_odds_ratio"], r_["sig_star"],
                        ha="center", va="bottom", fontsize=11, weight='bold')

    # cosmetics
    ax.set_title(sent, fontsize=13, fontproperties=font_prop if font_prop else None)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(section_order, fontsize=10,
                       fontproperties=font_prop if font_prop else None)
    ax.set_xlabel("Section", fontsize=11, fontproperties=font_prop if font_prop else None)
    if idx % n_cols == 0:
        ax.set_ylabel("Log-Odds", fontsize=11, fontproperties=font_prop if font_prop else None)
    # y tick font
    for lab in ax.get_yticklabels():
        if font_prop:
            lab.set_fontproperties(font_prop)
        lab.set_fontsize(10)

# Remove extra axes if fewer than 6 sentiments
if len(lineplot_sentiments) < n_rows * n_cols:
    for i in range(len(lineplot_sentiments), n_rows * n_cols):
        fig.delaxes(axes[i])

# Legend: two gender lines
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Gender",
           loc='center left', bbox_to_anchor=(0.87, 0.5),
           frameon=False, fontsize=11,
           title_fontsize=(12 if not font_prop else None))

plt.suptitle("Sentiment×Section Effects on Gender (Logit Log-Odds, Controlling for Age)\nlines = Female/Male; x = Sections",
             fontsize=14, y=0.95, fontproperties=font_prop if font_prop else None)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])
plt.show()

#%%
# =========================
# 3) Age one-vs-rest models: Age1..Age5 ~ 45 sentiments + Gender
# =========================
# Design for age models: 45 sentiments + Gender covariate
sent_all = pd.DataFrame(
    df["full_sentiment_vector"].apply(lambda v: list(map(int, v[:45]))).tolist(),
    index=df.index, columns=sentiment_feature_names
)
adf = df[df["AGE2"].isin([1,2,3,4,5]) & df["SEX"].isin([1,2])].copy()
adf["Gender"] = adf["SEX"].replace({1:1, 2:0}).astype(int)

X_age = pd.concat([sent_all.loc[adf.index], adf["Gender"]], axis=1)
X_age = X_age.apply(pd.to_numeric, errors="coerce")
X_age.replace([np.inf, -np.inf], np.nan, inplace=True)
X_age = sm.add_constant(X_age, has_constant="add")

group_results = {}
for g in [1,2,3,4,5]:
    y = (adf["AGE2"] == g).astype(int)
    mask = ~(X_age.isna().any(axis=1) | y.isna())
    Xi, yi = X_age.loc[mask], y.loc[mask]
    m = sm.Logit(yi, Xi).fit(disp=False)

    OR = np.exp(m.params)
    CI = np.exp(m.conf_int()); CI.columns = ["CI Lower","CI Upper"]
    res = pd.DataFrame({
        "Feature": m.params.index,
        "Coefficient": m.params.values,
        "Odds Ratio": OR.values,
        "p-value": m.pvalues.values,
        "CI Lower": CI["CI Lower"].values,
        "CI Upper": CI["CI Upper"].values,
        "Age Group": g
    })
    group_results[g] = res

# =========================
# 4) Combined heatmap (Age OVR + Gender (Male)) for 45 sentiments
# =========================
# Tidy gender results
gender_tbl = summary_df[summary_df["term"].isin(sentiment_feature_names)].copy()
gender_tbl = gender_tbl.rename(columns={"term":"Feature"})
gender_tbl["Demographic"] = "Gender (Male)"
gender_tbl["sig_star"] = gender_tbl["p-value"].apply(get_sig_star)
gender_tbl["annot"] = gender_tbl["log_odds_ratio"].round(2).astype(str) + gender_tbl["sig_star"]

# Tidy age results (drop const & Gender covariate)
age_tbl = pd.concat(group_results.values(), ignore_index=True)
age_tbl = age_tbl[age_tbl["Feature"].isin(sentiment_feature_names)].copy()
age_tbl["Demographic"] = age_tbl["Age Group"].apply(lambda x: f"Age{x}")
age_tbl["log_odds_ratio"] = np.log(age_tbl["Odds Ratio"])
age_tbl["sig_star"] = age_tbl["p-value"].apply(get_sig_star)
age_tbl["annot"] = age_tbl["log_odds_ratio"].round(2).astype(str) + age_tbl["sig_star"]

# Stack
common_cols = ['Feature','Coefficient','Odds Ratio','p-value','CI Lower','CI Upper',
               'Demographic','log_odds_ratio','sig_star','annot']
merged_df = pd.concat([age_tbl[common_cols], gender_tbl[common_cols]], ignore_index=True)

# Row/column orders
ordered_features = sentiment_feature_names  # section-first, as defined above
age_raw_labels = [f"Age{i}" for i in range(1,6)]
age_display_labels = ["≤18","19–34","35–49","50–64","65+"]
demographic_order = age_raw_labels + ["Gender (Male)"]
xtick_labels = age_display_labels + ["Gender (Male)"]

#%% Comprehensive heatmap with CI (two-line labels, adaptive text color)

# Safety check
required_cols = {'Feature','Demographic','log_odds_ratio','sig_star','CI Lower','CI Upper'}
missing = required_cols - set(merged_df.columns)
if missing:
    raise ValueError(f"merged_df is missing columns required for CI annotations: {missing}")

# Build main + CI-only annotation strings (log-scale CI)
eps = np.finfo(float).tiny
merged_df = merged_df.copy()
merged_df["CI Lower"] = pd.to_numeric(merged_df["CI Lower"], errors="coerce").clip(lower=eps)
merged_df["CI Upper"] = pd.to_numeric(merged_df["CI Upper"], errors="coerce").clip(lower=eps)
merged_df["log_odds_ratio"] = pd.to_numeric(merged_df["log_odds_ratio"], errors="coerce")

merged_df["ci_lo_log"] = np.log(merged_df["CI Lower"])
merged_df["ci_hi_log"] = np.log(merged_df["CI Upper"])

merged_df["annot_main"] = merged_df["log_odds_ratio"].round(2).astype(str) + merged_df["sig_star"].astype(str)
merged_df["annot_ci_only"] = (
    "[" + merged_df["ci_lo_log"].round(2).astype(str)
    + ", " + merged_df["ci_hi_log"].round(2).astype(str) + "]"
)

# Pivot for plotting
heatmap_data = pd.pivot_table(
    merged_df, index="Feature", columns="Demographic",
    values="log_odds_ratio", aggfunc="first"
).reindex(index=ordered_features, columns=demographic_order)

annot_main_tbl = pd.pivot_table(
    merged_df, index="Feature", columns="Demographic",
    values="annot_main", aggfunc="first"
).reindex(index=ordered_features, columns=demographic_order)

annot_ci_tbl = pd.pivot_table(
    merged_df, index="Feature", columns="Demographic",
    values="annot_ci_only", aggfunc="first"
).reindex(index=ordered_features, columns=demographic_order)

# Plot heatmap (manual two-line text so we can style sizes)
plt.figure(figsize=(12, max(14, 0.38 * len(ordered_features))))
ax = sns.heatmap(
    heatmap_data,
    annot=False,                 # we'll draw text manually
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    linecolor="gray",
    cbar_kws={'label': 'Log-Odds'}
)

# Adaptive text color based on the cell color
def pick_text_color_from_value(val, mappable, thresh=0.53):
    rgba = mappable.cmap(mappable.norm(val if pd.notna(val) else 0.0))
    r, g, b, _ = rgba
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    return "black" if lum > thresh else "white"

mesh = ax.collections[0]

# Draw two lines per cell: main (bigger) + CI (smaller)
for i, row_key in enumerate(heatmap_data.index):
    for j, col_key in enumerate(heatmap_data.columns):
        val = heatmap_data.loc[row_key, col_key]
        main_txt = annot_main_tbl.loc[row_key, col_key]
        ci_txt   = annot_ci_tbl.loc[row_key, col_key]
        if pd.isna(main_txt) and pd.isna(ci_txt):
            continue

        color = pick_text_color_from_value(val, mesh)

        # line 1: log-odds + stars
        if pd.notna(main_txt):
            ax.text(
                j + 0.5, i + 0.42, str(main_txt),
                ha="center", va="center",
                fontsize=11, color=color,
                fontproperties=font_prop if 'font_prop' in globals() and font_prop is not None else None
            )
        # line 2: CI (smaller font)
        if pd.notna(ci_txt):
            ax.text(
                j + 0.5, i + 0.68, str(ci_txt),
                ha="center", va="center",
                fontsize=8, color=color,
                fontproperties=font_prop if 'font_prop' in globals() and font_prop is not None else None
            )

# Colorbar label
cbar = ax.collections[0].colorbar
if 'font_prop' in globals() and font_prop is not None:
    cbar.ax.set_ylabel("Log-Odds", fontproperties=font_prop, fontsize=12)
else:
    cbar.ax.set_ylabel("Log-Odds", fontsize=12)

# X ticks
ax.set_xticks(np.arange(len(demographic_order)) + 0.5)
if 'font_prop' in globals() and font_prop is not None:
    ax.set_xticklabels(xtick_labels, rotation=0, fontproperties=font_prop, fontsize=12)
else:
    ax.set_xticklabels(xtick_labels, rotation=0, fontsize=12)

# Y ticks: show only the sentiment name (drop "Section X")
ytick_labels = [" ".join(lbl.split(" ")[2:]) for lbl in heatmap_data.index]
if 'font_prop' in globals() and font_prop is not None:
    ax.set_yticklabels(ytick_labels, rotation=0, fontproperties=font_prop, fontsize=12)
else:
    ax.set_yticklabels(ytick_labels, rotation=0, fontsize=12)

plt.title("Log(Odds Ratio) by Sentiment Feature across Demographics (Age & Gender)\n"
          "Top: log(OR) with stars • Bottom: 95% CI (log scale)",
          fontproperties=font_prop if 'font_prop' in globals() and font_prop is not None else None,
          fontsize=12)

plt.xlabel("Demographic",
           fontproperties=font_prop if 'font_prop' in globals() and font_prop is not None else None,
           fontsize=12)
ax.set_ylabel("Sentiment Feature",
              fontproperties=font_prop if 'font_prop' in globals() and font_prop is not None else None,
              fontsize=12)
ax.yaxis.set_label_coords(-0.3, 0.5)

# Optional: add vertical section labels on far left
# ---------- Optional: add vertical section labels on far left (custom names) ----------
import re

# Map the underlying prefixes in your Feature strings to display labels
# e.g., Feature values like "Section 1 — Apology", "Section 2 — Love", etc.
section_display_map = {
    "Section 1": "Opening",
    "Section 2": "Middle",
    "Section 3": "Ending",
}

# Where to draw the labels (to the left of the heatmap)
x_section_label = -1.25  # adjust leftward/rightward as needed

row_labels = list(heatmap_data.index)

# Find the block of rows for each section and place the label at the block midpoint
for section_prefix, display_name in section_display_map.items():
    rows_for_section = [
        i for i, lbl in enumerate(row_labels)
        if re.match(fr"^{re.escape(section_prefix)}\b", str(lbl))
    ]
    if not rows_for_section:
        continue  # skip if this section isn't present in the current data

    y_middle = (rows_for_section[0] + rows_for_section[-1]) / 2.0
    ax.text(
        x=x_section_label, y=y_middle, s=display_name,
        va="center", ha="center", rotation=90,
        fontsize=14, weight="bold", color="black",
        transform=ax.transData, clip_on=False,
        fontproperties=font_prop
    )

ax.tick_params(bottom=True, left=True) 

plt.tight_layout()
plt.show()

#%%


# Pivot robustly
heatmap_data = pd.pivot_table(
    merged_df, index="Feature", columns="Demographic",
    values="log_odds_ratio", aggfunc="first"
)
annot_data = pd.pivot_table(
    merged_df, index="Feature", columns="Demographic",
    values="annot", aggfunc="first"
)
heatmap_data = heatmap_data.reindex(index=ordered_features, columns=demographic_order)
annot_data   = annot_data.reindex(index=ordered_features, columns=demographic_order)

# Plot heatmap
plt.figure(figsize=(12, 18))
ax = sns.heatmap(
    heatmap_data, annot=annot_data, fmt="",
    cmap="coolwarm", center=0, linewidths=0.5, linecolor="gray",
    cbar_kws={'label': 'Log-Odds'}
)

for text in ax.texts:
    text.set_fontproperties(font_prop)
    text.set_fontsize(10)  # override size if needed


# Font for labels
try:
    custom_font = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family": custom_font})
    font_prop = fm.FontProperties(fname=font_path)
except Exception:
    font_prop = None

# Colorbar label
cbar = ax.collections[0].colorbar
if font_prop is not None:
    cbar.ax.set_ylabel("Log-Odds", fontproperties=font_prop, fontsize=12)
else:
    cbar.ax.set_ylabel("Log-Odds", fontsize=10)

# X ticks
ax.set_xticks(np.arange(len(demographic_order)) + 0.5)
if font_prop is not None:
    ax.set_xticklabels(xtick_labels, rotation=0, fontproperties=font_prop,fontsize=12)
else:
    ax.set_xticklabels(xtick_labels, rotation=0)

# Y ticks: show only sentiment (remove "Section X ")
ytick_labels = [" ".join(lbl.split(" ")[2:]) for lbl in heatmap_data.index]
if font_prop is not None:
    ax.set_yticklabels(ytick_labels, rotation=0, fontproperties=font_prop,fontsize=12 )
else:
    ax.set_yticklabels(ytick_labels, rotation=0)

plt.title("Log(Odds Ratio) by Sentiment Feature across Demographics (Age & Gender)",
          fontsize=12, fontproperties=font_prop if font_prop else None)
plt.xlabel("Demographic", fontproperties=font_prop if font_prop else None)
ax.set_ylabel("Sentiment Feature", fontproperties=font_prop if font_prop else None)
ax.yaxis.set_label_coords(-0.3, 0.5)

# Add vertical section labels on the far left
sents_per_section = len(sentiments)
x_section_label = -1.25
for i, sec in enumerate(sections):
    y_mid = i * sents_per_section + sents_per_section / 2 - 0.5
    ax.text(x_section_label, y_mid, sec, va="center", ha="center", rotation=90,
            fontproperties=font_prop if font_prop else None,
            fontsize=12, weight="bold", color="black",
            transform=ax.transData, clip_on=False)

plt.tight_layout()
plt.show()

# Optional: model fit info
print("\nGender model fit:")
print(f"  Log-Likelihood: {gmodel.llf: .3f}   AIC: {gmodel.aic: .2f}")
print("Reminder: SEX coded Male=1, Female=0. OR>1 ⇒ feature associated with higher odds of Male.")

