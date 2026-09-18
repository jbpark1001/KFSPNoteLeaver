# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 11:45:38 2025

@author: Jae Bin Park
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:42:11 2025
Requires the sentiment_environment environment
@author: Jae Bin Park
"""
import pandas as pd
raw= pd.read_pickle(r'C:\Users\Jae Bin Park\raw.pkl')

# =============================================================================
# notes= pd.read_excel(r'C:\Users\Jae Bin Park\kfsp_sectioned_ver4.xlsx')
# 
# 
# #%%
# 
# import pandas as pd
# raw= pd.read_pickle(r'C:\Users\Jae Bin Park\raw.pkl')
# 
# notes= pd.read_excel(r'C:\Users\Jae Bin Park\kfsp_sectioned_ver4.xlsx')
# morethan3=raw[raw["too_short"]==False]
# 
# raw=notes[notes.index.isin(morethan3.index)]
# =============================================================================
#%%


from tqdm import tqdm

def create_section_presence_vector(df, target_tokens, keyword_cols, label):
    tqdm.pandas(desc=f"Checking token presence for: {label}")

    def check_row(row):
        presence = []
        for col in keyword_cols:
            section = row[col]
            if not isinstance(section, list):
                presence.append(0)
                continue
            tokens = [t for t, _ in section]
            has_token = any(token in tokens for token in target_tokens)
            presence.append(int(has_token))
        return presence

    df[f"{label}_presence"] = df.progress_apply(check_row, axis=1)
    return df



sorry = [
    "미안", "죄송", "용서", "잘못", "후회", "죄", "책임", "민폐", "반성"
]

sorrycomprehensive = [
    "미안", "죄송", "용서", "잘못", "후회", "죄", "책임", "민폐", "반성",
    "변명", "참회", "사죄", "후회하", "잘못하", "죄책", "실수", "벌받", "속죄", "용서받",
    "부끄럽", "창피", "머쓱", "자책", "기억에남", "상처줬", "상처", "괴롭혔"
]


loveandgratitude = [
    "사랑", "고맙", "감사", "고생", "수고", "보고싶", "애틋", "소중", "그리"
]

loveandgratitudecomprehensive = [
    "사랑", "고맙", "감사", "고생", "수고", "보고싶", "애틋", "소중", "그리",
    "그립", "그리움", "좋았", "행복", "사랑했", "소중했", "잊지", "기억해", "행복하", "사랑받",
    "추억", "그때", "함께", "웃었", "따뜻", "힘이되", "위로받", "이해해줘서", "든든했", "친구야", "고마워", "정들"
]


burdensome = [
    "버겁", "부담", "짐", "감당", "힘들", "무겁", "지치", "헷갈"
]

burdensomecomprehensive = [
    "버겁", "부담", "짐", "감당", "힘들", "무겁", "지치", "헷갈",
    "포기하", "의지할", "기댈", "쉴곳", "기대고싶", "미안해서", "도움받", "걱정끼", "방해", "누가되", "무력", "불편하게", "귀찮게",
    "부담끼", "감정적", "답답", "억눌", "참아왔", "내성적", "무관심"
]


despair = [
    "포기", "좌절", "죽", "끝", "아무것", "헛되", "무의미", "잊히", "없어지", "그만두"
]

despaircomprehensive = [
    "포기", "좌절", "죽", "끝", "아무것", "헛되", "무의미", "잊히", "없어지", "그만두",
    "사라지", "떠나", "살기싫", "쓸모없", "죽고싶", "죽으려", "아무도", "절망", "지옥", "혼자", "외로", "무기력",
    "살고싶지", "끝내", "두렵지", "괴롭", "살의", "벗어나", "어둠", "힘빠", "고통", "피하고싶", "다잊고"
]


pmaffairs = [
    "부탁", "정리", "남기", "처리", "보험", "은행", "장례", "통장", "유서"
]

pmaffairscomprehensive = [
    "부탁", "정리", "남기", "처리", "보험", "은행", "장례", "통장", "유서",
    "정돈", "작성", "챙겨", "남겨두", "문서", "정산", "정리했", "지갑", "계좌", "연락처", "연락하",
    "친구한테", "부모님께", "동생한테", "건네", "유품", "물건", "사진", "메모", "마지막으로", "정리중"
]


import ast

section_columns = [
    "tokenizedkluekeywordsentencetransformer_1st_section",
    "tokenizedkluekeywordsentencetransformer_2nd_section",
    "tokenizedkluekeywordsentencetransformer_3rd_section",
    "tokenizedkluekeywordsentencetransformer_Intro",
    "tokenizedkluekeywordsentencetransformer_Body",
    "tokenizedkluekeywordsentencetransformer_Conclusion"]

for col in section_columns:
    raw[col] = raw[col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
    )
    

section_columns = [
    "tokenizedkluekeywordsentencetransformer_Intro",
    "tokenizedkluekeywordsentencetransformer_Body",
    "tokenizedkluekeywordsentencetransformer_Conclusion"
]

raw = create_section_presence_vector(raw, sorry, section_columns, label="sorrysentence1")
raw = create_section_presence_vector(raw, loveandgratitude, section_columns,label='loveandgratitudesentence1')
raw = create_section_presence_vector(raw, burdensome, section_columns,label='burdensomesentence1')
raw = create_section_presence_vector(raw, despair, section_columns,label='despairsentence1')
raw = create_section_presence_vector(raw, pmaffairs, section_columns,label='pmaffairssentence1')



section_columns = ["tokenizedkluekeywordsentencetransformer_1st_section", 
                   "tokenizedkluekeywordsentencetransformer_2nd_section", 
                   "tokenizedkluekeywordsentencetransformer_3rd_section"
]

raw = create_section_presence_vector(raw, sorry, section_columns, label="sorrysentence")
raw = create_section_presence_vector(raw, loveandgratitude, section_columns,label='loveandgratitudesentence')
raw = create_section_presence_vector(raw, burdensome, section_columns,label='burdensomesentence')
raw = create_section_presence_vector(raw, despair, section_columns,label='despairsentence')
raw = create_section_presence_vector(raw, pmaffairs, section_columns,label='pmaffairssentence')

raw = create_section_presence_vector(raw, sorrycomprehensive, section_columns,label='sorrysentencecomprehensive')
raw = create_section_presence_vector(raw, loveandgratitudecomprehensive, section_columns,label='loveandgratitudesentencecomprehensive')
raw = create_section_presence_vector(raw, burdensomecomprehensive, section_columns,label='burdensomesentencecomprehensive')
raw = create_section_presence_vector(raw, despaircomprehensive, section_columns,label='despairsentencecomprehensive')
raw = create_section_presence_vector(raw, pmaffairscomprehensive, section_columns,label='pmaffairssentencecomprehensive')

#%%
#morethan3=raw

morethan3=raw[raw["too_short"]==False]



from collections import Counter

# Get value counts
counts = morethan3["sorrysentence_presence"].value_counts()

for idx in range(3):  # 0 = Intro, 1 = Body, 2 = Conclusion
    # Total where section at idx has a 1
    section_1_total = sum(count for pattern, count in counts.items() if isinstance(pattern, list) and pattern[idx] == 1)

    # Total where any section has a 1
    any_1_total = sum(count for pattern, count in counts.items() if isinstance(pattern, list) and 1 in pattern)

    proportion = section_1_total / any_1_total if any_1_total > 0 else 0

    section_label = ["Intro", "Body", "Conclusion"][idx]
    print(f"Section: {section_label}")
    print(f"  Total with 1 in this section: {section_1_total}")
    print(f"  Total with any 1: {any_1_total}")
    print(f"  Proportion: {proportion:.4f}")
    print("-" * 40)

# Each contains a [0, 1, 0] for each section
presence_columns = {
    "Sorry and Shame": "sorrysentence_presence",
    "Love and Gratitude": "loveandgratitudesentence_presence",
    "Burden": "burdensomesentence_presence",
    "Despair": "despairsentence_presence",
    "Post-mortem Affairs": "pmaffairssentence_presence"
}



def get_section_vector(row, section_idx):
    return [
        row[presence_columns["Sorry and Shame"]][section_idx],
        row[presence_columns["Love and Gratitude"]][section_idx],
        row[presence_columns["Burden"]][section_idx],
        row[presence_columns["Despair"]][section_idx],
        row[presence_columns["Post-mortem Affairs"]][section_idx]
    ]



# Create one vector column per section
morethan3["section1_theme_vector"] = morethan3.apply(lambda row: get_section_vector(row, 0), axis=1)
morethan3["section2_theme_vector"] = morethan3.apply(lambda row: get_section_vector(row, 1), axis=1)
morethan3["section3_theme_vector"] = morethan3.apply(lambda row: get_section_vector(row, 2), axis=1)


def concatenate_theme_vectors(row):
    return (
        row["section1_theme_vector"] +
        row["section2_theme_vector"] +
        row["section3_theme_vector"]
    )

morethan3["full_theme_vector"] = morethan3.apply(concatenate_theme_vectors, axis=1)


theme_matrix = pd.DataFrame(
    morethan3["full_theme_vector"].tolist(),
    index=morethan3.index,
    columns=[f"Section {i+1} {theme}" for i in range(3) for theme in ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]]
)

existing_cols = [f"Section {i+1} {theme}" for i in range(3) for theme in ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]]
morethan3 = morethan3.drop(columns=[col for col in existing_cols if col in morethan3.columns])
morethan3 = pd.concat([morethan3, theme_matrix], axis=1)


# Assume your DataFrame is named `df` and the column is 'full_theme_vector'
# Step 1: Convert the list column into a new DataFrame
vector_matrix = pd.DataFrame(morethan3["full_theme_vector"].tolist())

# Step 2: Sum across rows (i.e., column-wise sum)
position_sums = vector_matrix.sum(axis=0)

# Step 3 (optional): Label the positions for clarity
position_sums.index = [f'Position_{i}' for i in position_sums.index]


malemorethan3 = morethan3[morethan3["SEX"] == 1]
femalemorethan3 = morethan3[morethan3["SEX"] == 2]
age1morethan3 = morethan3[morethan3["AGE2"] == 1]
age2morethan3 = morethan3[morethan3["AGE2"] == 2]
age3morethan3 = morethan3[morethan3["AGE2"] == 3]
age4morethan3 = morethan3[morethan3["AGE2"] == 4]
age5morethan3 = morethan3[morethan3["AGE2"] == 5]

df=morethan3[morethan3["SEX"].isin([1,2])]

# === Define theme presence column mapping ===
presence_columns = {
    "Sorry and Shame": "sorrysentence_presence",
    "Love and Gratitude": "loveandgratitudesentence_presence",
    "Burden": "burdensomesentence_presence",
    "Despair": "despairsentence_presence",
    "Post-mortem Affairs": "pmaffairssentence_presence"
}


# === Extract 5-theme vector for each section ===
def get_section_vector(row, section_idx):
    return [
        row[presence_columns["Sorry and Shame"]][section_idx],
        row[presence_columns["Love and Gratitude"]][section_idx],
        row[presence_columns["Burden"]][section_idx],
        row[presence_columns["Despair"]][section_idx],
        row[presence_columns["Post-mortem Affairs"]][section_idx]
    ]



# === Apply to create theme vectors for each section ===
df["section1_theme_vector"] = df.apply(lambda row: get_section_vector(row, 0), axis=1)
df["section2_theme_vector"] = df.apply(lambda row: get_section_vector(row, 1), axis=1)
df["section3_theme_vector"] = df.apply(lambda row: get_section_vector(row, 2), axis=1)

# === Combine into full theme vector with SEX appended ===
def concatenate_theme_vectors(row):
    return (
        row["section1_theme_vector"] +
        row["section2_theme_vector"] +
        row["section3_theme_vector"] +
        [row["SEX"]]
    )

df["full_theme_vector"] = df.apply(concatenate_theme_vectors, axis=1)

# === Build column names for the matrix (15 themes + 1 gender) ===
theme_feature_names = [f"Section {i+1} {theme}" for i in range(3) for theme in ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]]
theme_feature_names.append("SEX")

# === Create theme matrix DataFrame ===
theme_matrix = pd.DataFrame(
    df["full_theme_vector"].tolist(),
    index=df.index,
    columns=theme_feature_names
)

# === Drop any old theme columns from original dataframe if present ===
existing_cols = [col for col in theme_feature_names if col in df.columns]
df = df.drop(columns=existing_cols)

# === Merge matrix back to original dataframe ===
df = pd.concat([df, theme_matrix], axis=1)

# === Optional: Sum vector components to inspect distributions ===
vector_matrix = pd.DataFrame(df["full_theme_vector"].tolist())
position_sums = vector_matrix.sum(axis=0)
position_sums.index = [f'Position_{i}' for i in position_sums.index]

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 0: Filter AGE2 to keep 5 valid groups ===
filtered_df = df[df["AGE2"].isin([1, 2, 3, 4, 5])].copy()

# === Step 1: Define theme matrix ===
theme_labels = [f"Section {i+1} {theme}" for i in range(3) for theme in ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]]
theme_labels.append("Gender")  # Add this line to include gender as the final column
theme_matrix = pd.DataFrame(
    filtered_df["full_theme_vector"].apply(lambda v: list(map(int, v))).tolist(),
    columns=theme_labels,
    index=filtered_df.index
)

X = theme_matrix.copy()
X = sm.add_constant(X)  # Add intercept

group_results = {}

# === Step 2: Run One-vs-Rest Logistic Regression for each age group ===
for group in [1, 2, 3, 4, 5]:
    y = (filtered_df["AGE2"] == group).astype(int)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    odds_ratios = np.exp(result.params)
    conf = np.exp(result.conf_int())
    conf.columns = ['CI Lower', 'CI Upper']

    summary_df = pd.DataFrame({
        "term": result.params.index,
        "Coefficient": result.params,
        "Odds Ratio": odds_ratios,
        "p-value": result.pvalues,
        "CI Lower": conf["CI Lower"],
        "CI Upper": conf["CI Upper"],
        "Age Group": group
    })

    group_results[group] = summary_df.reset_index().rename(columns={'index': 'Feature'})

# === Step 3: Combine all group results ===
combined_df = pd.concat(group_results.values(), ignore_index=True)
combined_df = combined_df[combined_df["Feature"] != "const"].copy()

# === Step 4: Extract section and theme ===
combined_df["section"] = combined_df["Feature"].apply(lambda x: x.split("_")[0])
combined_df["theme"] = combined_df["Feature"].apply(lambda x: "_".join(x.split("_")[1:]))
combined_df["log_odds_ratio"] = np.log(combined_df["Odds Ratio"])



# === Step 5: Significance stars ===
def get_sig_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

combined_df["sig_star"] = combined_df["p-value"].apply(get_sig_star)
combined_df["annot"] = combined_df["log_odds_ratio"].round(2).astype(str) + combined_df["sig_star"]

# === Step 6: Plot heatmaps for each age group ===
for group in sorted(combined_df["Age Group"].unique()):
    sub_df = combined_df[combined_df["Age Group"] == group]

    heatmap_data = sub_df.pivot(index="theme", columns="section", values="log_odds_ratio")
    annot_data = sub_df.pivot(index="theme", columns="section", values="annot")

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        heatmap_data,
        annot=annot_data,
        fmt="",
        center=0,
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={'label': 'Log-Odds'}
    )
    plt.title(f"Log(Odds Ratio) Heatmap by Section and Theme (Age Group {group})")
    plt.ylabel("Theme")
    plt.xlabel("Section")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# === Step 7: Create final heatmap comparing all age groups ===

# Redefine significance star labeling for coefficient values
def add_stars(coef, p):
    if p < 0.001:
        return f"{coef:.2f}***"
    elif p < 0.01:
        return f"{coef:.2f}**"
    elif p < 0.05:
        return f"{coef:.2f}*"
    else:
        return f"{coef:.2f}"

combined_df["annot"] = combined_df.apply(lambda row: add_stars(row["Coefficient"], row["p-value"]), axis=1)

# Pivot for plotting
coef_matrix = combined_df.pivot(index="Feature", columns="Age Group", values="Coefficient")
annot_matrix = combined_df.pivot(index="Feature", columns="Age Group", values="annot")

# Reorder rows by theme-section convention
theme_order = []
for theme in ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]:
    for section in ["Section 1", "Section 2", "Section 3"]:
        theme_order.append(f"{section} {theme}")

coef_matrix = coef_matrix.reindex(theme_order)
annot_matrix = annot_matrix.reindex(theme_order)

# Final plot
plt.figure(figsize=(14, 8))
sns.heatmap(
    coef_matrix,
    cmap="coolwarm",
    center=0,
    annot=annot_matrix,
    fmt="",
    cbar_kws={"label": "Log-Odds"}
)
plt.title("One-vs-Rest Logistic Regression Coefficients by Age Group (with Significance)")
plt.ylabel("Thematic Feature")
plt.xlabel("Age Group")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# === Line Plot: Theme-Section Log-Odds Across Age Groups ===
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# === Optional: Use custom font ===
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
font_path = r'C:\Windows\Fonts\Arialbd.ttf'

custom_font = fm.FontProperties(fname=font_path).get_name()
font_prop = fm.FontProperties(fname=font_path)

# === Set plot style ===
sns.set(style="white", context="notebook", font_scale=1.2)
plt.rcParams.update({
    "font.family": custom_font,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.spines.right": False,
    "axes.spines.top": False
})

# === Filter to theme-only features (exclude gender if still present) ===
line_df = combined_df[combined_df["Feature"] != "Gender"].copy()

# Correctly extract section and theme
line_df["section"] = line_df["Feature"].apply(lambda x: " ".join(x.split()[:2]))     # e.g., 'Section 1'
line_df["theme"] = line_df["Feature"].apply(lambda x: " ".join(x.split()[2:]))       # e.g., 'Sorry and Shame'

# === Get theme, section, and age group settings ===
themes = ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]
sections = ["Section 1", "Section 2", "Section 3"]
age_groups = sorted(line_df["Age Group"].unique())

# === Set up subplots ===
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8), sharey=True)
fig.subplots_adjust(right=0.82)

# === Define significance star function (already applied but recheck) ===
def get_sig_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# === Plot each theme's line plot ===
for idx, theme in enumerate(themes):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    theme_data = line_df[line_df["theme"] == theme]

    for section in sections:
        line_data = theme_data[theme_data["section"] == section].sort_values("Age Group")

        ax.plot(line_data["Age Group"], line_data["log_odds_ratio"],
                marker="o", label=section, linewidth=2)

        for _, row_ in line_data.iterrows():
            x = row_["Age Group"]
            y = row_["log_odds_ratio"]
            star = get_sig_star(row_["p-value"])
            if star:
                ax.text(x, y, star, ha="center", va="bottom", fontsize=11, weight='bold')

    ax.set_title(f"{theme}", fontsize=13, fontproperties=font_prop)
    tick_labels = ["≤18", "19–34", "35–49", "50–64", "65+"]
    tick_positions = [1, 2, 3, 4, 5]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10, fontproperties=font_prop)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    
    ax.set_xlabel("Age Group", fontsize=11, fontproperties=font_prop)
    ax.tick_params(bottom=True, left=True) 

    if col == 0:
        ax.set_ylabel("Log-Odds", fontsize=11, fontproperties=font_prop)
    # Set font for y-axis tick labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(12)

    
    # Set font for x-axis tick labels (optional, but already done earlier with set_xticklabels)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(12)

# === Remove extra subplot if present ===
if len(themes) < n_rows * n_cols:
    fig.delaxes(axes[n_rows-1, n_cols-1])

# === Shared legend ===
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Section",
    loc='center left',
    bbox_to_anchor=(0.87, 0.5),
    frameon=False,
    fontsize=11,
    title_fontsize=12    
)

plt.suptitle("Theme-Section Effects on Age Group Membership\n(Controlling for Gender)", fontsize=14, weight='bold', y=0.95, fontproperties=font_prop)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])
plt.show()

# === Restore default rcParams ===
plt.rcdefaults()

#%%

# ============================================================
# FIGURE: x = Section 1/2/3, y = Odds Ratio, hue = Theme
# Each panel = one Age Group
# Confidence intervals match line colors
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ------------------------------------------------------------
# 1. Prepare plotting dataframe
# ------------------------------------------------------------

plot_df = combined_df.copy()

# Remove gender control variable from the age-group plot
plot_df = plot_df[plot_df["Feature"] != "Gender"].copy()

# Extract section and theme from feature names like:
# "Section 1 Sorry and Shame"
plot_df["section"] = plot_df["Feature"].str.extract(r"(Section \d)")
plot_df["theme"] = plot_df["Feature"].str.replace(
    r"^Section \d\s+",
    "",
    regex=True
)

# Define plotting order
section_order = ["Section 1", "Section 2", "Section 3"]

theme_order = [
#    "Sorry and Shame",
    "Love and Gratitude",
#    "Burden",
#    "Despair",
#    "Post-mortem Affairs"
]

age_label_map = {
    1: "≤18",
    2: "19–34",
    3: "35–49",
    4: "50–64",
    5: "65+"
}

age_order = ["≤18", "19–34", "35–49", "50–64", "65+"]

# Keep only valid rows
plot_df = plot_df[plot_df["section"].isin(section_order)].copy()
plot_df = plot_df[plot_df["theme"].isin(theme_order)].copy()

# Apply categorical ordering
plot_df["section"] = pd.Categorical(
    plot_df["section"],
    categories=section_order,
    ordered=True
)

plot_df["theme"] = pd.Categorical(
    plot_df["theme"],
    categories=theme_order,
    ordered=True
)

plot_df["Age Label"] = plot_df["Age Group"].map(age_label_map)

# Numeric x positions for sections
section_x_map = {
    "Section 1": 1,
    "Section 2": 2,
    "Section 3": 3
}

plot_df["section_x"] = plot_df["section"].map(section_x_map).astype(float)

# ------------------------------------------------------------
# 2. Significance stars
# ------------------------------------------------------------

def get_sig_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

plot_df["sig_star"] = plot_df["p-value"].apply(get_sig_star)

# ------------------------------------------------------------
# 3. Optional font
# ------------------------------------------------------------

try:
    font_path = r"C:\Windows\Fonts\Arialbd.ttf"
    custom_font = fm.FontProperties(fname=font_path).get_name()
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams.update({"font.family": custom_font})
except Exception:
    font_prop = fm.FontProperties()

# ------------------------------------------------------------
# 4. Style
# ------------------------------------------------------------

sns.set(style="white", context="notebook", font_scale=1.15)

plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.spines.right": False,
    "axes.spines.top": False
})

# ------------------------------------------------------------
# 5. Create figure
# ------------------------------------------------------------

n_rows = 2
n_cols = 3

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(17, 9),
    sharey=True
)

axes = axes.flatten()

# ------------------------------------------------------------
# 6. Plot one panel per age group
# ------------------------------------------------------------

for idx, age_label in enumerate(age_order):
    ax = axes[idx]

    age_df = plot_df[plot_df["Age Label"] == age_label].copy()

    for theme in theme_order:
        theme_df = age_df[age_df["theme"] == theme].sort_values("section_x")

        if theme_df.empty:
            continue

        # Main line
        line = ax.plot(
            theme_df["section_x"],
            theme_df["Odds Ratio"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=theme
        )

        # Get the same color as the line
        line_color = line[0].get_color()

        # Confidence intervals with matching color
        ax.errorbar(
            theme_df["section_x"],
            theme_df["Odds Ratio"],
            yerr=[
                theme_df["Odds Ratio"] - theme_df["CI Lower"],
                theme_df["CI Upper"] - theme_df["Odds Ratio"]
            ],
            fmt="none",
            linewidth=1.15,
            capsize=3,
            alpha=0.70,
            color=line_color
        )

        # Significance stars with matching color
        for _, row in theme_df.iterrows():
            if row["sig_star"] != "":
                ax.text(
                    row["section_x"],
                    row["CI Upper"] * 1.06,
                    row["sig_star"],
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    weight="bold",
                    color=line_color
                )

    # Reference line at OR = 1
    ax.axhline(
        1,
        linestyle="--",
        color="gray",
        linewidth=1
    )

    # Log scale is recommended for odds ratios
    ax.set_yscale("log")

    ax.set_title(
        f"Age Group: {age_label}",
        fontsize=13,
        fontproperties=font_prop
    )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(
        ["Section 1", "Section 2", "Section 3"],
        fontsize=11,
        fontproperties=font_prop
    )

    ax.set_xlabel(
        "Section",
        fontsize=12,
        fontproperties=font_prop
    )

    if idx % n_cols == 0:
        ax.set_ylabel(
            "Odds Ratio",
            fontsize=12,
            fontproperties=font_prop
        )

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(10)

# Remove unused final subplot
if len(age_order) < len(axes):
    fig.delaxes(axes[-1])

# ------------------------------------------------------------
# 7. Shared legend
# ------------------------------------------------------------

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    title="Theme",
    loc="center left",
    bbox_to_anchor=(0.91, 0.5),
    frameon=False,
    fontsize=11,
    title_fontsize=12
)

# ------------------------------------------------------------
# 8. Final title and layout
# ------------------------------------------------------------

fig.suptitle(
    "Theme-Specific Odds Ratios Across Note Sections by Age Group",
    fontsize=16,
    weight="bold",
    y=0.98,
    fontproperties=font_prop
)

plt.tight_layout(rect=[0, 0, 0.89, 0.94])
plt.show()

# Optional: restore matplotlib defaults
plt.rcdefaults()

#%%
# === Line Plot v2: x = Section 1/2/3, lines = Age Groups, one panel per Theme ===
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ---- 1) Prep tidy data (exclude the "Gender" row) ----
line_df = combined_df[combined_df["Feature"] != "Gender"].copy()

# Robust extraction of section/theme from "Feature" like "Section 1 Sorry and Shame"
line_df["section"] = line_df["Feature"].str.extract(r'(Section \d)')
line_df["theme"]   = line_df["Feature"].str.replace(r'^Section \d\s+', '', regex=True)

# Keep only valid sections and order them 1→3
section_order = ["Section 1", "Section 2", "Section 3"]
line_df = line_df[line_df["section"].isin(section_order)].copy()
line_df["section"] = pd.Categorical(line_df["section"], categories=section_order, ordered=True)

themes = ["Sorry and Shame", "Love and Gratitude", "Burden", "Despair", "Post-mortem Affairs"]
age_groups = sorted(line_df["Age Group"].unique())
age_labels_map = {1: "≤18", 2: "19–34", 3: "35–49", 4: "50–64", 5: "65+"}

# ---- 2) Optional font handling (graceful fallback) ----
try:
    font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
    custom_font = fm.FontProperties(fname=font_path).get_name()
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams.update({"font.family": custom_font})
except Exception:
    font_prop = fm.FontProperties()  # default font

# ---- 3) Style ----
sns.set(style="white", context="notebook", font_scale=1.2)
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.spines.right": False,
    "axes.spines.top": False
})

# ---- 4) Helper: significance stars ----
def star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# ---- 5) Plot (5 themes in a 2x3 grid, last panel blank) ----
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
axes = axes.flatten()

for idx, theme in enumerate(themes):
    ax = axes[idx]
    tdf = line_df[line_df["theme"] == theme].copy()

    # Draw one line per age group
    for ag in age_groups:
        sub = (tdf[tdf["Age Group"] == ag]
               .sort_values("section"))
        # x as numeric positions 1..3 for clean plotting, with labels set below
        x = np.arange(1, len(section_order) + 1)
        y = sub["log_odds_ratio"].to_numpy()

        ax.plot(x, y, marker="o", linewidth=2, label=age_labels_map.get(ag, str(ag)))

        # Add significance stars at each point
        for i, (_, row_) in enumerate(sub.iterrows(), start=1):
            s = star(row_["p-value"])
            if s:
                ax.text(i, row_["log_odds_ratio"], s, ha="center", va="bottom", fontsize=11, weight="bold")

    # Axes cosmetics
    ax.set_title(theme, fontsize=13, fontproperties=font_prop)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(section_order, rotation=0, fontproperties=font_prop, fontsize=10)
    if idx % n_cols == 0:
        ax.set_ylabel("Log-Odds", fontsize=11, fontproperties=font_prop)
    ax.set_xlabel("Section", fontsize=11, fontproperties=font_prop)

# Remove unused last axis (since 5 themes)
if len(themes) < len(axes):
    fig.delaxes(axes[-1])

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
# =============================================================================
# fig.legend(handles, labels, title="Age Group", loc='right', bbox_to_anchor=(0.98, 0.5),
#            frameon=False, fontsize=11, title_fontsize=12)
# =============================================================================

plt.suptitle("Theme-wise Effects Across Sections\n(lines = Age Groups, x = Sections; controlling for Gender)",
             fontsize=14, weight='bold', y=0.98, fontproperties=font_prop)
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.show()

# === (Optional) Restore defaults ===
plt.rcdefaults()


#%% SENTIMENT

import pandas as pd


morethan3rawsentiment= pd.read_pickle(r'C:\Users\Jae Bin Park\morethan3rawsentiment.pkl')


from tqdm import tqdm

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
    "절망",          # Despair
    "패배/자기혐오",  # Defeat / Self-loathing
    "힘듦/지침",      # Exhaustion / Fatigue
    "불안/걱정",      # Anxiety / Worry

    # Shame / Guilt / Self-directed Negativity
    "죄책감",        # Guilt
    "부끄러움",      # Shame
    "한심함",        # Worthlessness
    "불쌍함/연민",
    "안타까움/실망",

    # Anger / Resentment / Rejection
    "화남/분노",      # Anger / Rage
    "증오/혐오",      # Hatred / Disgust
    "어이없음",      # Resentment / Disbelief

    # Sadness / Isolation / Cry for Help
    "슬픔",          # Sadness
    "서러움",        # Deep sadness / Feeling wronged

    # Mixed / Ambivalent States (sometimes present in notes)
    "고마움",        # Gratitude
    "흐뭇함(귀여움/예쁨)",  # Pleased / Cute-Affection
    "행복",          # Happiness
    "안심/신뢰",       # Relief / Trust
    "없음"
]

for label in sentiment_targets:
    morethan3rawsentiment = create_sentiment_presence_vector(
        morethan3rawsentiment,
        sentiment_labels=[label],  # just one sentiment per run
        section_cols=section_cols,
        label_prefix=label  # will name the column "슬픔_presence", etc.
    )

# Step 0: Filter to valid genders and copy
df = morethan3rawsentiment[morethan3rawsentiment["SEX"].isin([1, 2])].copy()

# === Step 1: Define sentiment presence column mapping ===
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

# === Step 2: Define vector extraction function ===
def get_sentimentsection_vector(row, section_idx):
    return [
        row[sentiment_presence_columns[sentiment]][section_idx]
        for sentiment in sentiment_presence_columns
    ]

# === Step 3: Apply to create section vectors ===
df["section1_sentiment_vector"] = df.apply(lambda row: get_sentimentsection_vector(row, 0), axis=1)
df["section2_sentiment_vector"] = df.apply(lambda row: get_sentimentsection_vector(row, 1), axis=1)
df["section3_sentiment_vector"] = df.apply(lambda row: get_sentimentsection_vector(row, 2), axis=1)

# === Step 4: Combine section vectors with gender ===
def concatenate_sentiment_vectors(row):
    return (
        row["section1_sentiment_vector"] +
        row["section2_sentiment_vector"] +
        row["section3_sentiment_vector"] +
        [row["SEX"]]
    )

df["full_sentiment_vector"] = df.apply(concatenate_sentiment_vectors, axis=1)

# === Step 5: Build column names ===
sentiments = list(sentiment_presence_columns.keys())
sentiment_feature_names = [f"Section {i+1} {sentiment}" for i in range(3) for sentiment in sentiments]
sentiment_feature_names.append("SEX")

# === Step 6: Create sentiment matrix DataFrame ===
sentiment_matrix = pd.DataFrame(
    df["full_sentiment_vector"].tolist(),
    index=df.index,
    columns=sentiment_feature_names
)

# === Step 7: Drop old sentiment features if they exist ===
existing_cols = [col for col in sentiment_feature_names if col in df.columns]
df = df.drop(columns=existing_cols)

# === Step 8: Append matrix back to df ===
df = pd.concat([df, sentiment_matrix], axis=1)

# === Step 9: Optional sanity check (sum) ===
vector_matrix = pd.DataFrame(df["full_sentiment_vector"].tolist())
sentimentposition_sums = vector_matrix.sum(axis=0)
sentimentposition_sums.index = [f'Position_{i}' for i in sentimentposition_sums.index]



import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 0: Filter for valid AGE2 groups ===
filtered_df = df[df["AGE2"].isin([1, 2, 3, 4, 5])].copy()

# === Step 1: Define sentiment feature labels + gender ===
sentiments = [
    "Despair", "Defeat", "Exhaustion", "Anxious", "Disappointment",
    "Anger", "Hatred", "Resentment", "Sadness", "Sorrow",
    "Gratitude", "Affection", "Happiness", "Relief", "Neutral"
]
sentiment_labels = [f"Section {i+1} {sentiment}" for i in range(3) for sentiment in sentiments]
sentiment_labels.append("Gender")  # Include gender if needed as last element

# === Step 2: Extract matrix from full_sentiment_vector column ===
sentiment_matrix = pd.DataFrame(
    filtered_df["full_sentiment_vector"].apply(lambda v: list(map(int, v))).tolist(),
    columns=sentiment_labels,
    index=filtered_df.index
)

X = sentiment_matrix.copy()
X = sm.add_constant(X)

group_results = {}

# === Step 3: Run logistic regression for each AGE2 group ===
for group in [1, 2, 3, 4, 5]:
    y = (filtered_df["AGE2"] == group).astype(int)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    odds_ratios = np.exp(result.params)
    conf = np.exp(result.conf_int())
    conf.columns = ['CI Lower', 'CI Upper']

    summary_df = pd.DataFrame({
        "Coefficient": result.params,
        "Odds Ratio": odds_ratios,
        "p-value": result.pvalues,
        "CI Lower": conf["CI Lower"],
        "CI Upper": conf["CI Upper"],
        "Age Group": group
    })

    group_results[group] = summary_df.reset_index().rename(columns={'index': 'Feature'})

# === Step 4: Combine results across age groups ===
combined_df = pd.concat(group_results.values(), ignore_index=True)
combined_df = combined_df[combined_df["Feature"] != "const"].copy()

# === Step 5: Extract section and sentiment ===
combined_df["section"] = combined_df["Feature"].apply(lambda x: " ".join(x.split()[:2]))
combined_df["sentiment"] = combined_df["Feature"].apply(lambda x: " ".join(x.split()[2:]))
combined_df["log_odds_ratio"] = np.log(combined_df["Odds Ratio"])

# === Step 6: Significance stars ===
def get_sig_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

combined_df["sig_star"] = combined_df["p-value"].apply(get_sig_star)
combined_df["annot"] = combined_df["log_odds_ratio"].round(2).astype(str) + combined_df["sig_star"]

# === Step 7: Heatmaps by Age Group ===
for group in sorted(combined_df["Age Group"].unique()):
    sub_df = combined_df[(combined_df["Age Group"] == group) & (combined_df["section"] != "other")]

    heatmap_data = sub_df.pivot(index="sentiment", columns="section", values="log_odds_ratio")
    annot_data = sub_df.pivot(index="sentiment", columns="section", values="annot")

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        annot=annot_data,
        fmt="",
        center=0,
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={'label': 'Log-Odds'}
    )
    plt.title(f"Log(Odds Ratio) Heatmap by Section and Sentiment (Age Group {group})")
    plt.ylabel("Sentiment")
    plt.xlabel("Section")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# === Step 8: Final Combined Heatmap ===
def add_stars(coef, p):
    if p < 0.001:
        return f"{coef:.2f}***"
    elif p < 0.01:
        return f"{coef:.2f}**"
    elif p < 0.05:
        return f"{coef:.2f}*"
    else:
        return f"{coef:.2f}"

combined_df["annot"] = combined_df.apply(lambda row: add_stars(row["Coefficient"], row["p-value"]), axis=1)

coef_matrix = combined_df.pivot(index="Feature", columns="Age Group", values="Coefficient")
annot_matrix = combined_df.pivot(index="Feature", columns="Age Group", values="annot")

# Reorder: Section1–3 × Sentiments
sentiment_order = [f"Section {i} {s}" for s in sentiments for i in range(1, 4)]
coef_matrix = coef_matrix.reindex(sentiment_order)
annot_matrix = annot_matrix.reindex(sentiment_order)

# === Final Plot ===
plt.figure(figsize=(16, 10))
sns.heatmap(
    coef_matrix,
    cmap="coolwarm",
    center=0,
    annot=annot_matrix,
    fmt="",
    cbar_kws={"label": "Log-Odds"}
)
plt.title("Logistic Regression Coefficients by Age Group (Sentiment Features)")
plt.ylabel("Sentiment Feature")
plt.xlabel("Age Group")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# === Line Plot: Sentiment-Section Log-Odds Across Age Groups ===
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# === Optional: Use custom font ===
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
custom_font = fm.FontProperties(fname=font_path).get_name()
font_prop = fm.FontProperties(fname=font_path)

# === Set plot style ===
sns.set(style="white", context="notebook", font_scale=1.2)
plt.rcParams.update({
    "font.family": custom_font,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.spines.right": False,
    "axes.spines.top": False
})

# === Filter to sentiment-only rows (exclude gender if present) ===
line_df = combined_df[~combined_df["sentiment"].isin(["gender"])].copy()

# === Define sentiments, sections, and age group settings ===
sentiments = ["Defeat", "Exhaustion", "Neutral", "Sadness", "Happiness", "Disappointment"]
sections = ["Section 1", "Section 2", "Section 3"]
age_groups = sorted(line_df["Age Group"].unique())

# === Subplot layout ===
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8), sharey=True)
fig.subplots_adjust(right=0.85)

# === Significance stars ===
def get_sig_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# === Plot each sentiment ===
for idx, sentiment in enumerate(sentiments):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    sentiment_data = line_df[line_df["sentiment"] == sentiment]

    for section in sections:
        line_data = sentiment_data[sentiment_data["section"] == section].sort_values("Age Group")

        ax.plot(line_data["Age Group"], line_data["log_odds_ratio"],
                marker="o", label=section, linewidth=2)

        for _, row_ in line_data.iterrows():
            x = row_["Age Group"]
            y = row_["log_odds_ratio"]
            star = get_sig_star(row_["p-value"])
            if star:
                ax.text(x, y, star, ha="center", va="bottom", fontsize=11, weight='bold')

    ax.set_title(sentiment.capitalize(), fontsize=13, fontproperties=font_prop)
    tick_labels = ["≤18", "19–34", "35–49", "50–64", "65+"]
    tick_positions = [1, 2, 3, 4, 5]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10, fontproperties=font_prop)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Age Group", fontsize=12, fontproperties=font_prop)
    ax.tick_params(bottom=True, left=True) 

    if col == 0:
        ax.set_ylabel("Log-Odds", fontsize=12, fontproperties=font_prop)
    # Set font for y-axis tick labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(12)

    
    # Set font for x-axis tick labels (optional, but already done earlier with set_xticklabels)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(12)
# === Remove extra subplot if present ===
if len(sentiments) < n_rows * n_cols:
    for i in range(len(sentiments), n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

# === Shared legend ===
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Section",
    loc='center left',
    bbox_to_anchor=(0.87, 0.5),
    frameon=False,
    fontsize=11,
    title_fontsize=12    
)

plt.suptitle("Sentiment-Section Effects on Age Group Membership\n(Controlling for Gender)", fontsize=14, weight='bold', y=0.94, fontproperties=font_prop)
plt.tight_layout(rect=[0, 0, 0.85, 0.94])
plt.show()

# === Restore default rcParams ===
plt.rcdefaults()
#%%

# === Line Plot v2: Sentiment-wise panels; x = Sections, lines = Age Groups ===
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# === Optional: Use custom font ===
font_path = r'C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf'
try:
    custom_font = fm.FontProperties(fname=font_path).get_name()
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams.update({"font.family": custom_font})
except Exception:
    font_prop = fm.FontProperties()

# === Set plot style ===
sns.set(style="white", context="notebook", font_scale=1.2)
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.spines.right": False,
    "axes.spines.top": False
})

# === Filter to sentiment-only rows (exclude gender if present) ===
line_df = combined_df[~combined_df["sentiment"].isin(["gender"])].copy()

# Ensure clean section ordering
section_order = ["Section 1", "Section 2", "Section 3"]
line_df = line_df[line_df["section"].isin(section_order)].copy()
line_df["section"] = pd.Categorical(line_df["section"], categories=section_order, ordered=True)

# === Define sentiments, sections, and age group settings ===
sentiments = ["Defeat", "Exhaustion", "Neutral", "Sadness", "Happiness", "Disappointment"]
sections = section_order
age_groups = sorted(line_df["Age Group"].unique())
age_labels = {1: "≤18", 2: "19–34", 3: "35–49", 4: "50–64", 5: "65+"}

# === Subplot layout ===
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))
axes = axes.flatten()
fig.subplots_adjust(right=0.85)

# === Significance stars ===
def get_sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# === Plot each sentiment (x = sections; lines = age groups) ===
for idx, sentiment in enumerate(sentiments):
    ax = axes[idx]
    sdata = line_df[line_df["sentiment"] == sentiment].copy()

    # One line per age group across sections 1→3
    for ag in age_groups:
        sub = (sdata[sdata["Age Group"] == ag]
               .sort_values("section"))
        # numeric x for plotting; labels set below
        x = np.arange(1, len(sections) + 1)
        y = sub["log_odds_ratio"].to_numpy()

        ax.plot(x, y, marker="o", linewidth=2, label=age_labels.get(ag, str(ag)))

        # annotate significance stars at each point
        for i, (_, row_) in enumerate(sub.iterrows(), start=1):
            star = get_sig_star(row_["p-value"])
            if star:
                ax.text(i, row_["log_odds_ratio"], star, ha="center", va="bottom",
                        fontsize=11, weight='bold')

    # cosmetics
    ax.set_title(sentiment.capitalize(), fontsize=13, fontproperties=font_prop)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(sections, fontsize=10, fontproperties=font_prop)
    ax.set_xlabel("Section", fontsize=11, fontproperties=font_prop)
    if idx % n_cols == 0:
        ax.set_ylabel("Log-Odds", fontsize=11, fontproperties=font_prop)
    # y-tick font
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(font_prop)
        lab.set_fontsize(10)

# === Remove extra subplot if present ===
if len(sentiments) < n_rows * n_cols:
    for i in range(len(sentiments), n_rows * n_cols):
        fig.delaxes(axes[i])

# === Shared legend (Age Groups) ===
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Age Group",
    loc='center left',
    bbox_to_anchor=(0.87, 0.5),
    frameon=False,
    fontsize=11,
    title_fontsize=12
)

plt.suptitle("Sentiment-wise Effects Across Sections\n(lines = Age Groups; controlling for Gender)",
             fontsize=14, weight='bold', y=0.94, fontproperties=font_prop)
plt.tight_layout(rect=[0, 0, 0.85, 0.94])
plt.show()

# === Restore default rcParams (optional) ===
plt.rcdefaults()
