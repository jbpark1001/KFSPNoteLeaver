from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project import ROOT, input_path, output_path, load_frame
import ast
import os
import re
import warnings

from inference_checks import require_valid_logit
import numpy as np
import pandas as pd

from tqdm import tqdm

import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

warnings.filterwarnings("ignore")
tqdm.pandas()


# ============================================================
# 1. OPTIONS
# ============================================================

SAVE_OUTPUTS = True
OUTPUT_DIR = str(output_path("tables/sentence_robustness/.keep").parent)

if SAVE_OUTPUTS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. FIGURE FORMATTING
# ============================================================

FONT_CANDIDATES = [
    r"C:\Users\<user>\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf",
    r"C:\Windows\Fonts\Arialbd.ttf",
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
]

font_prop = None
custom_font = None

for fp in FONT_CANDIDATES:
    try:
        if os.path.exists(fp):
            font_prop = fm.FontProperties(fname=fp)
            custom_font = font_prop.get_name()
            break
    except Exception:
        continue

if custom_font is not None:
    plt.rcParams["font.family"] = custom_font


FIG_STYLE = {
    "cmap": "coolwarm",
    "center": 0,
    "line_width": 2,
    "marker": "o",
    "zero_line_style": "--",
    "zero_line_color": "gray",
    "zero_line_width": 1,
    "heatmap_linewidths": 0.5,
    "heatmap_linecolor": "gray",
    "title_fontsize": 14,
    "axis_label_fontsize": 12,
    "tick_fontsize": 11,
    "annot_fontsize": 10,
    "star_fontsize": 11,
    "legend_fontsize": 11,
    "legend_title_fontsize": 12,
}

AGE_ORDER = [1, 2, 3, 4, 5]

AGE_LABELS = {
    1: "≤18",
    2: "19–34",
    3: "35–49",
    4: "50–64",
    5: "65+",
}


def set_constant_plot_style():
    sns.set(style="white", context="notebook", font_scale=1.2)

    plt.rcParams.update({
        "axes.linewidth": 1.2,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 100,
    })

    if custom_font is not None:
        plt.rcParams["font.family"] = custom_font


def get_sig_star(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def safe_log_or(x):
    x = pd.to_numeric(x, errors="coerce")
    x = np.clip(x, np.finfo(float).tiny, np.inf)
    return np.log(x)


# ============================================================
# 3. PARSING HELPERS
# ============================================================

def parse_listlike(x):
    if isinstance(x, list):
        return x

    if pd.isna(x):
        return []

    if isinstance(x, str):
        x = x.strip()

        if x == "":
            return []

        if x.startswith("[") and x.endswith("]"):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return []

    return []


def parse_dictlike(x):
    if isinstance(x, dict):
        if "labels" not in x:
            x["labels"] = []
        if "scores" not in x:
            x["scores"] = []
        return x

    if pd.isna(x):
        return {"labels": [], "scores": []}

    if isinstance(x, str):
        x = x.strip()

        if x == "":
            return {"labels": [], "scores": []}

        if x.startswith("{") and x.endswith("}"):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, dict):
                    if "labels" not in parsed:
                        parsed["labels"] = []
                    if "scores" not in parsed:
                        parsed["scores"] = []
                    return parsed
            except Exception:
                return {"labels": [], "scores": []}

    return {"labels": [], "scores": []}


def extract_tokens_from_keyword_list(keyword_list):
    tokens = []

    keyword_list = parse_listlike(keyword_list)

    for item in keyword_list:
        if isinstance(item, tuple) and len(item) >= 1:
            tokens.append(str(item[0]))
        elif isinstance(item, list) and len(item) >= 1:
            tokens.append(str(item[0]))
        elif isinstance(item, str):
            tokens.append(item)

    return tokens


# ============================================================
# 4. THEME AND SENTIMENT DICTIONARIES
# ============================================================

THEME_TOKEN_DICT = {
    "Sorry and Shame": [
        "미안", "죄송", "용서", "잘못", "후회", "죄", "책임", "민폐", "반성"
    ],
    "Love and Gratitude": [
        "사랑", "고맙", "감사", "고생", "수고", "보고싶", "애틋", "소중", "그리"
    ],
    "Burden": [
        "버겁", "부담", "짐", "감당", "힘들", "무겁", "지치", "헷갈"
    ],
    "Despair": [
        "포기", "좌절", "죽", "끝", "아무것", "헛되", "무의미", "잊히", "없어지", "그만두"
    ],
    "Post-mortem Affairs": [
        "부탁", "정리", "남기", "처리", "보험", "은행", "장례", "통장", "유서"
    ],
}

THEMES = list(THEME_TOKEN_DICT.keys())


sentiment_label_map = {
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

SENTIMENTS = list(sentiment_label_map.keys())

LINEPLOT_SENTIMENTS = [
    "Defeat",
    "Exhaustion",
    "Neutral",
    "Sadness",
    "Happiness",
    "Disappointment",
]

sentence_df=load_frame(input_path("sentence_annotations"))
scored=load_frame(input_path("section_annotations"))

# ============================================================
# 5. PREPARE NOTE-LEVEL DEMOGRAPHIC DATA
# ============================================================

# You can use scored if it exists, otherwise kfsppororo.
# Important: note_id in sentence_df should correspond to the row index
# from the note-level dataframe.

if "scored" in globals():
    note_df = scored.copy()
elif "kfsppororo" in globals():
    note_df = kfsppororo.copy()
else:
    raise NameError("Need either `scored` or `kfsppororo` in memory.")

if "sentence_df" not in globals():
    raise NameError("`sentence_df` is not in memory.")

if "note_id" not in sentence_df.columns:
    raise KeyError("sentence_df must contain `note_id`.")

if "SEX" not in note_df.columns or "AGE2" not in note_df.columns:
    raise KeyError("note-level dataframe must contain SEX and AGE2.")

# Keep only raw notes if pred_label exists.
if "pred_label" in note_df.columns:
    note_df = note_df[note_df["pred_label"].eq("raw")].copy()

note_demo = note_df[["SEX", "AGE2"]].copy()
note_demo = note_demo.reset_index().rename(columns={"index": "note_id"})

if not note_demo["note_id"].is_unique:
    raise ValueError("Duplicate note IDs: verify source row linkage before sentence analysis.")
valid_note_ids = set(note_demo["note_id"])

sentence_work = sentence_df[sentence_df["note_id"].isin(valid_note_ids)].copy()

print("Sentence rows kept:", sentence_work.shape[0])
print("Unique notes kept:", sentence_work["note_id"].nunique())


# ============================================================
# 6. CREATE SHUFFLED SENTENCE POSITIONS
# ============================================================

def add_shuffled_sentence_positions(sentence_df_in, seed=42):
    """
    Keeps sentence content fixed but randomly reassigns sentence positions
    within each note.

    Adds:
        shuf_sentence_index
        shuf_rel_pos
        shuf_third_bin
        shuf_quart_bin
    """
    rng = np.random.default_rng(seed)
    parts = []

    for note_id, group in tqdm(
        sentence_df_in.groupby("note_id", sort=False),
        desc="Creating shuffled sentence positions"
    ):
        g = group.copy()
        n = len(g)

        if n == 0:
            continue

        shuffled_positions = rng.permutation(np.arange(1, n + 1))

        g["shuf_sentence_index"] = shuffled_positions
        g["shuf_rel_pos"] = (g["shuf_sentence_index"] - 0.5) / n

        g["shuf_third_bin"] = np.minimum(
            np.floor(g["shuf_rel_pos"] * 3).astype(int) + 1,
            3
        )

        g["shuf_quart_bin"] = np.minimum(
            np.floor(g["shuf_rel_pos"] * 4).astype(int) + 1,
            4
        )

        parts.append(g)

    return pd.concat(parts, ignore_index=True)


sentence_work = add_shuffled_sentence_positions(sentence_work, seed=42)


# ============================================================
# 7. CREATE SENTENCE-LEVEL THEME FLAGS
# ============================================================

keyword_col = "tokenizedkluekeywordsentencetransformer_sentence"

if keyword_col not in sentence_work.columns:
    raise KeyError(f"sentence_df must contain `{keyword_col}`.")

sentence_work[keyword_col] = sentence_work[keyword_col].apply(parse_listlike)


def sentence_has_theme(keyword_list, target_tokens):
    tokens = extract_tokens_from_keyword_list(keyword_list)

    return int(
        any(
            any(target in token for token in tokens)
            for target in target_tokens
        )
    )


for theme, target_tokens in THEME_TOKEN_DICT.items():
    flag_col = f"theme_{theme}"
    sentence_work[flag_col] = sentence_work[keyword_col].apply(
        lambda x: sentence_has_theme(x, target_tokens)
    )

theme_flag_cols = [f"theme_{theme}" for theme in THEMES]


# ============================================================
# 8. CREATE SENTENCE-LEVEL SENTIMENT FLAGS
# ============================================================

sentiment_col = "sentiment"

if sentiment_col not in sentence_work.columns:
    raise KeyError("sentence_df must contain `sentiment`.")

sentence_work[sentiment_col] = sentence_work[sentiment_col].apply(parse_dictlike)


for english_label, korean_label in sentiment_label_map.items():
    flag_col = f"sentiment_{english_label}"
    sentence_work[flag_col] = sentence_work[sentiment_col].apply(
        lambda d: int(korean_label in d.get("labels", []))
    )

sentiment_flag_cols = [f"sentiment_{sentiment}" for sentiment in SENTIMENTS]


# ============================================================
# 9. SENTENCE-LEVEL SECTION SCHEMES
# ============================================================

SENTENCE_SCHEMES = {
    "sent_real_third": {
        "bin_col": "third_bin",
        "n_bins": 3,
        "section_labels": ["Section 1", "Section 2", "Section 3"],
        "section_display": {
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        "min_sentences": 3,
    },
    "sent_real_quart": {
        "bin_col": "quart_bin",
        "n_bins": 4,
        "section_labels": ["Section 1", "Section 2", "Section 3", "Section 4"],
        "section_display": {
            "Section 1": "Q1",
            "Section 2": "Q2",
            "Section 3": "Q3",
            "Section 4": "Q4",
        },
        "min_sentences": 4,
    },
    "sent_shuf_third": {
        "bin_col": "shuf_third_bin",
        "n_bins": 3,
        "section_labels": ["Section 1", "Section 2", "Section 3"],
        "section_display": {
            "Section 1": "Shuffled 1",
            "Section 2": "Shuffled 2",
            "Section 3": "Shuffled 3",
        },
        "min_sentences": 3,
    },
    "sent_shuf_quart": {
        "bin_col": "shuf_quart_bin",
        "n_bins": 4,
        "section_labels": ["Section 1", "Section 2", "Section 3", "Section 4"],
        "section_display": {
            "Section 1": "Shuffled Q1",
            "Section 2": "Shuffled Q2",
            "Section 3": "Shuffled Q3",
            "Section 4": "Shuffled Q4",
        },
        "min_sentences": 4,
    },
}


# ============================================================
# 10. AGGREGATE SENTENCE FLAGS TO NOTE-LEVEL FEATURES
# ============================================================

def aggregate_sentence_flags_to_note_features(
    sentence_df_in,
    note_demo_df,
    flag_cols,
    label_names,
    bin_col,
    n_bins,
    scheme_name,
    min_sentences
):
    """
    Aggregates sentence-level binary flags into note-level section features.

    If any sentence in a note-section has the flag, the note-section feature = 1.

    Feature name format:
        sent_real_third Section 1 Sorry and Shame
        sent_shuf_quart Section 4 Despair
    """
    rows = []

    if bin_col not in sentence_df_in.columns:
        raise KeyError(f"Missing bin column: {bin_col}")

    for note_id, group in tqdm(
        sentence_df_in.groupby("note_id", sort=False),
        desc=f"Aggregating {scheme_name}"
    ):
        n_sent = int(group["n_sent"].iloc[0]) if "n_sent" in group.columns else len(group)

        if n_sent < min_sentences:
            continue

        row = {"note_id": note_id}

        for b in range(1, n_bins + 1):
            section_group = group[group[bin_col] == b]
            section_label = f"Section {b}"

            for flag_col, clean_label in zip(flag_cols, label_names):
                feature_name = f"{scheme_name} {section_label} {clean_label}"

                if len(section_group) == 0:
                    row[feature_name] = 0
                else:
                    row[feature_name] = int(section_group[flag_col].max())

        rows.append(row)

    feature_df = pd.DataFrame(rows)

    out = note_demo_df.merge(feature_df, on="note_id", how="inner")

    feature_names = [
        col for col in out.columns
        if col.startswith(f"{scheme_name} Section")
    ]

    return out, feature_names


# ============================================================
# 11. MODEL HELPERS
# ============================================================

def fit_age_ovr_logit(
    df,
    feature_names,
    outcome_col="AGE2",
    sex_col="SEX",
    gender_covariate_name="Gender",
    maxiter=200
):
    work = df[
        df[outcome_col].isin([1, 2, 3, 4, 5])
        & df[sex_col].isin([1, 2])
    ].copy()

    work[gender_covariate_name] = work[sex_col].replace({1: 1, 2: 0}).astype(int)

    X = work[feature_names + [gender_covariate_name]].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)

    group_results = {}

    for group in AGE_ORDER:
        y = (work[outcome_col] == group).astype(int)

        mask = ~(X.isna().any(axis=1) | y.isna())
        Xi = X.loc[mask].copy()
        yi = y.loc[mask].copy()

        fit_error = ''
        try:
            model = sm.Logit(yi, Xi)
            result = model.fit(disp=False, maxiter=maxiter)
            require_valid_logit(result)

            odds_ratios = np.exp(result.params)
            conf = np.exp(result.conf_int())
            conf.columns = ["CI Lower", "CI Upper"]

            summary_df = pd.DataFrame({
                "Feature": result.params.index,
                "Coefficient": result.params.values,
                "Odds Ratio": odds_ratios.values,
                "p-value": result.pvalues.values,
                "CI Lower": conf["CI Lower"].values,
                "CI Upper": conf["CI Upper"].values,
                "Age Group": group,
                "n_model": len(yi),
            })

        except Exception as e:
            fit_error = str(e)
            print(f"[WARN] Age model failed for group {group}: {e}")

            summary_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": np.nan,
                "Odds Ratio": np.nan,
                "p-value": np.nan,
                "CI Lower": np.nan,
                "CI Upper": np.nan,
                "Age Group": group,
                "n_model": len(yi),
            })

        summary_df['Inference Status'] = 'invalid' if fit_error else 'valid'
        summary_df['Inference Error'] = fit_error
        group_results[group] = summary_df

    combined = pd.concat(group_results.values(), ignore_index=True)

    combined = combined[combined["Feature"].isin(feature_names)].copy()
    combined["log_odds_ratio"] = safe_log_or(combined["Odds Ratio"])
    combined["sig_star"] = combined["p-value"].apply(get_sig_star)
    combined["annot"] = (
        combined["log_odds_ratio"].round(2).astype(str)
        + combined["sig_star"]
    )
    combined["Demographic"] = combined["Age Group"].apply(lambda x: f"Age{x}")

    return combined


def fit_gender_logit_controlling_age(
    df,
    feature_names,
    sex_col="SEX",
    age_col="AGE2",
    maxiter=200
):
    work = df[
        df[sex_col].isin([1, 2])
        & df[age_col].isin([1, 2, 3, 4, 5])
    ].copy()

    work["SEX_BINARY"] = work[sex_col].replace({1: 1, 2: 0}).astype(int)

    age_dummies = pd.get_dummies(
        work[age_col],
        prefix="AGE2",
        drop_first=True,
        dtype=int
    )

    for col in ["AGE2_2", "AGE2_3", "AGE2_4", "AGE2_5"]:
        if col not in age_dummies.columns:
            age_dummies[col] = 0

    age_dummies = age_dummies[["AGE2_2", "AGE2_3", "AGE2_4", "AGE2_5"]]

    X = pd.concat([work[feature_names], age_dummies], axis=1)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)

    y = work["SEX_BINARY"].astype(float)

    mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X.loc[mask].copy()
    y_clean = y.loc[mask].copy()

    print("Gender model diagnostics:")
    print("  X_clean shape:", X_clean.shape)
    print("  y_clean shape:", y_clean.shape)
    print("  Class balance:", y_clean.value_counts().to_dict())

    fit_error = ''
    try:
        model = sm.Logit(y_clean, X_clean)
        result = model.fit(disp=False, maxiter=maxiter)
        require_valid_logit(result)

        odds_ratios = np.exp(result.params)
        conf = np.exp(result.conf_int())
        conf.columns = ["CI Lower", "CI Upper"]

        summary_df = pd.DataFrame({
            "Feature": result.params.index,
            "Coefficient": result.params.values,
            "Odds Ratio": odds_ratios.values,
            "p-value": result.pvalues.values,
            "CI Lower": conf["CI Lower"].values,
            "CI Upper": conf["CI Upper"].values,
            "n_model": len(y_clean),
        })

    except Exception as e:
        fit_error = str(e)
        print(f"[WARN] Gender model failed: {e}")

        result = None

        summary_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": np.nan,
            "Odds Ratio": np.nan,
            "p-value": np.nan,
            "CI Lower": np.nan,
            "CI Upper": np.nan,
            "n_model": len(y_clean),
        })

    summary_df['Inference Status'] = 'invalid' if fit_error else 'valid'
    summary_df['Inference Error'] = fit_error
    summary_df = summary_df[summary_df["Feature"].isin(feature_names)].copy()
    summary_df["Demographic"] = "Gender (Male)"
    summary_df["log_odds_ratio"] = safe_log_or(summary_df["Odds Ratio"])
    summary_df["sig_star"] = summary_df["p-value"].apply(get_sig_star)
    summary_df["annot"] = (
        summary_df["log_odds_ratio"].round(2).astype(str)
        + summary_df["sig_star"]
    )

    return summary_df, result


def split_sentence_scheme_section_and_label(df, label_col_name):
    """
    Splits feature names like:
        sent_real_third Section 1 Sorry and Shame
        sent_shuf_quart Section 4 Despair
    """
    out = df.copy()

    extracted = out["Feature"].str.extract(
        r"^(?P<scheme>.+?)\s+(?P<section>Section\s+\d+)\s+(?P<label>.+)$"
    )

    out["scheme"] = extracted["scheme"]
    out["section"] = extracted["section"]
    out[label_col_name] = extracted["label"]

    return out


# ============================================================
# 12. PLOT HELPERS
# ============================================================

def plot_first_version_lineplot_by_age(
    combined_df,
    label_col,
    selected_labels,
    section_order,
    section_display,
    title,
    n_rows=2,
    n_cols=3,
    figsize=(15, 8)
):
    """
    First-version plot:
        x-axis = Age Group
        lines = Section
        panels = Theme/Sentiment
    """
    set_constant_plot_style()

    line_df = combined_df.copy()

    line_df = line_df[line_df[label_col].isin(selected_labels)].copy()
    line_df = line_df[line_df["section"].isin(section_order)].copy()

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True
    )

    axes = axes.flatten()
    fig.subplots_adjust(right=0.82)

    for idx, label in enumerate(selected_labels):
        if idx >= len(axes):
            break

        ax = axes[idx]
        label_data = line_df[line_df[label_col] == label].copy()

        for section in section_order:
            line_data = (
                label_data[label_data["section"] == section]
                .sort_values("Age Group")
            )

            if line_data.empty:
                continue

            ax.plot(
                line_data["Age Group"],
                line_data["log_odds_ratio"],
                marker="o",
                label=section_display.get(section, section),
                linewidth=2
            )

            for _, row_ in line_data.iterrows():
                x = row_["Age Group"]
                y = row_["log_odds_ratio"]
                star = get_sig_star(row_["p-value"])

                if star and pd.notna(y):
                    ax.text(
                        x,
                        y,
                        star,
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        weight="bold"
                    )

        ax.set_title(
            f"{label}",
            fontsize=13,
            fontproperties=font_prop
        )

        tick_positions = [1, 2, 3, 4, 5]
        tick_labels = ["≤18", "19–34", "35–49", "50–64", "65+"]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels,
            fontsize=10,
            fontproperties=font_prop
        )

        ax.axhline(0, linestyle="--", color="gray", linewidth=1)

        ax.set_xlabel(
            "Age Group",
            fontsize=11,
            fontproperties=font_prop
        )

        ax.tick_params(bottom=True, left=True)

        if idx % n_cols == 0:
            ax.set_ylabel(
                "Log-Odds",
                fontsize=11,
                fontproperties=font_prop
            )

        if font_prop is not None:
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontproperties(font_prop)
                tick_label.set_fontsize(12)

            for tick_label in ax.get_xticklabels():
                tick_label.set_fontproperties(font_prop)
                tick_label.set_fontsize(12)

    for idx in range(len(selected_labels), len(axes)):
        fig.delaxes(axes[idx])

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            title="Section",
            loc="center left",
            bbox_to_anchor=(0.87, 0.5),
            frameon=False,
            fontsize=11,
            title_fontsize=12
        )

    plt.suptitle(
        title,
        fontsize=14,
        weight="bold",
        y=0.95,
        fontproperties=font_prop
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()

    plt.rcdefaults()


def plot_combined_demographic_heatmap(
    age_df,
    gender_df,
    feature_order,
    row_label_kind,
    title,
    figsize=(12, 10),
    section_display_map=None
):
    set_constant_plot_style()

    if section_display_map is None:
        section_display_map = {
            "Section 1": "Opening",
            "Section 2": "Middle",
            "Section 3": "Ending",
            "Section 4": "Q4",
        }

    common_cols = [
        "Feature",
        "Coefficient",
        "Odds Ratio",
        "p-value",
        "CI Lower",
        "CI Upper",
        "Demographic",
        "log_odds_ratio",
        "sig_star",
        "annot",
    ]

    age_tbl = age_df.copy()
    gender_tbl = gender_df.copy()

    if "Demographic" not in age_tbl.columns:
        age_tbl["Demographic"] = age_tbl["Age Group"].apply(lambda x: f"Age{x}")

    if "Demographic" not in gender_tbl.columns:
        gender_tbl["Demographic"] = "Gender (Male)"

    age_tbl = age_tbl[[c for c in common_cols if c in age_tbl.columns]].copy()
    gender_tbl = gender_tbl[[c for c in common_cols if c in gender_tbl.columns]].copy()

    merged_df = pd.concat([age_tbl, gender_tbl], ignore_index=True)

    demographic_order = [f"Age{i}" for i in AGE_ORDER] + ["Gender (Male)"]
    xtick_labels = [AGE_LABELS[i] for i in AGE_ORDER] + ["Gender (Male)"]

    heatmap_data = pd.pivot_table(
        merged_df,
        index="Feature",
        columns="Demographic",
        values="log_odds_ratio",
        aggfunc="first"
    ).reindex(index=feature_order, columns=demographic_order)

    eps = np.finfo(float).tiny

    merged_df["CI Lower"] = pd.to_numeric(
        merged_df["CI Lower"],
        errors="coerce"
    ).clip(lower=eps)

    merged_df["CI Upper"] = pd.to_numeric(
        merged_df["CI Upper"],
        errors="coerce"
    ).clip(lower=eps)

    merged_df["ci_lo_log"] = np.log(merged_df["CI Lower"])
    merged_df["ci_hi_log"] = np.log(merged_df["CI Upper"])

    merged_df["annot_main"] = (
        merged_df["log_odds_ratio"].round(2).astype(str)
        + merged_df["sig_star"].astype(str)
    )

    merged_df["annot_ci_only"] = (
        "["
        + merged_df["ci_lo_log"].round(2).astype(str)
        + ", "
        + merged_df["ci_hi_log"].round(2).astype(str)
        + "]"
    )

    annot_main_tbl = pd.pivot_table(
        merged_df,
        index="Feature",
        columns="Demographic",
        values="annot_main",
        aggfunc="first"
    ).reindex(index=feature_order, columns=demographic_order)

    annot_ci_tbl = pd.pivot_table(
        merged_df,
        index="Feature",
        columns="Demographic",
        values="annot_ci_only",
        aggfunc="first"
    ).reindex(index=feature_order, columns=demographic_order)

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        heatmap_data,
        annot=False,
        fmt="",
        cmap=FIG_STYLE["cmap"],
        center=FIG_STYLE["center"],
        linewidths=FIG_STYLE["heatmap_linewidths"],
        linecolor=FIG_STYLE["heatmap_linecolor"],
        cbar_kws={"label": "Log-Odds"}
    )

    mesh = ax.collections[0]

    def pick_text_color_from_value(val, mappable, thresh=0.53):
        rgba = mappable.cmap(mappable.norm(val if pd.notna(val) else 0.0))
        r, g, b, _ = rgba
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if lum > thresh else "white"

    for i, row_key in enumerate(heatmap_data.index):
        for j, col_key in enumerate(heatmap_data.columns):
            val = heatmap_data.loc[row_key, col_key]
            main_txt = annot_main_tbl.loc[row_key, col_key]
            ci_txt = annot_ci_tbl.loc[row_key, col_key]

            if pd.isna(main_txt) and pd.isna(ci_txt):
                continue

            color = pick_text_color_from_value(val, mesh)

            if pd.notna(main_txt):
                ax.text(
                    j + 0.5,
                    i + 0.42,
                    str(main_txt),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=color,
                    fontproperties=font_prop
                )

            if pd.notna(ci_txt):
                ax.text(
                    j + 0.5,
                    i + 0.68,
                    str(ci_txt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontproperties=font_prop
                )

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Log-Odds", fontsize=12)

    if font_prop is not None:
        cbar.ax.yaxis.label.set_fontproperties(font_prop)

    ax.set_xticks(np.arange(len(demographic_order)) + 0.5)
    ax.set_xticklabels(
        xtick_labels,
        rotation=0,
        fontsize=11,
        fontproperties=font_prop
    )

    ytick_labels = [
        re.sub(r"^.+?\s+Section\s+\d+\s+", "", str(lbl))
        for lbl in heatmap_data.index
    ]

    ax.set_yticklabels(
        ytick_labels,
        rotation=0,
        fontsize=11,
        fontproperties=font_prop
    )

    ax.set_title(
        title,
        fontsize=14,
        fontproperties=font_prop
    )

    ax.set_xlabel(
        "Demographic",
        fontsize=12,
        fontproperties=font_prop
    )

    ax.set_ylabel(
        row_label_kind,
        fontsize=12,
        fontproperties=font_prop
    )

    ax.yaxis.set_label_coords(-0.35, 0.5)

    x_section_label = -1.4 if len(feature_order) > 20 else -1.2
    row_labels = list(heatmap_data.index)

    for section_prefix, display_name in section_display_map.items():
        rows_for_section = [
            i for i, lbl in enumerate(row_labels)
            if re.search(fr"\b{re.escape(section_prefix)}\b", str(lbl))
        ]

        if not rows_for_section:
            continue

        y_middle = (rows_for_section[0] + rows_for_section[-1]) / 2.0

        ax.text(
            x=x_section_label,
            y=y_middle,
            s=display_name,
            va="center",
            ha="center",
            rotation=90,
            fontsize=14,
            weight="bold",
            color="black",
            transform=ax.transData,
            clip_on=False,
            fontproperties=font_prop
        )

    ax.tick_params(bottom=True, left=True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 13. RUN SENTENCE_DF THEME ANALYSIS
# ============================================================

sentence_theme_outputs = {}

for scheme_name, scheme_config in SENTENCE_SCHEMES.items():
    print("\n" + "=" * 80)
    print(f"SENTENCE_DF THEME ANALYSIS: {scheme_name}")
    print("=" * 80)

    try:
        scheme_df, feature_names = aggregate_sentence_flags_to_note_features(
            sentence_df_in=sentence_work,
            note_demo_df=note_demo,
            flag_cols=theme_flag_cols,
            label_names=THEMES,
            bin_col=scheme_config["bin_col"],
            n_bins=scheme_config["n_bins"],
            scheme_name=scheme_name,
            min_sentences=scheme_config["min_sentences"]
        )

        age_results = fit_age_ovr_logit(
            scheme_df,
            feature_names=feature_names,
            outcome_col="AGE2",
            sex_col="SEX"
        )

        age_results = split_sentence_scheme_section_and_label(
            age_results,
            label_col_name="theme"
        )

        gender_results, gender_model = fit_gender_logit_controlling_age(
            scheme_df,
            feature_names=feature_names,
            sex_col="SEX",
            age_col="AGE2"
        )

        gender_results = split_sentence_scheme_section_and_label(
            gender_results,
            label_col_name="theme"
        )

        sentence_theme_outputs[scheme_name] = {
            "df": scheme_df,
            "feature_names": feature_names,
            "age_results": age_results,
            "gender_results": gender_results,
            "gender_model": gender_model,
        }

        if SAVE_OUTPUTS:
            scheme_df.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_theme_features_df.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            age_results.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_theme_age_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            gender_results.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_theme_gender_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    except Exception as e:
        print(f"[ERROR] Sentence theme analysis failed for {scheme_name}: {e}")


# ============================================================
# 14. RUN SENTENCE_DF SENTIMENT ANALYSIS
# ============================================================

sentence_sentiment_outputs = {}

for scheme_name, scheme_config in SENTENCE_SCHEMES.items():
    print("\n" + "=" * 80)
    print(f"SENTENCE_DF SENTIMENT ANALYSIS: {scheme_name}")
    print("=" * 80)

    try:
        scheme_df, feature_names = aggregate_sentence_flags_to_note_features(
            sentence_df_in=sentence_work,
            note_demo_df=note_demo,
            flag_cols=sentiment_flag_cols,
            label_names=SENTIMENTS,
            bin_col=scheme_config["bin_col"],
            n_bins=scheme_config["n_bins"],
            scheme_name=scheme_name,
            min_sentences=scheme_config["min_sentences"]
        )

        age_results = fit_age_ovr_logit(
            scheme_df,
            feature_names=feature_names,
            outcome_col="AGE2",
            sex_col="SEX"
        )

        age_results = split_sentence_scheme_section_and_label(
            age_results,
            label_col_name="sentiment"
        )

        gender_results, gender_model = fit_gender_logit_controlling_age(
            scheme_df,
            feature_names=feature_names,
            sex_col="SEX",
            age_col="AGE2"
        )

        gender_results = split_sentence_scheme_section_and_label(
            gender_results,
            label_col_name="sentiment"
        )

        sentence_sentiment_outputs[scheme_name] = {
            "df": scheme_df,
            "feature_names": feature_names,
            "age_results": age_results,
            "gender_results": gender_results,
            "gender_model": gender_model,
        }

        if SAVE_OUTPUTS:
            scheme_df.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_sentiment_features_df.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            age_results.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_sentiment_age_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            gender_results.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_sentiment_gender_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    except Exception as e:
        print(f"[ERROR] Sentence sentiment analysis failed for {scheme_name}: {e}")


# ============================================================
# 15. PLOT SENTENCE_DF THEME RESULTS
# ============================================================

for scheme_name, output in sentence_theme_outputs.items():
    print("\n" + "=" * 80)
    print(f"PLOTTING SENTENCE_DF THEME RESULTS: {scheme_name}")
    print("=" * 80)

    scheme_config = SENTENCE_SCHEMES[scheme_name]
    section_order = scheme_config["section_labels"]
    section_display = scheme_config["section_display"]

    age_df = output["age_results"]
    gender_df = output["gender_results"]
    feature_names = output["feature_names"]

    plot_first_version_lineplot_by_age(
        combined_df=age_df,
        label_col="theme",
        selected_labels=THEMES,
        section_order=section_order,
        section_display=section_display,
        title=(
            f"{scheme_name}: Sentence-Level Theme-Section Effects on Age Group Membership\n"
            "(Controlling for Gender)"
        ),
        n_rows=2,
        n_cols=3,
        figsize=(15, 8)
    )

    plot_combined_demographic_heatmap(
        age_df=age_df,
        gender_df=gender_df,
        feature_order=feature_names,
        row_label_kind="Sentence-Level Thematic Feature",
        title=(
            f"{scheme_name}: Sentence-Level Theme Features across Demographics\n"
            "Top: log(OR) with stars, Bottom: 95% CI on log scale"
        ),
        figsize=(12, max(10, 0.45 * len(feature_names))),
        section_display_map=section_display
    )


# ============================================================
# 16. PLOT SENTENCE_DF SENTIMENT RESULTS
# ============================================================

for scheme_name, output in sentence_sentiment_outputs.items():
    print("\n" + "=" * 80)
    print(f"PLOTTING SENTENCE_DF SENTIMENT RESULTS: {scheme_name}")
    print("=" * 80)

    scheme_config = SENTENCE_SCHEMES[scheme_name]
    section_order = scheme_config["section_labels"]
    section_display = scheme_config["section_display"]

    age_df = output["age_results"]
    gender_df = output["gender_results"]
    feature_names = output["feature_names"]

    plot_first_version_lineplot_by_age(
        combined_df=age_df,
        label_col="sentiment",
        selected_labels=LINEPLOT_SENTIMENTS,
        section_order=section_order,
        section_display=section_display,
        title=(
            f"{scheme_name}: Sentence-Level Sentiment-Section Effects on Age Group Membership\n"
            "(Controlling for Gender)"
        ),
        n_rows=2,
        n_cols=3,
        figsize=(15, 8)
    )

    plot_combined_demographic_heatmap(
        age_df=age_df,
        gender_df=gender_df,
        feature_order=feature_names,
        row_label_kind="Sentence-Level Sentiment Feature",
        title=(
            f"{scheme_name}: Sentence-Level Sentiment Features across Demographics\n"
            "Top: log(OR) with stars, Bottom: 95% CI on log scale"
        ),
        figsize=(12, max(14, 0.38 * len(feature_names))),
        section_display_map=section_display
    )


# ============================================================
# 17. COMBINE AND SAVE ALL SENTENCE_DF RESULTS
# ============================================================

all_sentence_theme_age = []
all_sentence_theme_gender = []

for scheme_name, output in sentence_theme_outputs.items():
    a = output["age_results"].copy()
    g = output["gender_results"].copy()

    a["scheme"] = scheme_name
    g["scheme"] = scheme_name

    all_sentence_theme_age.append(a)
    all_sentence_theme_gender.append(g)

all_sentence_sentiment_age = []
all_sentence_sentiment_gender = []

for scheme_name, output in sentence_sentiment_outputs.items():
    a = output["age_results"].copy()
    g = output["gender_results"].copy()

    a["scheme"] = scheme_name
    g["scheme"] = scheme_name

    all_sentence_sentiment_age.append(a)
    all_sentence_sentiment_gender.append(g)


all_sentence_theme_age = (
    pd.concat(all_sentence_theme_age, ignore_index=True)
    if all_sentence_theme_age else pd.DataFrame()
)

all_sentence_theme_gender = (
    pd.concat(all_sentence_theme_gender, ignore_index=True)
    if all_sentence_theme_gender else pd.DataFrame()
)

all_sentence_sentiment_age = (
    pd.concat(all_sentence_sentiment_age, ignore_index=True)
    if all_sentence_sentiment_age else pd.DataFrame()
)

all_sentence_sentiment_gender = (
    pd.concat(all_sentence_sentiment_gender, ignore_index=True)
    if all_sentence_sentiment_gender else pd.DataFrame()
)


if SAVE_OUTPUTS:
    sentence_work.to_csv(
        os.path.join(OUTPUT_DIR, "sentence_df_with_theme_sentiment_flags_and_shuffled_bins.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentence_theme_age.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_sentence_theme_age_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentence_theme_gender.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_sentence_theme_gender_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentence_sentiment_age.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_sentence_sentiment_age_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentence_sentiment_gender.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_sentence_sentiment_gender_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )


print("\n" + "=" * 80)
print("DONE: SENTENCE_DF ANALYSIS")
print("=" * 80)

print("\nSentence-level theme schemes completed:")
print(list(sentence_theme_outputs.keys()))

print("\nSentence-level sentiment schemes completed:")
print(list(sentence_sentiment_outputs.keys()))

if SAVE_OUTPUTS:
    print("\nSaved outputs to:")
    print(OUTPUT_DIR)
    

