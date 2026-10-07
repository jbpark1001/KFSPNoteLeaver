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
# 1. GLOBAL OPTIONS
# ============================================================

SAVE_OUTPUTS = True
OUTPUT_DIR = str(output_path("tables/section_robustness/.keep").parent)

if SAVE_OUTPUTS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. GLOBAL FIGURE FORMATTING
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


def apply_axis_font(ax):
    if font_prop is None:
        return

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(FIG_STYLE["tick_fontsize"])

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(FIG_STYLE["tick_fontsize"])

    ax.xaxis.label.set_fontproperties(font_prop)
    ax.yaxis.label.set_fontproperties(font_prop)


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
    """
    Converts Excel-saved stringified lists back into Python lists.

    Examples:
        "[('미안', 0.51), ('사랑', 0.44)]"
        "['문장1', '문장2']"
    """
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
    """
    Converts Excel-saved stringified dictionaries back into Python dicts.

    Example:
        "{'labels': ['슬픔'], 'scores': [0.91]}"
    """
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


def ensure_listn(x, n):
    """
    Makes sure a presence vector is an n-item list.
    """
    x = parse_listlike(x) if not isinstance(x, list) else x

    if not isinstance(x, list):
        return [0] * n

    x = list(x)

    if len(x) < n:
        x = x + [0] * (n - len(x))

    if len(x) > n:
        x = x[:n]

    return [int(v) if not pd.isna(v) else 0 for v in x]


def extract_tokens_from_keyword_list(section):
    """
    Expected forms:
        [('미안', 0.51), ('사랑', 0.44)]
        [['미안', 0.51], ['사랑', 0.44]]
        ['미안', '사랑']
    """
    tokens = []

    if not isinstance(section, list):
        return tokens

    for item in section:
        if isinstance(item, tuple) and len(item) >= 1:
            tokens.append(str(item[0]))
        elif isinstance(item, list) and len(item) >= 1:
            tokens.append(str(item[0]))
        elif isinstance(item, str):
            tokens.append(item)

    return tokens


# ============================================================
# 4. SECTION SCHEMES
# ============================================================

SECTION_SCHEMES = {
    "third": {
        "section_cols": ["third_1", "third_2", "third_3"],
        "tokenized_keyword_cols": [
            "tokenizedkluekeywordsentencetransformer_third_1",
            "tokenizedkluekeywordsentencetransformer_third_2",
            "tokenizedkluekeywordsentencetransformer_third_3",
        ],
        "sentiment_cols": [
            "third_1_sentiment",
            "third_2_sentiment",
            "third_3_sentiment",
        ],
        "section_labels": ["Section 1", "Section 2", "Section 3"],
        "section_display": {
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        "min_sent_col": "third_total_n_sent",
        "min_sentences": 3,
    },

    "quart": {
        "section_cols": ["quart_1", "quart_2", "quart_3", "quart_4"],
        "tokenized_keyword_cols": [
            "tokenizedkluekeywordsentencetransformer_quart_1",
            "tokenizedkluekeywordsentencetransformer_quart_2",
            "tokenizedkluekeywordsentencetransformer_quart_3",
            "tokenizedkluekeywordsentencetransformer_quart_4",
        ],
        "sentiment_cols": [
            "quart_1_sentiment",
            "quart_2_sentiment",
            "quart_3_sentiment",
            "quart_4_sentiment",
        ],
        "section_labels": ["Section 1", "Section 2", "Section 3", "Section 4"],
        "section_display": {
            "Section 1": "Q1",
            "Section 2": "Q2",
            "Section 3": "Q3",
            "Section 4": "Q4",
        },
        "min_sent_col": "quart_total_n_sent",
        "min_sentences": 4,
    },

    "third5": {
        "section_cols": ["third5_1", "third5_2", "third5_3"],
        "tokenized_keyword_cols": [
            "tokenizedkluekeywordsentencetransformer_third5_1",
            "tokenizedkluekeywordsentencetransformer_third5_2",
            "tokenizedkluekeywordsentencetransformer_third5_3",
        ],
        "sentiment_cols": [
            "third5_1_sentiment",
            "third5_2_sentiment",
            "third5_3_sentiment",
        ],
        "section_labels": ["Section 1", "Section 2", "Section 3"],
        "section_display": {
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        "min_sent_col": "third5_total_n_sent",
        "min_sentences": 5,
    },

    "shuf_third": {
        "section_cols": ["shuf_third_1", "shuf_third_2", "shuf_third_3"],
        "tokenized_keyword_cols": [
            "tokenizedkluekeywordsentencetransformer_shuf_third_1",
            "tokenizedkluekeywordsentencetransformer_shuf_third_2",
            "tokenizedkluekeywordsentencetransformer_shuf_third_3",
        ],
        "sentiment_cols": [
            "shuf_third_1_sentiment",
            "shuf_third_2_sentiment",
            "shuf_third_3_sentiment",
        ],
        "section_labels": ["Section 1", "Section 2", "Section 3"],
        "section_display": {
            "Section 1": "Shuffled 1",
            "Section 2": "Shuffled 2",
            "Section 3": "Shuffled 3",
        },
        "min_sent_col": "shuf_third_total_n_sent",
        "min_sentences": 3,
    },
}


# ============================================================
# 5. THEME AND SENTIMENT DICTIONARIES
# ============================================================

sorry = [
    "미안", "죄송", "용서", "잘못", "후회", "죄", "책임", "민폐", "반성"
]

loveandgratitude = [
    "사랑", "고맙", "감사", "고생", "수고", "보고싶", "애틋", "소중", "그리"
]

burdensome = [
    "버겁", "부담", "짐", "감당", "힘들", "무겁", "지치", "헷갈"
]

despair = [
    "포기", "좌절", "죽", "끝", "아무것", "헛되", "무의미", "잊히", "없어지", "그만두"
]

pmaffairs = [
    "부탁", "정리", "남기", "처리", "보험", "은행", "장례", "통장", "유서"
]


THEME_TOKEN_DICT = {
    "Sorry and Shame": sorry,
    "Love and Gratitude": loveandgratitude,
    "Burden": burdensome,
    "Despair": despair,
    "Post-mortem Affairs": pmaffairs,
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

import pandas as pd

scored=load_frame(input_path("section_annotations"))

#raw=scored[scored["pred_label"].isin(["raw"])]

#raw=pd.read_excel(r'C:\Users\<user>\oldrawnotes.xlsx')


# ============================================================
# 6. PREPARE RAW NOTES FROM SCORED
# ============================================================

if "scored" not in globals():
    raise NameError(
        "`scored` is not in memory. "
        "Run the raw-vs-summarized detector first:\n"
        "    scored = det.score_dataframe(notes_df, threshold=0.5)"
    )

if "pred_label" not in scored.columns:
    raise KeyError("`scored` must contain `pred_label`.")

raw = scored[scored["pred_label"].eq("raw")].copy()
print("\n==============================")
print("RAW FILTERING")
print("==============================")
print("Total scored rows:", scored.shape[0])
print("Raw rows:", raw.shape[0])
print(raw["pred_label"].value_counts(dropna=False))

#raw=pd.read_excel(r'C:\Users\<user>\oldrawnotes1.xlsx')

# ------------------------------------------------------------
# Define a general too_short marker if possible
# ------------------------------------------------------------
if "sentencesplit" in raw.columns:
    raw["sentencesplit"] = raw["sentencesplit"].apply(parse_listlike)
    raw["too_short"] = raw["sentencesplit"].apply(
        lambda x: len(x) < 3 if isinstance(x, list) else True
    )

elif "third_total_n_sent" in raw.columns:
    raw["too_short"] = raw["third_total_n_sent"].fillna(0).astype(int) < 3

else:
    raw["too_short"] = False

morethan3 = raw[raw["too_short"].eq(False)].copy()

print("Raw rows with >=3 sentences using general filter:", morethan3.shape[0])


# ============================================================
# 7. CHECK MISSING COLUMNS FOR ALL SCHEMES
# ============================================================

def check_scheme_columns(df, section_schemes):
    rows = []

    for scheme_name, scheme_config in section_schemes.items():
        for group_name in ["section_cols", "tokenized_keyword_cols", "sentiment_cols"]:
            for col in scheme_config[group_name]:
                rows.append({
                    "scheme": scheme_name,
                    "column_group": group_name,
                    "column": col,
                    "exists": col in df.columns,
                })

    out = pd.DataFrame(rows)
    return out


scheme_column_check = check_scheme_columns(raw, SECTION_SCHEMES)

print("\n==============================")
print("SCHEME COLUMN CHECK")
print("==============================")
print(scheme_column_check.groupby(["scheme", "column_group"])["exists"].mean())

missing_scheme_cols = scheme_column_check[scheme_column_check["exists"].eq(False)]

if not missing_scheme_cols.empty:
    print("\nWARNING: Some scheme columns are missing.")
    print(missing_scheme_cols.to_string(index=False))
    print(
        "\nIf sentiment columns are missing for quart/third5/shuf_third, "
        "rerun KOTE sentiment extraction for all section columns."
    )


# ============================================================
# 8. FEATURE BUILDERS
# ============================================================

def filter_scheme_df(df, scheme_name, scheme_config):
    """
    Applies scheme-specific minimum sentence filter.
    """
    out = df.copy()

    min_sent_col = scheme_config.get("min_sent_col")
    min_sentences = scheme_config.get("min_sentences")

    if min_sent_col in out.columns:
        out = out[out[min_sent_col].fillna(0).astype(int) >= min_sentences].copy()

    return out


def build_theme_features_for_scheme(df, scheme_name, scheme_config):
    """
    Builds theme presence features for one section scheme.

    Feature names:
        third Section 1 Sorry and Shame
        quart Section 4 Despair
        third5 Section 2 Burden
    """
    out = df.copy()

    keyword_cols = scheme_config["tokenized_keyword_cols"]
    section_labels = scheme_config["section_labels"]
    n_sections = len(section_labels)

    missing = [c for c in keyword_cols if c not in out.columns]
    if missing:
        raise KeyError(f"[{scheme_name}] Missing keyword columns:\n{missing}")

    for col in keyword_cols:
        out[col] = out[col].apply(parse_listlike)

    def create_presence_vector(row, target_tokens):
        presence = []

        for col in keyword_cols:
            tokens = extract_tokens_from_keyword_list(row[col])

            has_token = any(
                any(target in token for token in tokens)
                for target in target_tokens
            )

            presence.append(int(has_token))

        return presence

    for theme, target_tokens in THEME_TOKEN_DICT.items():
        presence_col = f"{scheme_name}_{theme}_presence"

        tqdm.pandas(desc=f"{scheme_name}: {theme}")

        out[presence_col] = out.progress_apply(
            lambda row: create_presence_vector(row, target_tokens),
            axis=1
        )

        out[presence_col] = out[presence_col].apply(
            lambda x: ensure_listn(x, n_sections)
        )

    feature_names = []

    for sec_idx, section_label in enumerate(section_labels):
        for theme in THEMES:
            feature_col = f"{scheme_name} {section_label} {theme}"
            presence_col = f"{scheme_name}_{theme}_presence"

            out[feature_col] = out[presence_col].apply(
                lambda v: ensure_listn(v, n_sections)[sec_idx]
            )

            feature_names.append(feature_col)

    return out, feature_names


def build_sentiment_features_for_scheme(df, scheme_name, scheme_config):
    """
    Builds sentiment presence features for one section scheme.

    Feature names:
        third Section 1 Sadness
        quart Section 4 Defeat
        third5 Section 2 Neutral
    """
    out = df.copy()

    sentiment_cols = scheme_config["sentiment_cols"]
    section_labels = scheme_config["section_labels"]
    n_sections = len(section_labels)

    missing = [c for c in sentiment_cols if c not in out.columns]
    if missing:
        raise KeyError(
            f"[{scheme_name}] Missing sentiment columns:\n{missing}\n"
            "You probably need to run KOTE sentiment prediction for all section columns."
        )

    for col in sentiment_cols:
        out[col] = out[col].apply(parse_dictlike)

    def create_sentiment_presence(row, korean_label):
        presence = []

        for col in sentiment_cols:
            entry = row[col]

            if isinstance(entry, dict):
                section_labels_found = entry.get("labels", [])
                has_label = int(korean_label in section_labels_found)
            else:
                has_label = 0

            presence.append(has_label)

        return presence

    for english_label, korean_label in sentiment_label_map.items():
        presence_col = f"{scheme_name}_{english_label}_presence"

        tqdm.pandas(desc=f"{scheme_name}: {english_label}")

        out[presence_col] = out.progress_apply(
            lambda row: create_sentiment_presence(row, korean_label),
            axis=1
        )

        out[presence_col] = out[presence_col].apply(
            lambda x: ensure_listn(x, n_sections)
        )

    feature_names = []

    for sec_idx, section_label in enumerate(section_labels):
        for sentiment in SENTIMENTS:
            feature_col = f"{scheme_name} {section_label} {sentiment}"
            presence_col = f"{scheme_name}_{sentiment}_presence"

            out[feature_col] = out[presence_col].apply(
                lambda v: ensure_listn(v, n_sections)[sec_idx]
            )

            feature_names.append(feature_col)

    return out, feature_names


# ============================================================
# 9. MODEL HELPERS
# ============================================================

def fit_age_ovr_logit(
    df,
    feature_names,
    outcome_col="AGE2",
    sex_col="SEX",
    gender_covariate_name="Gender",
    maxiter=200
):
    """
    One-vs-rest logistic regression:
        AGE2 group membership ~ features + Gender

    Gender coding:
        Male=1, Female=0
    """
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
    """
    Logistic regression:
        SEX_BINARY ~ features + AGE2 categorical dummies

    Outcome:
        Male=1, Female=0

    AGE2 reference group:
        AGE2 == 1
    """
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

    X = pd.concat(
        [
            work[feature_names],
            age_dummies
        ],
        axis=1
    )

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


def split_scheme_section_and_label(df, label_col_name):
    """
    Splits feature names like:
        third Section 1 Sorry and Shame
        quart Section 4 Despair

    into:
        scheme
        section
        theme/sentiment
    """
    out = df.copy()

    extracted = out["Feature"].str.extract(
        r"^(?P<scheme>\S+)\s+(?P<section>Section\s+\d+)\s+(?P<label>.+)$"
    )

    out["scheme"] = extracted["scheme"]
    out["section"] = extracted["section"]
    out[label_col_name] = extracted["label"]

    return out


# ============================================================
# 10. PLOT HELPERS
# ============================================================

def plot_age_heatmaps_by_feature(
    combined_df,
    label_col,
    label_order,
    section_order,
    section_display,
    title_prefix,
    figsize=(8, 5)
):
    set_constant_plot_style()

    for group in sorted(combined_df["Age Group"].dropna().unique()):
        sub_df = combined_df[combined_df["Age Group"] == group].copy()

        heatmap_data = sub_df.pivot(
            index=label_col,
            columns="section",
            values="log_odds_ratio"
        ).reindex(index=label_order, columns=section_order)

        annot_data = sub_df.pivot(
            index=label_col,
            columns="section",
            values="annot"
        ).reindex(index=label_order, columns=section_order)

        plt.figure(figsize=figsize)

        ax = sns.heatmap(
            heatmap_data,
            annot=annot_data,
            fmt="",
            center=FIG_STYLE["center"],
            cmap=FIG_STYLE["cmap"],
            linewidths=FIG_STYLE["heatmap_linewidths"],
            linecolor=FIG_STYLE["heatmap_linecolor"],
            cbar_kws={"label": "Log-Odds"}
        )

        for text in ax.texts:
            text.set_fontsize(FIG_STYLE["annot_fontsize"])
            if font_prop is not None:
                text.set_fontproperties(font_prop)

        ax.set_title(
            f"{title_prefix} by Age Group {group}",
            fontsize=FIG_STYLE["title_fontsize"],
            fontproperties=font_prop
        )
        ax.set_ylabel(label_col.capitalize(), fontsize=FIG_STYLE["axis_label_fontsize"])
        ax.set_xlabel("Section", fontsize=FIG_STYLE["axis_label_fontsize"])

        current_xticks = [x.get_text() for x in ax.get_xticklabels()]
        ax.set_xticklabels(
            [section_display.get(x, x) for x in current_xticks],
            rotation=0
        )

        apply_axis_font(ax)

        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel("Log-Odds", fontsize=FIG_STYLE["axis_label_fontsize"])
        if font_prop is not None:
            cbar.ax.yaxis.label.set_fontproperties(font_prop)

        plt.tight_layout()
        plt.show()


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
    FIRST-VERSION LINE PLOT.

    This matches the original structure:
        - one panel per theme/sentiment
        - x-axis = Age Group
        - lines = Section
        - y-axis = log_odds_ratio
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

        ax.axhline(
            0,
            linestyle="--",
            color="gray",
            linewidth=1
        )

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

    # Remove extra subplot if present
    for idx in range(len(selected_labels), len(axes)):
        fig.delaxes(axes[idx])

    # Shared legend
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
    include_ci=True,
    section_display_map=None
):
    """
    Combined heatmap:
        columns = Age1..Age5 + Gender(Male)
        rows = Section x Theme/Sentiment features

    If include_ci=True:
        first line = log OR + stars
        second line = 95% CI on log scale
    """
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

    for df_ in [age_tbl, gender_tbl]:
        if "annot" not in df_.columns:
            df_["annot"] = df_["log_odds_ratio"].round(2).astype(str) + df_["sig_star"]

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

    if include_ci:
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

    else:
        annot_tbl = pd.pivot_table(
            merged_df,
            index="Feature",
            columns="Demographic",
            values="annot",
            aggfunc="first"
        ).reindex(index=feature_order, columns=demographic_order)

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        heatmap_data,
        annot=False if include_ci else annot_tbl,
        fmt="",
        cmap=FIG_STYLE["cmap"],
        center=FIG_STYLE["center"],
        linewidths=FIG_STYLE["heatmap_linewidths"],
        linecolor=FIG_STYLE["heatmap_linecolor"],
        cbar_kws={"label": "Log-Odds"}
    )

    if include_ci:
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

    else:
        for text in ax.texts:
            text.set_fontsize(FIG_STYLE["annot_fontsize"])
            if font_prop is not None:
                text.set_fontproperties(font_prop)

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Log-Odds", fontsize=FIG_STYLE["axis_label_fontsize"])
    if font_prop is not None:
        cbar.ax.yaxis.label.set_fontproperties(font_prop)

    ax.set_xticks(np.arange(len(demographic_order)) + 0.5)
    ax.set_xticklabels(
        xtick_labels,
        rotation=0,
        fontsize=FIG_STYLE["tick_fontsize"],
        fontproperties=font_prop
    )

    ytick_labels = [
        re.sub(r"^\S+\s+Section\s+\d+\s+", "", str(lbl))
        for lbl in heatmap_data.index
    ]

    ax.set_yticklabels(
        ytick_labels,
        rotation=0,
        fontsize=FIG_STYLE["tick_fontsize"],
        fontproperties=font_prop
    )

    ax.set_title(
        title,
        fontsize=FIG_STYLE["title_fontsize"],
        fontproperties=font_prop
    )

    ax.set_xlabel(
        "Demographic",
        fontsize=FIG_STYLE["axis_label_fontsize"],
        fontproperties=font_prop
    )

    ax.set_ylabel(
        row_label_kind,
        fontsize=FIG_STYLE["axis_label_fontsize"],
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
# 11. RUN THEME ANALYSES FOR ALL SECTION SCHEMES
# ============================================================

theme_scheme_outputs = {}

for scheme_name, scheme_config in SECTION_SCHEMES.items():
    print("\n" + "=" * 80)
    print(f"THEME ANALYSIS FOR SECTION SCHEME: {scheme_name}")
    print("=" * 80)

    try:
        scheme_df = filter_scheme_df(morethan3, scheme_name, scheme_config)

        print(f"[{scheme_name}] N after scheme-specific sentence filter:", len(scheme_df))

        scheme_df, scheme_theme_features = build_theme_features_for_scheme(
            scheme_df,
            scheme_name,
            scheme_config
        )

        scheme_theme_age = fit_age_ovr_logit(
            scheme_df,
            feature_names=scheme_theme_features,
            outcome_col="AGE2",
            sex_col="SEX"
        )

        scheme_theme_age = split_scheme_section_and_label(
            scheme_theme_age,
            label_col_name="theme"
        )

        scheme_theme_gender, scheme_theme_gender_model = fit_gender_logit_controlling_age(
            scheme_df,
            feature_names=scheme_theme_features,
            sex_col="SEX",
            age_col="AGE2"
        )

        scheme_theme_gender = split_scheme_section_and_label(
            scheme_theme_gender,
            label_col_name="theme"
        )

        theme_scheme_outputs[scheme_name] = {
            "df": scheme_df,
            "feature_names": scheme_theme_features,
            "age_results": scheme_theme_age,
            "gender_results": scheme_theme_gender,
            "gender_model": scheme_theme_gender_model,
        }

        if SAVE_OUTPUTS:
            scheme_theme_age.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_theme_age_ovr_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            scheme_theme_gender.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_theme_gender_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            scheme_df.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_theme_features_df.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    except Exception as e:
        print(f"[ERROR] Theme analysis failed for scheme {scheme_name}: {e}")


# ============================================================
# 12. RUN SENTIMENT ANALYSES FOR ALL SECTION SCHEMES
# ============================================================

sentiment_scheme_outputs = {}

for scheme_name, scheme_config in SECTION_SCHEMES.items():
    print("\n" + "=" * 80)
    print(f"SENTIMENT ANALYSIS FOR SECTION SCHEME: {scheme_name}")
    print("=" * 80)

    try:
        scheme_df = filter_scheme_df(morethan3, scheme_name, scheme_config)

        print(f"[{scheme_name}] N after scheme-specific sentence filter:", len(scheme_df))

        scheme_df, scheme_sentiment_features = build_sentiment_features_for_scheme(
            scheme_df,
            scheme_name,
            scheme_config
        )

        scheme_sentiment_age = fit_age_ovr_logit(
            scheme_df,
            feature_names=scheme_sentiment_features,
            outcome_col="AGE2",
            sex_col="SEX"
        )

        scheme_sentiment_age = split_scheme_section_and_label(
            scheme_sentiment_age,
            label_col_name="sentiment"
        )

        scheme_sentiment_gender, scheme_sentiment_gender_model = fit_gender_logit_controlling_age(
            scheme_df,
            feature_names=scheme_sentiment_features,
            sex_col="SEX",
            age_col="AGE2"
        )

        scheme_sentiment_gender = split_scheme_section_and_label(
            scheme_sentiment_gender,
            label_col_name="sentiment"
        )

        sentiment_scheme_outputs[scheme_name] = {
            "df": scheme_df,
            "feature_names": scheme_sentiment_features,
            "age_results": scheme_sentiment_age,
            "gender_results": scheme_sentiment_gender,
            "gender_model": scheme_sentiment_gender_model,
        }

        if SAVE_OUTPUTS:
            scheme_sentiment_age.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentiment_age_ovr_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            scheme_sentiment_gender.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentiment_gender_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            scheme_df.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentiment_features_df.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    except Exception as e:
        print(f"[ERROR] Sentiment analysis failed for scheme {scheme_name}: {e}")


# ============================================================
# 13. PLOT ALL THEME RESULTS FOR ALL SECTION SCHEMES
#     FIRST LINE-PLOT VERSION ONLY
# ============================================================

for scheme_name, output in theme_scheme_outputs.items():
    print("\n" + "=" * 80)
    print(f"PLOTTING THEME RESULTS FOR: {scheme_name}")
    print("=" * 80)

    scheme_config = SECTION_SCHEMES[scheme_name]
    section_order = scheme_config["section_labels"]
    section_display = scheme_config["section_display"]

    theme_age_df = output["age_results"]
    theme_gender_df = output["gender_results"]
    theme_feature_names = output["feature_names"]

    # --------------------------------------------------------
    # Age heatmaps by section and theme
    # --------------------------------------------------------
    plot_age_heatmaps_by_feature(
        combined_df=theme_age_df,
        label_col="theme",
        label_order=THEMES,
        section_order=section_order,
        section_display=section_display,
        title_prefix=f"{scheme_name}: Theme × Section Log-Odds",
        figsize=(max(8, len(section_order) * 2.2), 5)
    )

    # --------------------------------------------------------
    # FIRST-VERSION LINE PLOT ONLY:
    # x-axis = Age Group
    # lines = Section
    # panels = Theme
    # --------------------------------------------------------
    plot_first_version_lineplot_by_age(
        combined_df=theme_age_df,
        label_col="theme",
        selected_labels=THEMES,
        section_order=section_order,
        section_display=section_display,
        title=(
            f"{scheme_name}: Theme-Section Effects on Age Group Membership\n"
            "(Controlling for Gender)"
        ),
        n_rows=2,
        n_cols=3,
        figsize=(15, 8)
    )

    # --------------------------------------------------------
    # Combined age + gender heatmap
    # --------------------------------------------------------
    plot_combined_demographic_heatmap(
        age_df=theme_age_df,
        gender_df=theme_gender_df,
        feature_order=theme_feature_names,
        row_label_kind="Thematic Feature",
        title=(
            f"{scheme_name}: Log(Odds Ratio) by Thematic Feature across Demographics\n"
            "Top: log(OR) with stars, Bottom: 95% CI on log scale"
        ),
        figsize=(12, max(10, 0.45 * len(theme_feature_names))),
        include_ci=True,
        section_display_map=section_display
    )


# ============================================================
# 14. PLOT ALL SENTIMENT RESULTS FOR ALL SECTION SCHEMES
#     FIRST LINE-PLOT VERSION ONLY
# ============================================================

for scheme_name, output in sentiment_scheme_outputs.items():
    print("\n" + "=" * 80)
    print(f"PLOTTING SENTIMENT RESULTS FOR: {scheme_name}")
    print("=" * 80)

    scheme_config = SECTION_SCHEMES[scheme_name]
    section_order = scheme_config["section_labels"]
    section_display = scheme_config["section_display"]

    sentiment_age_df = output["age_results"]
    sentiment_gender_df = output["gender_results"]
    sentiment_feature_names = output["feature_names"]

    # --------------------------------------------------------
    # Sentiment heatmaps by age group
    # --------------------------------------------------------
    plot_age_heatmaps_by_feature(
        combined_df=sentiment_age_df,
        label_col="sentiment",
        label_order=SENTIMENTS,
        section_order=section_order,
        section_display=section_display,
        title_prefix=f"{scheme_name}: Sentiment × Section Log-Odds",
        figsize=(max(10, len(section_order) * 2.6), 6)
    )

    # --------------------------------------------------------
    # FIRST-VERSION LINE PLOT ONLY:
    # x-axis = Age Group
    # lines = Section
    # panels = Sentiment
    # --------------------------------------------------------
    plot_first_version_lineplot_by_age(
        combined_df=sentiment_age_df,
        label_col="sentiment",
        selected_labels=LINEPLOT_SENTIMENTS,
        section_order=section_order,
        section_display=section_display,
        title=(
            f"{scheme_name}: Sentiment-Section Effects on Age Group Membership\n"
            "(Controlling for Gender)"
        ),
        n_rows=2,
        n_cols=3,
        figsize=(15, 8)
    )

    # --------------------------------------------------------
    # Combined age + gender sentiment heatmap
    # --------------------------------------------------------
    plot_combined_demographic_heatmap(
        age_df=sentiment_age_df,
        gender_df=sentiment_gender_df,
        feature_order=sentiment_feature_names,
        row_label_kind="Sentiment Feature",
        title=(
            f"{scheme_name}: Log(Odds Ratio) by Sentiment Feature across Demographics\n"
            "Top: log(OR) with stars, Bottom: 95% CI on log scale"
        ),
        figsize=(12, max(14, 0.38 * len(sentiment_feature_names))),
        include_ci=True,
        section_display_map=section_display
    )


# ============================================================
# 15. COMBINE AND SAVE ALL RESULTS
# ============================================================

all_theme_age_results = []
all_theme_gender_results = []

for scheme_name, output in theme_scheme_outputs.items():
    age_df = output["age_results"].copy()
    gender_df = output["gender_results"].copy()

    age_df["scheme"] = scheme_name
    gender_df["scheme"] = scheme_name

    all_theme_age_results.append(age_df)
    all_theme_gender_results.append(gender_df)

if all_theme_age_results:
    all_theme_age_results = pd.concat(all_theme_age_results, ignore_index=True)
else:
    all_theme_age_results = pd.DataFrame()

if all_theme_gender_results:
    all_theme_gender_results = pd.concat(all_theme_gender_results, ignore_index=True)
else:
    all_theme_gender_results = pd.DataFrame()


all_sentiment_age_results = []
all_sentiment_gender_results = []

for scheme_name, output in sentiment_scheme_outputs.items():
    age_df = output["age_results"].copy()
    gender_df = output["gender_results"].copy()

    age_df["scheme"] = scheme_name
    gender_df["scheme"] = scheme_name

    all_sentiment_age_results.append(age_df)
    all_sentiment_gender_results.append(gender_df)

if all_sentiment_age_results:
    all_sentiment_age_results = pd.concat(all_sentiment_age_results, ignore_index=True)
else:
    all_sentiment_age_results = pd.DataFrame()

if all_sentiment_gender_results:
    all_sentiment_gender_results = pd.concat(all_sentiment_gender_results, ignore_index=True)
else:
    all_sentiment_gender_results = pd.DataFrame()


if SAVE_OUTPUTS:
    all_theme_age_results.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_SCHEMES_theme_age_ovr_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_theme_gender_results.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_SCHEMES_theme_gender_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentiment_age_results.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_SCHEMES_sentiment_age_ovr_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentiment_gender_results.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_SCHEMES_sentiment_gender_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    scheme_column_check.to_csv(
        os.path.join(OUTPUT_DIR, "scheme_column_check.csv"),
        index=False,
        encoding="utf-8-sig"
    )


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print("\nTheme schemes completed:")
print(list(theme_scheme_outputs.keys()))

print("\nSentiment schemes completed:")
print(list(sentiment_scheme_outputs.keys()))

if SAVE_OUTPUTS:
    print("\nSaved outputs to:")
    print(OUTPUT_DIR)
    #%%
# ============================================================
# 16. REAL VS SHUFFLED NULL-MODEL COMPARISON FIGURES
# ============================================================
# Add this AFTER theme_scheme_outputs and sentiment_scheme_outputs are created.
#
# Purpose:
#   Compare real third sections against shuffled third sections.
#
# Main outputs:
#   1. Real-minus-shuffled delta dataframe
#   2. Delta heatmaps
#   3. Positional-signal attenuation heatmaps
#   4. Publication-resolution PDF/PNG files
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# Output directory for null-model figures
# ------------------------------------------------------------
NULL_FIG_DIR = os.path.join(OUTPUT_DIR, "real_vs_shuffled_figures")

if SAVE_OUTPUTS:
    os.makedirs(NULL_FIG_DIR, exist_ok=True)


# ------------------------------------------------------------
# Helper: save figures in publication-friendly formats
# ------------------------------------------------------------
def save_publication_figure(fig, filename_base, outdir=NULL_FIG_DIR, dpi=600):
    """
    Saves both PDF and PNG versions.
    PDF is preferred for vector graphics in manuscripts.
    PNG is useful for quick inspection.
    """
    if not SAVE_OUTPUTS:
        return

    pdf_path = os.path.join(outdir, f"{filename_base}.pdf")
    png_path = os.path.join(outdir, f"{filename_base}.png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# ------------------------------------------------------------
# Helper: build real-vs-shuffled comparison table
# ------------------------------------------------------------
def make_real_vs_shuffle_comparison(
    real_df,
    shuf_df,
    label_col,
    real_scheme_name="third",
    shuf_scheme_name="shuf_third",
    section_order=None
):
    """
    Creates a tidy comparison dataframe.

    Expected input:
        real_df = theme_scheme_outputs["third"]["age_results"]
        shuf_df = theme_scheme_outputs["shuf_third"]["age_results"]

    Output includes:
        log_odds_real
        log_odds_shuf
        delta_real_minus_shuf
        p_real
        p_shuf
    """
    if section_order is None:
        section_order = ["Section 1", "Section 2", "Section 3"]

    real_keep = real_df[
        [
            label_col,
            "section",
            "Age Group",
            "Feature",
            "log_odds_ratio",
            "p-value",
            "Odds Ratio",
            "CI Lower",
            "CI Upper",
        ]
    ].copy()

    shuf_keep = shuf_df[
        [
            label_col,
            "section",
            "Age Group",
            "Feature",
            "log_odds_ratio",
            "p-value",
            "Odds Ratio",
            "CI Lower",
            "CI Upper",
        ]
    ].copy()

    real_keep = real_keep.rename(
        columns={
            "Feature": "Feature_real",
            "log_odds_ratio": "log_odds_real",
            "p-value": "p_real",
            "Odds Ratio": "OR_real",
            "CI Lower": "CI_lower_real",
            "CI Upper": "CI_upper_real",
        }
    )

    shuf_keep = shuf_keep.rename(
        columns={
            "Feature": "Feature_shuf",
            "log_odds_ratio": "log_odds_shuf",
            "p-value": "p_shuf",
            "Odds Ratio": "OR_shuf",
            "CI Lower": "CI_lower_shuf",
            "CI Upper": "CI_upper_shuf",
        }
    )

    compare = real_keep.merge(
        shuf_keep,
        on=[label_col, "section", "Age Group"],
        how="inner"
    )

    compare["real_scheme"] = real_scheme_name
    compare["shuf_scheme"] = shuf_scheme_name

    compare["delta_real_minus_shuf"] = (
        compare["log_odds_real"] - compare["log_odds_shuf"]
    )

    compare["abs_delta"] = compare["delta_real_minus_shuf"].abs()

    compare["real_sig"] = compare["p_real"].apply(get_sig_star)
    compare["shuf_sig"] = compare["p_shuf"].apply(get_sig_star)

    compare["section"] = pd.Categorical(
        compare["section"],
        categories=section_order,
        ordered=True
    )

    compare["Age Group"] = pd.Categorical(
        compare["Age Group"],
        categories=AGE_ORDER,
        ordered=True
    )

    return compare


# ------------------------------------------------------------
# Helper: positional-signal attenuation table
# ------------------------------------------------------------
def make_positional_signal_table(compare_df, label_col):
    """
    Computes how much section differentiation is reduced after shuffling.

    For each label × age group:
        real_range = max(real section logOR) - min(real section logOR)
        shuf_range = max(shuffled section logOR) - min(shuffled section logOR)
        attenuation = real_range - shuf_range

    Positive attenuation means:
        real ordering has stronger section-specific structure than shuffled order.
    """
    rows = []

    for (label, age_group), g in compare_df.groupby([label_col, "Age Group"], observed=True):
        real_vals = pd.to_numeric(g["log_odds_real"], errors="coerce")
        shuf_vals = pd.to_numeric(g["log_odds_shuf"], errors="coerce")

        real_range = real_vals.max() - real_vals.min()
        shuf_range = shuf_vals.max() - shuf_vals.min()

        attenuation = real_range - shuf_range

        if pd.notna(real_range) and real_range != 0:
            attenuation_ratio = attenuation / real_range
        else:
            attenuation_ratio = np.nan

        rows.append({
            label_col: label,
            "Age Group": age_group,
            "real_sectional_range": real_range,
            "shuf_sectional_range": shuf_range,
            "attenuation": attenuation,
            "attenuation_ratio": attenuation_ratio,
        })

    out = pd.DataFrame(rows)

    out["Age Group"] = pd.Categorical(
        out["Age Group"],
        categories=AGE_ORDER,
        ordered=True
    )

    return out


# ------------------------------------------------------------
# Plot 1: Delta heatmaps, faceted by theme/sentiment
# ------------------------------------------------------------
def plot_delta_heatmap_facets(
    compare_df,
    label_col,
    label_order,
    section_order,
    section_display,
    title,
    filename_base=None,
    n_cols=3,
    figsize_per_panel=(4.2, 3.1),
    center=0
):
    """
    Faceted heatmap:
        rows = Section
        columns = Age Group
        values = real logOR - shuffled logOR

    This is the clearest null-model figure.
    """
    set_constant_plot_style()

    plot_df = compare_df.copy()
    plot_df = plot_df[plot_df[label_col].isin(label_order)].copy()

    n_labels = len(label_order)
    n_rows = int(np.ceil(n_labels / n_cols))

    fig_width = figsize_per_panel[0] * n_cols
    fig_height = figsize_per_panel[1] * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    axes_flat = axes.flatten()

    vmax = np.nanmax(np.abs(plot_df["delta_real_minus_shuf"]))
    if pd.isna(vmax) or vmax == 0:
        vmax = 1.0

    for idx, label in enumerate(label_order):
        ax = axes_flat[idx]

        sub = plot_df[plot_df[label_col] == label].copy()

        heatmap_data = sub.pivot(
            index="section",
            columns="Age Group",
            values="delta_real_minus_shuf"
        ).reindex(index=section_order, columns=AGE_ORDER)

        annot_data = heatmap_data.round(2).astype(str)

        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap=FIG_STYLE["cmap"],
            center=center,
            vmin=-vmax,
            vmax=vmax,
            annot=annot_data,
            fmt="",
            linewidths=0.5,
            linecolor="gray",
            cbar=(idx == n_labels - 1),
            cbar_kws={"label": "Real − shuffled log-odds"}
        )

        ax.set_title(label, fontsize=13, fontproperties=font_prop)

        ax.set_xlabel("Age group", fontsize=11, fontproperties=font_prop)
        ax.set_ylabel("Section", fontsize=11, fontproperties=font_prop)

        ax.set_xticklabels(
            [AGE_LABELS.get(int(x.get_text()), x.get_text()) for x in ax.get_xticklabels()],
            rotation=0,
            fontsize=10,
            fontproperties=font_prop
        )

        ax.set_yticklabels(
            [section_display.get(x.get_text(), x.get_text()) for x in ax.get_yticklabels()],
            rotation=0,
            fontsize=10,
            fontproperties=font_prop
        )

    for idx in range(n_labels, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle(
        title,
        fontsize=15,
        weight="bold",
        y=1.02,
        fontproperties=font_prop
    )

    plt.tight_layout()

    if filename_base is not None:
        save_publication_figure(fig, filename_base)

    plt.show()


# ------------------------------------------------------------
# Plot 2: Positional-signal attenuation heatmap
# ------------------------------------------------------------
def plot_positional_attenuation_heatmap(
    signal_df,
    label_col,
    label_order,
    title,
    filename_base=None,
    figsize=(8.5, 4.8)
):
    """
    Heatmap:
        rows = theme/sentiment
        columns = age group
        values = real sectional spread - shuffled sectional spread

    Positive values mean real sentence order produces stronger positional structure.
    """
    set_constant_plot_style()

    plot_df = signal_df.copy()
    plot_df = plot_df[plot_df[label_col].isin(label_order)].copy()

    heatmap_data = plot_df.pivot(
        index=label_col,
        columns="Age Group",
        values="attenuation"
    ).reindex(index=label_order, columns=AGE_ORDER)

    annot_data = heatmap_data.round(2).astype(str)

    vmax = np.nanmax(np.abs(heatmap_data.to_numpy()))
    if pd.isna(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=FIG_STYLE["cmap"],
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot_data,
        fmt="",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Real spread − shuffled spread"}
    )

    ax.set_title(
        title,
        fontsize=14,
        weight="bold",
        fontproperties=font_prop
    )

    ax.set_xlabel("Age group", fontsize=12, fontproperties=font_prop)
    ax.set_ylabel("", fontsize=12, fontproperties=font_prop)

    ax.set_xticklabels(
        [AGE_LABELS.get(int(x.get_text()), x.get_text()) for x in ax.get_xticklabels()],
        rotation=0,
        fontsize=11,
        fontproperties=font_prop
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=11,
        fontproperties=font_prop
    )

    plt.tight_layout()

    if filename_base is not None:
        save_publication_figure(fig, filename_base)

    plt.show()


# ------------------------------------------------------------
# Plot 3: Real vs shuffled overlay line plot
# ------------------------------------------------------------
def plot_real_vs_shuffled_overlay_lines(
    compare_df,
    label_col,
    selected_labels,
    section_order,
    section_display,
    title,
    filename_base=None,
    n_rows=2,
    n_cols=3,
    figsize=(15, 8)
):
    """
    Overlay plot:
        x-axis = Age group
        color/line group = Section
        solid line = real
        dashed line = shuffled

    This is useful as a supplementary figure.
    """
    set_constant_plot_style()

    plot_df = compare_df.copy()
    plot_df = plot_df[plot_df[label_col].isin(selected_labels)].copy()

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
        sub_label = plot_df[plot_df[label_col] == label].copy()

        for section in section_order:
            sub_sec = sub_label[sub_label["section"] == section].sort_values("Age Group")

            if sub_sec.empty:
                continue

            section_label = section_display.get(section, section)

            ax.plot(
                sub_sec["Age Group"].astype(int),
                sub_sec["log_odds_real"],
                marker="o",
                linewidth=2,
                linestyle="-",
                label=f"{section_label} real"
            )

            ax.plot(
                sub_sec["Age Group"].astype(int),
                sub_sec["log_odds_shuf"],
                marker="o",
                linewidth=1.6,
                linestyle="--",
                alpha=0.75,
                label=f"{section_label} shuffled"
            )

        ax.axhline(0, linestyle="--", color="gray", linewidth=1)

        ax.set_title(label, fontsize=13, fontproperties=font_prop)

        ax.set_xticks(AGE_ORDER)
        ax.set_xticklabels(
            [AGE_LABELS[i] for i in AGE_ORDER],
            fontsize=10,
            fontproperties=font_prop
        )

        ax.set_xlabel("Age group", fontsize=11, fontproperties=font_prop)

        if idx % n_cols == 0:
            ax.set_ylabel("Log-odds", fontsize=11, fontproperties=font_prop)

        ax.tick_params(bottom=True, left=True)

    for idx in range(len(selected_labels), len(axes)):
        fig.delaxes(axes[idx])

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            title="Section / null model",
            loc="center left",
            bbox_to_anchor=(0.87, 0.5),
            frameon=False,
            fontsize=9,
            title_fontsize=10
        )

    fig.suptitle(
        title,
        fontsize=14,
        weight="bold",
        y=0.96,
        fontproperties=font_prop
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.94])

    if filename_base is not None:
        save_publication_figure(fig, filename_base)

    plt.show()


# ============================================================
# 17. RUN REAL VS SHUFFLED COMPARISONS: THEMES
# ============================================================

if "third" in theme_scheme_outputs and "shuf_third" in theme_scheme_outputs:

    theme_real_shuf_compare = make_real_vs_shuffle_comparison(
        real_df=theme_scheme_outputs["third"]["age_results"],
        shuf_df=theme_scheme_outputs["shuf_third"]["age_results"],
        label_col="theme",
        real_scheme_name="third",
        shuf_scheme_name="shuf_third",
        section_order=["Section 1", "Section 2", "Section 3"]
    )

    theme_positional_signal = make_positional_signal_table(
        theme_real_shuf_compare,
        label_col="theme"
    )

    if SAVE_OUTPUTS:
        theme_real_shuf_compare.to_csv(
            os.path.join(NULL_FIG_DIR, "theme_real_vs_shuffled_delta_table.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        theme_positional_signal.to_csv(
            os.path.join(NULL_FIG_DIR, "theme_positional_signal_attenuation_table.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    # Main-paper candidate: delta heatmap
    plot_delta_heatmap_facets(
        compare_df=theme_real_shuf_compare,
        label_col="theme",
        label_order=THEMES,
        section_order=["Section 1", "Section 2", "Section 3"],
        section_display={
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        title=(
            "Theme positional effects beyond shuffled-order null model\n"
            "Delta = real third-section log-odds − shuffled third-section log-odds"
        ),
        filename_base="FIG_theme_real_minus_shuffled_delta_heatmaps",
        n_cols=3,
        figsize_per_panel=(4.2, 3.1)
    )

    # Main-paper or supplement: positional attenuation
    plot_positional_attenuation_heatmap(
        signal_df=theme_positional_signal,
        label_col="theme",
        label_order=THEMES,
        title=(
            "Theme positional signal attenuated by sentence-order shuffling\n"
            "Positive values indicate stronger section differentiation in real note order"
        ),
        filename_base="FIG_theme_positional_signal_attenuation",
        figsize=(8.5, 4.8)
    )

    # Supplement: direct overlay
    plot_real_vs_shuffled_overlay_lines(
        compare_df=theme_real_shuf_compare,
        label_col="theme",
        selected_labels=THEMES,
        section_order=["Section 1", "Section 2", "Section 3"],
        section_display={
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        title=(
            "Theme effects in real versus shuffled section order\n"
            "Solid = real order, dashed = shuffled-order null"
        ),
        filename_base="SUPP_theme_real_vs_shuffled_overlay_lines",
        n_rows=2,
        n_cols=3,
        figsize=(16, 8)
    )

else:
    print(
        "Theme real-vs-shuffled comparison skipped: "
        "need theme_scheme_outputs['third'] and theme_scheme_outputs['shuf_third']."
    )


# ============================================================
# 18. RUN REAL VS SHUFFLED COMPARISONS: SENTIMENTS
# ============================================================

if "third" in sentiment_scheme_outputs and "shuf_third" in sentiment_scheme_outputs:

    sentiment_real_shuf_compare = make_real_vs_shuffle_comparison(
        real_df=sentiment_scheme_outputs["third"]["age_results"],
        shuf_df=sentiment_scheme_outputs["shuf_third"]["age_results"],
        label_col="sentiment",
        real_scheme_name="third",
        shuf_scheme_name="shuf_third",
        section_order=["Section 1", "Section 2", "Section 3"]
    )

    sentiment_positional_signal = make_positional_signal_table(
        sentiment_real_shuf_compare,
        label_col="sentiment"
    )

    if SAVE_OUTPUTS:
        sentiment_real_shuf_compare.to_csv(
            os.path.join(NULL_FIG_DIR, "sentiment_real_vs_shuffled_delta_table.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        sentiment_positional_signal.to_csv(
            os.path.join(NULL_FIG_DIR, "sentiment_positional_signal_attenuation_table.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    # Main-paper candidate for selected sentiments
    plot_delta_heatmap_facets(
        compare_df=sentiment_real_shuf_compare,
        label_col="sentiment",
        label_order=LINEPLOT_SENTIMENTS,
        section_order=["Section 1", "Section 2", "Section 3"],
        section_display={
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        title=(
            "Sentiment positional effects beyond shuffled-order null model\n"
            "Delta = real third-section log-odds − shuffled third-section log-odds"
        ),
        filename_base="FIG_sentiment_real_minus_shuffled_delta_heatmaps_selected",
        n_cols=3,
        figsize_per_panel=(4.2, 3.1)
    )

    # Full sentiment attenuation heatmap
    plot_positional_attenuation_heatmap(
        signal_df=sentiment_positional_signal,
        label_col="sentiment",
        label_order=SENTIMENTS,
        title=(
            "Sentiment positional signal attenuated by sentence-order shuffling\n"
            "Positive values indicate stronger section differentiation in real note order"
        ),
        filename_base="FIG_sentiment_positional_signal_attenuation_all",
        figsize=(8.5, 8.5)
    )

    # Supplement overlay for selected sentiments
    plot_real_vs_shuffled_overlay_lines(
        compare_df=sentiment_real_shuf_compare,
        label_col="sentiment",
        selected_labels=LINEPLOT_SENTIMENTS,
        section_order=["Section 1", "Section 2", "Section 3"],
        section_display={
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        title=(
            "Sentiment effects in real versus shuffled section order\n"
            "Solid = real order, dashed = shuffled-order null"
        ),
        filename_base="SUPP_sentiment_real_vs_shuffled_overlay_lines_selected",
        n_rows=2,
        n_cols=3,
        figsize=(16, 8)
    )

else:
    print(
        "Sentiment real-vs-shuffled comparison skipped: "
        "need sentiment_scheme_outputs['third'] and sentiment_scheme_outputs['shuf_third']."
    )
    
#%%
