# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:50:51 2026

@author: Jae Bin Park
"""

# -*- coding: utf-8 -*-
"""
Markov theme & sentiment transitions
- Section 1 -> 2 -> 3
- Lift vs base prevalence
- ΔLift (POS vs NEG/REST)
- Permutation p-values (cellwise + optional FWER via max-statistic)
- Supports:
    * binary comparisons (e.g., SEX: male vs female)
    * one-vs-rest comparisons (e.g., AGE2: each age group vs rest)
Created on Thu Dec  4 2025 (revised for SEX/AGE2 one-vs-rest)
@author: Jae Bin Park
"""

#%% ============================================================
#                         IMPORTS
#===============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
import ast


#%% ============================================================
#          NEW: TABLES STORED IN WORKSPACE (NO EXPORT)
#===============================================================

# After running, you'll have all tables here:
# TABLES[analysis_type][comparison][transition] = {"delta":..., "annot":..., "p_cell":..., "p_max":...}
TABLES = {"SENTIMENT": {}, "THEME": {}}

def _ensure_keys(d, *keys):
    cur = d
    for k in keys:
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    return cur

def store_tables(analysis_type, comparison, transition, delta_df, annot_df, p_cell_df=None, p_max_df=None):
    """
    Stores the DataFrames in the global TABLES dict (workspace).
    """
    bucket = _ensure_keys(TABLES, analysis_type, comparison)
    bucket[transition] = {
        "delta": delta_df.copy(),
        "annot": annot_df.copy(),
        "p_cell": None if p_cell_df is None else p_cell_df.copy(),
        "p_max": None if p_max_df is None else p_max_df.copy(),
    }


#%% ============================================================
#                    SHARED HELPERS / CORE
#===============================================================

def compute_transition(df, from_cols, to_cols, state_names):
    """
    Generic Markov transition from one section to next.

    Returns:
        counts_df : K x K DataFrame of co-occurrence counts
        prob_df   : K x K DataFrame of P(j | i)  (conditional on from-state present)
        base_from : Series length-K: # notes with from-state i in from-section
    """
    M_from = df[from_cols].to_numpy(int)  # N x K
    M_to   = df[to_cols].to_numpy(int)    # N x K

    counts = M_from.T @ M_to              # K x K
    base_from = M_from.sum(axis=0)        # length K

    counts_df = pd.DataFrame(counts, index=state_names, columns=state_names)
    base_from = pd.Series(base_from, index=state_names, name="N_from")

    prob = counts_df.div(base_from.replace(0, np.nan), axis=0)
    prob_df = prob.fillna(0.0)

    return counts_df, prob_df, base_from


def compute_prob_and_base(df, from_cols, to_cols, state_names):
    """
    Minimal helper for group comparisons.

    Returns:
        prob      : K x K np.array of P(j | i)
        base_prev : length-K np.array of base prevalence P(j in next section)
    """
    M_from = df[from_cols].to_numpy(int)  # N x K
    M_to   = df[to_cols].to_numpy(int)    # N x K

    counts    = M_from.T @ M_to           # K x K
    base_from = M_from.sum(axis=0)        # length K

    prob = counts / np.maximum(base_from[:, None], 1e-9)
    base_prev = M_to.mean(axis=0)

    return prob, base_prev


def delta_lift_masks(df, from_cols, to_cols, state_names, mask_pos, mask_neg):
    """
    Observed ΔLift = Lift(pos) - Lift(neg), where Lift = P(next=j | from=i) / P(next=j)

    mask_pos: boolean mask for positive group (e.g., AGE2==g)
    mask_neg: boolean mask for negative/rest group (e.g., AGE2!=g)
    """
    df_pos = df.loc[mask_pos].copy()
    df_neg = df.loc[mask_neg].copy()

    prob_pos, base_pos = compute_prob_and_base(df_pos, from_cols, to_cols, state_names)
    prob_neg, base_neg = compute_prob_and_base(df_neg, from_cols, to_cols, state_names)

    lift_pos = prob_pos / np.maximum(base_pos, 1e-9)
    lift_neg = prob_neg / np.maximum(base_neg, 1e-9)

    delta = lift_pos - lift_neg
    return pd.DataFrame(delta, index=state_names, columns=state_names)


def permutation_pvalues_masks(
    df,
    from_cols,
    to_cols,
    state_names,
    mask_pos,
    mask_neg,
    n_perm=5000,
    use_max_stat=False,
    random_state=None,
):
    """
    Permutation test for ΔLift between two groups defined by masks.
      ΔLift = Lift_pos - Lift_neg
      Lift  = P(next=j | from=i) / P(next=j)

    Permutes group membership while preserving group sizes (within pooled pos+neg set).

    Returns:
        pvals_cell_df : K x K DataFrame of cellwise permutation p-values (two-sided)
        pvals_max_df  : K x K DataFrame of max-stat FWER p-values (if use_max_stat=True), else None
        delta_obs_df  : K x K DataFrame of observed ΔLift
    """
    rng = np.random.default_rng(random_state)

    df = df.copy()

    mask_pos = np.asarray(mask_pos, dtype=bool)
    mask_neg = np.asarray(mask_neg, dtype=bool)

    # --- Observed ΔLift ---
    df_pos = df.loc[mask_pos]
    df_neg = df.loc[mask_neg]

    prob_pos, base_pos = compute_prob_and_base(df_pos, from_cols, to_cols, state_names)
    prob_neg, base_neg = compute_prob_and_base(df_neg, from_cols, to_cols, state_names)

    lift_pos = prob_pos / np.maximum(base_pos, 1e-9)
    lift_neg = prob_neg / np.maximum(base_neg, 1e-9)

    delta_obs = lift_pos - lift_neg
    K = len(state_names)

    # --- Permutations ---
    idx_all = np.arange(len(df))
    pos_idx = idx_all[mask_pos]
    neg_idx = idx_all[mask_neg]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    pooled_idx = np.concatenate([pos_idx, neg_idx])
    pooled_df = df.iloc[pooled_idx].copy()
    n_pool = len(pooled_df)

    M_from_all = pooled_df[from_cols].to_numpy(int)  # n_pool x K
    M_to_all   = pooled_df[to_cols].to_numpy(int)    # n_pool x K

    deltas = np.zeros((n_perm, K, K), dtype=float)

    for b in range(n_perm):
        perm = rng.permutation(n_pool)
        perm_pos = perm[:n_pos]
        perm_neg = perm[n_pos:(n_pos + n_neg)]

        # POS
        M_from_pos = M_from_all[perm_pos]
        M_to_pos   = M_to_all[perm_pos]
        counts_pos = M_from_pos.T @ M_to_pos
        base_from_pos = M_from_pos.sum(axis=0)
        prob_pos_b = counts_pos / np.maximum(base_from_pos[:, None], 1e-9)
        base_prev_pos = M_to_pos.mean(axis=0)
        lift_pos_b = prob_pos_b / np.maximum(base_prev_pos, 1e-9)

        # NEG
        M_from_neg = M_from_all[perm_neg]
        M_to_neg   = M_to_all[perm_neg]
        counts_neg = M_from_neg.T @ M_to_neg
        base_from_neg = M_from_neg.sum(axis=0)
        prob_neg_b = counts_neg / np.maximum(base_from_neg[:, None], 1e-9)
        base_prev_neg = M_to_neg.mean(axis=0)
        lift_neg_b = prob_neg_b / np.maximum(base_prev_neg, 1e-9)

        deltas[b] = lift_pos_b - lift_neg_b

    abs_deltas = np.abs(deltas)
    abs_obs = np.abs(delta_obs)

    pvals_cell = (abs_deltas >= abs_obs[None, :, :]).mean(axis=0)
    pvals_cell_df = pd.DataFrame(pvals_cell, index=state_names, columns=state_names)
    delta_obs_df  = pd.DataFrame(delta_obs, index=state_names, columns=state_names)

    pvals_max_df = None
    if use_max_stat:
        max_abs = abs_deltas.reshape(n_perm, -1).max(axis=1)
        pvals_max = np.zeros((K, K), dtype=float)
        for i in range(K):
            for j in range(K):
                thresh = abs_obs[i, j]
                pvals_max[i, j] = (max_abs >= thresh).mean()
        pvals_max_df = pd.DataFrame(pvals_max, index=state_names, columns=state_names)

    return pvals_cell_df, pvals_max_df, delta_obs_df


def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def build_annot_matrix(values_df, pvals_df, decimals=2, show_ns=False):
    annot = pd.DataFrame("", index=values_df.index, columns=values_df.columns)
    for r in values_df.index:
        for c in values_df.columns:
            v = values_df.loc[r, c]
            p = pvals_df.loc[r, c] if (r in pvals_df.index and c in pvals_df.columns) else np.nan
            if pd.isna(v):
                annot.loc[r, c] = ""
                continue
            stars = p_to_stars(p)
            if (stars == "") and show_ns and (not pd.isna(p)):
                stars = "ns"
            annot.loc[r, c] = f"{v:.{decimals}f}{stars}"
    return annot


def plot_delta_heatmap(delta_df, annot_df, title, xlabel, ylabel, figsize=(9, 7)):
    plt.figure(figsize=figsize)
    sns.heatmap(
        delta_df,
        annot=annot_df,
        fmt="",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        linecolor="gray"
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


#%% ============================================================
#                      THEME FEATURES
#===============================================================

raw = pd.read_pickle(r'C:\Users\Jae Bin Park\raw.pkl')

def create_section_presence_vector(df, target_tokens, keyword_cols, label):
    tqdm.pandas(desc=f"Checking token presence for: {label}")

    def check_row(row):
        presence = []
        for col in keyword_cols:
            section = row.get(col, None)
            if not isinstance(section, list):
                presence.append(0)
                continue
            tokens = [t for t, _ in section]
            has_token = any(token in tokens for token in target_tokens)
            presence.append(int(has_token))
        return presence

    df[f"{label}_presence"] = df.progress_apply(check_row, axis=1)
    return df

sorry = ["미안", "죄송", "용서", "잘못", "후회", "죄", "책임", "민폐", "반성"]
loveandgratitude = ["사랑", "고맙", "감사", "고생", "수고", "보고싶", "애틋", "소중", "그리"]
burdensome = ["버겁", "부담", "짐", "감당", "힘들", "무겁", "지치", "헷갈"]
despair = ["포기", "좌절", "죽", "끝", "아무것", "헛되", "무의미", "잊히", "없어지", "그만두"]
pmaffairs = ["부탁", "정리", "남기", "처리", "보험", "은행", "장례", "통장", "유서"]

section_columns_all = [
    "tokenizedkluekeywordsentencetransformer_1st_section",
    "tokenizedkluekeywordsentencetransformer_2nd_section",
    "tokenizedkluekeywordsentencetransformer_3rd_section",
    "tokenizedkluekeywordsentencetransformer_Intro",
    "tokenizedkluekeywordsentencetransformer_Body",
    "tokenizedkluekeywordsentencetransformer_Conclusion"
]
for col in section_columns_all:
    if col in raw.columns:
        raw[col] = raw[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

sec_cols = [
    "tokenizedkluekeywordsentencetransformer_1st_section",
    "tokenizedkluekeywordsentencetransformer_2nd_section",
    "tokenizedkluekeywordsentencetransformer_3rd_section"
]

raw = create_section_presence_vector(raw, sorry, sec_cols, label="sorrysentence")
raw = create_section_presence_vector(raw, loveandgratitude, sec_cols, label='loveandgratitudesentence')
raw = create_section_presence_vector(raw, burdensome, sec_cols, label='burdensomesentence')
raw = create_section_presence_vector(raw, despair, sec_cols, label='despairsentence')
raw = create_section_presence_vector(raw, pmaffairs, sec_cols, label='pmaffairssentence')

morethan3 = raw[raw["too_short"] == False].copy()

presence_columns = {
    "Sorry and Shame": "sorrysentence_presence",
    "Love and Gratitude": "loveandgratitudesentence_presence",
    "Burden": "burdensomesentence_presence",
    "Despair": "despairsentence_presence",
    "Post-mortem Affairs": "pmaffairssentence_presence"
}

themes = list(presence_columns.keys())

def get_theme_section_vector(row, section_idx):
    return [row[presence_columns[t]][section_idx] for t in themes]

morethan3["section1_theme_vector"] = morethan3.apply(lambda r: get_theme_section_vector(r, 0), axis=1)
morethan3["section2_theme_vector"] = morethan3.apply(lambda r: get_theme_section_vector(r, 1), axis=1)
morethan3["section3_theme_vector"] = morethan3.apply(lambda r: get_theme_section_vector(r, 2), axis=1)

morethan3["full_theme_vector"] = morethan3.apply(
    lambda r: r["section1_theme_vector"] + r["section2_theme_vector"] + r["section3_theme_vector"],
    axis=1
)

theme_cols_15 = [f"Section {i+1} {t}" for i in range(3) for t in themes]

theme_matrix = pd.DataFrame(
    morethan3["full_theme_vector"].tolist(),
    index=morethan3.index,
    columns=theme_cols_15
)
morethan3 = morethan3.drop(columns=[c for c in theme_cols_15 if c in morethan3.columns], errors="ignore")
morethan3 = pd.concat([morethan3, theme_matrix], axis=1)

theme_sec1 = [f"Section 1 {t}" for t in themes]
theme_sec2 = [f"Section 2 {t}" for t in themes]
theme_sec3 = [f"Section 3 {t}" for t in themes]


#%% ============================================================
#                    SENTIMENT FEATURES
#===============================================================

morethan3rawsentiment = pd.read_pickle(r"C:\Users\Jae Bin Park\morethan3rawsentiment.pkl")
df_sent = morethan3rawsentiment.copy()

def create_sentiment_presence_vector(df, sentiment_labels, section_cols, label_prefix):
    tqdm.pandas(desc=f"Processing sentiment presence for: {label_prefix}")

    def check_row(row):
        presence = []
        for col in section_cols:
            entry = row.get(col, None)
            if isinstance(entry, dict):
                section_labels = entry.get("labels", [])
                has_any = int(any(lbl in sentiment_labels for lbl in section_labels))
            else:
                has_any = 0
            presence.append(has_any)
        return presence

    df[f"{label_prefix}_presence"] = df.progress_apply(check_row, axis=1)
    return df

section_cols_sent = ["1st_section_sentiment", "2nd_section_sentiment", "3rd_section_sentiment"]

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

sentiment_targets_kor = {
    "절망": "Despair",
    "패배/자기혐오": "Defeat",
    "힘듦/지침": "Exhaustion",
    "불안/걱정": "Anxious",
    "안타까움/실망": "Disappointment",
    "화남/분노": "Anger",
    "증오/혐오": "Hatred",
    "어이없음": "Resentment",
    "슬픔": "Sadness",
    "서러움": "Sorrow",
    "고마움": "Gratitude",
    "흐뭇함(귀여움/예쁨)": "Affection",
    "행복": "Happiness",
    "안심/신뢰": "Relief",
    "없음": "Neutral"
}

for kor_label in sentiment_targets_kor.keys():
    df_sent = create_sentiment_presence_vector(
        df_sent,
        sentiment_labels=[kor_label],
        section_cols=section_cols_sent,
        label_prefix=kor_label
    )

sentiments = list(sentiment_presence_columns.keys())
sections = ["Section 1", "Section 2", "Section 3"]
sentiment_feature_names = [f"{sec} {s}" for sec in sections for s in sentiments]

def get_sentiment_vector_by_section(row, section_idx):
    return [row[sentiment_presence_columns[s]][section_idx] for s in sentiments]

df_sent["section1_sentiment_vector"] = df_sent.apply(lambda r: get_sentiment_vector_by_section(r, 0), axis=1)
df_sent["section2_sentiment_vector"] = df_sent.apply(lambda r: get_sentiment_vector_by_section(r, 1), axis=1)
df_sent["section3_sentiment_vector"] = df_sent.apply(lambda r: get_sentiment_vector_by_section(r, 2), axis=1)

df_sent["full_sentiment_vector"] = df_sent.apply(
    lambda r: r["section1_sentiment_vector"] + r["section2_sentiment_vector"] + r["section3_sentiment_vector"],
    axis=1
)

sent_matrix = pd.DataFrame(
    df_sent["full_sentiment_vector"].tolist(),
    index=df_sent.index,
    columns=sentiment_feature_names
)
df_sent = df_sent.drop(columns=[c for c in sentiment_feature_names if c in df_sent.columns], errors="ignore")
df_sent = pd.concat([df_sent, sent_matrix], axis=1)

sent_sec1 = [f"Section 1 {s}" for s in sentiments]
sent_sec2 = [f"Section 2 {s}" for s in sentiments]
sent_sec3 = [f"Section 3 {s}" for s in sentiments]


#%% ============================================================
#           RUN COMPARISONS: SEX and AGE2 (ONE-VS-REST)
#===============================================================

sns.set(style="white", context="notebook", font_scale=1.1)

USE_MAXSTAT_FOR_STARS = False
N_PERM = 5000
DO_FWER = True
MIN_POS = 30
MIN_NEG = 30


# ------------------------------------------------------------
# A) SENTIMENT: SEX (Male vs Female)
# ------------------------------------------------------------

df_sent_sex = df_sent[df_sent["SEX"].isin([1, 2])].copy()

mask_male = df_sent_sex["SEX"] == 1
mask_fem  = df_sent_sex["SEX"] == 2

print("\n[SENTIMENT | SEX] n_male =", mask_male.sum(), " n_female =", mask_fem.sum())

if mask_male.sum() >= MIN_POS and mask_fem.sum() >= MIN_NEG:
    delta12_sent_sex = delta_lift_masks(df_sent_sex, sent_sec1, sent_sec2, sentiments, mask_male, mask_fem)
    delta23_sent_sex = delta_lift_masks(df_sent_sex, sent_sec2, sent_sec3, sentiments, mask_male, mask_fem)

    p12_cell, p12_max, _ = permutation_pvalues_masks(
        df_sent_sex, sent_sec1, sent_sec2, sentiments,
        mask_pos=mask_male, mask_neg=mask_fem,
        n_perm=N_PERM, use_max_stat=DO_FWER, random_state=111
    )
    p23_cell, p23_max, _ = permutation_pvalues_masks(
        df_sent_sex, sent_sec2, sent_sec3, sentiments,
        mask_pos=mask_male, mask_neg=mask_fem,
        n_perm=N_PERM, use_max_stat=DO_FWER, random_state=222
    )

    p12_for_stars = p12_max if (USE_MAXSTAT_FOR_STARS and p12_max is not None) else p12_cell
    p23_for_stars = p23_max if (USE_MAXSTAT_FOR_STARS and p23_max is not None) else p23_cell

    annot12 = build_annot_matrix(delta12_sent_sex, p12_for_stars, decimals=2, show_ns=False)
    annot23 = build_annot_matrix(delta23_sent_sex, p23_for_stars, decimals=2, show_ns=False)

    suffix = " (FWER max-stat)" if USE_MAXSTAT_FOR_STARS else " (cellwise)"

    plot_delta_heatmap(
        delta12_sent_sex, annot12,
        title=f"Sentiment ΔLift (Male − Female): Section 1 → Section 2{suffix}",
        xlabel="Section 2 Sentiment", ylabel="Section 1 Sentiment", figsize=(10, 8)
    )
    # NEW: store to workspace
    store_tables("SENTIMENT", "SEX_MaleMinusFemale", "S1_to_S2",
                 delta12_sent_sex, annot12, p12_cell, p12_max)

    plot_delta_heatmap(
        delta23_sent_sex, annot23,
        title=f"Sentiment ΔLift (Male − Female): Section 2 → Section 3{suffix}",
        xlabel="Section 3 Sentiment", ylabel="Section 2 Sentiment", figsize=(10, 8)
    )
    # NEW: store to workspace
    store_tables("SENTIMENT", "SEX_MaleMinusFemale", "S2_to_S3",
                 delta23_sent_sex, annot23, p23_cell, p23_max)

else:
    print("[SENTIMENT | SEX] Skipped due to small n.")


# ------------------------------------------------------------
# B) SENTIMENT: AGE2 one-vs-rest (1..5 vs rest)
# ------------------------------------------------------------

df_sent_age = df_sent[df_sent["AGE2"].isin([1, 2, 3, 4, 5])].copy()

age_groups = [1, 2, 3, 4, 5]
age_label_map = {1: "≤18", 2: "19–34", 3: "35–49", 4: "50–64", 5: "65+"}

for g in age_groups:
    mask_pos = df_sent_age["AGE2"] == g
    mask_neg = df_sent_age["AGE2"] != g

    print(f"\n[SENTIMENT | AGE2 {g} vs rest] n_pos={mask_pos.sum()} n_neg={mask_neg.sum()}")

    if mask_pos.sum() < MIN_POS or mask_neg.sum() < MIN_NEG:
        print("  -> skipped (small n)")
        continue

    delta12 = delta_lift_masks(df_sent_age, sent_sec1, sent_sec2, sentiments, mask_pos, mask_neg)
    delta23 = delta_lift_masks(df_sent_age, sent_sec2, sent_sec3, sentiments, mask_pos, mask_neg)

    p12_cell, p12_max, _ = permutation_pvalues_masks(
        df_sent_age, sent_sec1, sent_sec2, sentiments,
        mask_pos=mask_pos, mask_neg=mask_neg,
        n_perm=N_PERM, use_max_stat=DO_FWER, random_state=1000 + g
    )
    p23_cell, p23_max, _ = permutation_pvalues_masks(
        df_sent_age, sent_sec2, sent_sec3, sentiments,
        mask_pos=mask_pos, mask_neg=mask_neg,
        n_perm=N_PERM, use_max_stat=DO_FWER, random_state=2000 + g
    )

    p12_for_stars = p12_max if (USE_MAXSTAT_FOR_STARS and p12_max is not None) else p12_cell
    p23_for_stars = p23_max if (USE_MAXSTAT_FOR_STARS and p23_max is not None) else p23_cell

    annot12 = build_annot_matrix(delta12, p12_for_stars, decimals=2, show_ns=False)
    annot23 = build_annot_matrix(delta23, p23_for_stars, decimals=2, show_ns=False)

    suffix = " (FWER max-stat)" if USE_MAXSTAT_FOR_STARS else " (cellwise)"

    plot_delta_heatmap(
        delta12, annot12,
        title=f"Sentiment ΔLift (AGE2 {g} {age_label_map.get(g,'')} − Rest): Section 1 → Section 2{suffix}",
        xlabel="Section 2 Sentiment", ylabel="Section 1 Sentiment", figsize=(10, 8)
    )
    store_tables("SENTIMENT", f"AGE2_{g}_{age_label_map.get(g,'')}_MinusRest", "S1_to_S2",
                 delta12, annot12, p12_cell, p12_max)

    plot_delta_heatmap(
        delta23, annot23,
        title=f"Sentiment ΔLift (AGE2 {g} {age_label_map.get(g,'')} − Rest): Section 2 → Section 3{suffix}",
        xlabel="Section 3 Sentiment", ylabel="Section 2 Sentiment", figsize=(10, 8)
    )
    store_tables("SENTIMENT", f"AGE2_{g}_{age_label_map.get(g,'')}_MinusRest", "S2_to_S3",
                 delta23, annot23, p23_cell, p23_max)


# ------------------------------------------------------------
# C) THEME: SEX (Male vs Female)
# ------------------------------------------------------------

df_theme_sex = morethan3[morethan3["SEX"].isin([1, 2])].copy()

mask_male = df_theme_sex["SEX"] == 1
mask_fem  = df_theme_sex["SEX"] == 2

print("\n[THEME | SEX] n_male =", mask_male.sum(), " n_female =", mask_fem.sum())

if mask_male.sum() >= MIN_POS and mask_fem.sum() >= MIN_NEG:
    delta12_theme_sex = delta_lift_masks(df_theme_sex, theme_sec1, theme_sec2, themes, mask_male, mask_fem)
    delta23_theme_sex = delta_lift_masks(df_theme_sex, theme_sec2, theme_sec3, themes, mask_male, mask_fem)

    p12_cell, p12_max, _ = permutation_pvalues_masks(
        df_theme_sex, theme_sec1, theme_sec2, themes,
        mask_pos=mask_male, mask_neg=mask_fem,
        n_perm=N_PERM, use_max_stat=DO_FWER, random_state=333
    )
    p23_cell, p23_max, _ = permutation_pvalues_masks(
        df_theme_sex, theme_sec2, theme_sec3, themes,
        mask_pos=mask_male, mask_neg=mask_fem,
        n_perm=N_PERM, use_max_stat=DO_FWER, random_state=444
    )

    p12_for_stars = p12_max if (USE_MAXSTAT_FOR_STARS and p12_max is not None) else p12_cell
    p23_for_stars = p23_max if (USE_MAXSTAT_FOR_STARS and p23_max is not None) else p23_cell

    annot12 = build_annot_matrix(delta12_theme_sex, p12_for_stars, decimals=2, show_ns=False)
    annot23 = build_annot_matrix(delta23_theme_sex, p23_for_stars, decimals=2, show_ns=False)

    suffix = " (FWER max-stat)" if USE_MAXSTAT_FOR_STARS else " (cellwise)"

    plot_delta_heatmap(
        delta12_theme_sex, annot12,
        title=f"Theme ΔLift (Male − Female): Section 1 → Section 2{suffix}",
        xlabel="Section 2 Theme", ylabel="Section 1 Theme", figsize=(8, 6)
    )
    store_tables("THEME", "SEX_MaleMinusFemale", "S1_to_S2",
                 delta12_theme_sex, annot12, p12_cell, p12_max)

    plot_delta_heatmap(
        delta23_theme_sex, annot23,
        title=f"Theme ΔLift (Male − Female): Section 2 → Section 3{suffix}",
        xlabel="Section 3 Theme", ylabel="Section 2 Theme", figsize=(8, 6)
    )
    store_tables("THEME", "SEX_MaleMinusFemale", "S2_to_S3",
                 delta23_theme_sex, annot23, p23_cell, p23_max)

else:
    print("[THEME | SEX] Skipped due to small n.")


# ------------------------------------------------------------
# D) THEME: AGE2 one-vs-rest (1..5 vs rest)
# ------------------------------------------------------------

df_theme_age = morethan3[morethan3["AGE2"].isin([1, 2, 3, 4, 5])].copy()

for g in age_groups:
    mask_pos = df_theme_age["AGE2"] == g
    mask_neg = df_theme_age["AGE2"] != g

    print(f"\n[THEME | AGE2 {g} vs rest] n_pos={mask_pos.sum()} n_neg={mask_neg.sum()}")

    if mask_pos.sum() < MIN_POS or mask_neg.sum() < MIN_NEG:
        print("  -> skipped (small n)")
        continue

    delta12 = delta_lift_masks(df_theme_age, theme_sec1, theme_sec2, themes, mask_pos, mask_neg)
    delta23 = delta_lift_masks(df_theme_age, theme_sec2, theme_sec3, themes, mask_pos, mask_neg)

    p12_cell, p12_max, _ = permutation_pvalues_masks(
        df_theme_age, theme_sec1, theme_sec2, themes,
        mask_pos=mask_pos, mask_neg=mask_neg,
        n_perm=N_PERM, use_max_stat=DO_FWER, random_state=3000 + g
    )
    p23_cell, p23_max, _ = permutation_pvalues_masks(
        df_theme_age, theme_sec2, theme_sec3, themes,
        mask_pos=mask_pos, mask_neg=mask_neg,
        n_perm=N_PERM, use_max_stat=DO_FWER, random_state=4000 + g
    )

    p12_for_stars = p12_max if (USE_MAXSTAT_FOR_STARS and p12_max is not None) else p12_cell
    p23_for_stars = p23_max if (USE_MAXSTAT_FOR_STARS and p23_max is not None) else p23_cell

    annot12 = build_annot_matrix(delta12, p12_for_stars, decimals=2, show_ns=False)
    annot23 = build_annot_matrix(delta23, p23_for_stars, decimals=2, show_ns=False)

    suffix = " (FWER max-stat)" if USE_MAXSTAT_FOR_STARS else " (cellwise)"

    plot_delta_heatmap(
        delta12, annot12,
        title=f"Theme ΔLift (AGE2 {g} {age_label_map.get(g,'')} − Rest): Section 1 → Section 2{suffix}",
        xlabel="Section 2 Theme", ylabel="Section 1 Theme", figsize=(8, 6)
    )
    store_tables("THEME", f"AGE2_{g}_{age_label_map.get(g,'')}_MinusRest", "S1_to_S2",
                 delta12, annot12, p12_cell, p12_max)

    plot_delta_heatmap(
        delta23, annot23,
        title=f"Theme ΔLift (AGE2 {g} {age_label_map.get(g,'')} − Rest): Section 2 → Section 3{suffix}",
        xlabel="Section 3 Theme", ylabel="Section 2 Theme", figsize=(8, 6)
    )
    store_tables("THEME", f"AGE2_{g}_{age_label_map.get(g,'')}_MinusRest", "S2_to_S3",
                 delta23, annot23, p23_cell, p23_max)


#%% ============================================================
# AFTER RUN: HOW TO ACCESS IN WORKSPACE
#===============================================================
# Examples:
# TABLES["SENTIMENT"].keys()
# TABLES["SENTIMENT"]["SEX_MaleMinusFemale"].keys()
# TABLES["SENTIMENT"]["SEX_MaleMinusFemale"]["S1_to_S2"]["delta"]   # numeric ΔLift
# TABLES["SENTIMENT"]["SEX_MaleMinusFemale"]["S1_to_S2"]["annot"]   # "ΔLift + stars"
# TABLES["SENTIMENT"]["SEX_MaleMinusFemale"]["S1_to_S2"]["p_cell"]  # p-values
# TABLES["SENTIMENT"]["SEX_MaleMinusFemale"]["S1_to_S2"]["p_max"]   # max-stat p-values (if computed)

import pickle

in_path = r"C:\Users\Jae Bin Park\TABLES.pkl"
with open(in_path, "rb") as f:
    TABLES = pickle.load(f)
