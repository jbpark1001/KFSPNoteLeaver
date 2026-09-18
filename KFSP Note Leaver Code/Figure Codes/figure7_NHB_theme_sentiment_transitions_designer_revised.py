#%%
# -*- coding: utf-8 -*-
r"""
NHB-style main figure for structured theme and sentiment transitions.

This file is standalone: it loads the theme and sentiment dataframes from
pickle files, validates and aligns them, computes the statistics, and exports
the figures and source-data tables.

Default run:
    python NHB_theme_sentiment_transitions_designer_revised.py

Quick validation run:
    python NHB_theme_sentiment_transitions_designer_revised.py --n-perm 200

Use --help to override either dataframe path or the output directory.

This version:
1. Uses separate theme_df and sentiment_df.
2. Avoids collision between theme "Despair" and sentiment "Despair"
   by prefixing sentiment columns as "Sentiment Section ...".
3. Removes the schematic / panel a entirely.
4. Shows overall theme transitions.
5. Shows overall sentiment transitions.
6. Shows age-specific theme motifs separately for:
      opening -> middle
      middle -> ending
7. Shows age-specific sentiment motifs separately for:
      opening -> middle
      middle -> ending
8. Shows sex-contrast sentiment transitions in one row.
9. Shows sex-contrast theme transitions in a separate row below.
10. Uses the figure designer's muted rainbow palette.
11. Uses uniform filled circles and solid lines across motif series.
12. Adds light horizontal guides and a blue dotted zero reference line.
13. Makes the motif rows taller and exports standalone portrait panels.
14. Uses stars, not black rings, for significance in age-motif line plots.
15. Clips age-specific motif panels for visualization only.
    Original values are still saved in source-data CSVs.
"""

# ============================================================
# 0. Imports
# ============================================================

import argparse
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================================================
# 1. User settings
# ============================================================

DEFAULT_THEME_DATA_PATH = Path(
    r"C:\Users\Jae Bin Park\morethan3rawformainfigures"
)
DEFAULT_SENTIMENT_DATA_PATH = Path(
    r"C:\Users\Jae Bin Park\morethan3rawsentimentformainfigures"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "NHB_transition_figure"

# These globals are updated from command-line arguments in main().
OUT_DIR = str(DEFAULT_OUT_DIR)

SAVE_PREFIX = "NHB_theme_sentiment_transitions_combined_designer_layout"
OUTPUT_DPI = 600

# For debugging, use 200 or 500.
# For final manuscript-quality output, use 5000.
N_PERM = 5000

MIN_GROUP_N = 30
RANDOM_SEED = 20260706

# Visual clipping ranges for age-specific motif panels.
# These affect displayed y-axis only, not saved source data.
THEME_MOTIF_Y_CLIP = 3.0
SENTIMENT_MOTIF_Y_CLIP = 3.0


# ============================================================
# 2. Theme and sentiment states
# ============================================================

THEMES = [
    "Sorry and Shame",
    "Love and Gratitude",
    "Burden",
    "Despair",
    "Post-mortem Affairs"
]

THEME_LABELS = {
    "Sorry and Shame": "Sorry /\nshame",
    "Love and Gratitude": "Love /\ngratitude",
    "Burden": "Burden",
    "Despair": "Despair",
    "Post-mortem Affairs": "Affairs"
}


SENTIMENT_STATES = [
    "Defeat",
    "Exhaustion",
    "Despair",
    "Sorrow",
    "Neutral",
    "Sadness",
    "Disappointment",
    "Anxious",
    "Happiness",
    "Gratitude",
    "Relief"
]

SENTIMENT_LABELS = {
    "Defeat": "Defeat",
    "Exhaustion": "Exhaustion",
    "Despair": "Despair",
    "Sorrow": "Sorrow",
    "Neutral": "Neutral",
    "Sadness": "Sadness",
    "Disappointment": "Disappoint.",
    "Anxious": "Anxiety",
    "Happiness": "Happiness",
    "Gratitude": "Gratitude",
    "Relief": "Relief"
}


SENTIMENT_KOR_MAP = {
    "Defeat": ["패배/자기혐오"],
    "Exhaustion": ["힘듦/지침"],
    "Despair": ["절망"],
    "Sorrow": ["서러움"],
    "Neutral": ["없음"],
    "Sadness": ["슬픔"],
    "Disappointment": ["안타까움/실망"],
    "Anxious": ["불안/걱정"],
    "Happiness": ["행복"],
    "Gratitude": ["고마움"],
    "Relief": ["안심/신뢰"]
}


SENTIMENT_PRESENCE_COLS = {
    "Defeat": "패배/자기혐오_presence",
    "Exhaustion": "힘듦/지침_presence",
    "Despair": "절망_presence",
    "Sorrow": "서러움_presence",
    "Neutral": "없음_presence",
    "Sadness": "슬픔_presence",
    "Disappointment": "안타까움/실망_presence",
    "Anxious": "불안/걱정_presence",
    "Happiness": "행복_presence",
    "Gratitude": "고마움_presence",
    "Relief": "안심/신뢰_presence"
}


AGE_LABELS = {
    1: "≤18",
    2: "19–34",
    3: "35–49",
    4: "50–64",
    5: "≥65"
}


# ============================================================
# 3. Theme and sentiment columns
# ============================================================

THEME_SEC1_COLS = [f"Section 1 {t}" for t in THEMES]
THEME_SEC2_COLS = [f"Section 2 {t}" for t in THEMES]
THEME_SEC3_COLS = [f"Section 3 {t}" for t in THEMES]

ALL_THEME_COLS = THEME_SEC1_COLS + THEME_SEC2_COLS + THEME_SEC3_COLS


SENTIMENT_PREFIX = "Sentiment Section"

SENT_SEC1_COLS = [f"{SENTIMENT_PREFIX} 1 {s}" for s in SENTIMENT_STATES]
SENT_SEC2_COLS = [f"{SENTIMENT_PREFIX} 2 {s}" for s in SENTIMENT_STATES]
SENT_SEC3_COLS = [f"{SENTIMENT_PREFIX} 3 {s}" for s in SENTIMENT_STATES]

ALL_SENT_COLS = SENT_SEC1_COLS + SENT_SEC2_COLS + SENT_SEC3_COLS

UNPREFIXED_SENT_COLS = (
    [f"Section 1 {s}" for s in SENTIMENT_STATES] +
    [f"Section 2 {s}" for s in SENTIMENT_STATES] +
    [f"Section 3 {s}" for s in SENTIMENT_STATES]
)


# ============================================================
# 4. Motifs aligned with manuscript text
# ============================================================

# Same motif definitions are evaluated separately for S1->S2 and S2->S3.
# This makes early versus late transition-window differences directly visible.

THEME_MOTIFS = [
    ("Burden", "Burden"),
    ("Burden", "Post-mortem Affairs"),
    ("Love and Gratitude", "Sorry and Shame"),
    ("Sorry and Shame", "Love and Gratitude"),
    ("Post-mortem Affairs", "Post-mortem Affairs"),
    ("Sorry and Shame", "Sorry and Shame")
]

THEME_MOTIF_LABELS = {
    ("Burden", "Burden"): "Burden → burden",
    ("Burden", "Post-mortem Affairs"): "Burden → affairs",
    ("Love and Gratitude", "Sorry and Shame"): "Love → shame",
    ("Sorry and Shame", "Love and Gratitude"): "Shame → love",
    ("Post-mortem Affairs", "Post-mortem Affairs"): "Affairs → affairs",
    ("Sorry and Shame", "Sorry and Shame"): "Shame → shame"
}


SENTIMENT_MOTIFS = [
    ("Defeat", "Defeat"),
    ("Defeat", "Despair"),
    ("Defeat", "Exhaustion"),
    ("Despair", "Sorrow"),
    ("Sorrow", "Sorrow"),
    ("Neutral", "Neutral"),
    ("Despair", "Gratitude"),
    ("Despair", "Relief")
]

SENTIMENT_MOTIF_LABELS = {
    ("Defeat", "Defeat"): "Defeat → defeat",
    ("Defeat", "Despair"): "Defeat → despair",
    ("Defeat", "Exhaustion"): "Defeat → exhaustion",
    ("Despair", "Sorrow"): "Despair → sorrow",
    ("Sorrow", "Sorrow"): "Sorrow → sorrow",
    ("Neutral", "Neutral"): "Neutral → neutral",
    ("Despair", "Gratitude"): "Despair → gratitude",
    ("Despair", "Relief"): "Despair → relief"
}


# ============================================================
# 5. Motif-specific visual encoding
# ============================================================

# Exact solid colors sampled from the supplied designer PNG.
DESIGNER_BLUE = "#3C69A3"
DESIGNER_GREEN = "#5EA58A"
DESIGNER_RED = "#C44C38"
DESIGNER_YELLOW = "#F4E07B"
DESIGNER_TEAL = "#36819A"
DESIGNER_GRAY = "#707070"
DESIGNER_PURPLE = "#8A6FA8"
DESIGNER_ORANGE = "#D58A4B"
DESIGNER_GRID = "#C9C9C9"
DESIGNER_ZERO = "#396EA9"

# Six theme motifs use the six colors visible in the reference figure,
# in top-to-bottom palette order. All series intentionally share the
# same circle marker and solid line style.
THEME_MOTIF_COLORS = {
    "Burden → burden": DESIGNER_BLUE,
    "Burden → affairs": DESIGNER_GREEN,
    "Love → shame": DESIGNER_RED,
    "Shame → love": DESIGNER_YELLOW,
    "Affairs → affairs": DESIGNER_TEAL,
    "Shame → shame": DESIGNER_GRAY
}

THEME_MOTIF_MARKERS = {
    motif: "o" for motif in THEME_MOTIF_COLORS
}

THEME_MOTIF_LINESTYLES = {
    motif: "-" for motif in THEME_MOTIF_COLORS
}


# The sentiment panel has eight motifs, so two complementary muted hues
# extend the same designer palette while retaining clear separation.
SENTIMENT_MOTIF_COLORS = {
    "Defeat → defeat": DESIGNER_BLUE,
    "Defeat → despair": DESIGNER_GREEN,
    "Defeat → exhaustion": DESIGNER_RED,
    "Despair → sorrow": DESIGNER_YELLOW,
    "Sorrow → sorrow": DESIGNER_TEAL,
    "Neutral → neutral": DESIGNER_GRAY,
    "Despair → gratitude": DESIGNER_PURPLE,
    "Despair → relief": DESIGNER_ORANGE
}

SENTIMENT_MOTIF_MARKERS = {
    motif: "o" for motif in SENTIMENT_MOTIF_COLORS
}

SENTIMENT_MOTIF_LINESTYLES = {
    motif: "-" for motif in SENTIMENT_MOTIF_COLORS
}


# ============================================================
# 6. Visual style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.titlesize": 8.0,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 5.4,
    "figure.titlesize": 10,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

GRAY_TEXT = "#2B2B2B"
MID_GRAY = "#999999"

CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "muted_blue_white_red",
    ["#3B6EA8", "#F7F7F7", "#B64A3A"],
    N=256
)


# ============================================================
# 7. Data preparation
# ============================================================

def _is_binary_list_like(x, expected_len=3):
    return isinstance(x, (list, tuple, np.ndarray)) and len(x) >= expected_len


def stable_seed_from_label(label, base_seed=RANDOM_SEED):
    return int(base_seed + zlib.adler32(str(label).encode("utf-8")) % 100000)


def ensure_theme_section_columns(theme_df):
    df = theme_df.copy()

    if all(c in df.columns for c in ALL_THEME_COLS):
        for c in ALL_THEME_COLS:
            df[c] = df[c].fillna(0).astype(int)
        return df

    if "full_theme_vector" in df.columns:
        mat = pd.DataFrame(
            df["full_theme_vector"].apply(lambda v: list(map(int, v[:15]))).tolist(),
            index=df.index,
            columns=ALL_THEME_COLS
        )

        df = df.drop(
            columns=[c for c in ALL_THEME_COLS if c in df.columns],
            errors="ignore"
        )

        df = pd.concat([df, mat], axis=1)

        for c in ALL_THEME_COLS:
            df[c] = df[c].fillna(0).astype(int)

        return df

    missing = [c for c in ALL_THEME_COLS if c not in df.columns]

    raise ValueError(
        "Theme dataframe is missing required columns and cannot be rebuilt:\n"
        + "\n".join(missing)
    )


def ensure_sentiment_section_columns(sentiment_df):
    df = sentiment_df.copy()

    if all(c in df.columns for c in ALL_SENT_COLS):
        for c in ALL_SENT_COLS:
            df[c] = df[c].fillna(0).astype(int)
        return df

    if all(c in df.columns for c in UNPREFIXED_SENT_COLS):
        for sec in [1, 2, 3]:
            for state in SENTIMENT_STATES:
                old_col = f"Section {sec} {state}"
                new_col = f"{SENTIMENT_PREFIX} {sec} {state}"
                df[new_col] = df[old_col].fillna(0).astype(int)

        return df

    has_presence_cols = all(
        SENTIMENT_PRESENCE_COLS[s] in df.columns
        for s in SENTIMENT_STATES
    )

    if has_presence_cols:
        for state in SENTIMENT_STATES:
            presence_col = SENTIMENT_PRESENCE_COLS[state]

            for section_idx in range(3):
                new_col = f"{SENTIMENT_PREFIX} {section_idx + 1} {state}"
                df[new_col] = df[presence_col].apply(
                    lambda x: int(x[section_idx])
                    if _is_binary_list_like(x, expected_len=3)
                    else 0
                )

        for c in ALL_SENT_COLS:
            df[c] = df[c].fillna(0).astype(int)

        return df

    raw_section_cols = [
        "1st_section_sentiment",
        "2nd_section_sentiment",
        "3rd_section_sentiment"
    ]

    if all(c in df.columns for c in raw_section_cols):
        def has_label(entry, kor_targets):
            if not isinstance(entry, dict):
                return 0

            labels = entry.get("labels", [])

            if labels is None:
                return 0

            return int(any(lbl in labels for lbl in kor_targets))

        for state in SENTIMENT_STATES:
            kor_targets = SENTIMENT_KOR_MAP[state]

            for section_idx, raw_col in enumerate(raw_section_cols):
                new_col = f"{SENTIMENT_PREFIX} {section_idx + 1} {state}"
                df[new_col] = df[raw_col].apply(
                    lambda entry: has_label(entry, kor_targets)
                )

        for c in ALL_SENT_COLS:
            df[c] = df[c].fillna(0).astype(int)

        return df

    missing = [c for c in ALL_SENT_COLS if c not in df.columns]

    raise ValueError(
        "Sentiment dataframe is missing required sentiment columns and cannot be rebuilt:\n"
        + "\n".join(missing)
    )


def make_transition_input_df(theme_df, sentiment_df=None):
    theme_ready = ensure_theme_section_columns(theme_df)

    if sentiment_df is None:
        sentiment_ready = ensure_sentiment_section_columns(theme_ready)
    else:
        sentiment_ready = ensure_sentiment_section_columns(sentiment_df)

    sentiment_part = sentiment_ready[ALL_SENT_COLS].copy()

    common_index = theme_ready.index.intersection(sentiment_part.index)

    if len(common_index) == 0:
        raise ValueError(
            "Theme and sentiment dataframes have no overlapping index values."
        )

    merged = theme_ready.loc[common_index].copy()

    merged = merged.drop(
        columns=[c for c in ALL_SENT_COLS if c in merged.columns],
        errors="ignore"
    )

    merged = merged.join(sentiment_part.loc[common_index], how="inner")

    if "SEX" not in merged.columns:
        if "SEX" in sentiment_ready.columns:
            merged["SEX"] = sentiment_ready.loc[merged.index, "SEX"]
        else:
            raise ValueError("SEX column is missing.")

    if "AGE2" not in merged.columns:
        if "AGE2" in sentiment_ready.columns:
            merged["AGE2"] = sentiment_ready.loc[merged.index, "AGE2"]
        else:
            raise ValueError("AGE2 column is missing.")

    merged = merged[
        merged["SEX"].isin([1, 2]) &
        merged["AGE2"].isin([1, 2, 3, 4, 5])
    ].copy()

    for c in ALL_THEME_COLS + ALL_SENT_COLS:
        merged[c] = merged[c].fillna(0).astype(int)

    print("\nAnalytic transition dataframe created.")
    print(f"Theme rows:      {len(theme_ready):,}")
    print(f"Sentiment rows:  {len(sentiment_ready):,}")
    print(f"Merged rows:     {len(merged):,}")
    print(f"Sentiment states visualized: {len(SENTIMENT_STATES)}")

    return merged


def check_required_columns(df):
    required = ALL_THEME_COLS + ALL_SENT_COLS + ["SEX", "AGE2"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Analytic dataframe is missing required columns:\n"
            + "\n".join(missing)
        )


# ============================================================
# 8. Transition computations
# ============================================================

def compute_transition_probability(df, from_cols, to_cols, state_names):
    M_from = df[from_cols].to_numpy(dtype=int)
    M_to = df[to_cols].to_numpy(dtype=int)

    counts = M_from.T @ M_to
    base_from = M_from.sum(axis=0)
    base_next = M_to.mean(axis=0)

    prob = counts / np.maximum(base_from[:, None], 1e-12)

    counts_df = pd.DataFrame(
        counts,
        index=state_names,
        columns=state_names
    )

    prob_df = pd.DataFrame(
        prob,
        index=state_names,
        columns=state_names
    )

    base_from_s = pd.Series(
        base_from,
        index=state_names,
        name="N_from"
    )

    base_next_s = pd.Series(
        base_next,
        index=state_names,
        name="base_next"
    )

    return counts_df, prob_df, base_from_s, base_next_s


def compute_lift(df, from_cols, to_cols, state_names):
    _, prob_df, _, base_next = compute_transition_probability(
        df,
        from_cols,
        to_cols,
        state_names
    )

    lift = prob_df.to_numpy() / np.maximum(base_next.to_numpy()[None, :], 1e-12)

    return pd.DataFrame(
        lift,
        index=state_names,
        columns=state_names
    )


def compute_log2_lift(df, from_cols, to_cols, state_names):
    lift_df = compute_lift(
        df,
        from_cols,
        to_cols,
        state_names
    )

    return pd.DataFrame(
        np.log2(np.maximum(lift_df.to_numpy(), 1e-12)),
        index=state_names,
        columns=state_names
    )


def compute_delta_log2_lift(df, from_cols, to_cols, state_names, mask_pos, mask_neg):
    df_pos = df.loc[mask_pos].copy()
    df_neg = df.loc[mask_neg].copy()

    log2_pos = compute_log2_lift(
        df_pos,
        from_cols,
        to_cols,
        state_names
    )

    log2_neg = compute_log2_lift(
        df_neg,
        from_cols,
        to_cols,
        state_names
    )

    return log2_pos - log2_neg


def permutation_pvalues_delta_log2_lift(
    df,
    from_cols,
    to_cols,
    state_names,
    mask_pos,
    mask_neg,
    n_perm=5000,
    random_state=20260706,
    use_max_stat=False
):
    rng = np.random.default_rng(random_state)

    mask_pos = np.asarray(mask_pos, dtype=bool)
    mask_neg = np.asarray(mask_neg, dtype=bool)

    idx_all = np.arange(len(df))
    pos_idx = idx_all[mask_pos]
    neg_idx = idx_all[mask_neg]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos < MIN_GROUP_N or n_neg < MIN_GROUP_N:
        raise ValueError(f"Small group size: n_pos={n_pos}, n_neg={n_neg}")

    delta_obs_df = compute_delta_log2_lift(
        df,
        from_cols,
        to_cols,
        state_names,
        mask_pos,
        mask_neg
    )

    delta_obs = delta_obs_df.to_numpy()
    abs_obs = np.abs(delta_obs)

    pooled_idx = np.concatenate([pos_idx, neg_idx])
    pooled_df = df.iloc[pooled_idx].copy().reset_index(drop=True)

    M_from_all = pooled_df[from_cols].to_numpy(dtype=int)
    M_to_all = pooled_df[to_cols].to_numpy(dtype=int)

    K = len(state_names)
    deltas = np.zeros((n_perm, K, K), dtype=float)

    def _log2_lift_from_matrices(M_from, M_to):
        counts = M_from.T @ M_to
        base_from = M_from.sum(axis=0)
        prob = counts / np.maximum(base_from[:, None], 1e-12)
        base_next = M_to.mean(axis=0)
        lift = prob / np.maximum(base_next[None, :], 1e-12)
        return np.log2(np.maximum(lift, 1e-12))

    for b in range(n_perm):
        perm = rng.permutation(len(pooled_df))
        perm_pos = perm[:n_pos]
        perm_neg = perm[n_pos:n_pos + n_neg]

        log2_pos = _log2_lift_from_matrices(
            M_from_all[perm_pos],
            M_to_all[perm_pos]
        )

        log2_neg = _log2_lift_from_matrices(
            M_from_all[perm_neg],
            M_to_all[perm_neg]
        )

        deltas[b] = log2_pos - log2_neg

    abs_deltas = np.abs(deltas)
    p_cell = (abs_deltas >= abs_obs[None, :, :]).mean(axis=0)

    p_cell_df = pd.DataFrame(
        p_cell,
        index=state_names,
        columns=state_names
    )

    p_max_df = None

    if use_max_stat:
        max_abs = abs_deltas.reshape(n_perm, -1).max(axis=1)
        p_max = np.zeros((K, K), dtype=float)

        for i in range(K):
            for j in range(K):
                p_max[i, j] = (max_abs >= abs_obs[i, j]).mean()

        p_max_df = pd.DataFrame(
            p_max,
            index=state_names,
            columns=state_names
        )

    return delta_obs_df, p_cell_df, p_max_df


def compute_age_motif_table(
    df,
    from_cols,
    to_cols,
    state_names,
    motif_pairs,
    motif_labels,
    transition_label,
    n_perm=5000
):
    """
    Computes age group vs rest contrasts for motif pairs for one transition window.
    """
    rows = []

    df_age = df[df["AGE2"].isin([1, 2, 3, 4, 5])].copy()

    for age in [1, 2, 3, 4, 5]:
        console_transition = transition_label.replace("→", "->")
        print(
            f"    {console_transition}: age group {age} "
            f"({age}/5; {n_perm:,} permutations)",
            flush=True
        )

        mask_pos = df_age["AGE2"].eq(age).to_numpy()
        mask_neg = ~mask_pos

        if mask_pos.sum() < MIN_GROUP_N or mask_neg.sum() < MIN_GROUP_N:
            continue

        seed_label = f"{transition_label}_{age}_{state_names[0]}"
        delta_df, p_df, _ = permutation_pvalues_delta_log2_lift(
            df_age,
            from_cols,
            to_cols,
            state_names,
            mask_pos=mask_pos,
            mask_neg=mask_neg,
            n_perm=n_perm,
            random_state=stable_seed_from_label(seed_label),
            use_max_stat=False
        )

        for source, target in motif_pairs:
            motif_label = motif_labels.get((source, target), f"{source} → {target}")

            rows.append({
                "transition": transition_label,
                "age": age,
                "age_label": AGE_LABELS[age],
                "source": source,
                "target": target,
                "motif": motif_label,
                "delta_log2_lift": delta_df.loc[source, target],
                "p": p_df.loc[source, target],
                "significant": p_df.loc[source, target] < 0.05
            })

    return pd.DataFrame(rows)


# ============================================================
# 9. Plotting helpers
# ============================================================

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_attached_colorbar(ax, norm, cmap, label, size="4%", pad=0.18):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, cax=cax)

    cbar.set_label(label, fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)

    return cbar


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


def draw_heatmap(
    ax,
    data_df,
    title,
    state_labels,
    vlim=None,
    significance_df=None,
    show_ylabels=True
):
    values = data_df.to_numpy(dtype=float)

    if vlim is None:
        max_abs = np.nanpercentile(np.abs(values), 95)
        max_abs = max(max_abs, 0.5)
    else:
        max_abs = vlim

    norm = TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0,
        vmax=max_abs
    )

    im = ax.imshow(
        values,
        cmap=CMAP_DIVERGING,
        norm=norm,
        aspect="equal"
    )

    K = values.shape[0]

    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))

    ax.set_xticklabels(
        [state_labels[t] for t in data_df.columns],
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    if show_ylabels:
        ax.set_yticklabels([state_labels[t] for t in data_df.index])
    else:
        ax.set_yticklabels([])

    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.8)

    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if significance_df is not None:
        for i, row in enumerate(data_df.index):
            for j, col in enumerate(data_df.columns):
                p = significance_df.loc[row, col]

                if pd.notna(p) and p < 0.05:
                    ax.scatter(
                        j,
                        i,
                        s=8.5,
                        facecolor="#222222",
                        edgecolor="none",
                        zorder=3
                    )

    ax.set_title(
        title,
        pad=8,
        fontsize=7.8,
        fontweight="bold",
        color=GRAY_TEXT
    )

    return im, norm


def draw_motif_panel(
    ax,
    motif_df,
    motif_order,
    motif_colors,
    motif_markers,
    motif_linestyles,
    panel_title,
    y_label,
    y_clip=None,
    show_stars=True,
    mark_clipped=True,
    legend_ncol=2
):
    """
    Draw an age-specific motif panel using the designer's visual system.

    The six reference colors are sampled directly from the supplied PNG.
    Series use uniform filled circles and solid lines; light horizontal
    rules help reveal small differences without adding vertical clutter.
    """
    age_order = [1, 2, 3, 4, 5]
    x = np.arange(len(age_order))

    ax.set_axisbelow(True)
    ax.yaxis.grid(
        True,
        color=DESIGNER_GRID,
        linewidth=0.60,
        linestyle="-",
        zorder=0
    )
    ax.xaxis.grid(False)

    for motif in motif_order:
        sub = motif_df[motif_df["motif"].eq(motif)].copy()
        sub = sub.set_index("age").reindex(age_order)

        y_raw = sub["delta_log2_lift"].to_numpy(dtype=float)
        pvals = sub["p"].to_numpy(dtype=float)

        color = motif_colors.get(motif, "#333333")
        marker = motif_markers.get(motif, "o")
        linestyle = motif_linestyles.get(motif, "-")

        y_plot = y_raw.copy()

        clipped_high = np.zeros_like(y_plot, dtype=bool)
        clipped_low = np.zeros_like(y_plot, dtype=bool)

        if y_clip is not None:
            clipped_high = y_raw > y_clip
            clipped_low = y_raw < -y_clip
            y_plot = np.clip(y_raw, -y_clip, y_clip)

        ax.plot(
            x,
            y_plot,
            lw=1.45,
            linestyle=linestyle,
            marker=marker,
            markersize=5.2,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.0,
            color=color,
            alpha=1.0,
            label=motif,
            zorder=3
        )

        if show_stars:
            finite_vals = y_plot[np.isfinite(y_plot)]

            if finite_vals.size > 0:
                y_range = np.nanmax(finite_vals) - np.nanmin(finite_vals)
            else:
                y_range = 0.2

            if not np.isfinite(y_range) or y_range == 0:
                y_range = 0.2

            for xi, yi, p in zip(x, y_plot, pvals):
                star = p_to_stars(p)

                if star and np.isfinite(yi):
                    ax.text(
                        xi + 0.07,
                        yi + 0.045 * y_range,
                        star,
                        ha="left",
                        va="bottom",
                        fontsize=6.0,
                        fontweight="bold",
                        color="#000000",
                        zorder=5
                    )

        if y_clip is not None and mark_clipped:
            for xi, yi, hi, lo in zip(x, y_plot, clipped_high, clipped_low):
                if hi:
                    ax.scatter(
                        xi,
                        y_clip,
                        marker="^",
                        s=36,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.0,
                        zorder=6
                    )

                if lo:
                    ax.scatter(
                        xi,
                        -y_clip,
                        marker="v",
                        s=36,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.0,
                        zorder=6
                    )

    ax.axhline(
        0,
        color=DESIGNER_ZERO,
        lw=1.25,
        linestyle=(0, (2.2, 2.2)),
        zorder=2
    )

    ax.set_xticks(x)
    ax.set_xticklabels([AGE_LABELS[a] for a in age_order])
    ax.set_ylabel(y_label)

    ax.set_title(
        panel_title,
        fontsize=8.0,
        fontweight="bold",
        color="#000000",
        pad=8
    )

    if y_clip is not None:
        ax.set_ylim(-y_clip * 1.10, y_clip * 1.10)

        ax.text(
            0.99,
            0.02,
            f"Values clipped at ±{y_clip:g}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#555555"
        )

    ax.margins(x=0.09)
    clean_axis(ax)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(axis="both", colors="#000000")

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.25),
        ncol=legend_ncol,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.5,
        fontsize=5.35
    )


def save_designer_motif_panel(
    motif_df,
    motif_order,
    motif_colors,
    motif_markers,
    motif_linestyles,
    panel_title,
    y_label,
    filename_stem,
    y_clip=None
):
    """
    Export one near-square/portrait motif panel matching the designer PNG.

    The main composite retains its compact legend. Standalone panels omit
    the legend, as in the supplied reference, and use manuscript-size text.
    PNG and JPEG files are written at 600 dpi; an editable SVG is also saved.
    """
    fig, ax = plt.subplots(
        figsize=(7.8, 7.5),
        constrained_layout=False
    )

    draw_motif_panel(
        ax,
        motif_df,
        motif_order,
        motif_colors=motif_colors,
        motif_markers=motif_markers,
        motif_linestyles=motif_linestyles,
        panel_title=panel_title,
        y_label=y_label,
        y_clip=y_clip,
        show_stars=True,
        mark_clipped=True,
        legend_ncol=2
    )

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    # Scale the compact composite-panel styling to the standalone reference.
    for line in ax.lines:
        if line.get_label() in motif_order:
            line.set_linewidth(1.6)
            line.set_markersize(8.2)

    for annotation in ax.texts:
        if annotation.get_text().startswith("Values clipped"):
            annotation.set_visible(False)
        else:
            annotation.set_fontsize(12.0)

    ax.set_title(
        panel_title,
        fontsize=17.0,
        fontweight="bold",
        color="#000000",
        pad=42
    )
    ax.set_xlabel(
        "Age group",
        fontsize=16.0,
        color="#000000",
        labelpad=24
    )
    ax.set_ylabel(
        y_label,
        fontsize=15.0,
        color="#000000",
        labelpad=20
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=11.0,
        width=1.2,
        length=9,
        pad=8,
        colors="#000000"
    )
    ax.spines["left"].set_linewidth(1.35)
    ax.spines["bottom"].set_linewidth(1.35)

    if y_clip is not None:
        ax.set_ylim(-y_clip * 1.22, y_clip * 1.16)
        if float(y_clip).is_integer():
            clip_int = int(y_clip)
            ax.set_yticks(np.arange(-clip_int, clip_int + 1, 1))

    fig.subplots_adjust(
        left=0.19,
        right=0.965,
        top=0.80,
        bottom=0.19
    )

    png_path = os.path.join(OUT_DIR, f"{filename_stem}.png")
    jpeg_path = os.path.join(OUT_DIR, f"{filename_stem}.jpeg")
    svg_path = os.path.join(OUT_DIR, f"{filename_stem}.svg")

    fig.savefig(
        png_path,
        dpi=OUTPUT_DPI,
        facecolor="white",
        bbox_inches=None
    )
    fig.savefig(
        jpeg_path,
        dpi=OUTPUT_DPI,
        facecolor="white",
        bbox_inches=None,
        pil_kwargs={
            "quality": 95,
            "subsampling": 0
        }
    )
    fig.savefig(
        svg_path,
        facecolor="white",
        bbox_inches=None
    )
    plt.close(fig)

    return {
        "png": png_path,
        "jpeg": jpeg_path,
        "svg": svg_path
    }


# ============================================================
# 10. Main figure function
# ============================================================

def make_nhb_transition_figure(
    theme_df,
    sentiment_df=None,
    show_figure=False
):
    os.makedirs(OUT_DIR, exist_ok=True)

    df = make_transition_input_df(theme_df, sentiment_df)
    check_required_columns(df)

    print("[1/8] Computing overall theme transitions...", flush=True)

    # ------------------------------------------------------------
    # Overall theme transitions
    # ------------------------------------------------------------
    _, theme_prob12, _, _ = compute_transition_probability(
        df,
        THEME_SEC1_COLS,
        THEME_SEC2_COLS,
        THEMES
    )

    _, theme_prob23, _, _ = compute_transition_probability(
        df,
        THEME_SEC2_COLS,
        THEME_SEC3_COLS,
        THEMES
    )

    theme_log2_lift12 = compute_log2_lift(
        df,
        THEME_SEC1_COLS,
        THEME_SEC2_COLS,
        THEMES
    )

    theme_log2_lift23 = compute_log2_lift(
        df,
        THEME_SEC2_COLS,
        THEME_SEC3_COLS,
        THEMES
    )

    print("[2/8] Computing overall sentiment transitions...", flush=True)

    # ------------------------------------------------------------
    # Overall sentiment transitions
    # ------------------------------------------------------------
    _, sent_prob12, _, _ = compute_transition_probability(
        df,
        SENT_SEC1_COLS,
        SENT_SEC2_COLS,
        SENTIMENT_STATES
    )

    _, sent_prob23, _, _ = compute_transition_probability(
        df,
        SENT_SEC2_COLS,
        SENT_SEC3_COLS,
        SENTIMENT_STATES
    )

    sent_log2_lift12 = compute_log2_lift(
        df,
        SENT_SEC1_COLS,
        SENT_SEC2_COLS,
        SENTIMENT_STATES
    )

    sent_log2_lift23 = compute_log2_lift(
        df,
        SENT_SEC2_COLS,
        SENT_SEC3_COLS,
        SENTIMENT_STATES
    )

    print(
        "[3/8] Computing age-specific theme motifs "
        "(opening to middle)...",
        flush=True
    )

    # ------------------------------------------------------------
    # Age motif contrasts: separated by transition window
    # ------------------------------------------------------------
    age_theme_motif_12_df = compute_age_motif_table(
        df,
        THEME_SEC1_COLS,
        THEME_SEC2_COLS,
        THEMES,
        THEME_MOTIFS,
        THEME_MOTIF_LABELS,
        transition_label="Opening→middle",
        n_perm=N_PERM
    )

    print(
        "[4/8] Computing age-specific theme motifs "
        "(middle to ending)...",
        flush=True
    )

    age_theme_motif_23_df = compute_age_motif_table(
        df,
        THEME_SEC2_COLS,
        THEME_SEC3_COLS,
        THEMES,
        THEME_MOTIFS,
        THEME_MOTIF_LABELS,
        transition_label="Middle→ending",
        n_perm=N_PERM
    )

    print(
        "[5/8] Computing age-specific sentiment motifs "
        "(opening to middle)...",
        flush=True
    )

    age_sent_motif_12_df = compute_age_motif_table(
        df,
        SENT_SEC1_COLS,
        SENT_SEC2_COLS,
        SENTIMENT_STATES,
        SENTIMENT_MOTIFS,
        SENTIMENT_MOTIF_LABELS,
        transition_label="Opening→middle",
        n_perm=N_PERM
    )

    print(
        "[6/8] Computing age-specific sentiment motifs "
        "(middle to ending)...",
        flush=True
    )

    age_sent_motif_23_df = compute_age_motif_table(
        df,
        SENT_SEC2_COLS,
        SENT_SEC3_COLS,
        SENTIMENT_STATES,
        SENTIMENT_MOTIFS,
        SENTIMENT_MOTIF_LABELS,
        transition_label="Middle→ending",
        n_perm=N_PERM
    )

    print("[7/8] Computing sex-contrast permutation tests...", flush=True)

    # ------------------------------------------------------------
    # Sex contrasts
    # ------------------------------------------------------------
    mask_male = df["SEX"].eq(1).to_numpy()
    mask_female = df["SEX"].eq(2).to_numpy()

    sex_theme_delta12, sex_theme_p12, _ = permutation_pvalues_delta_log2_lift(
        df,
        THEME_SEC1_COLS,
        THEME_SEC2_COLS,
        THEMES,
        mask_pos=mask_male,
        mask_neg=mask_female,
        n_perm=N_PERM,
        random_state=RANDOM_SEED + 90,
        use_max_stat=False
    )

    sex_theme_delta23, sex_theme_p23, _ = permutation_pvalues_delta_log2_lift(
        df,
        THEME_SEC2_COLS,
        THEME_SEC3_COLS,
        THEMES,
        mask_pos=mask_male,
        mask_neg=mask_female,
        n_perm=N_PERM,
        random_state=RANDOM_SEED + 100,
        use_max_stat=False
    )

    sex_sent_delta12, sex_sent_p12, _ = permutation_pvalues_delta_log2_lift(
        df,
        SENT_SEC1_COLS,
        SENT_SEC2_COLS,
        SENTIMENT_STATES,
        mask_pos=mask_male,
        mask_neg=mask_female,
        n_perm=N_PERM,
        random_state=RANDOM_SEED + 200,
        use_max_stat=False
    )

    sex_sent_delta23, sex_sent_p23, _ = permutation_pvalues_delta_log2_lift(
        df,
        SENT_SEC2_COLS,
        SENT_SEC3_COLS,
        SENTIMENT_STATES,
        mask_pos=mask_male,
        mask_neg=mask_female,
        n_perm=N_PERM,
        random_state=RANDOM_SEED + 300,
        use_max_stat=False
    )

    print("[8/8] Saving source data and rendering figures...", flush=True)

    # ------------------------------------------------------------
    # Save source data
    # ------------------------------------------------------------
    df.to_pickle(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_analytic_df.pkl"
        )
    )

    theme_prob12.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_theme_probability_S1_to_S2.csv"
        ),
        encoding="utf-8-sig"
    )

    theme_prob23.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_theme_probability_S2_to_S3.csv"
        ),
        encoding="utf-8-sig"
    )

    theme_log2_lift12.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_theme_log2lift_S1_to_S2.csv"
        ),
        encoding="utf-8-sig"
    )

    theme_log2_lift23.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_theme_log2lift_S2_to_S3.csv"
        ),
        encoding="utf-8-sig"
    )

    sent_prob12.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_sentiment_probability_S1_to_S2.csv"
        ),
        encoding="utf-8-sig"
    )

    sent_prob23.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_sentiment_probability_S2_to_S3.csv"
        ),
        encoding="utf-8-sig"
    )

    sent_log2_lift12.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_sentiment_log2lift_S1_to_S2.csv"
        ),
        encoding="utf-8-sig"
    )

    sent_log2_lift23.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_sentiment_log2lift_S2_to_S3.csv"
        ),
        encoding="utf-8-sig"
    )

    age_theme_motif_12_df.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_age_theme_motifs_opening_to_middle.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    age_theme_motif_23_df.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_age_theme_motifs_middle_to_ending.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    age_sent_motif_12_df.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_age_sentiment_motifs_opening_to_middle_UNCLIPPED.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    age_sent_motif_23_df.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_age_sentiment_motifs_middle_to_ending_UNCLIPPED.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    sex_theme_delta12.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_sex_theme_delta_S1_to_S2.csv"
        ),
        encoding="utf-8-sig"
    )

    sex_theme_delta23.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_sex_theme_delta_S2_to_S3.csv"
        ),
        encoding="utf-8-sig"
    )

    sex_sent_delta12.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_sex_sentiment_delta_S1_to_S2.csv"
        ),
        encoding="utf-8-sig"
    )

    sex_sent_delta23.to_csv(
        os.path.join(
            OUT_DIR,
            f"{SAVE_PREFIX}_sex_sentiment_delta_S2_to_S3.csv"
        ),
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Shared color scales
    # ------------------------------------------------------------
    theme_vlim = np.nanpercentile(
        np.abs(
            np.concatenate([
                theme_log2_lift12.to_numpy().flatten(),
                theme_log2_lift23.to_numpy().flatten()
            ])
        ),
        95
    )
    theme_vlim = max(theme_vlim, 0.75)

    sent_vlim = np.nanpercentile(
        np.abs(
            np.concatenate([
                sent_log2_lift12.to_numpy().flatten(),
                sent_log2_lift23.to_numpy().flatten()
            ])
        ),
        95
    )
    sent_vlim = max(sent_vlim, 0.75)

    sex_theme_vlim = np.nanpercentile(
        np.abs(
            np.concatenate([
                sex_theme_delta12.to_numpy().flatten(),
                sex_theme_delta23.to_numpy().flatten()
            ])
        ),
        95
    )
    sex_theme_vlim = max(sex_theme_vlim, 0.5)

    sex_sent_vlim = np.nanpercentile(
        np.abs(
            np.concatenate([
                sex_sent_delta12.to_numpy().flatten(),
                sex_sent_delta23.to_numpy().flatten()
            ])
        ),
        95
    )
    sex_sent_vlim = max(sex_sent_vlim, 0.5)

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    fig = plt.figure(
        figsize=(12.2, 11.1),
        constrained_layout=False,
        facecolor="white"
    )

    # Four-row layout matching the supplied combined-figure reference:
    # overall theme and sentiment heatmaps;
    # age-specific theme motifs;
    # age-specific sentiment motifs; and
    # sex-contrast theme and sentiment heatmaps.
    outer = gridspec.GridSpec(
        nrows=4,
        ncols=2,
        figure=fig,
        width_ratios=[0.94, 1.06],
        height_ratios=[1.40, 0.86, 0.86, 1.40],
        wspace=0.25,
        hspace=0.86
    )

    # Top-left: overall theme transitions
    gs_theme_overall = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[0, 0],
        wspace=0.62
    )
    ax_b1 = fig.add_subplot(gs_theme_overall[0, 0])
    ax_b2 = fig.add_subplot(gs_theme_overall[0, 1])

    # Top-right: overall sentiment transitions
    gs_sent_overall = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[0, 1],
        wspace=0.54
    )
    ax_c1 = fig.add_subplot(gs_sent_overall[0, 0])
    ax_c2 = fig.add_subplot(gs_sent_overall[0, 1])

    # Age-specific theme motifs.
    # The blank outer columns center the two plots within 80% of the row.
    # Increase 0.25 to make both panels narrower; decrease it to widen them.
    gs_age_theme = gridspec.GridSpecFromSubplotSpec(
        1,
        4,
        subplot_spec=outer[1, :],
        width_ratios=[0.25, 1.0, 1.0, 0.25],
        wspace=0.20
    )
    ax_d1 = fig.add_subplot(gs_age_theme[0, 1])
    ax_d2 = fig.add_subplot(gs_age_theme[0, 2])

    # Age-specific sentiment motifs, centered to match the theme motifs.
    gs_age_sent = gridspec.GridSpecFromSubplotSpec(
        1,
        4,
        subplot_spec=outer[2, :],
        width_ratios=[0.25, 1.0, 1.0, 0.25],
        wspace=0.20
    )
    ax_e1 = fig.add_subplot(gs_age_sent[0, 1])
    ax_e2 = fig.add_subplot(gs_age_sent[0, 2])

    # Bottom-left: sex-contrast theme transitions
    gs_sex_theme = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[3, 0],
        wspace=0.62
    )
    ax_g1 = fig.add_subplot(gs_sex_theme[0, 0])
    ax_g2 = fig.add_subplot(gs_sex_theme[0, 1])

    # Bottom-right: sex-contrast sentiment transitions
    gs_sex_sent = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[3, 1],
        wspace=0.54
    )
    ax_f1 = fig.add_subplot(gs_sex_sent[0, 0])
    ax_f2 = fig.add_subplot(gs_sex_sent[0, 1])

    # ------------------------------------------------------------
    # Row 1: Overall theme transitions
    # ------------------------------------------------------------
    _, norm_b = draw_heatmap(
        ax_b1,
        theme_log2_lift12,
        title="Theme transitions: opening to middle",
        state_labels=THEME_LABELS,
        vlim=theme_vlim,
        significance_df=None,
        show_ylabels=True
    )
    ax_b1.set_ylabel("Opening theme")
    ax_b1.set_xlabel("Middle theme")

    draw_heatmap(
        ax_b2,
        theme_log2_lift23,
        title="Theme transitions: middle to ending",
        state_labels=THEME_LABELS,
        vlim=theme_vlim,
        significance_df=None,
        show_ylabels=True
    )
    ax_b2.set_ylabel("Middle theme")
    ax_b2.set_xlabel("Ending theme")

    add_attached_colorbar(
        ax_b2,
        norm_b,
        CMAP_DIVERGING,
        "log2 lift"
    )

    # ------------------------------------------------------------
    # Row 2: Overall sentiment transitions
    # ------------------------------------------------------------
    _, norm_c = draw_heatmap(
        ax_c1,
        sent_log2_lift12,
        title="Sentiment transitions: opening to middle",
        state_labels=SENTIMENT_LABELS,
        vlim=sent_vlim,
        significance_df=None,
        show_ylabels=True
    )
    ax_c1.set_ylabel("Opening sentiment")
    ax_c1.set_xlabel("Middle sentiment")

    draw_heatmap(
        ax_c2,
        sent_log2_lift23,
        title="Sentiment transitions: middle to ending",
        state_labels=SENTIMENT_LABELS,
        vlim=sent_vlim,
        significance_df=None,
        show_ylabels=True
    )
    ax_c2.set_ylabel("Middle sentiment")
    ax_c2.set_xlabel("Ending sentiment")

    add_attached_colorbar(
        ax_c2,
        norm_c,
        CMAP_DIVERGING,
        "log2 lift"
    )

    # ------------------------------------------------------------
    # Row 3: Age-specific theme motifs, separated by window
    # ------------------------------------------------------------
    theme_motif_order = [
        THEME_MOTIF_LABELS[m]
        for m in THEME_MOTIFS
    ]

    draw_motif_panel(
        ax_d1,
        age_theme_motif_12_df,
        theme_motif_order,
        motif_colors=THEME_MOTIF_COLORS,
        motif_markers=THEME_MOTIF_MARKERS,
        motif_linestyles=THEME_MOTIF_LINESTYLES,
        panel_title="Age-specific theme motifs: opening to middle",
        y_label="Age group vs rest, Δlog2 lift",
        y_clip=THEME_MOTIF_Y_CLIP,
        show_stars=True,
        mark_clipped=True,
        legend_ncol=2
    )

    draw_motif_panel(
        ax_d2,
        age_theme_motif_23_df,
        theme_motif_order,
        motif_colors=THEME_MOTIF_COLORS,
        motif_markers=THEME_MOTIF_MARKERS,
        motif_linestyles=THEME_MOTIF_LINESTYLES,
        panel_title="Age-specific theme motifs: middle to ending",
        y_label="Age group vs rest, Δlog2 lift",
        y_clip=THEME_MOTIF_Y_CLIP,
        show_stars=True,
        mark_clipped=True,
        legend_ncol=2
    )

    # ------------------------------------------------------------
    # Row 4: Age-specific sentiment motifs, separated by window
    # ------------------------------------------------------------
    sent_motif_order = [
        SENTIMENT_MOTIF_LABELS[m]
        for m in SENTIMENT_MOTIFS
    ]

    draw_motif_panel(
        ax_e1,
        age_sent_motif_12_df,
        sent_motif_order,
        motif_colors=SENTIMENT_MOTIF_COLORS,
        motif_markers=SENTIMENT_MOTIF_MARKERS,
        motif_linestyles=SENTIMENT_MOTIF_LINESTYLES,
        panel_title="Age-specific sentiment motifs: opening to middle",
        y_label="Age group vs rest, Δlog2 lift",
        y_clip=SENTIMENT_MOTIF_Y_CLIP,
        show_stars=True,
        mark_clipped=True,
        legend_ncol=2
    )

    draw_motif_panel(
        ax_e2,
        age_sent_motif_23_df,
        sent_motif_order,
        motif_colors=SENTIMENT_MOTIF_COLORS,
        motif_markers=SENTIMENT_MOTIF_MARKERS,
        motif_linestyles=SENTIMENT_MOTIF_LINESTYLES,
        panel_title="Age-specific sentiment motifs: middle to ending",
        y_label="Age group vs rest, Δlog2 lift",
        y_clip=SENTIMENT_MOTIF_Y_CLIP,
        show_stars=True,
        mark_clipped=True,
        legend_ncol=2
    )

    # ------------------------------------------------------------
    # Row 5: Sex contrast, sentiment transitions
    # ------------------------------------------------------------
    _, norm_f_sent = draw_heatmap(
        ax_f1,
        sex_sent_delta12,
        title="Sex contrast: sentiment opening to middle",
        state_labels=SENTIMENT_LABELS,
        vlim=sex_sent_vlim,
        significance_df=sex_sent_p12,
        show_ylabels=True
    )
    ax_f1.set_ylabel("Opening sentiment")
    ax_f1.set_xlabel("Middle sentiment")

    draw_heatmap(
        ax_f2,
        sex_sent_delta23,
        title="Sex contrast: sentiment middle to ending",
        state_labels=SENTIMENT_LABELS,
        vlim=sex_sent_vlim,
        significance_df=sex_sent_p23,
        show_ylabels=True
    )
    ax_f2.set_ylabel("Middle sentiment")
    ax_f2.set_xlabel("Ending sentiment")

    add_attached_colorbar(
        ax_f2,
        norm_f_sent,
        CMAP_DIVERGING,
        "Male - female, Δlog2 lift"
    )

    # ------------------------------------------------------------
    # Row 6: Sex contrast, theme transitions
    # ------------------------------------------------------------
    _, norm_g_theme = draw_heatmap(
        ax_g1,
        sex_theme_delta12,
        title="Sex contrast: theme opening to middle",
        state_labels=THEME_LABELS,
        vlim=sex_theme_vlim,
        significance_df=sex_theme_p12,
        show_ylabels=True
    )
    ax_g1.set_ylabel("Opening theme")
    ax_g1.set_xlabel("Middle theme")

    draw_heatmap(
        ax_g2,
        sex_theme_delta23,
        title="Sex contrast: theme middle to ending",
        state_labels=THEME_LABELS,
        vlim=sex_theme_vlim,
        significance_df=sex_theme_p23,
        show_ylabels=True
    )
    ax_g2.set_ylabel("Middle theme")
    ax_g2.set_xlabel("Ending theme")

    add_attached_colorbar(
        ax_g2,
        norm_g_theme,
        CMAP_DIVERGING,
        "Male - female, Δlog2 lift"
    )

    # ------------------------------------------------------------
    # Standalone designer-style motif panels
    # ------------------------------------------------------------
    designer_panel_paths = {
        "theme_opening_to_middle": save_designer_motif_panel(
            age_theme_motif_12_df,
            theme_motif_order,
            motif_colors=THEME_MOTIF_COLORS,
            motif_markers=THEME_MOTIF_MARKERS,
            motif_linestyles=THEME_MOTIF_LINESTYLES,
            panel_title="Age-specific theme motifs: opening to middle",
            y_label="Age group vs rest, Δlog2 lift",
            filename_stem=f"{SAVE_PREFIX}_designer_theme_opening_to_middle",
            y_clip=THEME_MOTIF_Y_CLIP
        ),
        "theme_middle_to_ending": save_designer_motif_panel(
            age_theme_motif_23_df,
            theme_motif_order,
            motif_colors=THEME_MOTIF_COLORS,
            motif_markers=THEME_MOTIF_MARKERS,
            motif_linestyles=THEME_MOTIF_LINESTYLES,
            panel_title="Age-specific theme motifs: middle to ending",
            y_label="Age group vs rest, Δlog2 lift",
            filename_stem=f"{SAVE_PREFIX}_designer_theme_middle_to_ending",
            y_clip=THEME_MOTIF_Y_CLIP
        ),
        "sentiment_opening_to_middle": save_designer_motif_panel(
            age_sent_motif_12_df,
            sent_motif_order,
            motif_colors=SENTIMENT_MOTIF_COLORS,
            motif_markers=SENTIMENT_MOTIF_MARKERS,
            motif_linestyles=SENTIMENT_MOTIF_LINESTYLES,
            panel_title="Age-specific sentiment motifs: opening to middle",
            y_label="Age group vs rest, Δlog2 lift",
            filename_stem=f"{SAVE_PREFIX}_designer_sentiment_opening_to_middle",
            y_clip=SENTIMENT_MOTIF_Y_CLIP
        ),
        "sentiment_middle_to_ending": save_designer_motif_panel(
            age_sent_motif_23_df,
            sent_motif_order,
            motif_colors=SENTIMENT_MOTIF_COLORS,
            motif_markers=SENTIMENT_MOTIF_MARKERS,
            motif_linestyles=SENTIMENT_MOTIF_LINESTYLES,
            panel_title="Age-specific sentiment motifs: middle to ending",
            y_label="Age group vs rest, Δlog2 lift",
            filename_stem=f"{SAVE_PREFIX}_designer_sentiment_middle_to_ending",
            y_clip=SENTIMENT_MOTIF_Y_CLIP
        )
    }

    # ------------------------------------------------------------
    # Combined-figure spacing, row labels, and border
    # ------------------------------------------------------------
    fig.subplots_adjust(
        left=0.065,
        right=0.975,
        top=0.965,
        bottom=0.065
    )

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------
    pdf_path = os.path.join(
        OUT_DIR,
        f"{SAVE_PREFIX}.pdf"
    )

    svg_path = os.path.join(
        OUT_DIR,
        f"{SAVE_PREFIX}.svg"
    )

    png_path = os.path.join(
        OUT_DIR,
        f"{SAVE_PREFIX}.png"
    )

    jpeg_path = os.path.join(
        OUT_DIR,
        f"{SAVE_PREFIX}.jpeg"
    )

    tiff_path = os.path.join(
        OUT_DIR,
        f"{SAVE_PREFIX}.tiff"
    )

    fig.savefig(
        pdf_path,
        dpi=OUTPUT_DPI,
        facecolor="white",
        bbox_inches=None
    )

    fig.savefig(
        svg_path,
        facecolor="white",
        bbox_inches=None
    )

    fig.savefig(
        png_path,
        dpi=OUTPUT_DPI,
        facecolor="white",
        bbox_inches=None
    )

    fig.savefig(
        jpeg_path,
        dpi=OUTPUT_DPI,
        facecolor="white",
        bbox_inches=None,
        pil_kwargs={
            "quality": 95,
            "subsampling": 0
        }
    )

    fig.savefig(
        tiff_path,
        dpi=OUTPUT_DPI,
        facecolor="white",
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"}
    )

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    print("\nSaved figure files:")
    print(pdf_path)
    print(svg_path)
    print(png_path)
    print(jpeg_path)
    print(tiff_path)

    print("\nSaved source-data tables and standalone designer panels in:")
    print(OUT_DIR)

    print("\nStandalone designer panel files:")
    for panel_name, panel_files in designer_panel_paths.items():
        print(f"{panel_name}:")
        print(f"  {panel_files['png']}")
        print(f"  {panel_files['jpeg']}")
        print(f"  {panel_files['svg']}")

    print("\nInterpretation note:")
    print("Overall transition heatmaps show log2 lift.")
    print("Age motif panels show age group vs rest delta log2 lift.")
    print("Age motif panels are separated by transition window.")
    print("Stars in age-motif panels indicate permutation significance.")
    print("Sex transition heatmaps show male - female delta log2 lift.")
    print("Black dots in sex-contrast heatmaps indicate cellwise permutation p < .05.")
    print(
        "Theme age-motif panels are visually clipped at "
        f"+/-{THEME_MOTIF_Y_CLIP}."
    )
    print(
        "Sentiment age-motif panels are visually clipped at "
        f"+/-{SENTIMENT_MOTIF_Y_CLIP}."
    )
    print("Saved age-motif CSVs contain the original unclipped values.")

    return {
        "figure": fig,
        "figure_paths": {
            "pdf": pdf_path,
            "svg": svg_path,
            "png": png_path,
            "jpeg": jpeg_path,
            "tiff": tiff_path
        },
        "analytic_df": df,

        "theme_prob12": theme_prob12,
        "theme_prob23": theme_prob23,
        "theme_log2_lift12": theme_log2_lift12,
        "theme_log2_lift23": theme_log2_lift23,

        "sent_prob12": sent_prob12,
        "sent_prob23": sent_prob23,
        "sent_log2_lift12": sent_log2_lift12,
        "sent_log2_lift23": sent_log2_lift23,

        "age_theme_motif_12_df": age_theme_motif_12_df,
        "age_theme_motif_23_df": age_theme_motif_23_df,
        "age_sent_motif_12_df": age_sent_motif_12_df,
        "age_sent_motif_23_df": age_sent_motif_23_df,

        "sex_theme_delta12": sex_theme_delta12,
        "sex_theme_p12": sex_theme_p12,
        "sex_theme_delta23": sex_theme_delta23,
        "sex_theme_p23": sex_theme_p23,

        "sex_sent_delta12": sex_sent_delta12,
        "sex_sent_p12": sex_sent_p12,
        "sex_sent_delta23": sex_sent_delta23,
        "sex_sent_p23": sex_sent_p23,

        "designer_panel_paths": designer_panel_paths
    }


# ============================================================
# 11. Standalone data loading and command-line entry point
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the NHB theme/sentiment transition figure and the "
            "standalone designer-style age-motif panels."
        )
    )
    parser.add_argument(
        "--theme-data",
        type=Path,
        default=DEFAULT_THEME_DATA_PATH,
        help=(
            "Pickle containing the theme dataframe "
            f"(default: {DEFAULT_THEME_DATA_PATH})"
        )
    )
    parser.add_argument(
        "--sentiment-data",
        type=Path,
        default=DEFAULT_SENTIMENT_DATA_PATH,
        help=(
            "Pickle containing the sentiment dataframe "
            f"(default: {DEFAULT_SENTIMENT_DATA_PATH})"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for figures and source data (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=N_PERM,
        help=(
            "Number of permutations. Use 200-500 for a quick test and "
            "5000 for the final figure (default: 5000)."
        )
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the composite figure after saving it."
    )
    return parser.parse_args(argv)


def load_dataframe(path, data_label):
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"{data_label} dataframe file was not found: {path}\n"
            f"Supply it with --{data_label.lower()}-data."
        )

    print(f"Loading {data_label.lower()} dataframe: {path}")
    loaded = pd.read_pickle(path)

    if not isinstance(loaded, pd.DataFrame):
        raise TypeError(
            f"{data_label} input must contain a pandas DataFrame; "
            f"found {type(loaded).__name__}."
        )

    if not loaded.index.is_unique:
        raise ValueError(
            f"{data_label} dataframe index must be unique for safe alignment."
        )

    print(
        f"Loaded {data_label.lower()} dataframe: "
        f"{len(loaded):,} rows x {loaded.shape[1]:,} columns"
    )
    return loaded


def validate_input_alignment(theme_df, sentiment_df):
    common_index = theme_df.index.intersection(sentiment_df.index)

    if len(common_index) == 0:
        raise ValueError(
            "Theme and sentiment dataframes have no overlapping index values."
        )

    if len(common_index) != len(theme_df) or len(common_index) != len(sentiment_df):
        print(
            "Warning: the two dataframes do not have identical index sets. "
            f"The analysis will use their {len(common_index):,} shared rows."
        )
    else:
        print(
            f"Input alignment validated: {len(common_index):,} shared rows."
        )

    # Run the existing reconstruction checks before starting permutations.
    ensure_theme_section_columns(theme_df)
    ensure_sentiment_section_columns(sentiment_df)
    print("Required theme and sentiment columns validated.")


def main(argv=None):
    global OUT_DIR, N_PERM

    args = parse_args(argv)

    if args.n_perm <= 0:
        raise ValueError("--n-perm must be a positive integer.")

    theme_path = args.theme_data.expanduser().resolve()
    sentiment_path = args.sentiment_data.expanduser().resolve()
    output_path = args.output_dir.expanduser().resolve()

    N_PERM = args.n_perm
    OUT_DIR = str(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    status_path = output_path / "RUN_STATUS.txt"
    status_path.write_text(
        "RUNNING\n"
        f"Permutation count: {N_PERM:,}\n"
        "Figure files are written after the permutation analyses finish.\n",
        encoding="utf-8"
    )

    try:
        theme_df = load_dataframe(theme_path, "Theme")

        if sentiment_path == theme_path:
            print("Theme and sentiment paths are identical; reusing one dataframe.")
            sentiment_df = theme_df
        else:
            sentiment_df = load_dataframe(sentiment_path, "Sentiment")

        validate_input_alignment(theme_df, sentiment_df)

        print(f"Permutation count: {N_PERM:,}")
        print(f"Output directory: {OUT_DIR}")

        results = make_nhb_transition_figure(
            theme_df=theme_df,
            sentiment_df=sentiment_df,
            show_figure=args.show
        )
    except Exception as exc:
        status_path.write_text(
            "FAILED\n"
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8"
        )
        raise

    status_path.write_text(
        "COMPLETE\n"
        f"PNG: {results['figure_paths']['png']}\n"
        f"JPEG: {results['figure_paths']['jpeg']}\n",
        encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    results = main()
