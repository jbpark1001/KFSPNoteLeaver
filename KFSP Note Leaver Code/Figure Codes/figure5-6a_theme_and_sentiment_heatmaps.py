# -*- coding: utf-8 -*-
"""
Create publication-ready theme and sentiment heatmaps.

The script is standalone: it reads the two pickle files, reconstructs the
section-level indicators, fits the demographic logistic-regression models,
and exports both heatmaps plus their underlying coefficient tables.

Default inputs
--------------
    C:/Users/<user>/raw.pkl
    C:/Users/<user>/morethan3rawsentiment.pkl

Default outputs
---------------
    Figure Images/figure6a_theme_heatmap_style2.png
    Figure Images/figure6a_theme_heatmap_style2.pdf
    Figure Images/figure6a_theme_heatmap_estimates.csv
    Figure Images/figure6b_sentiment_heatmap_style2.png
    Figure Images/figure6b_sentiment_heatmap_style2.pdf
    Figure Images/figure6b_sentiment_heatmap_estimates.csv

Examples
--------
    python figure6_theme_and_sentiment_heatmaps.py
    python figure6_theme_and_sentiment_heatmaps.py --style 1
    python figure6_theme_and_sentiment_heatmaps.py --style 3 --text-color black

Statistical specification
-------------------------
* Gender column: Male vs Female logit, adjusted for categorical age.
* Age columns: one-vs-rest logits for each age group, adjusted for gender.
* Cell top line: log-odds coefficient and significance stars.
* Cell bottom line: 95% confidence interval on the log-odds scale.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

# Save figures without opening GUI windows. Remove this line for interactive use.
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

THEME_PRESENCE_COLUMNS: Mapping[str, str] = {
    "Sorry and Shame": "sorrysentence_presence",
    "Love and Gratitude": "loveandgratitudesentence_presence",
    "Burden": "burdensomesentence_presence",
    "Despair": "despairsentence_presence",
    "Post-mortem Affairs": "pmaffairssentence_presence",
}

# English display name -> exact Korean label stored in *_section_sentiment.
SENTIMENT_LABELS: Mapping[str, str] = {
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

SENTIMENT_SOURCE_COLUMNS: Sequence[str] = (
    "1st_section_sentiment",
    "2nd_section_sentiment",
    "3rd_section_sentiment",
)

SECTION_KEYS: Sequence[str] = ("Section 1", "Section 2", "Section 3")
SECTION_DISPLAY: Sequence[str] = ("Opening", "Middle", "Ending")

# AGE2 codes used in the supplied scripts and their publication labels.
AGE_CODES: Sequence[int] = (1, 2, 3, 4, 5)
AGE_LABELS: Sequence[str] = ("<18", "19–34", "35–49", "50–64", "65+")

THEME_TITLE = "Theme–Section Associations Across Demographic Groups"
SENTIMENT_TITLE = "Sentiment–Section Associations Across Demographic Groups"
SUBTITLE = (
    "Log-odds coefficients with significance stars; brackets show 95% confidence intervals"
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _required_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{source} is missing required columns: {joined}")


def _presence_at(value: object, section_index: int) -> int:
    """Return a clean 0/1 section indicator from a list-like presence value."""
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > section_index:
        item = value[section_index]
        if pd.isna(item):
            return 0
        return int(bool(item))
    return 0


def build_theme_features(
    source: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Read raw.pkl and return metadata plus the 3 x 5 theme matrix."""
    raw = pd.read_pickle(source)
    _required_columns(
        raw,
        ["SEX", "AGE2", "too_short", *THEME_PRESENCE_COLUMNS.values()],
        source,
    )

    data = raw.loc[raw["too_short"].eq(False)].copy()
    feature_columns: Dict[str, pd.Series] = {}

    for section_index, section in enumerate(SECTION_KEYS):
        for display_name, source_column in THEME_PRESENCE_COLUMNS.items():
            feature_name = f"{section} {display_name}"
            feature_columns[feature_name] = data[source_column].map(
                lambda value, idx=section_index: _presence_at(value, idx)
            )

    features = pd.DataFrame(feature_columns, index=data.index, dtype=float)
    metadata = data[["SEX", "AGE2"]].copy()
    ordered_features = list(feature_columns)
    item_labels = list(THEME_PRESENCE_COLUMNS)
    return metadata, features, ordered_features, item_labels


def _sentiment_is_present(entry: object, target_label: str) -> int:
    """Return 1 when target_label appears in a stored sentiment dictionary."""
    if not isinstance(entry, dict):
        return 0
    labels = entry.get("labels", [])
    if not isinstance(labels, (list, tuple, np.ndarray)):
        return 0
    return int(target_label in labels)


def build_sentiment_features(
    source: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Read the sentiment pickle and return metadata plus the 3 x 15 matrix."""
    data = pd.read_pickle(source)
    _required_columns(
        data,
        ["SEX", "AGE2", *SENTIMENT_SOURCE_COLUMNS],
        source,
    )

    feature_columns: Dict[str, pd.Series] = {}
    for section, source_column in zip(SECTION_KEYS, SENTIMENT_SOURCE_COLUMNS):
        for display_name, target_label in SENTIMENT_LABELS.items():
            feature_name = f"{section} {display_name}"
            feature_columns[feature_name] = data[source_column].map(
                lambda entry, label=target_label: _sentiment_is_present(entry, label)
            )

    features = pd.DataFrame(feature_columns, index=data.index, dtype=float)
    metadata = data[["SEX", "AGE2"]].copy()
    ordered_features = list(feature_columns)
    item_labels = list(SENTIMENT_LABELS)
    return metadata, features, ordered_features, item_labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _fit_logit(y: pd.Series, predictors: pd.DataFrame, model_name: str):
    """Fit one Logit model after numeric conversion and complete-case filtering."""
    x = predictors.apply(pd.to_numeric, errors="coerce").astype(float)
    x = sm.add_constant(x, has_constant="add")
    y_numeric = pd.to_numeric(y, errors="coerce")
    valid = ~(x.isna().any(axis=1) | y_numeric.isna())
    x = x.loc[valid]
    y_numeric = y_numeric.loc[valid].astype(float)

    if y_numeric.nunique() != 2:
        raise ValueError(f"{model_name}: outcome must contain both 0 and 1.")

    try:
        return sm.Logit(y_numeric, x).fit(disp=False, maxiter=200)
    except Exception as exc:
        raise RuntimeError(
            f"{model_name} failed. Check for perfect prediction or duplicate/"
            "constant feature columns."
        ) from exc


def _result_table(model_result, feature_names: Sequence[str]) -> pd.DataFrame:
    """Convert a fitted statsmodels result to a tidy coefficient table."""
    ci = model_result.conf_int()
    table = pd.DataFrame(
        {
            "Feature": model_result.params.index,
            "Coefficient": model_result.params.to_numpy(),
            "Odds Ratio": np.exp(model_result.params.to_numpy()),
            "p-value": model_result.pvalues.to_numpy(),
            "CI Lower": ci.iloc[:, 0].to_numpy(),
            "CI Upper": ci.iloc[:, 1].to_numpy(),
        }
    )
    return table.loc[table["Feature"].isin(feature_names)].copy()


def fit_demographic_models(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    Fit the gender model and five age one-vs-rest models.

    Gender coefficient: log odds of Male (1) vs Female (0), controlling for age.
    Age coefficient: log odds of membership in that age group vs all other age
    groups, controlling for gender.
    """
    valid = metadata["SEX"].isin([1, 2]) & metadata["AGE2"].isin(AGE_CODES)
    meta = metadata.loc[valid].copy()
    x_features = features.loc[meta.index, feature_names].astype(float)

    # Gender model: Male=1, Female=0; AGE2=1 is the reference age category.
    age_category = pd.Series(
        pd.Categorical(meta["AGE2"], categories=AGE_CODES),
        index=meta.index,
        name="AGE2",
    )
    age_dummies = pd.get_dummies(
        age_category, prefix="AGE2", drop_first=True, dtype=float
    )
    gender_predictors = pd.concat([x_features, age_dummies], axis=1)
    gender_outcome = meta["SEX"].eq(1).astype(float)
    gender_model = _fit_logit(
        gender_outcome,
        gender_predictors,
        "Gender model",
    )
    gender_table = _result_table(gender_model, feature_names)
    gender_table["Demographic"] = "Gender (Male)"

    # Age models: each group vs all other groups; Gender is the covariate.
    age_predictors = x_features.copy()
    age_predictors["Gender"] = meta["SEX"].eq(1).astype(float)
    age_tables: List[pd.DataFrame] = []
    for age_code in AGE_CODES:
        age_outcome = meta["AGE2"].eq(age_code).astype(float)
        age_model = _fit_logit(
            age_outcome,
            age_predictors,
            f"Age {age_code} one-vs-rest model",
        )
        table = _result_table(age_model, feature_names)
        table["Demographic"] = f"Age{age_code}"
        age_tables.append(table)

    combined = pd.concat([*age_tables, gender_table], ignore_index=True)
    combined["sig_star"] = combined["p-value"].map(significance_stars)
    combined["Section"] = combined["Feature"].str.extract(r"^(Section \d+)")
    combined["Feature label"] = combined["Feature"].str.replace(
        r"^Section \d+\s+", "", regex=True
    )
    combined["Model interpretation"] = np.where(
        combined["Demographic"].eq("Gender (Male)"),
        "Male vs Female, adjusted for age",
        "Age group vs all other age groups, adjusted for gender",
    )
    return combined


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def designer_style(style: int) -> Tuple[LinearSegmentedColormap, str, float]:
    """
    Return the requested designer option.

    1: vivid two-tone gradient with darker cell borders.
    2: vivid two-tone gradient with white borders (recommended default).
    3: higher-variation multi-stop gradient with white borders.
    """
    if style in (1, 2):
        colors = [
            (0.00, "#2474C6"),
            (0.24, "#70A5D7"),
            (0.50, "#F1EFEA"),
            (0.76, "#F3AA8D"),
            (1.00, "#EE6F4F"),
        ]
        edge_color = "#666666" if style == 1 else "#FFFFFF"
        edge_width = 0.75 if style == 1 else 0.90
    elif style == 3:
        colors = [
            (0.00, "#2868B2"),
            (0.22, "#6FA0D2"),
            (0.40, "#B9CDDF"),
            (0.50, "#E8E5E1"),
            (0.62, "#F0CFC2"),
            (0.82, "#F19A78"),
            (1.00, "#D85B49"),
        ]
        edge_color = "#FFFFFF"
        edge_width = 0.90
    else:
        raise ValueError("style must be 1, 2, or 3")

    cmap = LinearSegmentedColormap.from_list(
        f"designer_style_{style}", colors, N=256
    )
    cmap.set_bad("#FFFFFF")
    return cmap, edge_color, edge_width


def _annotation_color(
    face_color: Tuple[float, float, float, float],
    selection: str,
) -> str:
    if selection in {"black", "white"}:
        return selection
    red, green, blue, _ = face_color
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.57 else "white"


def _load_font(font_path: Optional[Path]) -> fm.FontProperties:
    candidates = []
    if font_path is not None:
        candidates.append(font_path)
    candidates.extend(
        [
            Path("C:/Windows/Fonts/Arial.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/Arialbd.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return fm.FontProperties(fname=str(candidate))
    return fm.FontProperties(family="DejaVu Sans")


def _finite_color_limit(values: np.ndarray) -> float:
    """
    Choose a robust symmetric limit so a few extremes do not wash out the plot.

    The most extreme 2.5% of absolute coefficients are color-clipped, but their
    printed values and confidence intervals remain unchanged.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    limit = float(np.quantile(np.abs(finite), 0.975))
    if limit >= 0.1:
        limit = np.ceil(limit * 10.0) / 10.0
    else:
        limit = np.ceil(limit * 100.0) / 100.0
    return max(limit, 1e-6)


def plot_heatmap(
    results: pd.DataFrame,
    ordered_features: Sequence[str],
    item_labels: Sequence[str],
    *,
    title: str,
    y_axis_label: str,
    output_stem: Path,
    style: int,
    text_color: str,
    formats: Sequence[str],
    dpi: int,
    font_path: Optional[Path],
    subtitle: str = SUBTITLE,
    color_limit: Optional[float] = None,
) -> None:
    demographics = [f"Age{code}" for code in AGE_CODES] + ["Gender (Male)"]
    x_labels = [*AGE_LABELS, "Gender (Male)"]

    estimates = (
        results.pivot_table(
            index="Feature",
            columns="Demographic",
            values="Coefficient",
            aggfunc="first",
        )
        .reindex(index=ordered_features, columns=demographics)
        .astype(float)
    )
    p_values = (
        results.pivot_table(
            index="Feature",
            columns="Demographic",
            values="p-value",
            aggfunc="first",
        )
        .reindex(index=ordered_features, columns=demographics)
        .astype(float)
    )
    ci_lower = (
        results.pivot_table(
            index="Feature",
            columns="Demographic",
            values="CI Lower",
            aggfunc="first",
        )
        .reindex(index=ordered_features, columns=demographics)
        .astype(float)
    )
    ci_upper = (
        results.pivot_table(
            index="Feature",
            columns="Demographic",
            values="CI Upper",
            aggfunc="first",
        )
        .reindex(index=ordered_features, columns=demographics)
        .astype(float)
    )

    if estimates.isna().any().any():
        missing = estimates.isna().stack()
        missing = missing[missing].index.tolist()
        raise ValueError(f"Missing heatmap estimates for: {missing[:5]}")

    n_sections = len(SECTION_KEYS)
    n_items = len(item_labels)
    n_columns = len(demographics)
    section_gap = 0.34
    total_height = n_sections * n_items + (n_sections - 1) * section_gap
    positions: List[Tuple[int, str, float]] = []
    for section_index, section in enumerate(SECTION_KEYS):
        offset = section_index * (n_items + section_gap)
        for item_index, item_label in enumerate(item_labels):
            feature = f"{section} {item_label}"
            positions.append((section_index, feature, offset + item_index))

    # Keep annotations legible in the 45-row sentiment heatmap.
    is_dense = len(ordered_features) > 20
    
    figure_width = 13.0 if is_dense else 12.0
    figure_height = 18.0 if is_dense else 15.0
    y_tick_size = 9.2 if is_dense else 12.0
    main_text_size = 8.2 if is_dense else 10.6
    ci_text_size = 6.2 if is_dense else 7.8
    section_text_size = 13.0 if is_dense else 14.5

    font = _load_font(font_path)
    cmap, edge_color, edge_width = designer_style(style)
    limit = color_limit if color_limit is not None else _finite_color_limit(
        estimates.to_numpy()
    )
    if limit <= 0:
        raise ValueError("color_limit must be positive")
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height),
        facecolor="white"
    )
    ax.set_facecolor("white")

    for _, feature, y_position in positions:
        for column_index, demographic in enumerate(demographics):
            value = estimates.loc[feature, demographic]
            face_color = cmap(norm(value))
            ax.add_patch(
                Rectangle(
                    (column_index, y_position),
                    1.0,
                    1.0,
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                )
            )

            annotation_color = _annotation_color(face_color, text_color)
            p_value = p_values.loc[feature, demographic]
            stars = significance_stars(p_value)
            lower = ci_lower.loc[feature, demographic]
            upper = ci_upper.loc[feature, demographic]

            ax.text(
                column_index + 0.5,
                y_position + 0.38,
                f"{value:.2f}{stars}",
                ha="center",
                va="center",
                fontsize=main_text_size,
                fontproperties=font,
                fontweight="bold",
                color=annotation_color,
            )
            ax.text(
                column_index + 0.5,
                y_position + 0.69,
                f"[{lower:.2f}, {upper:.2f}]",
                ha="center",
                va="center",
                fontsize=ci_text_size,
                fontproperties=font,
                color=annotation_color,
            )

    y_positions = [position + 0.5 for _, _, position in positions]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [feature.split(" ", 2)[2] for _, feature, _ in positions],
        rotation=0,
        fontsize=y_tick_size,
        fontproperties=font,
    )
    ax.set_xticks(np.arange(n_columns) + 0.5)
    ax.set_xticklabels(
        x_labels,
        rotation=0,
        fontsize=12.0,
        fontproperties=font,
    )

    # Section block labels, placed to the left of the repeated row names.
    for section_index, section_label in enumerate(SECTION_DISPLAY):
        first_y = section_index * (n_items + section_gap)
        middle_y = first_y + n_items / 2.0
        ax.text(
            -1.10 if is_dense else -1.7,
            middle_y,
            section_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=section_text_size,
            fontproperties=font,
            fontweight="bold",
            clip_on=False,
        )

    ax.set_xlim(0, n_columns)
    ax.set_ylim(total_height, 0)
    ax.tick_params(axis="x", bottom=True, top=False, length=7, width=0.8, pad=8)
    ax.tick_params(axis="y", left=True, right=False, length=7, width=0.8, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel(
        "Demographic",
        fontsize=14.5,
        fontproperties=font,
        labelpad=20,
    )
    ax.set_ylabel(
        y_axis_label,
        fontsize=14.5,
        fontproperties=font,
        labelpad=62,
    )
    # Keep the overall feature label separate from the vertical section label,
    # especially in the taller 45-row sentiment figure.
    ax.yaxis.set_label_coords(-0.34, 0.5)

    # Set the final axes geometry before adding a dedicated colorbar. A
    # dedicated axis keeps the bar exactly as tall as both the 15-row theme
    # heatmap and the much taller 45-row sentiment heatmap.
    left_margin = 0.34 if is_dense else 0.335
    fig.subplots_adjust(
        left=left_margin,
        right=0.855,
        top=0.905,
        bottom=0.075 if is_dense else 0.105,
    )
    heatmap_box = ax.get_position()
    colorbar_ax = fig.add_axes(
        [heatmap_box.x1 + 0.026, heatmap_box.y0, 0.020, heatmap_box.height]
    )

    # Colorbar uses the same negative -> neutral -> positive normalization.
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    actual_limit = float(np.nanmax(np.abs(estimates.to_numpy())))
    extend = "both" if actual_limit > limit else "neither"
    colorbar = fig.colorbar(
        mappable,
        cax=colorbar_ax,
        extend=extend,
        extendfrac=0.012 if is_dense else 0.025,
    )
    colorbar.set_ticks(np.linspace(-limit, limit, 5))
    colorbar.outline.set_visible(False)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.ax.tick_params(labelsize=10.5, length=6, width=0.7)
    colorbar.set_label(
        "Log-odds coefficient",
        rotation=90,
        fontsize=12.5,
        fontproperties=font,
        labelpad=16,
    )

    fig.suptitle(
        title,
        x=0.53,
        y=0.975,
        fontsize=22,
        fontproperties=font,
        fontweight="bold",
    )
    fig.text(
        0.53,
        0.944,
        subtitle,
        ha="center",
        va="top",
        fontsize=12.5,
        fontproperties=font,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output_path = output_stem.with_suffix(f".{extension.lower()}")
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        print(f"Saved {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent if script_dir.name == "Figure Codes" else script_dir
    default_output = project_dir / "Figure Images"

    parser = argparse.ArgumentParser(
        description="Fit and plot the theme and sentiment demographic heatmaps."
    )
    parser.add_argument(
        "--theme-data",
        type=Path,
        default=Path.home() / "raw.pkl",
        help="Path to raw.pkl.",
    )
    parser.add_argument(
        "--sentiment-data",
        type=Path,
        default=Path.home() / "morethan3rawsentiment.pkl",
        help="Path to morethan3rawsentiment.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for figures and coefficient CSV files.",
    )
    parser.add_argument(
        "--style",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="Designer option: 1=dark borders, 2=white borders, 3=varied gradient.",
    )
    parser.add_argument(
        "--text-color",
        choices=("adaptive", "black", "white"),
        default="adaptive",
        help="Annotation text color. Adaptive chooses black/white per cell.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg", "jpg", "tiff"),
        default=("png", "pdf"),
        help="One or more output formats.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Raster output DPI.")
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="Optional .ttf/.otf font file.",
    )
    parser.add_argument("--theme-title", default=THEME_TITLE)
    parser.add_argument("--sentiment-title", default=SENTIMENT_TITLE)
    parser.add_argument("--subtitle", default=SUBTITLE)
    parser.add_argument(
        "--theme-color-limit",
        type=float,
        default=None,
        help="Optional symmetric theme color limit, e.g. 1.0 gives -1 to +1.",
    )
    parser.add_argument(
        "--sentiment-color-limit",
        type=float,
        default=None,
        help="Optional symmetric sentiment color limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for input_path in (args.theme_data, args.sentiment_data):
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}\n"
                "Pass the correct location with --theme-data or --sentiment-data."
            )

    warnings.filterwarnings(
        "ignore",
        message="Maximum Likelihood optimization failed to converge",
    )

    print("Preparing theme indicators and fitting demographic models...")
    theme_meta, theme_features, theme_order, theme_labels = build_theme_features(
        args.theme_data
    )
    theme_results = fit_demographic_models(
        theme_meta,
        theme_features,
        theme_order,
    )
    theme_csv = args.output_dir / "figure6a_theme_heatmap_estimates.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    theme_results.to_csv(theme_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {theme_csv}")

    plot_heatmap(
        theme_results,
        theme_order,
        theme_labels,
        title=args.theme_title,
        y_axis_label="Thematic Feature",
        output_stem=args.output_dir / f"figure6a_theme_heatmap_style{args.style}",
        style=args.style,
        text_color=args.text_color,
        formats=args.formats,
        dpi=args.dpi,
        font_path=args.font,
        subtitle=args.subtitle,
        color_limit=args.theme_color_limit,
    )

    print("Preparing sentiment indicators and fitting demographic models...")
    sent_meta, sent_features, sent_order, sent_labels = build_sentiment_features(
        args.sentiment_data
    )
    sent_results = fit_demographic_models(
        sent_meta,
        sent_features,
        sent_order,
    )
    sent_csv = args.output_dir / "figure6b_sentiment_heatmap_estimates.csv"
    sent_results.to_csv(sent_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {sent_csv}")

    plot_heatmap(
        sent_results,
        sent_order,
        sent_labels,
        title=args.sentiment_title,
        y_axis_label="Sentiment Feature",
        output_stem=args.output_dir / f"figure6b_sentiment_heatmap_style{args.style}",
        style=args.style,
        text_color=args.text_color,
        formats=args.formats,
        dpi=args.dpi,
        font_path=args.font,
        subtitle=args.subtitle,
        color_limit=args.sentiment_color_limit,
    )

    print("Done.")


if __name__ == "__main__":
    main()
