# -*- coding: utf-8 -*-
"""
Standalone exporter for the two PCA/K-means cluster-center heatmaps.

Outputs:
    Figure Images/figure4b_theme_heatmap.png
    Figure Images/figure4b_theme_heatmap.jpg
    Figure Images/figure4c_sentiment_heatmap.png
    Figure Images/figure4c_sentiment_heatmap.jpg

The calculations intentionally reproduce the supplied original code:
    - Theme: PCA(5) -> KMeans(15) -> PCA inverse-transformed centers.
    - Sentiment: PCA(6) -> KMeans(10) -> PCA inverse-transformed centers.
    - The original sentiment column-name assignment and display reorder are
      retained so the displayed values match the original heatmap exactly.
"""

from pathlib import Path
import re
import textwrap

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ===========================================================================
# User-editable configuration
# ===========================================================================
THEME_DATA_PATH = Path(r"C:\Users\Jae Bin Park\morethan3rawformainfigures")
SENTIMENT_DATA_PATH = Path(
    r"C:\Users\Jae Bin Park\morethan3rawsentimentformainfigures"
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Figure Images"
OUTPUT_DPI = 600

# "white" is the designer-recommended version. Change to "black" if desired.
BORDER_VERSION = "white"


# ===========================================================================
# Shared styling
# ===========================================================================
SECTION_DISPLAY_NAMES = ["Opening", "Middle", "Ending"]

BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "cluster_center_blues",
    [
        "#EEF2F6",
        "#D7E2EE",
        "#AFC6DF",
        "#7EA9D6",
        "#4A8CCF",
        "#2E6BA7",
        "#173E69",
    ],
)


def configure_fonts():
    """Use regular Arial for details and bold Arial for hierarchy."""
    arial_regular = Path(r"C:\Windows\Fonts\arial.ttf")
    arial_bold = Path(r"C:\Windows\Fonts\arialbd.ttf")

    for font_path in (arial_regular, arial_bold):
        if font_path.exists():
            # str() keeps compatibility with older Matplotlib versions.
            fm.fontManager.addfont(str(font_path))

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def contrasting_text_color(rgba):
    """Choose dark or white annotation text using cell luminance."""
    red, green, blue = rgba[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111111" if luminance > 0.60 else "#FFFFFF"


def format_cluster_label(cluster_name_map, cluster_id, wrap_width):
    """Format labels as 'Type n — description' and wrap long descriptions."""
    raw_label = cluster_name_map.get(
        cluster_id,
        f"{cluster_id + 1}. Cluster",
    )
    description = re.sub(r"^\d+\.\s*", "", raw_label)
    wrapped_description = textwrap.fill(
        description,
        width=wrap_width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return f"Type {cluster_id + 1} — {wrapped_description}"


def get_border_style():
    if BORDER_VERSION == "white":
        return {
            "cell_edge": "#FFFFFF",
            "cell_linewidth": 0.8,
            "section_edge": "#FFFFFF",
        }
    if BORDER_VERSION == "black":
        return {
            "cell_edge": "#202020",
            "cell_linewidth": 0.4,
            "section_edge": "#202020",
        }
    raise ValueError("BORDER_VERSION must be 'white' or 'black'")


def export_heatmap(
    values_df,
    x_labels,
    row_labels,
    columns_per_section,
    title,
    subtitle,
    output_stem,
    figsize,
    x_rotation,
    x_fontsize,
    annotation_fontsize,
    section_linewidth,
    group_y,
    left,
    right,
    top,
    bottom,
    title_x,
    colorbar_fraction,
    colorbar_aspect,
):
    """Draw one styled heatmap and export it as both PNG and JPG."""
    values = values_df.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    border = get_border_style()

    # This matches imshow's original automatic min/max normalization.
    # Small values outside [0, 1] are retained PCA reconstruction artifacts.
    norm = mcolors.Normalize(
        vmin=float(np.nanmin(values)),
        vmax=float(np.nanmax(values)),
    )

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    heatmap = ax.pcolormesh(
        values,
        cmap=BLUE_CMAP,
        norm=norm,
        shading="flat",
        edgecolors=border["cell_edge"],
        linewidth=border["cell_linewidth"],
        antialiased=True,
    )
    ax.invert_yaxis()
    ax.set_aspect("auto")

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(
        x_labels,
        rotation=x_rotation,
        ha="right",
        rotation_mode="anchor",
        fontsize=x_fontsize,
        fontweight="normal",
    )
    ax.tick_params(axis="x", length=6, width=0.6, pad=7)

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(
        row_labels,
        fontsize=10,
        fontweight="normal",
        ha="right",
    )
    ax.tick_params(axis="y", length=0, pad=10)

    # Separate Opening, Middle, and Ending.
    for boundary in (
        columns_per_section,
        2 * columns_per_section,
    ):
        ax.axvline(
            boundary,
            color=border["section_edge"],
            linewidth=section_linewidth,
            zorder=4,
        )

    # Cell values use the same two-decimal formatting as the original code.
    for row in range(n_rows):
        for col in range(n_cols):
            value = values[row, col]
            ax.text(
                col + 0.5,
                row + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                fontweight="bold",
                color=contrasting_text_color(BLUE_CMAP(norm(value))),
            )

    for section_index, section_name in enumerate(SECTION_DISPLAY_NAMES):
        centre = (
            section_index * columns_per_section
            + columns_per_section / 2
        )
        ax.text(
            centre,
            group_y,
            section_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        title,
        x=title_x,
        y=0.965,
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        title_x,
        0.920,
        subtitle,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="normal",
    )

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
        fraction=colorbar_fraction,
        pad=0.03,
        aspect=colorbar_aspect,
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=9, width=0.6, length=6, pad=6)
    colorbar.set_label(
        "Reconstructed cluster-center value",
        fontsize=11,
        fontweight="normal",
        labelpad=14,
    )

    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "jpg"):
        output_path = OUTPUT_DIR / f"{output_stem}.{extension}"
        fig.savefig(
            output_path,
            format=extension,
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.15,
        )
        print(f"Saved: {output_path}")

    plt.close(fig)


# ===========================================================================
# Figure 4b: theme heatmap
# ===========================================================================
THEMES = [
    "Sorry and Shame",
    "Love and Gratitude",
    "Burden",
    "Despair",
    "Post-mortem Affairs",
]
THEME_SECTIONS = ["Section 1", "Section 2", "Section 3"]
THEME_COLUMNS = [
    f"{section} {theme}"
    for section in THEME_SECTIONS
    for theme in THEMES
]

THEME_CLUSTER_NAMES = {
    0: "1. Love and Gratitude",
    1: "2. Sorry and Shame (First & Last Section)",
    2: "3. Sorry and Shame (First & Middle Section)",
    3: "4. Sorry and Shame + Love and Gratitude",
    4: "5. Theme-absent",
    5: "6. Sorry and Shame (Middle & Last Section)",
    6: "7. Sorry and Shame (First Section)",
    7: "8. Pervasive Sorry and Shame",
    8: "9. Sorry and Shame (Last Section)",
    9: "10. Pervasive Love and Gratitude + Sorry and Shame (Last Section)",
    10: (
        "11. Pervasive Love and Gratitude + Sorry and Shame "
        "(First & Middle Section)"
    ),
    11: "12. Sorry and Shame (Middle Section)",
    12: (
        "13. Sorry and Shame + Love and Gratitude "
        "(First & Last Section)"
    ),
    13: (
        "14. Pervasive Sorry and Shame + Post-Mortem Affairs "
        "(Middle & Last Section)"
    ),
    14: (
        "15. Sorry and Shame + Love and Gratitude (First Section) "
        "+ Post-Mortem Affairs (Last Section)"
    ),
}


def calculate_theme_centers():
    """Reproduce the original Figure 4b matrix exactly."""
    theme_data = pd.read_pickle(THEME_DATA_PATH)
    theme_matrix = pd.DataFrame(
        theme_data["full_theme_vector"].tolist(),
        index=theme_data.index,
        columns=THEME_COLUMNS,
        dtype=float,
    )

    pca = PCA(n_components=5, random_state=0)
    x_pca = pca.fit_transform(theme_matrix)

    kmeans = KMeans(
        n_clusters=15,
        random_state=0,
        n_init=10,
    )
    kmeans.fit(x_pca)

    theme_centers = pd.DataFrame(
        pca.inverse_transform(kmeans.cluster_centers_),
        columns=THEME_COLUMNS,
    )
    theme_centers.index = range(15)
    return theme_centers


def export_theme_heatmap(theme_centers):
    row_labels = [
        format_cluster_label(THEME_CLUSTER_NAMES, cluster_id, 54)
        for cluster_id in range(15)
    ]

    export_heatmap(
        values_df=theme_centers[THEME_COLUMNS],
        x_labels=THEMES * len(THEME_SECTIONS),
        row_labels=row_labels,
        columns_per_section=len(THEMES),
        title="Theme Profiles of PCA–K-Means Clusters",
        subtitle=(
            "PCA-reconstructed cluster centers across opening, middle, "
            "and ending sections"
        ),
        output_stem="figure4b_theme_heatmap",
        figsize=(16, 11),
        x_rotation=36,
        x_fontsize=10,
        annotation_fontsize=8,
        section_linewidth=8.0 if BORDER_VERSION == "white" else 2.4,
        group_y=-0.18,
        left=0.38,
        right=0.91,
        top=0.86,
        bottom=0.18,
        title_x=0.57,
        colorbar_fraction=0.035,
        colorbar_aspect=34,
    )


# ===========================================================================
# Figure 4c: sentiment heatmap
# ===========================================================================
SENTIMENT_SECTION_COLUMNS = [
    "1st_section_sentiment",
    "2nd_section_sentiment",
    "3rd_section_sentiment",
]

# English display label -> Korean label stored in each section's dictionary.
SENTIMENT_SOURCE_LABELS = {
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
SENTIMENT_LABELS = list(SENTIMENT_SOURCE_LABELS)

# Preserve the original sentiment-major column-name assignment.
SENTIMENT_PCA_COLUMNS = [
    f"Section {section_index + 1} {sentiment}"
    for sentiment in SENTIMENT_LABELS
    for section_index in range(3)
]

# Preserve the original section-major reorder used for the heatmap.
SENTIMENT_DISPLAY_COLUMNS = [
    f"Section {section_index + 1} {sentiment}"
    for section_index in range(3)
    for sentiment in SENTIMENT_LABELS
]

SENTIMENT_CLUSTER_NAMES = {
    0: "1. Pervasive Hostility",
    1: "2. Defeat + Exhaustion",
    2: "3. Hatred + Resentment",
    3: "4. Emotionally Flat",
    4: "5. Affection + Happiness",
    5: "6. Hopelessness + Affiliative Remarks",
    6: "7. Hopelessness + Hostility + Affiliative Remarks",
    7: "8. Emotional Turmoil",
    8: "9. Defeat + Exhaustion + Happiness",
    9: "10. Despair + Defeat + Exhaustion",
}


def section_sentiment_vector(entry):
    """Return the original 15 binary indicators for one note section."""
    labels = set(entry.get("labels", [])) if isinstance(entry, dict) else set()
    return [
        float(source_label in labels)
        for source_label in SENTIMENT_SOURCE_LABELS.values()
    ]


def calculate_sentiment_centers():
    """Reproduce the original Figure 4c matrix exactly."""
    sentiment_data = pd.read_pickle(SENTIMENT_DATA_PATH)

    section_matrices = []
    for section_column in SENTIMENT_SECTION_COLUMNS:
        section_matrix = np.asarray(
            [
                section_sentiment_vector(entry)
                for entry in sentiment_data[section_column]
            ],
            dtype=float,
        )
        section_matrices.append(section_matrix)

    # The numeric vector is section-major, as in the supplied original code.
    full_sentiment_matrix = np.hstack(section_matrices)

    # The supplied code assigned sentiment-major names to that numeric vector.
    # This mapping is intentionally retained for exact output parity.
    sentiment_matrix = pd.DataFrame(
        full_sentiment_matrix,
        index=sentiment_data.index,
        columns=SENTIMENT_PCA_COLUMNS,
    )

    pca = PCA(n_components=6, random_state=42)
    x_pca = pca.fit_transform(sentiment_matrix)

    kmeans = KMeans(
        n_clusters=10,
        random_state=42,
        n_init=10,
    )
    kmeans.fit(x_pca)

    sentiment_centers = pd.DataFrame(
        pca.inverse_transform(kmeans.cluster_centers_),
        columns=SENTIMENT_PCA_COLUMNS,
    )
    sentiment_centers = sentiment_centers[SENTIMENT_DISPLAY_COLUMNS]
    sentiment_centers.index = range(10)
    return sentiment_centers


def export_sentiment_heatmap(sentiment_centers):
    row_labels = [
        format_cluster_label(SENTIMENT_CLUSTER_NAMES, cluster_id, 48)
        for cluster_id in range(10)
    ]

    export_heatmap(
        values_df=sentiment_centers[SENTIMENT_DISPLAY_COLUMNS],
        x_labels=SENTIMENT_LABELS * len(SENTIMENT_SECTION_COLUMNS),
        row_labels=row_labels,
        columns_per_section=len(SENTIMENT_LABELS),
        title="Sentiment Profiles of PCA–K-Means Clusters",
        subtitle=(
            "PCA-reconstructed cluster centers across opening, middle, "
            "and ending sections"
        ),
        output_stem="figure4c_sentiment_heatmap",
        figsize=(24, 9.5),
        x_rotation=48,
        x_fontsize=9,
        annotation_fontsize=6.3,
        section_linewidth=7.0 if BORDER_VERSION == "white" else 2.2,
        group_y=-0.18,
        left=0.28,
        right=0.94,
        top=0.84,
        bottom=0.25,
        title_x=0.59,
        colorbar_fraction=0.018,
        colorbar_aspect=30,
    )


# ===========================================================================
# Run both figures
# ===========================================================================
def main():
    configure_fonts()

    print("Calculating Figure 4b theme cluster centers...")
    theme_centers = calculate_theme_centers()
    export_theme_heatmap(theme_centers)

    print("Calculating Figure 4c sentiment cluster centers...")
    sentiment_centers = calculate_sentiment_centers()
    export_sentiment_heatmap(sentiment_centers)

    print("Finished exporting both heatmaps.")


if __name__ == "__main__":
    main()
