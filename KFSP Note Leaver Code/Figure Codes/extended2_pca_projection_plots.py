# -*- coding: utf-8 -*-
"""
Standalone PCA/K-means projection plots for Figure 4b and Figure 4c.

The script:
    1. Loads the theme and sentiment pickle files.
    2. Reconstructs the theme and sentiment feature matrices.
    3. Fits the same PCA and K-means models as the original analysis.
    4. Projects observations onto PC1-PC2 without refitting K-means in 2D.
    5. Draws small, uniformly circular points and optional convex hulls.
    6. Saves publication-resolution PNG and vector PDF files.

Outputs:
    Figure Images/figure4b_theme_pca_projection.png
    Figure Images/figure4b_theme_pca_projection.pdf
    Figure Images/figure4c_sentiment_pca_projection.png
    Figure Images/figure4c_sentiment_pca_projection.pdf
    Figure Images/figure4b_theme_cluster_key.csv
    Figure Images/figure4c_sentiment_cluster_key.csv

Required packages:
    numpy, pandas, matplotlib, scipy, scikit-learn
"""

from pathlib import Path
import re
import textwrap

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError
except ImportError:
    # Compatibility with the SciPy version in sentiment_environment.
    from scipy.spatial.qhull import QhullError
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ===========================================================================
# User-editable configuration
# ===========================================================================
THEME_DATA_PATH = Path(r"C:\Users\Jae Bin Park\morethan3rawformainfigures")
SENTIMENT_DATA_PATH = Path(
    r"C:\Users\Jae Bin Park\morethan3rawsentimentformainfigures"
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Figure Images"
OUTPUT_DPI = 600

# The designer's example contains translucent convex-hull polygons.
# Change to False if the hulls make the 15-cluster theme plot too crowded.
SHOW_CONVEX_HULLS = True

# Small points make dense areas and individual positions easier to see.
POINT_SIZE = 9
POINT_ALPHA = 0.72
HULL_ALPHA = 0.09

# False preserves the original cluster_name_map labels in the plot legends.
# Set either value to True only if you later want a compact "Type n" legend.
THEME_COMPACT_LEGEND = False
SENTIMENT_COMPACT_LEGEND = False

# The figures are always saved. Change this to True to also open plot windows.
SHOW_FIGURES = False

# Computing the exact silhouette score for all 18,000+ observations can use a
# large amount of memory. A reproducible sample is sufficient for reporting.
SILHOUETTE_SAMPLE_SIZE = 5000


# ===========================================================================
# Labels and model settings
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
    9: (
        "10. Pervasive Love and Gratitude + "
        "Sorry and Shame (Last Section)"
    ),
    10: (
        "11. Pervasive Love and Gratitude + "
        "Sorry and Shame (First & Middle Section)"
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

SENTIMENT_SECTION_COLUMNS = [
    "1st_section_sentiment",
    "2nd_section_sentiment",
    "3rd_section_sentiment",
]

# English feature name -> Korean label stored in each section dictionary.
# Unicode escapes keep the source file portable across Windows encodings.
SENTIMENT_SOURCE_LABELS = {
    "Despair": "\uc808\ub9dd",
    "Defeat": "\ud328\ubc30/\uc790\uae30\ud610\uc624",
    "Exhaustion": "\ud798\ub4e6/\uc9c0\uce68",
    "Anxious": "\ubd88\uc548/\uac71\uc815",
    "Disappointment": "\uc548\ud0c0\uae4c\uc6c0/\uc2e4\ub9dd",
    "Anger": "\ud654\ub0a8/\ubd84\ub178",
    "Hatred": "\uc99d\uc624/\ud610\uc624",
    "Resentment": "\uc5b4\uc774\uc5c6\uc74c",
    "Sadness": "\uc2ac\ud514",
    "Sorrow": "\uc11c\ub7ec\uc6c0",
    "Gratitude": "\uace0\ub9c8\uc6c0",
    "Affection": (
        "\ud750\ubb47\ud568(\uadc0\uc5ec\uc6c0/\uc608\uc068)"
    ),
    "Happiness": "\ud589\ubcf5",
    "Relief": "\uc548\uc2ec/\uc2e0\ub8b0",
    "Neutral": "\uc5c6\uc74c",
}
SENTIMENT_LABELS = list(SENTIMENT_SOURCE_LABELS)
SENTIMENT_COLUMNS = [
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


# ===========================================================================
# Shared helpers
# ===========================================================================
def configure_fonts():
    """Use regular Arial for details and bold Arial for hierarchy."""
    arial_regular = Path(r"C:\Windows\Fonts\arial.ttf")
    arial_bold = Path(r"C:\Windows\Fonts\arialbd.ttf")

    for font_path in (arial_regular, arial_bold):
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def distinct_palette(number_of_colors):
    """Return separated colors from Matplotlib's categorical palettes."""
    base = list(plt.get_cmap("tab20").colors)
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    palette = [base[index] for index in order]

    if number_of_colors > len(palette):
        extra_count = number_of_colors - len(palette)
        hsv = plt.get_cmap("hsv")
        palette.extend(
            hsv(value)
            for value in np.linspace(0, 1, extra_count, endpoint=False)
        )

    return palette[:number_of_colors]


def clean_cluster_description(label):
    """Remove the leading numeric prefix from a descriptive cluster label."""
    return re.sub(r"^\d+\.\s*", "", label)


def validate_input_path(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} pickle was not found:\n{path}\n"
            "Update the corresponding path in the configuration section."
        )


def vector_series_to_matrix(series, expected_width, vector_name):
    """Convert a Series of list-like vectors to a validated numeric matrix."""
    lengths = series.map(
        lambda value: len(value)
        if isinstance(value, (list, tuple, np.ndarray))
        else -1
    )
    invalid = lengths.ne(expected_width)

    if invalid.any():
        bad_rows = list(series.index[invalid][:10])
        raise ValueError(
            f"{vector_name} must contain {expected_width} values per row. "
            f"Invalid example indices: {bad_rows}"
        )

    matrix = np.asarray(series.tolist(), dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError(f"{vector_name} contains NaN or infinite values.")

    return matrix


def fit_pca_kmeans(
    matrix,
    n_components,
    n_clusters,
    random_state,
    analysis_name,
):
    """Fit PCA and K-means using the original analysis settings."""
    pca = PCA(
        n_components=n_components,
        random_state=random_state,
    )
    pca_coordinates = pca.fit_transform(matrix)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(pca_coordinates)

    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(pca_coordinates))
    silhouette = silhouette_score(
        pca_coordinates,
        cluster_labels,
        sample_size=sample_size,
        random_state=random_state,
    )

    first_two = pca.explained_variance_ratio_[:2] * 100
    retained = pca.explained_variance_ratio_.sum() * 100

    print(f"\n{analysis_name}")
    print(f"  Observations: {len(matrix):,}")
    print(f"  Features: {matrix.shape[1]}")
    print(f"  Retained PCs: {n_components}")
    print(f"  Retained variance: {retained:.2f}%")
    print(f"  PC1 variance: {first_two[0]:.2f}%")
    print(f"  PC2 variance: {first_two[1]:.2f}%")
    print(
        f"  Silhouette score ({sample_size:,}-row sample): "
        f"{silhouette:.3f}"
    )

    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  Type {cluster_id + 1}: {count:,} observations")

    return pca, pca_coordinates, cluster_labels


def draw_cluster_hull(ax, points, color):
    """Draw one convex hull when at least three non-collinear points exist."""
    unique_points = np.unique(points, axis=0)
    if len(unique_points) < 3:
        return

    try:
        hull = ConvexHull(unique_points)
    except QhullError:
        return

    boundary = unique_points[hull.vertices]
    boundary = np.vstack([boundary, boundary[0]])

    ax.fill(
        boundary[:, 0],
        boundary[:, 1],
        facecolor=color,
        edgecolor="none",
        alpha=HULL_ALPHA,
        zorder=1,
    )
    ax.plot(
        boundary[:, 0],
        boundary[:, 1],
        color=color,
        linewidth=0.9,
        alpha=0.72,
        zorder=2,
    )


def plot_pca_projection(
    pca_coordinates,
    cluster_labels,
    explained_variance,
    cluster_name_map,
    title,
    output_stem,
    legend_title,
    compact_legend,
):
    """
    Plot a PC1-PC2 projection while preserving high-dimensional clustering.

    K-means was fitted on all retained components. This function displays only
    the first two components and therefore does not alter cluster membership.
    """
    xy = np.asarray(pca_coordinates)[:, :2]
    labels = np.asarray(cluster_labels)
    cluster_ids = sorted(np.unique(labels))

    palette = distinct_palette(len(cluster_ids))
    color_by_cluster = dict(zip(cluster_ids, palette))

    if compact_legend:
        figure_width = 10.4
    elif len(cluster_ids) >= 15:
        figure_width = 16.5
    else:
        figure_width = 13.0
    fig, ax = plt.subplots(
        figsize=(figure_width, 7.5),
        facecolor="white",
    )

    for cluster_id in cluster_ids:
        points = xy[labels == cluster_id]
        color = color_by_cluster[cluster_id]

        if SHOW_CONVEX_HULLS:
            draw_cluster_hull(ax, points, color)

        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=POINT_SIZE,
            marker="o",
            facecolor=color,
            edgecolor="none",
            alpha=POINT_ALPHA,
            rasterized=True,
            zorder=3,
        )

    legend_handles = []
    for cluster_id in cluster_ids:
        if compact_legend:
            legend_label = f"Type {cluster_id + 1}"
        else:
            # Preserve the exact original cluster_name_map label. Wrapping only
            # inserts line breaks so long labels fit beside the PCA panel.
            legend_label = textwrap.fill(
                cluster_name_map[cluster_id],
                width=58,
                break_long_words=False,
                break_on_hyphens=False,
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                label=legend_label,
                markerfacecolor=color_by_cluster[cluster_id],
                markeredgecolor="none",
                markersize=6,
            )
        )

    legend_columns = 3 if compact_legend else 1
    legend_fontsize = 8.3 if len(cluster_ids) >= 15 else 9.5
    legend = ax.legend(
        handles=legend_handles,
        title=legend_title,
        bbox_to_anchor=(1.015, 1),
        loc="upper left",
        ncol=legend_columns,
        frameon=False,
        fontsize=legend_fontsize,
        title_fontsize=10.5,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0,
    )
    legend.get_title().set_fontweight("bold")

    pc1_variance = explained_variance[0] * 100
    pc2_variance = explained_variance[1] * 100

    ax.set_title(
        title,
        fontsize=19,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(
        f"PC1 ({pc1_variance:.1f}% variance)",
        fontsize=14,
        labelpad=10,
    )
    ax.set_ylabel(
        f"PC2 ({pc2_variance:.1f}% variance)",
        fontsize=14,
        labelpad=10,
    )

    ax.tick_params(axis="both", labelsize=11, width=0.8, length=5)
    ax.set_axisbelow(True)
    ax.grid(
        True,
        color="#D4D4D4",
        linewidth=0.65,
        alpha=0.80,
    )
    ax.margins(x=0.04, y=0.06)

    for spine in ax.spines.values():
        spine.set_color("#202020")
        spine.set_linewidth(1.1)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        output_path = OUTPUT_DIR / f"{output_stem}.{extension}"
        fig.savefig(
            output_path,
            format=extension,
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.12,
        )
        print(f"Saved: {output_path}")

    return fig


def save_cluster_key(cluster_name_map, output_stem):
    """Save the compact legend's full cluster descriptions."""
    cluster_key = pd.DataFrame(
        {
            "type": [
                f"Type {cluster_id + 1}"
                for cluster_id in sorted(cluster_name_map)
            ],
            "description": [
                clean_cluster_description(cluster_name_map[cluster_id])
                for cluster_id in sorted(cluster_name_map)
            ],
        }
    )
    output_path = OUTPUT_DIR / f"{output_stem}.csv"
    cluster_key.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")


# ===========================================================================
# Theme projection
# ===========================================================================
def calculate_theme_projection():
    validate_input_path(THEME_DATA_PATH, "Theme")
    theme_data = pd.read_pickle(THEME_DATA_PATH)

    if "full_theme_vector" not in theme_data.columns:
        raise KeyError(
            "Theme data does not contain the 'full_theme_vector' column."
        )

    theme_matrix = vector_series_to_matrix(
        theme_data["full_theme_vector"],
        expected_width=len(THEME_COLUMNS),
        vector_name="full_theme_vector",
    )

    return fit_pca_kmeans(
        matrix=theme_matrix,
        n_components=5,
        n_clusters=15,
        random_state=0,
        analysis_name="Theme PCA/K-means",
    )


# ===========================================================================
# Sentiment projection
# ===========================================================================
def section_sentiment_matrix(section_series):
    """Create one observations-by-sentiments binary matrix."""
    source_labels = list(SENTIMENT_SOURCE_LABELS.values())
    rows = []

    for entry in section_series:
        labels = (
            set(entry.get("labels", []))
            if isinstance(entry, dict)
            else set()
        )
        rows.append(
            [float(source_label in labels) for source_label in source_labels]
        )

    return np.asarray(rows, dtype=float)


def calculate_sentiment_projection():
    validate_input_path(SENTIMENT_DATA_PATH, "Sentiment")
    sentiment_data = pd.read_pickle(SENTIMENT_DATA_PATH)

    missing_columns = [
        column
        for column in SENTIMENT_SECTION_COLUMNS
        if column not in sentiment_data.columns
    ]
    if missing_columns:
        raise KeyError(
            "Sentiment data is missing section columns: "
            + ", ".join(missing_columns)
        )

    available_labels = set()
    for section_column in SENTIMENT_SECTION_COLUMNS:
        for entry in sentiment_data[section_column]:
            if isinstance(entry, dict):
                available_labels.update(entry.get("labels", []))

    missing_features = [
        feature_name
        for feature_name, source_label in SENTIMENT_SOURCE_LABELS.items()
        if source_label not in available_labels
    ]
    if missing_features:
        raise ValueError(
            "These configured sentiment features were not found in the data: "
            + ", ".join(missing_features)
        )

    # The full vector is section-major:
    # all 15 sentiments in Section 1, then Section 2, then Section 3.
    section_matrices = [
        section_sentiment_matrix(sentiment_data[column])
        for column in SENTIMENT_SECTION_COLUMNS
    ]
    sentiment_matrix = np.hstack(section_matrices)

    if sentiment_matrix.shape[1] != len(SENTIMENT_COLUMNS):
        raise ValueError(
            "Unexpected sentiment matrix width: "
            f"{sentiment_matrix.shape[1]}"
        )

    return fit_pca_kmeans(
        matrix=sentiment_matrix,
        n_components=6,
        n_clusters=10,
        random_state=42,
        analysis_name="Sentiment PCA/K-means",
    )


# ===========================================================================
# Run both analyses
# ===========================================================================
def main():
    configure_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Calculating the theme PCA projection...")
    theme_pca, theme_coordinates, theme_clusters = (
        calculate_theme_projection()
    )
    theme_figure = plot_pca_projection(
        pca_coordinates=theme_coordinates,
        cluster_labels=theme_clusters,
        explained_variance=theme_pca.explained_variance_ratio_,
        cluster_name_map=THEME_CLUSTER_NAMES,
        title="Theme-based PCA - K-means clustering",
        output_stem="figure4b_theme_pca_projection",
        legend_title="Theme cluster",
        compact_legend=THEME_COMPACT_LEGEND,
    )
    save_cluster_key(
        THEME_CLUSTER_NAMES,
        "figure4b_theme_cluster_key",
    )

    print("\nCalculating the sentiment PCA projection...")
    sentiment_pca, sentiment_coordinates, sentiment_clusters = (
        calculate_sentiment_projection()
    )
    sentiment_figure = plot_pca_projection(
        pca_coordinates=sentiment_coordinates,
        cluster_labels=sentiment_clusters,
        explained_variance=sentiment_pca.explained_variance_ratio_,
        cluster_name_map=SENTIMENT_CLUSTER_NAMES,
        title="Sentiment-based PCA - K-means clustering",
        output_stem="figure4c_sentiment_pca_projection_with",
        legend_title="Sentiment cluster",
        compact_legend=SENTIMENT_COMPACT_LEGEND,
    )
    save_cluster_key(
        SENTIMENT_CLUSTER_NAMES,
        "figure4c_sentiment_cluster_key",
    )

    print("\nFinished exporting both PCA projection plots.")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(theme_figure)
        plt.close(sentiment_figure)


if __name__ == "__main__":
    main()
