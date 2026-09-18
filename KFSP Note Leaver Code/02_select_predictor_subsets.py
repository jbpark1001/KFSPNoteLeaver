"""Order predictors by complete-case retention and export eight candidate cuts.

The greedy selection rule and the manually selected cut sizes are unchanged
from the manuscript analysis. The eight cut sizes are 32, 43, 50, 58, 68,
84, 97, and 143 columns. Because filenames are zero-indexed, manuscript
"Subset 3" is ``Cut_2_DF_50Columns.xlsx``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "derived" / "cleaned_demographic_data.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "derived" / "cuts"
DEFAULT_QC_DIR = PROJECT_ROOT / "outputs" / "tables"

MISSING_VALUE = -1
CUT_SIZES = (32, 43, 50, 58, 68, 84, 97, 143)


def maximize_complete_rows_greedy(
    df: pd.DataFrame,
    missing_value: int = MISSING_VALUE,
    candidate_columns: list[str] | None = None,
) -> tuple[list[str], list[int], list[int]]:
    """Greedily add the column that preserves the most complete rows.

    Ties are resolved by the source-data column order, matching the original
    implementation's strict ``kept > best_kept`` comparison.
    """
    if candidate_columns is None:
        candidate_columns = df.columns.tolist()

    observed = df[candidate_columns].to_numpy() != missing_value
    column_index = {column: index for index, column in enumerate(candidate_columns)}

    selected: list[str] = []
    remaining = candidate_columns.copy()
    current_mask = np.ones(len(df), dtype=bool)
    kept_counts: list[int] = []
    marginal_drops: list[int] = []

    while remaining:
        best_column: str | None = None
        best_kept = -1
        best_mask: np.ndarray | None = None

        for column in remaining:
            candidate_mask = current_mask & observed[:, column_index[column]]
            kept = int(candidate_mask.sum())
            if kept > best_kept:
                best_column = column
                best_kept = kept
                best_mask = candidate_mask

        if best_column is None or best_mask is None:
            raise RuntimeError("Greedy selection failed to choose a remaining column.")

        previous_kept = int(current_mask.sum())
        selected.append(best_column)
        kept_counts.append(best_kept)
        marginal_drops.append(previous_kept - best_kept)
        current_mask = best_mask
        remaining.remove(best_column)

    return selected, kept_counts, marginal_drops


def export_retention_plot(
    kept_counts: list[int], cut_sizes: tuple[int, ...], output_path: Path
) -> None:
    """Save the complete-case retention curve used to identify candidate cuts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(kept_counts) + 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x, kept_counts, color="#2166AC", linewidth=1.2)
    ax.scatter(cut_sizes, [kept_counts[k - 1] for k in cut_sizes], s=24, color="#B2182B")
    for subset_number, size in enumerate(cut_sizes, start=1):
        ax.annotate(
            f"Subset {subset_number}",
            (size, kept_counts[size - 1]),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Number of included columns")
    ax.set_ylabel("Rows retained as complete cases")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qc-dir", type=Path, default=DEFAULT_QC_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Cleaned dataframe not found: {args.input}\n"
            "Run analysis/01_reorganize_demographic_data.py first."
        )

    dataframe = pd.read_excel(args.input)
    if max(CUT_SIZES) > dataframe.shape[1]:
        raise ValueError(
            f"Largest cut requires {max(CUT_SIZES)} columns, but the cleaned "
            f"dataframe has only {dataframe.shape[1]}."
        )

    selected, kept_counts, marginal_drops = maximize_complete_rows_greedy(dataframe)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.qc_dir.mkdir(parents=True, exist_ok=True)

    selection_table = pd.DataFrame(
        {
            "selection_step": np.arange(1, len(selected) + 1),
            "column": selected,
            "complete_rows_retained": kept_counts,
            "marginal_rows_dropped": marginal_drops,
        }
    )
    selection_table.to_csv(args.qc_dir / "greedy_column_order.csv", index=False)

    manifest_rows: list[dict[str, object]] = []
    for cut_index, column_count in enumerate(CUT_SIZES):
        filename = f"Cut_{cut_index}_DF_{column_count}Columns.xlsx"
        columns = selected[:column_count]
        dataframe[columns].to_excel(args.output_dir / filename, index=False)
        for position, column in enumerate(columns, start=1):
            manifest_rows.append(
                {
                    "manuscript_subset": cut_index + 1,
                    "file_cut_index": cut_index,
                    "column_count": column_count,
                    "expected_complete_rows": kept_counts[column_count - 1],
                    "column_position": position,
                    "column": column,
                    "filename": filename,
                }
            )

    pd.DataFrame(manifest_rows).to_csv(
        args.qc_dir / "candidate_cut_manifest.csv", index=False
    )
    export_retention_plot(
        kept_counts,
        CUT_SIZES,
        args.qc_dir / "complete_case_retention_curve.png",
    )

    selected_n = kept_counts[CUT_SIZES[2] - 1]
    print(f"Saved {len(CUT_SIZES)} candidate cuts to: {args.output_dir}")
    print(
        "Selected manuscript model: Subset 3 = Cut_2 "
        f"({CUT_SIZES[2]} columns; {selected_n:,} complete rows before R exclusions)."
    )


if __name__ == "__main__":
    main()
