"""
Recreate Supplementary Data 1-4 as designer-style table lollipop plots.

The dark blue point is the observed note-leaver proportion.  Because the two
groups partition each row, the non-leaver proportion is calculated as:

    non_leavers = 100 - note_leavers

Usage
-----
python recreate_supplementary_lollipops.py
python recreate_supplementary_lollipops.py --dpi 300 --formats png pdf svg
python recreate_supplementary_lollipops.py --no-composites
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "supplementary_lollipop_note_leaver_proportions.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "designer_lollipop_outputs"

DARK_BLUE = "#3B73B9"
LIGHT_BLUE = "#9FC3E9"
GRID = "#C7C7C7"
TEXT = "#111111"
WHITE = "#FFFFFF"

PANEL_ORDER = [
    "demographics",
    "incident_context",
    "health_services",
    "psychiatric_symptoms",
    "psychiatric_diagnoses",
    "job_economic",
    "familial_relational",
    "verbal_warning",
    "emotional_warning",
    "behavioral_warning",
]

COMPOSITE_LAYOUTS = {
    1: [["demographics", "incident_context"]],
    2: [["health_services", None], ["psychiatric_symptoms", "psychiatric_diagnoses"]],
    3: [["job_economic", "familial_relational"]],
    4: [["verbal_warning", "emotional_warning"], ["behavioral_warning", None]],
}

EXPECTED_PANEL_ROWS = {
    "demographics": 31,
    "incident_context": 57,
    "health_services": 14,
    "psychiatric_symptoms": 24,
    "psychiatric_diagnoses": 24,
    "job_economic": 26,
    "familial_relational": 28,
    "verbal_warning": 32,
    "emotional_warning": 32,
    "behavioral_warning": 32,
}


def load_data(path: Path) -> dict[str, dict]:
    panels: dict[str, dict] = {}
    seen_rows: set[tuple[str, int, int]] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "supplement",
            "panel_order",
            "panel_id",
            "panel_title",
            "group_order",
            "group",
            "detail_order",
            "detail",
            "note_leavers",
        }
        if set(reader.fieldnames or []) != required:
            missing = required - set(reader.fieldnames or [])
            extra = set(reader.fieldnames or []) - required
            raise ValueError(f"Unexpected CSV columns. Missing={missing}; extra={extra}")

        for raw in reader:
            panel_id = raw["panel_id"].strip()
            group = raw["group"].strip()
            detail = raw["detail"].strip()
            row_key = (
                panel_id,
                int(raw["group_order"]),
                int(raw["detail_order"]),
            )
            if row_key in seen_rows:
                raise ValueError(f"Duplicate row: {row_key}")
            seen_rows.add(row_key)

            value = float(raw["note_leavers"])
            if not 0 <= value <= 100:
                raise ValueError(f"Out-of-range proportion in {row_key}: {value}")

            panel = panels.setdefault(
                panel_id,
                {
                    "supplement": int(raw["supplement"]),
                    "panel_order": int(raw["panel_order"]),
                    "panel_id": panel_id,
                    "panel_title": raw["panel_title"].strip(),
                    "rows": [],
                },
            )
            panel["rows"].append(
                {
                    "group_order": int(raw["group_order"]),
                    "group": group,
                    "detail_order": int(raw["detail_order"]),
                    "detail": detail,
                    "note_leavers": value,
                }
            )

    missing_panels = set(EXPECTED_PANEL_ROWS) - set(panels)
    if missing_panels:
        raise ValueError(f"Missing panels: {sorted(missing_panels)}")

    for panel_id, panel in panels.items():
        panel["rows"].sort(key=lambda row: (row["group_order"], row["detail_order"]))
        expected = EXPECTED_PANEL_ROWS.get(panel_id)
        actual = len(panel["rows"])
        if expected is not None and actual != expected:
            raise ValueError(f"{panel_id}: expected {expected} rows, found {actual}")

    return panels


def wrapped(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def render_panel(panel: dict, output_dir: Path, formats: list[str], dpi: int) -> Path:
    rows = panel["rows"]
    n_rows = len(rows)
    figure_height = max(6.2, 2.8 + 0.39 * n_rows)
    figure_width = 14.0

    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(figure_width, figure_height), facecolor=WHITE)
    ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
    ax.set_facecolor(WHITE)
    ax.set_xlim(-46, 103)
    ax.set_ylim(-1.45, n_rows + 2.45)
    ax.axis("off")

    title_y = n_rows + 1.90
    header_y = n_rows + 0.62
    table_top = n_rows - 0.25
    table_bottom = -0.55
    type_detail_divider = -25
    plot_start = 0

    ax.text(
        28,
        title_y,
        panel["panel_title"],
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=DARK_BLUE,
            markeredgecolor=DARK_BLUE,
            markersize=9,
            label="Note leavers",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=LIGHT_BLUE,
            markeredgecolor=LIGHT_BLUE,
            markersize=9,
            label="Non-leavers",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.975),
        frameon=True,
        fancybox=False,
        framealpha=1,
        facecolor=WHITE,
        edgecolor=TEXT,
        fontsize=11,
        borderpad=0.8,
        labelspacing=0.7,
        handletextpad=0.7,
    )

    ax.text(-35.5, header_y, "Type", ha="center", va="center", fontsize=13, color=TEXT)
    ax.text(-12.5, header_y, "Details", ha="center", va="center", fontsize=13, color=TEXT)
    ax.text(50, header_y, "Proportion (%)", ha="center", va="center", fontsize=13, color=TEXT)

    for x in range(0, 101, 20):
        ax.vlines(x, table_bottom, table_top, color=GRID, linewidth=0.75, zorder=0)
    ax.vlines(
        50,
        table_bottom,
        table_top,
        color=DARK_BLUE,
        linewidth=1.0,
        linestyles=(0, (3, 3)),
        zorder=1,
    )

    ax.vlines(type_detail_divider, table_bottom, table_top, color=TEXT, linewidth=1.0)
    ax.vlines(plot_start, table_bottom, table_top, color=TEXT, linewidth=1.0)

    ax.hlines(table_top, -46, 100, color=TEXT, linewidth=1.8)
    ax.hlines(table_bottom, -46, 100, color=TEXT, linewidth=1.8)

    grouped: OrderedDict[tuple[int, str], list[int]] = OrderedDict()
    for index, row in enumerate(rows):
        grouped.setdefault((row["group_order"], row["group"]), []).append(index)

    for row_index, row in enumerate(rows):
        y = n_rows - 1 - row_index
        note = row["note_leavers"]
        non = 100.0 - note

        ax.hlines(y - 0.5, 0, 100, color=GRID, linewidth=0.55, zorder=0)
        ax.plot([note, non], [y, y], color=LIGHT_BLUE, linewidth=2.8, zorder=2)
        ax.scatter(
            [note],
            [y],
            s=53,
            marker="o",
            facecolor=DARK_BLUE,
            edgecolor=DARK_BLUE,
            linewidth=0.5,
            zorder=4,
        )
        ax.scatter(
            [non],
            [y],
            s=53,
            marker="o",
            facecolor=LIGHT_BLUE,
            edgecolor=LIGHT_BLUE,
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            note - 1.45,
            y,
            f"{note:.1f}%",
            ha="right",
            va="center",
            fontsize=11.5,
            fontweight="medium",
            color=TEXT,
            zorder=5,
        )
        ax.text(
            -2.2,
            y,
            wrapped(row["detail"], 27),
            ha="right",
            va="center",
            fontsize=11.5,
            color=TEXT,
        )

    for (_, group), indices in grouped.items():
        first = min(indices)
        last = max(indices)
        y_center = n_rows - 1 - (first + last) / 2
        y_boundary = n_rows - 1 - last - 0.5
        ax.text(
            -35.5,
            y_center,
            wrapped(group, 19),
            ha="center",
            va="center",
            fontsize=12,
            color=TEXT,
        )
        ax.hlines(y_boundary, -46, 100, color=TEXT, linewidth=1.45, zorder=6)

    for x in (20, 40, 60, 80):
        ax.text(x, -1.06, f"{x}%", ha="center", va="center", fontsize=10.5, color=TEXT)

    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    png_path = panel_dir / f"{panel['panel_id']}.png"

    save_formats = list(dict.fromkeys(formats))
    if "png" not in save_formats:
        save_formats.append("png")
    for extension in save_formats:
        destination = panel_dir / f"{panel['panel_id']}.{extension}"
        fig.savefig(
            destination,
            dpi=dpi if extension == "png" else 300,
            facecolor=WHITE,
            bbox_inches="tight",
            pad_inches=0.12,
        )

    plt.close(fig)
    return png_path


def build_composite(
    supplement: int,
    layout: list[list[str | None]],
    panel_paths: dict[str, Path],
    output_dir: Path,
    dpi: int,
    composite_width_inches: float = 16.4,
) -> Path:
    images = {
        panel_id: Image.open(panel_paths[panel_id]).convert("RGB")
        for row in layout
        for panel_id in row
        if panel_id is not None
    }
    columns = max(len(row) for row in layout)
    rows = len(layout)
    raw_column_widths = []
    for column in range(columns):
        raw_column_widths.append(
            max(
                (
                    images[layout[row][column]].width
                    for row in range(rows)
                    if column < len(layout[row]) and layout[row][column] is not None
                ),
                default=1,
            )
        )
    gap = max(1, round(0.15 * dpi))
    target_width = round(composite_width_inches * dpi)
    usable_width = target_width - gap * (columns - 1)
    scale = usable_width / sum(raw_column_widths)
    for panel_id, image in list(images.items()):
        images[panel_id] = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    column_widths = []
    for column in range(columns):
        column_widths.append(
            max(
                (
                    images[layout[row][column]].width
                    for row in range(rows)
                    if column < len(layout[row]) and layout[row][column] is not None
                ),
                default=1,
            )
        )
    row_heights = []
    for row in range(rows):
        row_heights.append(
            max(
                (
                    images[panel_id].height
                    for panel_id in layout[row]
                    if panel_id is not None
                ),
                default=1,
            )
        )

    canvas_width = sum(column_widths) + gap * (columns - 1)
    canvas_height = sum(row_heights) + gap * (rows - 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)

    y = 0
    for row_index, row in enumerate(layout):
        x = 0
        for column_index in range(columns):
            panel_id = row[column_index] if column_index < len(row) else None
            if panel_id is not None:
                canvas.paste(images[panel_id], (x, y))
            x += column_widths[column_index] + gap
        y += row_heights[row_index] + gap

    composite_dir = output_dir / "composites"
    composite_dir.mkdir(parents=True, exist_ok=True)
    destination = composite_dir / f"supplementary_data_{supplement}_designer.png"
    canvas.save(destination, dpi=(dpi, dpi), optimize=True)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
    )
    parser.add_argument(
        "--no-composites",
        action="store_true",
        help="Skip the four composite PNG figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panels = load_data(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel_paths: dict[str, Path] = {}
    for panel_id in PANEL_ORDER:
        panel_paths[panel_id] = render_panel(
            panels[panel_id],
            args.output_dir,
            list(args.formats),
            args.dpi,
        )

    composite_paths = []
    if not args.no_composites:
        for supplement, layout in COMPOSITE_LAYOUTS.items():
            composite_paths.append(
                build_composite(
                    supplement,
                    layout,
                    panel_paths,
                    args.output_dir,
                    args.dpi,
                )
            )

    print(f"Validated {sum(len(panel['rows']) for panel in panels.values())} rows.")
    print(f"Panel files: {args.output_dir / 'panels'}")
    if composite_paths:
        print(f"Composite files: {args.output_dir / 'composites'}")


if __name__ == "__main__":
    main()
