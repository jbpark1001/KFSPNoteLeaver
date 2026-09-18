# Reordered_Levels_Comparative_ForestPlots_Nature_TableStyle_OnePage.py
# Nature-ready table-style forest plots from the existing R-exported Excel tables.
# This revision preserves the source data workflow and does not modify the original script.
# - Level-wise uses: or_table_*.xlsx
# - Factor-wise uses: var_tests_*.xlsx + or_table_*.xlsx
#
# Save figures as PNGs. Replace the example file paths below with your own.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple,Optional
import re
import matplotlib.transforms as mtransforms
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import textwrap
#%%
# --- Fonts: use Roboto everywhere ---
from matplotlib import font_manager as fm, rcParams

# If you bundle the font file with the script, point to it once:

fm.fontManager.addfont(r"C:\windows\fonts\arialbd.ttf")   # (optional)

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],#,"Roboto", "DejaVu Sans"],
    "axes.unicode_minus": False,   # so minus signs look right
})
#%%
REF_FLAG_COL = "_is_reference_row"

# --- add to imports near top ---
import re

# --- helper: strip "[Reference: ...]" -> "..."
def _strip_reference_caption(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    m = re.match(r"^\[Reference:\s*(.*?)\s*\]$", s.strip())
    return m.group(1) if m else s

# --- helper: build sections from a label order that includes headers ---
def _sections_from_label_order(label_order: List[str],
                               headers: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Returns a list of (caption, labels_in_section) in the order they appear.
    Only labels that are actually present in the plot data will be kept later.
    """
    hdr_set = set(headers)
    sections = []
    cur_caption = None
    cur_items = []
    for lab in label_order:
        if lab in hdr_set:
            if cur_caption is not None:
                sections.append((cur_caption, cur_items))
            cur_caption = lab
            cur_items = []
        else:
            cur_items.append(lab)
    if cur_caption is not None:
        sections.append((cur_caption, cur_items))
    return sections

#%%
# -----------------------------
# Helpers to read & clean tables
# -----------------------------
def _make_unique(seq):
    """Return a list with duplicates removed, preserving first occurrence."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_variable_from_term(term: str) -> str:
    """
    Mirror your R regex logic to pull the factor name from a dummy 'term'.
    Works for 'VAR:level', 'VAR.level', or 'VAR<digits>' patterns.
    """
    v = term
    # if R used var:level naming
    if ":" in v:
        v = v.split(":", 1)[0]
    # if var.level naming
    if "." in v:
        v = v.split(".", 1)[0]
    # fallback: strip trailing digits
    i = 0
    for i, ch in enumerate(v):
        pass
    # simple regex-free fallback: cut trailing digits
    while len(v) and v[-1].isdigit():
        v = v[:-1]
    return v

def load_or_table(or_xlsx: str) -> pd.DataFrame:
    """
    Read the level-wise OR table exported by R (broom::tidy with exponentiate=TRUE).
    Expected columns (per your example): term, estimate (OR), conf.low, conf.high, std.error, p.value
    Returns a cleaned DataFrame with a 'Variable' column.
    """
    df = pd.read_excel(or_xlsx)
    # standardize columns (robust to minor renamings)
    cols = {c.lower(): c for c in df.columns}
    # drop intercept, NA ORs
    term_col = cols.get("term", "term")
    est_col  = cols.get("estimate", "estimate")
    lo_col   = cols.get("conf.low", "conf.low")
    hi_col   = cols.get("conf.high", "conf.high")
    se_col   = cols.get("std.error", "std.error")
    p_col    = cols.get("p.value", "p.value")

    out = df.copy()
    out = out[out[term_col] != "(Intercept)"].copy()
    out = out.loc[~out[est_col].isna()].copy()

    # add factor name
    out["Variable"] = out[term_col].astype(str).map(_extract_variable_from_term)
    # convenience: log OR and weights (SE is on log-odds scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["logOR"] = np.log(out[est_col].astype(float))
    out["SE"] = pd.to_numeric(out[se_col], errors="coerce")
    out["weight"] = 1.0 / (out["SE"]**2)
    out["weight"] = out["weight"].replace([np.inf, -np.inf], np.nan)

    # keep key cols with consistent names
    out = out.rename(columns={
        term_col: "term",
        est_col: "OR",
        lo_col: "CI_low",
        hi_col: "CI_high",
        p_col: "p_value"
    })
    return out[["term", "Variable", "OR", "CI_low", "CI_high", "SE", "logOR", "weight", "p_value"]]

def load_var_tests(var_xlsx: str) -> pd.DataFrame:
    """
    Read the factor-wise p-values table from R (drop1/ANOVA).
    Expected columns: 'Variable', 'Factor_p_value' (per your code).
    """
    df = pd.read_excel(var_xlsx)
    # Be tolerant to column naming
    cols = {c.lower(): c for c in df.columns}
    var_col = cols.get("variable", "Variable")
    p_col   = None
    for k in cols:
        if "p_value" in k or "pr(>chi" in k or "factor_p_value" in k:
            p_col = cols[k]
            break
    if p_col is None:
        raise ValueError("Couldn't find a factor p-value column in var_tests file.")
    out = df.rename(columns={var_col: "Variable", p_col: "Factor_p_value"})
    return out[["Variable", "Factor_p_value"]]

# ---------------------------------
# Aggregation for factor-wise ORs
# ---------------------------------

def aggregate_factor_or(or_df: pd.DataFrame, how: str = "max_abs") -> pd.DataFrame:
    """
    Collapse level-wise ORs into one representative OR per factor (Variable).
    Options:
      - how='max_abs'  : pick the level with the largest |log(OR)|
      - how='median'   : exponentiate(median(log(OR)) across levels)
      - how='ivw'      : inverse-variance weighted mean on log(OR) (requires SE)
    Returns DataFrame: Variable, OR, CI_low, CI_high, method
    """
    method = how
    out_rows = []
    for var, g in or_df.groupby("Variable", dropna=False):
        g = g.dropna(subset=["OR", "CI_low", "CI_high", "logOR"])
        if g.empty:
            continue

        if how == "max_abs":
            # choose the level with maximum absolute log(OR)
            idx = np.argmax(np.abs(g["logOR"].values))
            row = g.iloc[idx]
            out_rows.append({"Variable": var, "OR": row["OR"], "CI_low": row["CI_low"], "CI_high": row["CI_high"], "method": method, "term": row["term"]})

        elif how == "median":
            med_log = np.median(g["logOR"].values)
            OR = float(np.exp(med_log))
            # conservative CI via levels' CIs envelope
            lo = float(g["CI_low"].min())
            hi = float(g["CI_high"].max())
            out_rows.append({"Variable": var, "OR": OR, "CI_low": lo, "CI_high": hi, "method": method, "term": "(median of levels)"})

        elif how == "ivw":
            gg = g.dropna(subset=["weight"])
            if gg.empty or (gg["weight"] <= 0).all():
                # fallback to max_abs if no SE/weights
                idx = np.argmax(np.abs(g["logOR"].values))
                row = g.iloc[idx]
                out_rows.append({"Variable": var, "OR": row["OR"], "CI_low": row["CI_low"], "CI_high": row["CI_high"], "method": "max_abs(fallback)", "term": row["term"]})
            else:
                w = gg["weight"].values
                L = gg["logOR"].values
                mu = np.sum(w * L) / np.sum(w)
                se_mu = math.sqrt(1.0 / np.sum(w))
                OR = float(np.exp(mu))
                lo = float(np.exp(mu - 1.96 * se_mu))
                hi = float(np.exp(mu + 1.96 * se_mu))
                out_rows.append({"Variable": var, "OR": OR, "CI_low": lo, "CI_high": hi, "method": method, "term": "(IVW across levels)"})
        else:
            raise ValueError("how must be one of {'max_abs', 'median', 'ivw'}")
    return pd.DataFrame(out_rows)

# -----------------------------
# Plotting
# -----------------------------
def forest_plot(data: pd.DataFrame,
                y_col: str,
                x_col: str = "OR",
                lo_col: str = "CI_low",
                hi_col: str = "CI_high",
                group_col: str = "model",
                title: str = "",
                filename: str = "forest.png",
                max_items: int = 40,
                logx: bool = True,
                figsize: Tuple[int, int] = (8, 0),
                # --- NEW ---
                section_headers: Optional[List[str]] = None,
                draw_section_dividers: bool = True,
                label_order: Optional[List[str]] = None,
                reference_labels: Optional[List[str]] = None,
                category_map: Optional[Dict[str, str]] = None,  # {y_key -> category}
                category_order: Optional[List[str]] = None,     # ["Demographic","Context",...]
                label_map: Optional[Dict[str, str]] = None,     # {y_key -> pretty label}
                y_key_col: Optional[str] = None,                # e.g. "term" (levelwise)
                within_order: str = "effect"                   # "effect"|"alpha"|"none"
               ):
    """
    If category_map is provided, items are ordered by category (per category_order).
    Labels can be customized via label_map.
    If no category_map is provided, falls back to effect-size ordering like before.
    """
    df = data.copy()
    df["dist"] = np.abs(np.log(df[x_col].astype(float)))

    # Default behavior (no categories): order by effect size like before
    if category_map is None:
        if label_order is not None:
        # Manual order overrides everything
            order = [lab for lab in label_order if lab in set(df[y_col].astype(str))]
        else:
            order = (df.groupby(y_col)["dist"].max()
                       .sort_values(ascending=False).index.tolist())

        df[y_col] = pd.Categorical(df[y_col], categories=order, ordered=True)
        df = df.sort_values([y_col, group_col], ascending=[True, True])

        n = len(order)
        if figsize[1] == 0:
            height = max(3, min(20, 0.4 * n + 2))
            figsize = (figsize[0], height)

        fig, ax = plt.subplots(figsize=figsize)
        models = df[group_col].unique().tolist()
        y_positions = {key: i for i, key in enumerate(order)}
        k = len(models)
        offsets = np.linspace(-0.25, 0.25, k) if k > 1 else [0]
        
        palette = sns.color_palette("deep", n_colors=len(models))
        
        if REF_FLAG_COL not in df.columns:
            df[REF_FLAG_COL] = False
            
            # prepare palette
        if len(models) > 1:
            # multiple models → color by model
            palette = sns.color_palette("deep", n_colors=len(models))
            color_map = {m: palette[i] for i, m in enumerate(models)}
        else:
            # single model → color each variable differently
            palette = sns.color_palette("deep", n_colors=len(order))
            color_map = {lab: palette[i] for i, lab in enumerate(order)}

        for mi, m in enumerate(models):
            sub  = df[df[group_col] == m]
            draw = sub[~sub[REF_FLAG_COL]].copy()
        
            y = np.array([y_positions[val] + offsets[mi] for val in draw[y_col]])
        
            # asymmetric x-errors: distance from point to each CI edge
            x = draw[x_col].to_numpy(float)
            xerr = np.vstack([x - draw[lo_col].to_numpy(float),
                              draw[hi_col].to_numpy(float) - x])
            
            for xi, yi, lo, hi, lab in zip(x, y, draw[lo_col], draw[hi_col], draw[y_col]):
                if len(models) > 1:
                    c = color_map[m]         # color by model
                else:
                    c = color_map[lab]       # color by variable label
        
                ax.errorbar(
                    xi, yi,
                    xerr=[[xi - lo], [hi - xi]],
                    fmt='o',
                    capsize=3.0,
                    elinewidth=1.5,
                    capthick=1.2,
                    markersize=7.0,
                    linestyle='none',
                    color=c,
                    label=m if (mi == 0 and len(models) > 1) else None,
                    zorder=3,
                    markeredgecolor  = "none"
                )
            
            # ax.scatter(x, y, s=18, label=m, zorder=3,
            #    color=col, edgecolor="none")

        ax.axvline(1.0, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(order,fontdict={"family": "Arial"}, fontsize=17)
        ax.invert_yaxis()
        ax.set_xlabel("Odds Ratio",fontdict={"family": "Arial"})
        ax.set_title(title,fontdict={"family": "Arial"})
        if logx:
            ax.set_xscale("log")
        if len(models) > 1:
            #ax.legend(title="Model", loc="best", fontsize=6)
            handles, labels = ax.get_legend_handles_labels()
            by_label = {}
            for h, l in zip(handles, labels):
                if l and l != "_nolegend_" and l not in by_label:
                    by_label[l] = h
            
            fp_label = fm.FontProperties(family='Arial', size=6)
            fp_title = fm.FontProperties(family='Arial', weight='bold', size=7)
            
            ax.legend(by_label.values(), by_label.keys(), title="Model", prop=fp_label).set_title("Model", prop=fp_title)
            
        fig.tight_layout()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=600)
        plt.close(fig)
        return

    # --- NEW path: category-driven ordering & labels ---
    if y_key_col is None:
        y_key_col = y_col  # usually "term" (levelwise) or "Variable" (factorwise)
    
    df2, y_order = _apply_labels_and_categories(
        df.assign(key=df[y_key_col]),
        key_col="key",
        label_map=label_map,
        category_map=category_map,
        category_order=category_order,
        within_order=within_order,
        label_order=label_order
    )
    
    # respect max_items AFTER ordering
    if max_items is not None and len(y_order) > max_items:
        y_order = y_order[:max_items]
        df2 = df2[df2["label"].isin(y_order)]
    
    # --- Use the order we just built; do NOT recompute by effect size ---
    y_order = _make_unique([o for o in y_order if pd.notna(o)])
    
    # --- NEW: convert "[Reference: ...]" rows into OUTER CAPTIONS & DIVIDERS ---
    # Detect headers: either explicitly passed via section_headers OR inferred from label_order
    if section_headers is None and label_order is not None:
        section_headers = [lab for lab in label_order if isinstance(lab, str) and lab.startswith("[Reference:")]
    section_headers = section_headers or []
    
    # If we have any headers, build sections from the provided label_order;
    # then filter y_order to drop the header rows from the axis.
    sections = []
    if section_headers and label_order is not None:
        sections = _sections_from_label_order(label_order, section_headers)
    
        # Keep only labels present in y_order (respecting max_items already applied)
        present = set(y_order)
        sections = [(cap, [x for x in items if x in present]) for cap, items in sections]
        # Rebuild y_order without the header rows, preserving current order
        y_order = [lab for lab in y_order if lab not in set(section_headers)]
    
        # If a section got emptied by filtering, drop it
        sections = [(cap, items) for cap, items in sections if items]
    
    # print(y_order)
    # print(len(y_order))
    # Apply the (header-free) order to the dataframe
    df2 = df2[df2["label"].isin(y_order)]
    df2["label"] = pd.Categorical(df2["label"], categories=y_order, ordered=True)
    df2 = df2.sort_values(["label", group_col], ascending=[True, True])

    
    df2["label"] = pd.Categorical(df2["label"], categories=y_order, ordered=True)
    df2 = df2.sort_values(["label", group_col], ascending=[True, True])
    
    # y-axis ticks for items only (no headers)
    # y-axis ticks for items only (no headers)
    n = len(y_order)
    if figsize[1] == 0:
        height = max(3, min(24, 0.42 * n + 2))
        figsize = (figsize[0], height)
   
    fig, ax = plt.subplots(figsize=figsize)
   
    # draw CIs/points
    models = df2[group_col].unique().tolist()
    y_positions = {key: i for i, key in enumerate(y_order)}
    k = len(models)
    offsets = np.linspace(-0.25, 0.25, k) if k > 1 else [0]
    
        # prepare palette
    if len(models) > 1:
        # multiple models → color by model
        palette = sns.color_palette("deep", n_colors=len(models))
        color_map = {m: palette[i] for i, m in enumerate(models)}
    else:
        # single model → color each variable differently
        palette = sns.color_palette("deep", n_colors=len(y_order))
        color_map = {lab: palette[i] for i, lab in enumerate(y_order)}
    
    
    for mi, m in enumerate(models):
        sub = df2[df2[group_col] == m]
        col = palette[mi]
        y = [y_positions[val] + offsets[mi] for val in sub["label"]]
        ax.hlines(y, sub[lo_col], sub[hi_col], alpha=0.9, linewidth=1.5)
        #ax.scatter(sub[x_col], y, color = col, s=18, label=m, zorder=3)
        
        sub  = df2[df2[group_col] == m]
    
        y = np.array([y_positions[val] + offsets[mi] for val in sub[y_col]])
        # asymmetric x-errors: distance from point to each CI edge
        x = sub[x_col].to_numpy(float)
        xerr = np.vstack([x - sub[lo_col].to_numpy(float),
                          sub[hi_col].to_numpy(float) - x])
    
        for xi, yi, lo, hi, lab in zip(x, y, sub[lo_col], sub[hi_col], sub[y_col]):
            if len(models) > 1:
                c = color_map[m]         # color by model
            else:
                c = color_map[lab]       # color by variable label
    
            ax.errorbar(
                xi, yi,
                xerr=[[xi - lo], [hi - xi]],
                fmt='o',
                capsize=5.0,
                elinewidth=2.5,
                capthick=3.0,
                markersize=10.0,
                linestyle='none',
                markeredgecolor  = 'none',
                color=c,
                label=m if (mi == 0 and len(models) > 1) else None,
                zorder=3
            )
   
    # axis labels
    ax.axvline(1.0, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_order,fontdict={"family": "Arial"}, fontsize=17)
    ax.invert_yaxis()
    ax.set_xlabel("Odds Ratio",fontdict={"family": "Arial"}, fontsize=20)
    ax.set_title(title)
    #print(y_order)
   
    # --- NEW: draw dashed dividers and left "outer captions" ---
    if sections:
        # give room for captions
        fig.subplots_adjust(left=0.25)
        ypos = {lab: i for i, lab in enumerate(y_order)}
        for cap, items in sections:
            idxs = [ypos[it] for it in items if it in ypos]
            if not idxs:
                continue
            s, e = min(idxs), max(idxs)
            cap_text = _strip_reference_caption(cap)
            # ax.annotate(cap_text,
            #             xy=(-0.3, (s + e) / 2.0),
            #             xycoords=('axes fraction', 'data'),
            #             ha='right', va='center',
            #             fontsize=9, fontweight='bold')
            # put this once after you create ax
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            
            # inside the loop
            cap_text = _strip_reference_caption(cap)
            ax.text(
                -0.8,                       # push further left (more negative = further left)
                (s + e) / 2.0,
                cap_text,
                transform=trans,             # x in axes coords, y in data coords
                ha='right', va='center',
                fontsize=17, fontweight='bold',
                clip_on=False,               # don't clip outside the axes
            )
            # optional: divider line
            if draw_section_dividers:
                ax.axhline(e + 0.5, linestyle='-', linewidth=0.8, alpha=0.6, color = 'black')
            if draw_section_dividers:
                ax.axhline(e + 0.5, linestyle='-', linewidth=0.8, alpha=0.6, color = 'black')
   
    if logx:
        ax.set_xscale("log")
    if len(models) > 1:
        # build a unique legend (preserve order)
        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            if l and l != "_nolegend_" and l not in by_label:
                by_label[l] = h
        
        fp_label = fm.FontProperties(family='Arial', size=6)
        fp_title = fm.FontProperties(family='Arial', weight='bold', size=7)
        
        ax.legend(by_label.values(), by_label.keys(), title="Model", prop=fp_label).set_title("Model", prop=fp_title)

        #ax.legend(title="Model", loc="best", fontsize=8)
   
    #fig.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=600, bbox_inches = 'tight')
    plt.close(fig)
    return


NATURE_BLUE = "#2166AC"
NATURE_WIDTH_MM = 183.0
NATURE_HEIGHT_MM = 247.0
NATURE_FONT_SIZE_PT = 5.5
NATURE_HEADER_SIZE_PT = 6.5
NATURE_ROWS_PER_PAGE = 50


def _parse_reference_header(header: str) -> Tuple[str, str]:
    """
    Split source captions such as
    "[Reference: Gender [Reference: Female]]" into ("Gender", "Female").

    Captions without a nested reference retain an empty Reference cell.
    """
    text = str(header).strip()
    if text.startswith("[Reference:"):
        text = text[len("[Reference:"):].strip()
        if text.endswith("]"):
            text = text[:-1].strip()

    nested = re.match(r"^(.*?)\s*\[Reference:\s*(.*?)\]\s*$", text)
    if nested:
        return nested.group(1).strip(), nested.group(2).strip()
    return text, ""


def _wrap_cell(text: str, width: int) -> str:
    return "\n".join(
        textwrap.wrap(
            str(text),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _ordered_table_sections(
        available_labels: List[str],
        label_order: List[str],
        section_headers: List[str],
) -> List[Dict[str, object]]:
    """Build ordered Type/Reference blocks from the source label order."""
    available = set(available_labels)
    headers = set(section_headers or [])
    sections: List[Dict[str, object]] = []
    current = None

    def finish_current():
        nonlocal current
        if current is not None and current["details"]:
            sections.append(current)
        current = None

    for item in label_order:
        if item in headers:
            finish_current()
            type_text, reference_text = _parse_reference_header(item)
            current = {
                "type": type_text,
                "reference": reference_text,
                "details": [],
            }
            continue

        if item not in available:
            continue

        # SUICIDE_TIME is the only ordered source row without its own caption.
        # Give it a distinct table block while preserving its exact row position.
        if item == "00:00-05:59":
            finish_current()
            sections.append({
                "type": "Suicide Time",
                "reference": "",
                "details": [item],
            })
            continue

        if current is None:
            current = {"type": "", "reference": "", "details": []}
        current["details"].append(item)

    finish_current()
    return sections


def _paginate_sections(
        sections: List[Dict[str, object]],
        rows_per_page: int,
) -> List[List[Dict[str, object]]]:
    """Pack complete groups onto pages and split only groups that exceed a page."""
    pages: List[List[Dict[str, object]]] = []
    page: List[Dict[str, object]] = []
    used = 0

    for section in sections:
        details = list(section["details"])
        offset = 0
        part_number = 1

        while offset < len(details):
            remaining = rows_per_page - used
            if remaining == 0:
                pages.append(page)
                page = []
                used = 0
                remaining = rows_per_page

            rows_left = len(details) - offset
            if rows_left <= remaining:
                take = rows_left
            elif used > 0 and rows_left <= rows_per_page:
                pages.append(page)
                page = []
                used = 0
                continue
            else:
                take = remaining

            piece = {
                "type": section["type"] if part_number == 1 else f"{section['type']} (cont.)",
                "reference": section["reference"],
                "details": details[offset:offset + take],
            }
            page.append(piece)
            used += take
            offset += take
            part_number += 1

    if page:
        pages.append(page)
    return pages


def _forest_tick_label(value, _position):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:g}"
    if value >= 0.1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def nature_table_forest_plot(
        data: pd.DataFrame,
        filename: str,
        label_order: List[str],
        section_headers: List[str],
        label_map: Dict[str, str],
        group_col: str = "model",
        x_col: str = "OR",
        lo_col: str = "CI_low",
        hi_col: str = "CI_high",
        point_color: str = NATURE_BLUE,
        rows_per_page: int = NATURE_ROWS_PER_PAGE,
        max_items: Optional[int] = None,
        dpi: int = 600,
        font_size: float = NATURE_FONT_SIZE_PT,
):
    """
    Render a Nature-sized table-forest hybrid.

    Output:
      - one multipage vector PDF containing every ordered row;
      - one PDF and SVG per page for editable vector submission;
      - one 600-dpi PNG per page for review and manuscript drafting.
    """
    df = data.copy()
    df["label"] = df["term"].map(label_map).fillna(df["term"]).astype(str)
    for col in (x_col, lo_col, hi_col):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[x_col, lo_col, hi_col])
    df = df[
        (df[x_col] > 0)
        & (df[lo_col] > 0)
        & (df[hi_col] > 0)
    ].copy()
    if df.empty:
        raise ValueError("No finite positive OR and confidence-interval values to plot.")

    ordered_labels = [
        label for label in label_order
        if label not in set(section_headers or [])
        and label in set(df["label"])
    ]
    ordered_labels = _make_unique(ordered_labels)
    if max_items is not None:
        ordered_labels = ordered_labels[:max_items]
    df = df[df["label"].isin(ordered_labels)].copy()
    if df.empty:
        raise ValueError("None of the source labels matched label_order.")

    sections = _ordered_table_sections(
        available_labels=ordered_labels,
        label_order=label_order,
        section_headers=section_headers,
    )
    # This dedicated variant keeps the complete ordered table on one page.
    pages = [sections]

    source_path = Path(filename)
    output_dir = source_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    multipage_pdf = output_dir / f"{stem}.pdf"

    low = min(float(df[lo_col].min()), 1.0)
    high = max(float(df[hi_col].max()), 1.0)
    log_low, log_high = np.log10(low), np.log10(high)
    pad = max(0.08 * (log_high - log_low), 0.04)
    x_limits = (10 ** (log_low - pad), 10 ** (log_high + pad))

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
    width_in = NATURE_WIDTH_MM / 25.4
    models = list(dict.fromkeys(df[group_col].astype(str)))
    model_offsets = (
        np.linspace(-0.18, 0.18, len(models))
        if len(models) > 1 else np.array([0.0])
    )

    with PdfPages(multipage_pdf) as combined_pdf:
        for page_number, page_sections in enumerate(pages, start=1):
            page_labels = [
                detail
                for section in page_sections
                for detail in section["details"]
            ]
            n_rows = len(page_labels)
            y_lookup = {label: row for row, label in enumerate(page_labels)}
            page_height_mm = NATURE_HEIGHT_MM
            height_in = page_height_mm / 25.4

            with plt.rc_context({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "font.size": font_size,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "xtick.major.size": 2.5,
            }):
                fig, axes = plt.subplots(
                    1,
                    4,
                    figsize=(width_in, height_in),
                    gridspec_kw={
                        "width_ratios": [1.36, 1.16, 1.45, 3.23],
                        "wspace": 0.0,
                        "left": 0.025,
                        "right": 0.985,
                        "top": 0.98,
                        "bottom": 0.115,
                    },
                )
                type_ax, ref_ax, detail_ax, forest_ax = axes

                for text_ax in (type_ax, ref_ax, detail_ax):
                    text_ax.set_xlim(0, 1)
                    text_ax.set_ylim(n_rows - 0.5, -1.65)
                    text_ax.set_xticks([])
                    text_ax.set_yticks([])
                    for side in ("left", "top", "bottom"):
                        text_ax.spines[side].set_visible(False)
                    text_ax.spines["right"].set_visible(True)
                    text_ax.spines["right"].set_linewidth(0.5)
                    text_ax.spines["right"].set_color("#4D4D4D")

                forest_ax.set_ylim(n_rows - 0.5, -1.65)
                forest_ax.set_xscale("log")
                forest_ax.set_xlim(*x_limits)
                forest_ax.set_yticks([])
                for side in ("left", "right", "top"):
                    forest_ax.spines[side].set_visible(False)
                forest_ax.spines["bottom"].set_linewidth(0.5)

                header_y = -1.06
                for ax, heading, x in (
                    (type_ax, "Type", 0.5),
                    (ref_ax, "Reference", 0.5),
                    (detail_ax, "Details", 0.5),
                ):
                    ax.text(
                        x, header_y, heading,
                        ha="center", va="center",
                        fontsize=NATURE_HEADER_SIZE_PT,
                        fontweight="bold",
                    )
                forest_ax.text(
                    0.5, header_y, "Odds Ratio", 
                    transform=mtransforms.blended_transform_factory(
                        forest_ax.transAxes, forest_ax.transData
                    ),
                    ha="left", va="center",
                    fontsize=NATURE_HEADER_SIZE_PT,
                    fontweight="bold",
                )

                row_start = 0
                for section in page_sections:
                    details = list(section["details"])
                    row_end = row_start + len(details) - 1
                    center = (row_start + row_end) / 2.0
                    type_ax.text(
                        0.5, center,
                        _wrap_cell(section["type"], 20),
                        ha="center", va="center",
                        fontsize=font_size,
                        fontweight="bold",
                        linespacing=1.05,
                    )
                    ref_ax.text(
                        0.5, center,
                        _wrap_cell(section["reference"], 20),
                        ha="center", va="center",
                        fontsize=font_size,
                        linespacing=1.05,
                    )
                    for row, detail in enumerate(details, start=row_start):
                        detail_ax.text(
                            0.96, row,
                            _wrap_cell(detail, 34),
                            ha="right", va="center",
                            
                            fontsize=font_size,
                            linespacing=1.02,
                        )

                    divider_y = row_end + 0.5
                    for ax in axes:
                        ax.axhline(
                            divider_y,
                            color="#333333",
                            linewidth=0.65,
                            zorder=1,
                            clip_on=False,
                        )
                    row_start = row_end + 1

                for ax in axes:
                    ax.axhline(-0.5, color="#111111", linewidth=0.8, zorder=1)

                page_df = df[df["label"].isin(page_labels)]
                for model_index, model in enumerate(models):
                    sub = page_df[page_df[group_col].astype(str) == model]
                    for _, row in sub.iterrows():
                        y = y_lookup[row["label"]] + model_offsets[model_index]
                        forest_ax.errorbar(
                            row[x_col],
                            y,
                            xerr=np.array([[
                                row[x_col] - row[lo_col],
                                row[hi_col] - row[x_col],
                            ]]).T,
                            fmt=marker_cycle[model_index % len(marker_cycle)],
                            color=point_color,
                            ecolor=point_color,
                            markerfacecolor=point_color,
                            markeredgecolor=point_color,
                            markersize=3.4,
                            markeredgewidth=0.35,
                            elinewidth=0.8,
                            capsize=1.8,
                            capthick=0.7,
                            linestyle="none",
                            zorder=4,
                        )

                # Grid boundaries
                top_of_body = -0.5
                bottom_of_body = n_rows - 0.5
                
                # Vertical gridlines at visible major log ticks
                vertical_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
                vertical_ticks = [
                    tick for tick in vertical_ticks
                    if x_limits[0] <= tick <= x_limits[1]
                ]
                
                forest_ax.vlines(
                    vertical_ticks,
                    ymin=top_of_body,
                    ymax=bottom_of_body,
                    color="#D9D9D9",
                    linewidth=0.45,
                    zorder=0,
                )
                
                # Horizontal gridlines around each estimate row
                row_boundaries = np.arange(-0.5, n_rows + 0.5, 1.0)
                
                forest_ax.hlines(
                    row_boundaries,
                    xmin=x_limits[0],
                    xmax=x_limits[1],
                    color="#E3E3E3",
                    linewidth=0.35,
                    zorder=0,
                )
                
                # Dashed null line, drawn over the grid
                forest_ax.vlines(
                    1.0,
                    ymin=top_of_body,
                    ymax=bottom_of_body,
                    color="#555555",
                    linestyle=(0, (3, 2)),
                    linewidth=0.7,
                    zorder=2,
                )
                forest_ax.xaxis.set_major_formatter(
                    FuncFormatter(_forest_tick_label)
                )
                forest_ax.tick_params(
                    axis="x",
                    which="major",
                    labelsize=font_size,
                    pad=2,
                )
                forest_ax.tick_params(axis="x", which="minor", length=0)
                forest_ax.set_xlabel(
                    "Odds ratio (95% CI)",
                    fontsize=NATURE_HEADER_SIZE_PT,
                    labelpad=4,
                )

                if len(models) > 1:
                    handles = [
                        plt.Line2D(
                            [0], [0],
                            marker=marker_cycle[i % len(marker_cycle)],
                            color=point_color,
                            linestyle="none",
                            markersize=3.4,
                            label=model,
                        )
                        for i, model in enumerate(models)
                    ]
                    forest_ax.legend(
                        handles=handles,
                        loc="upper right",
                        frameon=False,
                        fontsize=font_size,
                        ncol=min(len(models), 3),
                        handletextpad=0.35,
                        columnspacing=0.8,
                    )

                page_pdf = output_dir / f"{stem}_page_{page_number:02d}.pdf"
                page_svg = output_dir / f"{stem}_page_{page_number:02d}.svg"
                page_png = output_dir / f"{stem}_page_{page_number:02d}.png"
                fig.savefig(page_pdf, format="pdf", facecolor="white")
                fig.savefig(page_svg, format="svg", facecolor="white")
                fig.savefig(page_png, format="png", dpi=dpi, facecolor="white")
                combined_pdf.savefig(fig, facecolor="white")
                plt.close(fig)

    print(
        f"Saved {len(pages)} Nature-sized page(s): "
        f"{multipage_pdf} plus page-level PDF, SVG, and {dpi}-dpi PNG files."
    )





# --------------------------------
# Public APIs you’ll typically run
# --------------------------------
def make_levelwise_forest(or_files: Dict[str, str],
                          common_only: bool = True,
                          top_k: int = 40,
                          out_png: str = "forest_levelwise.png",
                          reference_labels: Optional[List[str]] = None,
                          # --- NEW ---
                          category_map: Optional[Dict[str, str]] = None,  # {term -> category}
                          category_order: Optional[List[str]] = None,     # ["Demographic","Context",...]
                          label_map: Optional[Dict[str, str]] = None,     # {term -> pretty label}
                          within_order: str = "effect" ,                 # "effect"|"alpha"|"none"
                          only_significant: bool = True,   # <---
                          alpha: float = 0.05,
                          label_order: Optional[List[str]] = None
                          ):
    frames = []
    for model, path in or_files.items():
        df = load_or_table(path)
        df = df.dropna(subset=["OR", "CI_low", "CI_high"])
        if only_significant:
            df = df[df["p_value"].fillna(1.0) < alpha]
        df["model"] = model
        frames.append(df[["term", "OR", "CI_low", "CI_high", "model", "p_value"]])

    if not frames:
        raise ValueError("No OR tables loaded.")

    if common_only:
        common_terms = set(frames[0]["term"])
        for f in frames[1:]:
            common_terms &= set(f["term"])
        frames = [f[f["term"].isin(common_terms)].copy() for f in frames]

    plot_df = pd.concat(frames, ignore_index=True)
    # default y label is the term (we’ll override with label_map inside forest_plot if provided)
    plot_df["label"] = plot_df["term"]

    # if reference_labels:
    #     plot_df = inject_reference_rows(plot_df, reference_labels, label_col="label", group_col="model")
    
    nature_table_forest_plot(
        data=plot_df,
        filename=out_png,
        label_order=label_order,
        section_headers=reference_labels,
        label_map=label_map,
        group_col="model",
        max_items=top_k,
    )

    
    return plot_df

def make_factorwise_forest(var_files: Dict[str, str],
                           or_files: Dict[str, str],
                           agg: str = "max_abs",  # 'max_abs', 'median', 'ivw'
                           keep_only_significant_factors: bool = False,
                           alpha: float = 0.05,
                           top_k: int = 40,
                           out_png: str = "forest_factorwise.png"):
    """
    Build a factor-wise forest using one representative OR per factor.
    - agg controls how levels are collapsed into a single factor OR.
    - If keep_only_significant_factors is True, filters factors by Factor_p_value < alpha (per model),
      then keeps the union of significant factors across models so you can compare them side-by-side.
    """
    # load each model's var_tests and or_table
    model_rows = []
    all_vars = None

    for model, vpath in var_files.items():
        if model not in or_files:
            raise ValueError(f"Missing or_table for model '{model}' in or_files.")

        vt = load_var_tests(vpath)
        ot = load_or_table(or_files[model])

        # filter factors by significance if requested
        if keep_only_significant_factors:
            sig_vars = set(vt.loc[vt["Factor_p_value"] < alpha, "Variable"])
            ot = ot[ot["Variable"].isin(sig_vars)].copy()
            vt = vt[vt["Variable"].isin(sig_vars)].copy()

        # aggregate to one OR per factor
        agg_df = aggregate_factor_or(ot, how=agg)
        # attach p-values (factor-wise)
        agg_df = agg_df.merge(vt, on="Variable", how="left")
        agg_df["model"] = model
        model_rows.append(agg_df)

        if all_vars is None:
            all_vars = set(agg_df["Variable"])
        else:
            all_vars |= set(agg_df["Variable"])

    if not model_rows:
        raise ValueError("No factor-wise data assembled.")

    plot_df = pd.concat(model_rows, ignore_index=True)
    # keep only factors that have finite OR and CIs
    plot_df = plot_df.dropna(subset=["OR", "CI_low", "CI_high"])
    # Label y as the factor
    plot_df["label"] = plot_df["Variable"]

    forest_plot(
        data=plot_df,
        y_col="label",
        group_col="model",
        title=f"Factor-wise Forest Plot (OR agg={agg}, 95% CI)",
        filename=out_png,
        max_items=top_k
    )
    
# --- NEW: helpers for custom categories and labels ---

from typing import Optional

def _apply_labels_and_categories(
    df: pd.DataFrame,
    key_col: str,
    label_map: Optional[Dict[str, str]] = None,
    category_map: Optional[Dict[str, str]] = None,
    category_order: Optional[List[str]] = None,
    within_order: str = "effect",
    label_order: Optional[List[str]] = None
):
    df = df.copy()

    # 1) label
    if label_map:
        df["label"] = df[key_col].map(label_map).fillna(df.get("label", df[key_col]))
    else:
        df["label"] = df.get("label", df[key_col])

    # 2) category
    if category_map:
        df["category"] = df[key_col].map(category_map).fillna("Other")
    else:
        df["category"] = "All"

    # 3) distance / within-key
    if "dist" not in df.columns:
        df["dist"] = np.abs(np.log(df["OR"].astype(float)))
    if within_order == "effect":
        df["_within_key"] = -df["dist"]
    elif within_order == "alpha" and "p_value" in df.columns:
        df["_within_key"] = df["p_value"].fillna(1.0)
    elif within_order == "alphabetical":
        df["_within_key"] = df["label"].astype(str)
    else:
        df["_within_key"] = np.arange(len(df))

    # --- NEW: short-circuit if label_order is given
    if label_order is not None:
        wanted = [lab for lab in label_order if lab in set(df["label"].astype(str))]
        out = df[df["label"].astype(str).isin(wanted)].copy()  # <- don't filter by category
        out["label"] = pd.Categorical(out["label"].astype(str), categories=wanted, ordered=True)
        out = out.sort_values("label", kind="mergesort")
        out = out.drop(columns=["_within_key"], errors="ignore")
        return out, wanted

    # Otherwise, do category-driven ordering
    if category_order is None:
        cats = sorted(df["category"].unique().tolist())
    else:
        cats = [c for c in category_order if c in set(df["category"])]

    y_order = []
    out_frames = []
    for cat in cats:
        sub = df[df["category"] == cat].sort_values("_within_key", kind="mergesort")
        out_frames.append(sub)
        y_order.extend(sub["label"].tolist())

    out = pd.concat(out_frames, ignore_index=True) if out_frames else df
    out = out.drop(columns=["_within_key"], errors="ignore")
    return out, y_order

#%%
def inject_reference_rows(plot_df: pd.DataFrame,
                          reference_labels: List[str],
                          label_col: str = "label",
                          group_col: str = "model") -> pd.DataFrame:
    """
    Insert placeholder rows (one per model) for each label in reference_labels.
    These rows have NaN OR/CI so nothing is plotted, but the label appears on the y-axis.
    """
    if not reference_labels:
        return plot_df

    models = plot_df[group_col].unique().tolist()
    ref_frames = []
    for lab in reference_labels:
        for m in models:
            ref_frames.append({
                label_col: lab,
                group_col: m,
                "term": f"__ref__{lab}",     # harmless placeholder
                "OR": np.nan,
                "CI_low": np.nan,
                "CI_high": np.nan,
                "p_value": np.nan,
                REF_FLAG_COL: True
            })
    ref_df = pd.DataFrame(ref_frames)
    out = pd.concat([plot_df, ref_df], ignore_index=True)
    if REF_FLAG_COL not in out.columns:
        out[REF_FLAG_COL] = False
    else:
        out[REF_FLAG_COL] = out[REF_FLAG_COL].fillna(False)
    return out


def category_for(term: str) -> str:
    if term.startswith("YEAR"):
        return "MetaInfo"
    if term.startswith(("AGE", "SEX")):
        return "Demographics"
    if term.startswith("MARITAL_TP"):
        return "Demographics"
    if term.startswith(("COHABITATION_EX","SPOUSE","CHILD","PARENT","SIBSHIP","RELATIVE","ETC1")):
        return "Demographics"
    if term.startswith("HAB_TP"):
        return "Demographics"
    if term.startswith("HOME_SIDO"):
        return "Demographics"
    if term.startswith("FIND_SIDO"):
        return "Context"
    if term.startswith("FIND_PLACETP"):
        return "Context"
    if term.startswith("FINDER"):
        return "Context"
    if term.startswith("SUICIDE_MOMENT"):
        return "Context"
    if term.startswith("SUICIDE_TIME"):
        return "Context"
    if term.startswith("MAIN_METHOD"):
        return "Context"
    if term.startswith("MAIN_CAUSE"):
        return "Context"
    if term.startswith("EMPLOYMENT_TP"):
        return "Demographics"
    if term.startswith("WARNSPEAK"):
        return "Warning Signs"
    if term.startswith("TIME"):
        return "Warning Signs"
    if term.startswith("PC_"):
        return "MetaInfo"
    # singletons / miscellaneous
    if term.startswith("M_"):
        return "Context"
    if term.startswith("G_"):
        return "Context"
    return "MetaInfo"


terms = [
    'YEAR1','YEAR2','YEAR3','SUICIDE_MOMENT1','SUICIDE_MOMENT2','SUICIDE_MOMENT3',
    'FIND_PLACETP1','FIND_PLACETP3','FIND_PLACETP7','COHABITATION_EX_None1',
    'AGE21','AGE22','AGE23','AGE24','SEX1','FINDER1','FINDER2','FINDER3','FINDER4',
    'MAIN_METHOD2','MAIN_METHOD3','MAIN_METHOD6','MAIN_METHOD7','MAIN_METHOD9',
    'MAIN_METHOD10','MAIN_METHOD11','MAIN_METHOD12','MAIN_METHOD13','MAIN_METHOD16',
    'MAIN_METHOD18','MAIN_METHOD19','PC_ETC1','PC_SPOUSE1','PC_PARENT1','PC_SIBSHIP1',
    'PC_RELATIVE1','PC_FRIEND1','HOME_SIDO3','HOME_SIDO7','YEAR5','YEAR6','YEAR7',
    'FIND_PLACETP2','FIND_PLACETP5','FIND_PLACETP8','FIND_SIDO3','MAIN_METHOD17',
    'PC_CHILD1','PC_LOVER1','M_LOVER1','HAB_TP1','HAB_TP2','HAB_TP5','HAB_TP6',
    'MAIN_CAUSE1','MAIN_CAUSE3','MAIN_CAUSE5','EMPLOYMENT_TP1','EMPLOYMENT_TP2',
    'EMPLOYMENT_TP3','EMPLOYMENT_TP4','MARITAL_TP1','MARITAL_TP3','WARNSPEAK11',
    'WARNSPEAK41','TIME31','TIME23','TIME41','TIME42','TIME43','TIME44','TIME45',
    'HAB_TP4','TIME113','TIME115','TIME125'
]

terms = ['HOME_SIDO3','FIND_PLACETP4','MARITAL_TP1','ANXIETY_SX1','FIND_SIDO8','MAIN_METHOD7','DEPRESSION_SX1','SUICIDE_MOMENT1',
     'WARNSPEAK21','COGDECLINE_SX1','SUICIDE_MOMENT3','FIND_PLACETP3','MAIN_CAUSE3','MARITAL_TP3','FIND_SIDO6','TIME45',
     'MAIN_METHOD9','MAIN_METHOD12','WARNBEHAV41','INSOMNIA_SX1','MAIN_METHOD16','EMPLOYMENT_TP6','TIME113','MAIN_METHOD17','AGE24','FIND_PLACETP8',
     'SPOUSE1','HAB_TP1','EMPLOYMENT_TP2','MAIN_CAUSE1','HAB_TP2','FAM_ISOLATED1','FAM_PARENT1','SUICIDE_TIME1','HAB_TP5','MAIN_METHOD4','TIME95',
     'COHABITATION_EX_None1','M_LOVER1','SIBSHIP1','ACUTE_SX1','FIND_SIDO10','FINDER3','MAIN_METHOD11','TIME44','MAIN_METHOD3','MAIN_METHOD2',
     'FINDER4','JOB_TP6','EMPLOYMENT_TP3','SEX1','FIND_PLACETP2','TIME115','WARNSPEAK11','ALCOHOL_SX1','TIME43','FIND_SIDO1','FIND_PLACETP6',
     'TIME52','BEHAVADD_SX1','PSYCHOSIS1','TIME25','EMPLOYMENT_TP1','WARNBEHAV21','RELATIVE1','TIME12','FIND_SIDO7','SOMATIC_DISAB1','TIME63',
     'HOME_SIDO8','AGE22','CHILD1','MANIC_SX1','AGE21','MAIN_METHOD13','FAM_RELATIVE1','PARENT1','WARNBEHAV31','MAIN_METHOD19','TIME41','MAIN_METHOD18',
     'SOMATIZATION_SX1','FAM_SIBSHIP1','FIND_PLACETP5','MAIN_METHOD6','FINDER2','RELAT_JOB1','REDUCE_INCOME1','NONNP_THERAPY1','TIME42','EMPLOYMENT_TP7','TIME125',
     'FIND_PLACETP7','MAIN_METHOD10','HAB_TP4','EMPLOYMENT_TP4','FIND_PLACETP1','HAB_TP6','AGE23',
     'WARNSPEAK41','TIME23','FINDER1','MAIN_CAUSE2','SUICIDE_MOMENT2','FIND_SIDO4','FIND_SIDO3','JOB_TP8','ETC1','MAIN_CAUSE4','TIME24','WARNSPEAK31',
     'FIND_SIDO12','MAIN_CAUSE5','NP_THERAPYEX1']



category_map_level = {t: category_for(t) for t in sorted(set(terms))}

#%%
Level_Labels = {
    'YEAR1': '2019',
    'YEAR2': '2013',
    'YEAR3': '2020',
    'YEAR4': '2014',
    'YEAR5': '2015',
    'YEAR6': '2016',
    'YEAR7': '2017',

    # Suicide moment timing
    'SUICIDE_MOMENT1': 'After 24 Hours',
    'SUICIDE_MOMENT2': 'Alive',
    'SUICIDE_MOMENT3': 'Skeletonized',

    # Discovery / finding location
    'FIND_PLACETP1': 'Public Places',
    'FIND_PLACETP2': 'Lodging Facilities',
    'FIND_PLACETP3': 'School/Workplace',
    'FIND_PLACETP5': 'Hospital',
    'FIND_PLACETP7': 'Other Places',
    'FIND_PLACETP8': 'Acquaintance house',
    'FIND_SIDO3': 'Found in Busan',
    'HOME_SIDO8': 'Jeonllabuk-do',

    # Demographics
    'COHABITATION_EX_None1': 'Lives Alone',
    'AGE21': '35-49',
    'AGE22': '>65',
    'AGE23': '19-34',
    'AGE24': '<18',
    'SEX1': 'Male',

    # Discoverer identity
    'FINDER1': 'Stranger',
    'FINDER2': 'Relative/Acquaintance',
    'FINDER3': 'Police|Emergency Services',
    'FINDER4': 'Property Manager',
    'FINDER5': 'Social Worker',

    # Method of suicide
    'MAIN_METHOD1': 'Jumping',
    'MAIN_METHOD2': 'Gas Asphyxiation',
    'MAIN_METHOD3': 'Pesticide',
    'MAIN_METHOD4': 'Drowning',
    'MAIN_METHOD5': 'Prescription Drugs',
    'MAIN_METHOD6': 'Knife/Awl',
    'MAIN_METHOD7': 'Sleeping Pills',
    'MAIN_METHOD8': 'Other Checmical Drugs',
    'MAIN_METHOD9': 'Other Asphyxiation',
    'MAIN_METHOD10': 'Immolation',
    'MAIN_METHOD11': 'Other SelfHarm',
    'MAIN_METHOD12': 'Running into car/subway',
    'MAIN_METHOD13': 'Other Methods',
    'MAIN_METHOD15': 'Other Drugs',
    'MAIN_METHOD16': 'Insecticide',
    'MAIN_METHOD17': 'Rat Poison',
    'MAIN_METHOD18': 'Gun',
    'MAIN_METHOD19': 'Herbicide',

    # Personal connection
    'PC_ETC1': 'Interviewed: Etc',
    'PC_SPOUSE1': 'Interviewed: Spouse',
    'PC_PARENT1': 'Interviewed: Parent',
    'PC_CHILD1': 'Interviewed: Child',
    'PC_SIBSHIP1': 'Interviewed: Brother/Sister',
    'PC_RELATIVE1': 'Interviewed: Relative',
    'PC_LOVER1': 'Interviewed: Lover',
    'PC_FRIEND1': 'Interviewed: Friend',
    'M_LOVER1': 'Murder Suicide: Lover',

    # Residence region
    'HOME_SIDO3': 'kyeongsan-NamDo',
    'HOME_SIDO7': 'Daegu-Kwangyeoksi',

    # Habits
    'HAB_TP1': 'Individual house',
    'HAB_TP2': 'Multi-unit residential building',
    'HAB_TP3': 'Other than houses/apartment (e.g dorm)',
    'HAB_TP4': 'OfficeTel',
    'HAB_TP5': 'House within commercial building',
    'HAB_TP6': 'Other Residence Types',

    # Main cause / trigger
    'MAIN_CAUSE1': 'Economic Problems',
    'MAIN_CAUSE2': 'Physical Health Problems',
    'MAIN_CAUSE3': 'Family Problems',
    'MAIN_CAUSE4': 'Relationship Problems',
    'MAIN_CAUSE5': 'Job Problems',
    'MAIN_CAUSE6': 'Other Problems',

    # Employment
    'EMPLOYMENT_TP1': 'Employee',
    'EMPLOYMENT_TP2': 'Self-Employed',
    'EMPLOYMENT_TP3': 'Unemployed',
    'EMPLOYMENT_TP4': 'Homemaker',
    'EMPLOYMENT_TP5': 'Student',

    # Marital status
    'MARITAL_TP1': 'Single(Unmarried)',
    'MARITAL_TP3': 'Widowed',
    'MARITAL_TP4': 'Separated',

    # Warning speech
    'WARNSPEAK11': 'Talked about Death and Suicide',
    'WARNSPEAK41': 'Other Verbal Warning Signs',

    # Time of day / duration
    'TIME41': 'Other Verbal Warning Signs (Within 1 Week)',
    'TIME35': 'Self-Deprecating comments (Within 6 Months)',
    'TIME53': 'Lack of interest in self-appearance (Within 3 months)',
}

Level_Labels.update({
    # New FIND_PLACETP categories
    'FIND_PLACETP4': 'Suburbs',
    'FIND_PLACETP6': 'Relative House',
    
    # New SUICIDE_TIME
    'SUICIDE_TIME1': '00:00-05:59',

    
    
    # Additional FIND_SIDO
    'FIND_SIDO1': 'Found in Seoul',
    'FIND_SIDO2': 'Found in KyeongSangNamDo',
    'FIND_SIDO4': 'Found in KyeongSangBukDo',
    'FIND_SIDO5': 'Found in Incheon',
    'FIND_SIDO6': 'Found in ChungCheongNamDo',
    'FIND_SIDO7': 'Found in Daegu',
    'FIND_SIDO8': 'Found in KangWonDo',
    'FIND_SIDO9': 'Found in JeollaNamDo',
    'FIND_SIDO10': 'Found in JeollaBukDo',
    'FIND_SIDO12': 'Found in Daejeon',
    'FIND_SIDO11': 'Found in ChungCheongBukDo',
    'FIND_SIDO15': 'Found in Jeju',
    'FIND_SIDO16': 'Found in Sejong',
    'HOME_SIDO1': 'Seoul',
    'HOME_SIDO2': 'KyeongSangNamDo',
    'HOME_SIDO5': 'Incheon',
    'HOME_SIDO6': 'ChungCheongNamDo',

    # Employment extensions
    'EMPLOYMENT_TP6': 'Mandatory Military Service',
    'EMPLOYMENT_TP7': 'Others',
    'JOB_TP6': 'Agriculture',
    'JOB_TP8': 'Crafts Worker',
    'RELAT_JOB1': 'Relationship Problem related with Job',
    
    # Family structure
    'FAM_ISOLATED1': 'Isolation in Family Context',
    'FAM_PARENT1': 'Problem with Parent',
    'FAM_RELATIVE1': 'Problem with Relative',
    'FAM_SIBSHIP1': 'Problem with Sibling',

    # Psychiatric / Clinical symptoms
    'ANXIETY_SX1': 'Anxiety Symptoms',
    'DEPRESSION_SX1': 'Depression Symptoms',
    'COGDECLINE_SX1': 'Cognitive Decline',
    'INSOMNIA_SX1': 'Insomnia Symptoms',
    'ACUTE_SX1': 'Acute Symptoms Due To Environmental Stressors',
    'ALCOHOL_SX1': 'Alcohol Abuse',
    'BEHAVADD_SX1': 'Behavioral Addiction Symptoms',#
    'PSYCHOSIS1': 'Psychosis',#
    'MANIC_SX1': 'Manic Symptoms',
    'SOMATIC_DISAB1': 'Physical Disability',#
    'SOMATIZATION_SX1': 'Somatic Symptoms Related To Psychological Stress',#

    # Warning behaviors (non-speech)
    'WARNBEHAV21': 'Self-Harm/Substance Abuse',
    'WARNBEHAV31': 'Anhedonia',
    'WARNBEHAV41': 'Other Behavioral Warning Signs',

    # Additional warning speech
    'WARNSPEAK21': 'Talked about Physical Discomfort',
    'WARNSPEAK31': 'Self-Deprecating comments',

    # Time-related (new codes)
    'TIME25': 'Talked about Physical Discomfort (Within 6 Months)',

    # Therapy / treatment
    'NP_THERAPYEX1': 'Mental health treatment history',
    'NONNP_THERAPY1': 'Non-Mental health treatment history',

    # Economic hardship
    'REDUCE_INCOME1': 'Reduced Income',

    # Relatives & family interview (new codes)
    'SPOUSE1': 'Lives with Spouse',
    'CHILD1': 'Lives with Child',
    'PARENT1': 'Lives with Parent',
    'SIBSHIP1': 'Lives with Sibling',
    'RELATIVE1': 'Lives with Extended',
    'ETC1': 'Lives with Other',
    
    'S_AFTERMURDER_None1': 'Murder Suicide',
    'DRINKING_EX1': 'Alcohol Consumption at Death',
    'TIME112': 'Emotional Distress (Within 1 Week)',
    'DEBT1': 'Debt',
    'WARNEMOTION11': 'Depressed Mood',
    'FAM_SPOUSE1': 'Problem with Spouse',
})
#%%

my_label_order = [
    
    "[Reference: Gender [Reference: Female]]",
    'Male',
    
    "[Reference: Age [Reference: 50-65]]",
    '<18',
    '19-34',
    '35-49',
    '>65',
    
    '[Reference: Marital Status [Reference: Married]]',
    'Single(Unmarried)',
    'Separated',
    'Widowed',
    
    
    "[Reference: Cohabitation Status]",
    'Lives Alone',
    'Lives with Child',
    'Lives with Extended',
    'Lives with Parent',
    'Lives with Spouse',
    'Lives with Sibling',
    'Lives with Other',
    
    
    
    '[Reference: Employment Status [Reference: Non-Economic Activities]]',
    'Employee',
    'Homemaker',
    'Mandatory Military Service',
    'Self-Employed',
    'Student',
    'Unemployed',
    'Others',
    
    '[Reference: Occupation [Reference: None/HomeMaker/Non-EconActivities]]',
    'Agriculture',
    'Crafts Worker',
    
    '[Reference: Murder Suicide]',
    'Murder Suicide',
    'Murder Suicide: Lover',
    
    '[Reference: Main Cause [Reference: Mental Health Problems]]',
    'Economic Problems',
    'Family Problems',
    'Job Problems',
    'Physical Health Problems',
    'Relationship Problems',
    'Other Problems',
    
    '00:00-05:59',
    
    '[Reference: Discovered Location [Reference: Own Home]]',
    'Acquaintance house',
    'Hospital',
    'Lodging Facilities',
    'Public Places',
    'Relative House',
    'School/Workplace',
    'Suburbs',
    'Other Places',
    
    '[Reference: First Responder [Reference: Family]]',
    'Police|Emergency Services',
    'Property Manager',
    'Relative/Acquaintance',
    'Social Worker',
    'Stranger',
    
    '[Reference: Main Method [Reference: Hanging]]',
    'Drowning',
    'Gas Asphyxiation',
    'Gun',
    'Herbicide',
    'Immolation',
    'Insecticide',
    'Jumping',
    'Knife/Awl',
    'Pesticide',
    'Prescription Drugs',
    'Rat Poison',
    'Running into car/subway',
    'Sleeping Pills',
    'Other Checmical Drugs',
    'Other Asphyxiation',
    'Other SelfHarm',
    'Other Drugs',
    'Other Methods',
    
    '[Reference: Discovered State [Reference: Within 24 Hours]]',
    'Alive',
    'After 24 Hours',
    'Skeletonized',
    
    '[Reference: Mental/Physical Health Symptoms]',
    'Anxiety Symptoms',
    'Acute Symptoms Due To Environmental Stressors',
    'Alcohol Consumption at Death',
    'Alcohol Abuse',
    'Behavioral Addiction Symptoms',
    'Cognitive Decline',
    'Depression Symptoms',
    'Insomnia Symptoms',
    'Manic Symptoms',
    'Mental health treatment history',
    'Non-Mental health treatment history',
    'Physical Disability',
    'Psychosis',
    'Somatic Symptoms Related To Psychological Stress',
    
    '[Reference: Life Stressors]',
    'Debt',
    'Isolation in Family Context',
    'Problem with Parent',
    'Problem with Relative',
    'Problem with Sibling',
    'Problem with Spouse',
    'Reduced Income',
    'Relationship Problem related with Job',
    
    
    '[Reference: Warning Signs]',
    'Anhedonia',
    'Depressed Mood',
    'Self-Deprecating comments',
    'Self-Harm/Substance Abuse',
    'Talked about Death and Suicide',
    'Talked about Physical Discomfort',
    'Other Behavioral Warning Signs',
    'Other Verbal Warning Signs',
    
    '[Reference: Warning Sign Timing[Reference: Emotional Distress (Within 1 Month)]]',
    'Emotional Distress (Within 1 Week)',
    
    '[Reference: Warning Sign Timing[Reference: Talked about Physical Discomfort (Within 1 Month)]]',
    'Talked about Physical Discomfort (Within 6 Months)',
    
    '[Reference: Warning Sign Timing[Reference: Other Verbal Warning Signs (Within 1 Month)]]',
    'Other Verbal Warning Signs (Within 1 Week)',
    
    '[Reference: Warning Sign Timing[Reference: Self-Deprecating comments (Within 1 Month)]]',
    'Self-Deprecating comments (Within 6 Months)',
    
    '[Reference: Warning Sign Timing[Reference: Lack of interest in self-appearance (Within 1 Month)]]',
    'Lack of interest in self-appearance (Within 3 months)',

    

    
]

#reference_targets = ["2018", "Aged: 50-65"]
#reference_labels = [f"[Reference: {x}]" for x in reference_targets]
reference_labels = [
    "[Reference: Gender [Reference: Female]]",
    "[Reference: Age [Reference: 50-65]]",
    "[Reference: Marital Status [Reference: Married]]",
    "[Reference: Cohabitation Status]",
    
    "[Reference: Employment Status [Reference: Non-Economic Activities]]",
    "[Reference: Occupation [Reference: None/HomeMaker/Non-EconActivities]]",
    
    '[Reference: Murder Suicide]',
    "[Reference: Main Cause [Reference: Mental Health Problems]]",
    
    "[Reference: Discovered Location [Reference: Own Home]]",
    "[Reference: First Responder [Reference: Family]]",
    
    
    "[Reference: Main Method [Reference: Hanging]]",
    "[Reference: Discovered State [Reference: Within 24 Hours]]",
    
    "[Reference: Mental/Physical Health Symptoms]",
    
    "[Reference: Life Stressors]",
    
    "[Reference: Warning Signs]",
    '[Reference: Warning Sign Timing[Reference: Emotional Distress (Within 1 Month)]]',
    '[Reference: Warning Sign Timing[Reference: Talked about Physical Discomfort (Within 1 Month)]]',
    '[Reference: Warning Sign Timing[Reference: Other Verbal Warning Signs (Within 1 Month)]]',
    '[Reference: Warning Sign Timing[Reference: Self-Deprecating comments (Within 1 Month)]]',
    '[Reference: Warning Sign Timing[Reference: Lack of interest in self-appearance (Within 1 Month)]]',
]

#%%
# -----------------------------
# Example usage (edit paths)
# -----------------------------
if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "Nature_TableStyle_OnePage_Exports"

    # Replace these with your file paths for each model you want to compare
    # (You can compare 1, 2, or many models.)
    level_or_files = {
        #"Cut_0(N=97k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut0\or_table_Cut_0_DF_32Columns.xlsx",
        #"Cut_1(N=91k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut1\or_table_Cut_1_DF_43Columns.xlsx",
        "Cut_2(N=66k)": r"C:\Users\Jae Bin Park\or_table_Cut_2_DF_50Columns.xlsx",
        #"Cut_3(N=54k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut3\or_table_Cut_3_DF_58Columns.xlsx",
        #"Cut_4(N=44k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut4\or_table_Cut_4_DF_68Columns.xlsx",
        #"Cut_5(N=31k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut5\or_table_Cut_5_DF_84Columns.xlsx",
        #"Cut_6(N=13k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut6\or_table_Cut_6_DF_97Columns.xlsx",
        #"Cut_7(N=4k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut7\or_table_Cut_7_DF_143Columns.xlsx",
    }

    factor_var_files = {
        #"Cut_0(N=97k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut0\var_tests_Cut_0_DF_32Columns.xlsx",
        #"Cut_1(N=91k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut1\var_tests_Cut_1_DF_43Columns.xlsx",
        "Cut_2(N=66k)": r"C:\Users\Jae Bin Park\var_tests_Cut_2_DF_50Columns.xlsx",
        #"Cut_3(N=54k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut3\var_tests_Cut_3_DF_58Columns.xlsx",
        #"Cut_4(N=44k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut4\var_tests_Cut_4_DF_68Columns.xlsx",
        #"Cut_5(N=31k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut5\var_tests_Cut_5_DF_84Columns.xlsx",
        #"Cut_6(N=13k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut6\var_tests_Cut_6_DF_97Columns.xlsx",
        #"Cut_7(N=4k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut7\var_tests_Cut_7_DF_143Columns.xlsx",
    }
    factor_or_files = {
        #"Cut_0(N=97k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut0\or_table_Cut_0_DF_32Columns.xlsx",
        #"Cut_1(N=91k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut1\or_table_Cut_1_DF_43Columns.xlsx",
        "Cut_2(N=66k)": r"C:\Users\Jae Bin Park\or_table_Cut_2_DF_50Columns.xlsx",
        #"Cut_3(N=54k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut3\or_table_Cut_3_DF_58Columns.xlsx",
        #"Cut_4(N=44k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut4\or_table_Cut_4_DF_68Columns.xlsx",
        #"Cut_5(N=31k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut5\or_table_Cut_5_DF_84Columns.xlsx",
        #"Cut_6(N=13k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut6\or_table_Cut_6_DF_97Columns.xlsx",
        #"Cut_7(N=4k)": r"C:\Users\User\OneDrive\Me\PhD Research\Suicide\Manuscripts\Additional Analyses\KFSP_New_LogReg\NewResults\MultiCut\Cut7\or_table_Cut_7_DF_143Columns.xlsx",
    }
    
    
    # Order of blocks (top -> bottom)
    category_order_level = ["Demographics", "Context", "Warning Signs", "MetaInfo"]
    
    
    # Call:
    P_DF = make_levelwise_forest(
    or_files=level_or_files,
    common_only=False,
    top_k=800,
    out_png=str(output_dir / "Reordered_Levels_Comparative_ForestPlots_Nature_TableStyle_OnePage.png"),
    category_map=category_map_level,      # optional; ignored if label_order is given
    category_order=category_order_level,  # optional; ignored if label_order is given
    label_map=Level_Labels,
    within_order="none",
    label_order=my_label_order,           # <- exact y-axis order
    reference_labels=reference_labels,    # <- rows to render as headers (no OR/CI)
    only_significant=True,                # preserve the original p < 0.05 workflow
    alpha=0.05,
)


    # 2) FACTOR-WISE forest (aggregate OR per factor)
    make_factorwise_forest(
        var_files=factor_var_files,
        or_files=factor_or_files,
        agg="max_abs",                  # 'max_abs' (default), 'median', or 'ivw'
        keep_only_significant_factors=True,  # set True to filter by drop1 p<0.05
        alpha=0.05,
        top_k=75,
        out_png=str(output_dir / "ReorderedReferenced_forest_factorwise.png")
    )

    print(f"Outputs saved in: {output_dir}")
#%%
''' SANITY CHECKS'''

# A =list(set(P_DF['term']))
# B = list(set(Level_Labels.keys()))
# missing = [x for x in A if not x in B]

# C = list(set(Level_Labels.values()))
# missing = [x for x in A if not Level_Labels[x] in C]

#%%
# df,df_clean = pickle.load(open(r'ReorderedLevelsdiag_dfclean_generalizedreduced_NoSIDOs','rb'))
# import pandas as pd

# # Step 1: Parse Codebook into lookup dicts
# def parse_codelist(codestr):
#     mapping = {}
#     for part in str(codestr).split("|"):
#         if "=" in part:
#             k, v = part.split("=", 1)  # only split at the first '='
#             try:
#                 mapping[int(k)] = v
#             except ValueError:
#                 mapping[k] = v  # fallback if key isn't an int
#     return mapping

# codebook_dict = {
#     row["변수명"]: parse_codelist(row["코드리스트"])
#     for _, row in Codebook.dropna(subset=["코드리스트"]).iterrows()
# }

# # Step 2: Update df["mapping"]
# def fix_mapping(row):
#     var = row["column"]
#     mapping = row["mapping"]
#     if var in codebook_dict:
#         cb_map = codebook_dict[var]
#         return {k: cb_map.get(v, v) for k, v in mapping.items()}
#     return mapping

# df["mapping_fixed"] = df.apply(fix_mapping, axis=1)
#%%
import pandas as pd

''' load df, and df_cleaned''' 
# # pick mapping source
# src_col = "mapping_fixed" if "mapping_fixed" in df.columns else "mapping"
# lookup_map = df.set_index("column")[src_col].to_dict()

# # longest-prefix match over known variable names
# varnames = tuple(sorted(df["column"].astype(str).unique(), key=len, reverse=True))

# def split_label(label: str):
#     s = str(label)
#     for name in varnames:
#         if s.startswith(name):
#             return name, s[len(name):]  # suffix is the potential key
#     return None, None

# def fetch_value_from_label(label):
#     var, key_str = split_label(label)
#     if not var or not key_str:
#         return pd.NA
#     # parse key as int if possible (handles negatives too)
#     try:
#         key = int(key_str)
#     except ValueError:
#         return pd.NA
#     m = lookup_map.get(var)
#     if not isinstance(m, dict):
#         return pd.NA
#     # keys may be ints or strings in mapping
#     return m.get(key, m.get(str(key), pd.NA))

# P_DF["label_value"] = [fetch_value_from_label(x) for x in P_DF["label"]]


# mask = P_DF["label"] == "S_AFTERMURDER_None1"
# P_DF.loc[mask, "label_value"] = "Murder Suicide Commited"

# mask = P_DF["label"] == "COHABITATION_EX_None1"
# P_DF.loc[mask, "label_value"] = "Lived Alone"
