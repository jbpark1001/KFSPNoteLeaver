
"""Create the noun and verb alluvial figures as a standalone script.

Run this file directly. It loads the two pickled count tables, applies the
included translations and semantic-category mappings, and exports publication
PDF and PNG files in the same directory as this script.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgba
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from pathlib import Path

# -----------------------------
# Font helpers
# -----------------------------
def _fontprop(font_path=None, size=11):
    return fm.FontProperties(fname=font_path, size=size) if font_path else None

def _maybe_set_global_font(font_path):
    """
    Registers the font and sets it as the default family so seaborn/mpl inherit it.
    Safe no-op if font_path is None or registration fails.
    """
    if not font_path:
        return
    try:
        fm.fontManager.addfont(font_path)
        name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = name
    except Exception:
        # If anything goes wrong, we simply don't override the global font.
        pass

def _apply_axis_font(ax, fp, size=11, xtick_size=None, ytick_size=None):
    """Apply FontProperties to labels, ticks, and legend on a given axis."""
    if fp is None:
        return
    ax.set_xlabel(ax.get_xlabel(), fontproperties=fp, fontsize=size-2)
    ax.set_ylabel(ax.get_ylabel(), fontproperties=fp, fontsize=size-2)

    xtick_size = size if xtick_size is None else xtick_size
    ytick_size = size if ytick_size is None else ytick_size

    for lab in ax.get_xticklabels():
        lab.set_fontproperties(fp)
        lab.set_fontsize(xtick_size)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(fp)
        lab.set_fontsize(ytick_size)

    leg = ax.get_legend()
    if leg is not None:
        leg.get_title().set_fontproperties(fp)
        leg.get_title().set_fontsize(size)
        for txt in leg.get_texts():
            txt.set_fontproperties(fp)
            txt.set_fontsize(size)


# -----------------------------
# Palettes
# -----------------------------
def build_palette(categories, scheme="npg20", include_other_gray=True):
    """
    Returns {category -> color} using a Nature/NHB-friendly discrete palette.
    - scheme="npg20": 20 curated colors (extends NPG tones).
    - include_other_gray=True: any category starting with 'Other' gets a neutral gray.
    """
    NPG20 = [
        # Core NPG tones
        "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
        "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
        # Extended but harmonized (distinct hues / luminance)
        "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
        "#CC79A7", "#EE6677", "#228833", "#9467BD", "#8C564B",
    ]

    PALETTES = {
        "npg20": NPG20,
    }

    base = PALETTES.get(scheme, NPG20)
    mapping = {}
    i = 0
    for cat in categories:
        if include_other_gray and str(cat).lower().startswith("other"):
            mapping[cat] = "#8C8C8C"  # neutral gray for "Other"
        else:
            mapping[cat] = base[i % len(base)]
            i += 1
    return mapping


# -----------------------------
# 2) Prep function: add labels & categories; choose metric
# -----------------------------
def prepare_word_table(df,
                       word_col="Word",
                       count_col="Count",
                       translation_dict=None,
                       category_map=None,
                       top_n=50,
                       proportion=True):
    d = df[[word_col, count_col]].copy()
    d[count_col] = pd.to_numeric(d[count_col], errors="coerce").fillna(0)

    if proportion:
        total = d[count_col].sum()
        d["value"] = (d[count_col] / total) if total > 0 else 0.0
        xlab = "Share of tokens"
        xfmt = PercentFormatter(1.0)
    else:
        d["value"] = d[count_col].astype(float)
        xlab = "Frequency"
        xfmt = None

    # --- English-only labels (fallback to the original token if no translation) ---
    if translation_dict:
        eng = d[word_col].map(translation_dict).fillna("")
        d["Label"] = np.where(eng != "", eng, d[word_col])
    else:
        d["Label"] = d[word_col]

    # Categories
    if category_map:
        d["Category"] = d[word_col].map(category_map).fillna("Other")
    else:
        d["Category"] = "Other"

    # Rank & take top_n overall
    d = d.sort_values("value", ascending=False).head(top_n).reset_index(drop=True)
    return d, xlab, xfmt


# -----------------------------
# 3) Alluvial layout helpers
# -----------------------------
def _ribbon(x0, x1, source_bottom, source_top,
            target_bottom, target_top, color, alpha):
    """Return one smooth category-to-word ribbon."""
    bend = 0.46 * (x1 - x0)
    vertices = [
        (x0, source_top),
        (x0 + bend, source_top),
        (x1 - bend, target_top),
        (x1, target_top),
        (x1, target_bottom),
        (x1 - bend, target_bottom),
        (x0 + bend, source_bottom),
        (x0, source_bottom),
        (x0, source_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(
        MplPath(vertices, codes),
        facecolor=to_rgba(color, alpha),
        edgecolor="none",
        linewidth=0,
        zorder=1,
    )


def _spread_label_positions(desired_positions, lower, upper, minimum_gap):
    """Move crowded category labels apart without moving their nodes."""
    ordered = sorted(desired_positions, key=desired_positions.get)
    if len(ordered) < 2:
        return desired_positions.copy()

    gap = min(minimum_gap, (upper - lower) / (len(ordered) - 1))
    positions = [max(desired_positions[ordered[0]], lower)]
    for category in ordered[1:]:
        positions.append(max(desired_positions[category], positions[-1] + gap))

    if positions[-1] > upper:
        positions[-1] = upper
        for i in range(len(positions) - 2, -1, -1):
            positions[i] = min(positions[i], positions[i + 1] - gap)

    if positions[0] < lower:
        shift = lower - positions[0]
        positions = [position + shift for position in positions]
    return dict(zip(ordered, positions))


def plot_words_alluvial(
    df_plot,
    ax,
    palette,
    xfmt,
    font_path=r"C:\Windows\Fonts\arial.ttf",
    font_size=11,
    value_decimals=2,
    ribbon_alpha=0.34,
    category_gap=0.010,
    word_gap=None,
):
    """Draw category nodes, word nodes, ribbons, and direct value labels."""
    if df_plot.empty:
        raise ValueError("No positive values are available to plot.")

    fp = _fontprop(font_path, font_size)
    fp_header = _fontprop(font_path, font_size + 0.5)
    fp_value = _fontprop(font_path, max(font_size - 0.25, 8))
    use_percent = isinstance(xfmt, PercentFormatter)

    category_totals = (
        df_plot.groupby("Category", as_index=False, sort=False)["value"]
        .sum()
        .sort_values(["value", "Category"], ascending=[False, True])
        .reset_index(drop=True)
    )
    category_order = category_totals["Category"].tolist()
    category_rank = {
        category: rank for rank, category in enumerate(category_order)
    }

    # Keep every category's words together and sort them by value. This avoids
    # ribbon crossings and creates one readable block per semantic category.
    d = df_plot.copy()
    d["_category_rank"] = d["Category"].map(category_rank)
    d = d.sort_values(
        ["_category_rank", "value", "Label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    top, bottom = 0.965, 0.035
    available = top - bottom
    if word_gap is None:
        # Reserve about half of the vertical plotting area for separation
        # between word rows. The cap avoids excessive gaps for small top_n.
        word_gap = min(
            0.012,
            0.50 * available / max(len(d) - 1, 1),
        )
    category_gap_total = category_gap * max(len(category_order) - 1, 0)
    word_gap_total = word_gap * max(len(d) - 1, 0)
    data_height = available - max(category_gap_total, word_gap_total)
    if data_height <= 0:
        raise ValueError(
            "The gaps do not fit. Reduce word_gap or plot fewer words."
        )

    selected_total = d["value"].sum()
    scale = data_height / selected_total
    left_total_height = data_height + category_gap_total
    right_total_height = data_height + word_gap_total
    left_cursor = top - (available - left_total_height) / 2
    right_cursor = top - (available - right_total_height) / 2

    category_bounds = {}
    source_bounds = {}
    for category in category_order:
        category_rows = d.index[d["Category"].eq(category)].tolist()
        category_height = d.loc[category_rows, "value"].sum() * scale
        category_top = left_cursor
        category_bottom = category_top - category_height
        category_bounds[category] = (category_bottom, category_top)

        source_cursor = category_top
        for row_i in category_rows:
            word_height = d.at[row_i, "value"] * scale
            source_bounds[row_i] = (
                source_cursor - word_height,
                source_cursor,
            )
            source_cursor -= word_height
        left_cursor = category_bottom - category_gap

    target_bounds = {}
    for row_i, row in d.iterrows():
        word_height = row["value"] * scale
        target_bounds[row_i] = (
            right_cursor - word_height,
            right_cursor,
        )
        right_cursor -= word_height + word_gap

    x_category_label = 0.010
    x_category_value = 0.315
    x_left_node = 0.355
    left_node_width = 0.022
    x_right_node = 0.690
    right_node_width = 0.017
    x_word_label = 0.730
    x_word_value = 0.990

    for row_i, row in d.iterrows():
        source_bottom, source_top = source_bounds[row_i]
        target_bottom, target_top = target_bounds[row_i]
        color = palette.get(row["Category"], "#7A7A7A")
        ax.add_patch(
            _ribbon(
                x_left_node + left_node_width,
                x_right_node,
                source_bottom,
                source_top,
                target_bottom,
                target_top,
                color,
                ribbon_alpha,
            )
        )

    desired_category_labels = {
        category: np.mean(category_bounds[category])
        for category in category_order
    }
    category_label_positions = _spread_label_positions(
        desired_category_labels,
        lower=bottom,
        upper=top,
        minimum_gap=0.030,
    )
    total_lookup = category_totals.set_index("Category")["value"].to_dict()

    for category in category_order:
        node_bottom, node_top = category_bounds[category]
        node_center = (node_bottom + node_top) / 2
        label_y = category_label_positions[category]
        color = palette.get(category, "#7A7A7A")

        ax.add_patch(
            Rectangle(
                (x_left_node, node_bottom),
                left_node_width,
                node_top - node_bottom,
                facecolor=color,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.plot(
            [x_category_value + 0.010, x_left_node - 0.006],
            [label_y, node_center],
            color=to_rgba(color, 0.60),
            linewidth=0.7,
            zorder=2,
        )
        ax.text(
            x_category_label,
            label_y,
            category,
            ha="left",
            va="center",
            fontproperties=fp,
            color="#111111",
            zorder=4,
        )
        category_value = total_lookup[category]
        category_label = (
            f"{category_value:.{value_decimals}%}"
            if use_percent
            else f"{category_value:,.0f}"
        )
        ax.text(
            x_category_value,
            label_y,
            category_label,
            ha="right",
            va="center",
            fontproperties=fp_value,
            color="#6F6F6F",
            zorder=4,
        )

    for row_i, row in d.iterrows():
        node_bottom, node_top = target_bounds[row_i]
        node_center = (node_bottom + node_top) / 2
        color = palette.get(row["Category"], "#7A7A7A")
        ax.add_patch(
            Rectangle(
                (x_right_node, node_bottom),
                right_node_width,
                node_top - node_bottom,
                facecolor=color,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x_word_label,
            node_center,
            row["Label"],
            ha="left",
            va="center",
            fontproperties=fp,
            color="#111111",
            zorder=4,
        )
        word_label = (
            f"{row['value']:.{value_decimals}%}"
            if use_percent
            else f"{row['value']:,.0f}"
        )
        ax.text(
            x_word_value,
            node_center,
            word_label,
            ha="right",
            va="center",
            fontproperties=fp_value,
            color="#6F6F6F",
            zorder=4,
        )

    header_y = 1.035
    line_y = 1.012
    ax.text(
        x_category_label,
        header_y,
        "Semantic category",
        ha="left",
        va="bottom",
        fontproperties=fp_header,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        x_right_node,
        header_y,
        "Detail words",
        ha="left",
        va="bottom",
        fontproperties=fp_header,
        fontweight="bold",
        color="#111111",
    )
    ax.plot(
        [x_category_label, x_left_node + left_node_width],
        [line_y, line_y],
        color="#222222",
        linewidth=0.8,
    )
    ax.plot(
        [x_right_node, x_word_value],
        [line_y, line_y],
        color="#222222",
        linewidth=0.8,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.075)
    ax.axis("off")


# -----------------------------
# 4) Public figure function
# -----------------------------
def figure_words_with_categories(
    df,
    word_col="Word",
    count_col="Count",
    translation_dict=None,
    category_map=None,
    top_n=50,
    proportion=True,
    suptitle="",
    font_path=r"C:\Windows\Fonts\arial.ttf",
    font_size=11,
    set_global_font=True,
    value_decimals=2,
    ribbon_alpha=0.34,
    figsize=None,
):
    """Create a tall alluvial figure with one readable row per detail word."""
    if set_global_font:
        _maybe_set_global_font(font_path)

    df_plot, _, xfmt = prepare_word_table(
        df,
        word_col=word_col,
        count_col=count_col,
        translation_dict=translation_dict,
        category_map=category_map,
        top_n=top_n,
        proportion=proportion,
    )
    category_totals = (
        df_plot.groupby("Category", sort=False)["value"]
        .sum()
        .sort_values(ascending=False)
    )
    palette = build_palette(category_totals.index)

    # Reserve roughly 0.36 inch per word. At top_n=50 this produces a
    # 12 x 21 inch canvas, leaving clear separation between 11-point labels.
    if figsize is None:
        figsize = (12.0, max(10.0, 3.0 + 1.5 * len(df_plot)))

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    plot_words_alluvial(
        df_plot,
        ax,
        palette,
        xfmt,
        font_path=font_path,
        font_size=font_size,
        value_decimals=value_decimals,
        ribbon_alpha=ribbon_alpha,
    )
    if suptitle:
        fig.suptitle(
            suptitle,
            y=0.995,
            fontproperties=_fontprop(font_path, font_size + 1),
        )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.025)
    return fig

DEFAULT_NOUN_DATA = Path.home() / "whole_nng_counts_results"
DEFAULT_VERB_DATA = Path.home() / "whole_vv_counts_results"
# =========================
# Category definitions (brief, for Methods/Supplement)
# =========================
noun_category_defs = {
    "Family & Kinship": "Terms denoting relatives and kinship roles.",
    "Finance & Assets": "Money, debt, banking, and material assets.",
    "Health & Clinical": "Illness, treatment, health states, and substances.",
    "Affect & Values": "Positive affect/values (love, hope, courage, etc.).",
    "Distress/Suffering": "Negative affect, hardship, pain, tears, harm.",
    "Mortality & Funerary": "Death and funerary references.",
    "Moral/Accountability": "Sin, forgiveness, truth/lying, choice as agency.",
    "Communication & Contact": "Messages, contact, phone, suicide note as object.",
    "Time & Duration": "Day, time, moment, lifetime, end/last, age.",
    "Self & Identity": "Self-reference and personal appearance.",
    "Body & Physical": "Body parts and somatic references.",
    "Social Relations & People": "People, social ties, gender/role labels, isolation.",
    "Work & Roles": "Work, workplace, positions/roles.",
    "Place & Environment": "Home, landscape, sky, path.",
    "Objects & Technology": "Physical artifacts/devices/vehicles.",
    "Security & Access": "Credentials and access keys.",
    "Situation & Circumstance": "General problems/situations/methods.",
    "Life & Existence": "Life as a condition or life-course concept.",
    "Other/Ambiguous": "Polysemous or context-dependent items."
}

verb_category_defs = {
    "Apology & Forgiveness": "Apologizing, forgiving, gratitude/repair acts.",
    "Love & Attachment": "Expressions of love/attachment.",
    "Life & Death": "Living, dying, birth, suicide, survival.",
    "Requests & Wants": "Requests, wishes, wants, pleading.",
    "Communication": "Saying, informing, calling, contacting, relaying.",
    "Cognition & Memory": "Thinking, knowing, understanding, remembering/forgetting, belief.",
    "Perception & Discovery": "Seeing, hearing, showing, finding, being found.",
    "Giving & Transfer": "Giving, receiving, sending, leaving behind, offering.",
    "Caretaking & Support": "Helping, caring, protecting, accompanying, raising.",
    "Organization/Documentation/Resolution": "Writing, composing, handling, resolving, arranging, placing/disposal.",
    "Motion & Transition": "Going/coming/leaving/starting; movement and change.",
    "Persistence & Effort": "Enduring, trying, persevering.",
    "Decision & Choice": "Choosing, giving up.",
    "Work & Economy": "Working, buying, selling, repaying.",
    "Existence/State & Outcomes": "Existing, remaining, ability/inability, things going wrong, reduction.",
    "Emotion Expression": "Crying, grieving, feeling, resentment, exhaustion.",
    "Creation/Construction": "Making, building.",
    "Harm/Impact": "Hitting, causing (impact on others).",
    "Ritual/Funeral": "Cremation, burial, scattering ashes.",
    "Self-care & Daily Actions": "Eating, drinking, resting.",
    "Control/Causation": "Causing others to act.",
    "Social Interaction": "Meeting/relating directly.",
    "Generic Action": "Generic honorific 'do'.",
    "Other/Ambiguous": "Polysemous or context-dependent items."
}

# =========================
# Noun -> Category (primary assignment; one per lemma)
# =========================
noun_category_map = {
    # Family & Kinship
    "엄마":"Family & Kinship","아빠":"Family & Kinship","아들":"Family & Kinship","가족":"Family & Kinship",
    "형":"Family & Kinship","아버지":"Family & Kinship","딸":"Family & Kinship","누나":"Family & Kinship",
    "어머니":"Family & Kinship","동생":"Family & Kinship","부모":"Family & Kinship","오빠":"Family & Kinship",
    "언니":"Family & Kinship","남편":"Family & Kinship","아내":"Family & Kinship","형제":"Family & Kinship",
    "할머니":"Family & Kinship","식구":"Family & Kinship","아이":"Family & Kinship","어머님":"Family & Kinship",

    # Finance & Assets
    "돈":"Finance & Assets","빚":"Finance & Assets","통장":"Finance & Assets","재산":"Finance & Assets","카드":"Finance & Assets",

    # Health & Clinical
    "병원":"Health & Clinical","병":"Health & Clinical","우울증":"Health & Clinical","약":"Health & Clinical",
    "건강":"Health & Clinical","술":"Health & Clinical",

    # Affect & Values (positive)
    "마음":"Affect & Values","사랑":"Affect & Values","희망":"Affect & Values","행복":"Affect & Values",
    "용기":"Affect & Values","힘":"Affect & Values","맘":"Affect & Values",

    # Distress/Suffering (negative affect)
    "고생":"Distress/Suffering","고통":"Distress/Suffering","눈물":"Distress/Suffering","걱정":"Distress/Suffering",
    "피해":"Distress/Suffering","짐":"Distress/Suffering","상처":"Distress/Suffering",

    # Mortality & Funerary
    "죽음":"Mortality & Funerary","자살":"Mortality & Funerary","사망":"Mortality & Funerary",
    "장례":"Mortality & Funerary","장례식":"Mortality & Funerary","시신":"Mortality & Funerary",

    # Moral/Accountability
    "죄":"Moral/Accountability","용서":"Moral/Accountability","거짓말":"Moral/Accountability","선택":"Moral/Accountability","진심":"Moral/Accountability",

    # Communication & Contact
    "메시지":"Communication & Contact","연락":"Communication & Contact","전화":"Communication & Contact",
    "연락처":"Communication & Contact","유서":"Communication & Contact",

    # Time & Duration
    "시간":"Time & Duration","하루":"Time & Duration","날":"Time & Duration","순간":"Time & Duration",
    "평생":"Time & Duration","마지막":"Time & Duration","끝":"Time & Duration","나이":"Time & Duration",

    # Self & Identity
    "자신":"Self & Identity","모습":"Self & Identity",

    # Body & Physical
    "몸":"Body & Physical","가슴":"Body & Physical",

    # Social Relations & People
    "사람":"Social Relations & People","친구":"Social Relations & People","지인":"Social Relations & People",
    "남":"Social Relations & People","여자":"Social Relations & People","인간":"Social Relations & People",
    "혼자":"Social Relations & People","도움":"Social Relations & People",

    # Work & Roles
    "일":"Work & Roles","회사":"Work & Roles","자리":"Work & Roles",

    # Place & Environment
    "집":"Place & Environment","길":"Place & Environment","산":"Place & Environment","하늘":"Place & Environment",

    # Objects & Technology
    "차":"Objects & Technology","핸드폰":"Objects & Technology",

    # Security & Access
    "비밀번호":"Security & Access",

    # Situation & Circumstance
    "상황":"Situation & Circumstance","문제":"Situation & Circumstance","방법":"Situation & Circumstance",

    # Life & Existence
    "삶":"Life & Existence","인생":"Life & Existence",
}

# =========================
# Verb -> Category (primary assignment; one per lemma)
# =========================
verb_category_map = {
    # Apology & Forgiveness
    "미안하":"Apology & Forgiveness","죄송하":"Apology & Forgiveness","용서하":"Apology & Forgiveness","감사하":"Apology & Forgiveness",

    # Love & Attachment
    "사랑하":"Love & Attachment",

    # Life & Death
    "살":"Life & Death","살았":"Life & Death","살아가":"Life & Death","죽":"Life & Death","죽었":"Life & Death",
    "자살하":"Life & Death","태어나":"Life & Death","먼저가":"Life & Death",

    # Requests & Wants
    "부탁하":"Requests & Wants","부탁드리":"Requests & Wants","원하":"Requests & Wants","바라":"Requests & Wants","빌":"Requests & Wants",

    # Communication
    "말하":"Communication","연락하":"Communication","전화하":"Communication","전하":"Communication","알리":"Communication",

    # Cognition & Memory
    "생각하":"Cognition & Memory","알":"Cognition & Memory","알았":"Cognition & Memory","이해하":"Cognition & Memory",
    "믿":"Cognition & Memory","모르":"Cognition & Memory","모르겠":"Cognition & Memory","잊":"Cognition & Memory",

    # Perception & Discovery
    "보":"Perception & Discovery","보이":"Perception & Discovery","듣":"Perception & Discovery",
    "찾":"Perception & Discovery","발견되":"Perception & Discovery",

    # Giving & Transfer
    "주":"Giving & Transfer","드리":"Giving & Transfer","받":"Giving & Transfer","보내":"Giving & Transfer","남기":"Giving & Transfer",

    # Caretaking & Support
    "돕":"Caretaking & Support","보살피":"Caretaking & Support","챙기":"Caretaking & Support",
    "모시":"Caretaking & Support","지키":"Caretaking & Support","키우":"Caretaking & Support","위하":"Caretaking & Support",

    # Organization/Documentation/Resolution
    "쓰":"Organization/Documentation/Resolution","작성하":"Organization/Documentation/Resolution","적":"Organization/Documentation/Resolution",
    "정리하":"Organization/Documentation/Resolution","처리하":"Organization/Documentation/Resolution","해결하":"Organization/Documentation/Resolution",
    "두":"Organization/Documentation/Resolution","버리":"Organization/Documentation/Resolution",

    # Motion & Transition
    "가":"Motion & Transition","오":"Motion & Transition","떠나":"Motion & Transition",
    "나":"Motion & Transition","나오":"Motion & Transition","시작하":"Motion & Transition",
    "가었":"Motion & Transition","오았":"Motion & Transition",  # tolerate past-stem variants if present
    "가았":"Motion & Transition",  # as provided in your list

    # Persistence & Effort
    "버티":"Persistence & Effort","노력하":"Persistence & Effort",

    # Decision & Choice
    "선택하":"Decision & Choice","포기하":"Decision & Choice",

    # Work & Economy
    "일하":"Work & Economy","사":"Work & Economy","팔":"Work & Economy","갚":"Work & Economy",

    # Existence/State & Outcomes
    "있었":"Existence/State & Outcomes","남":"Existence/State & Outcomes","지내":"Existence/State & Outcomes",
    "안되":"Existence/State & Outcomes","잘못되":"Existence/State & Outcomes","인하":"Existence/State & Outcomes","가지":"Existence/State & Outcomes",
    "못하":"Existence/State & Outcomes","잘하":"Existence/State & Outcomes",

    # Emotion Expression
    "울":"Emotion Expression","느끼":"Emotion Expression","슬퍼하":"Emotion Expression","원망하":"Emotion Expression","지치":"Emotion Expression",

    # Creation/Construction
    "만들":"Creation/Construction","짓":"Creation/Construction",

    # Harm/Impact
    "치":"Harm/Impact","끼치":"Harm/Impact",

    # Ritual/Funeral
    "화장하":"Ritual/Funeral","묻":"Ritual/Funeral","뿌리":"Ritual/Funeral",

    # Self-care & Daily Actions
    "먹":"Self-care & Daily Actions","마시":"Self-care & Daily Actions","쉬":"Self-care & Daily Actions",

    # Control/Causation
    "시키":"Control/Causation",

    # Social Interaction
    "만나":"Social Interaction",

    # Generic Action
    "하시":"Generic Action",
}


korean_wholenouns_to_english = {
    "엄마": "Mom",
    "아빠": "Dad",
    "사람": "Person",
    "아들": "Son",
    "가족": "Family",
    "돈": "Money",
    "마음": "Heart",
    "일": "Work",
    "형": "Older brother",
    "아버지": "Father",
    "마지막": "Last",
    "딸": "Daughter",
    "집": "Home",
    "유서": "Will",
    "친구": "Friend",
    "누나": "Older sister (by male)",
    "어머니": "Mother",
    "생각": "Thought",
    "동생": "Younger sibling",
    "자신": "Oneself",
    "몸": "Body",
    "삶": "Life",
    "부탁": "Request",
    "인생": "Life (whole)",
    "부모": "Parents",
    "날": "Day",
    "끝": "End",
    "오빠": "Older brother (by female)",
    "길": "Path",
    "시간": "Time",
    "짐": "Burden",
    "자식": "Children",
    "고생": "Hardship",
    "언니": "Older sister (by female)",
    "빚": "Debt",
    "선택": "Choice",
    "고통": "Pain",
    "힘": "Strength",
    "모습": "Appearance",
    "통장": "Bank account",
    "죽음": "Death",
    "아이": "Child",
    "사랑": "Love",
    "남편": "Husband",
    "죄": "Sin",
    "용서": "Forgiveness",
    "아내": "Wife",
    "병원": "Hospital",
    "가슴": "Chest / Heart",
    "상처": "Wound / Hurt",
    "진심": "Sincerity",
    "장례": "Funeral",
    "형제": "Siblings",
    "메시지": "Message",
    "술": "Alcohol",
    "방법": "Method",
    "남": "Others / Male",
    "연락": "Contact",
    "장례식": "Funeral Ceremony",
    "자살": "Suicide",
    "전화": "Phone call",
    "하루": "Day",
    "할머니": "Grandmother",
    "병": "Illness",
    "사망": "Death (formal)",
    "도움": "Help",
    "지인": "Acquaintance",
    "피해": "Harm",
    "혼자": "Alone",
    "우울증": "Depression",
    "여자": "Woman",
    "순간": "Moment",
    "평생": "Lifetime",
    "어머님": "Mother (honorific)",
    "눈물": "Tears",
    "걱정": "Worry",
    "정신": "Mind",
    "희망": "Hope",
    "인간": "Human",
    "회사": "Company",
    "상황": "Situation",
    "재산": "Assets",
    "카드": "Card",
    "비밀번호": "Password",
    "문제": "Problem",
    "자리": "Position",
    "약": "Medicine",
    "연락처": "Contact info",
    "산": "Mountain",
    "행복": "Happiness",
    "차": "Car",
    "건강": "Health",
    "맘": "Heart (colloquial)",
    "거짓말": "Lie",
    "용기": "Courage",
    "핸드폰": "Cell phone",
    "식구": "Family member",
    "나이": "Age",
    "시신": "Corpse",
    "하늘": "Sky"
}

korean_verbs_to_english = {
    "미안하": "Be sorry",
    "사랑하": "Love",
    "살": "Live",
    "죄송하": "Apologize",
    "가": "Go",
    "죽": "Die",
    "부탁하": "Ask",
    "바라": "Hope",
    "보": "See",
    "생각하": "Think",
    "용서하": "Forgive",
    "받": "Receive",
    "알": "Know",
    "만나": "Meet",
    "못하": "Not do",
    "화장하": "Cremate",
    "안되": "Not work",
    "오": "Come",
    "뿌리": "Scatter",
    "쓰": "Write",
    "떠나": "Leave",
    "잘하": "Do well",
    "남기": "Leave behind",
    "주": "Give",
    "감사하": "Thank",
    "드리": "Offer",
    "먹": "Eat",
    "살았": "Have lived",
    "보내": "Send",
    "말하": "Say",
    "지키": "Keep",
    "보이": "Show",
    "지내": "Spend time",
    "믿": "Believe",
    "모르": "Not know",
    "모르겠": "Not know",
    "위하": "Care for",
    "잊": "Forget",
    "돕": "Help",
    "챙기": "Take care of",
    "연락하": "Contact",
    "사": "Buy",
    "전하": "Deliver",
    "알리": "Inform",
    "이해하": "Understand",
    "키우": "Raise",
    "태어나": "Be born",
    "갚": "Repay",
    "마시": "Drink",
    "처리하": "Handle",
    "찾": "Find",
    "슬퍼하": "Grieve",
    "나": "Come out",
    "정리하": "Organize",
    "인하": "Reduce",
    "가지": "Have",
    "선택하": "Choose",
    "나오": "Come out",
    "남": "Remain",
    "원망하": "Resent",
    "가았": "Have gone",
    "듣": "Hear",
    "대하": "Face",
    "만들": "Make",
    "먼저가": "Go first",
    "살아가": "Live on",
    "노력하": "Try",
    "시키": "Make someone do",
    "전화하": "Call",
    "알았": "Have known",
    "있었": "Have existed",
    "버리": "Throw away",
    "쉬": "Rest",
    "울": "Cry",
    "오았": "Have come",
    "포기하": "Give up",
    "모시": "Accompany",
    "잘못되": "Go wrong",
    "적": "Write down",
    "작성하": "Write",
    "죽었": "Have died",
    "끼치": "Cause",
    "두": "Put",
    "해결하": "Resolve",
    "치": "Hit",
    "보살피": "Take care of",
    "하시": "Do (honorific)",
    "자살하": "Commit suicide",
    "일하": "Work",
    "팔": "Sell",
    "느끼": "Feel",
    "부탁드리": "Humbly ask",
    "묻": "Bury",
    "짓": "Build",
    "원하": "Want",
    "지치": "Be exhausted",
    "시작하": "Start",
    "버티": "Endure",
    "빌": "Beg",
    "발견되": "Be found"
}


# List of words you want to remove
stopwords  = ['말', '이상', '애', '내용', '앞', '동안', '이제', '중략','밖','곳', '다음', '옆', '글', '이유', 
              '속', '모두', '문자', '이름', '눈', '전', '그동안', '변사자', '장', '후', "지금", "미안", "조금", "곁",
              "뒤", "이번", "처음", "잘못", "위", "나중", "그때", "층", '테', '당']

# List of words you want to remove
stopverbs  = ["하","되","있","그러", "하았", "안하", "들", "되었", "그러었", "말", "잘살", "다하"]

TOP_N = 30
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
FONT_SIZE = 11
PNG_DPI = 600


def load_and_clean_data(
    noun_data_path=DEFAULT_NOUN_DATA,
    verb_data_path=DEFAULT_VERB_DATA,
):
    """Load the two source tables and reproduce the original cleaning."""
    noun_data_path = Path(noun_data_path)
    verb_data_path = Path(verb_data_path)
    if not noun_data_path.exists():
        raise FileNotFoundError(f"Noun data file not found: {noun_data_path}")
    if not verb_data_path.exists():
        raise FileNotFoundError(f"Verb data file not found: {verb_data_path}")

    noun_counts = pd.read_pickle(noun_data_path)
    verb_counts = pd.read_pickle(verb_data_path)

    clean_nouns = noun_counts.loc[
        ~noun_counts["Word"].isin(stopwords)
    ].reset_index(drop=True)

    # Merge the two forms safely. The earlier .loc[len(df)] approach could
    # overwrite an existing row because filtering preserves the old index.
    merge_mask = verb_counts["Word"].isin(["모르", "모르겠"])
    merged_count = verb_counts.loc[merge_mask, "Count"].sum()
    verb_counts = verb_counts.loc[~merge_mask].reset_index(drop=True)
    if merged_count > 0:
        merged_row = pd.DataFrame(
            [{"Word": "모르", "Count": merged_count}]
        )
        verb_counts = pd.concat(
            [verb_counts, merged_row],
            ignore_index=True,
        )

    verb_counts = verb_counts.sort_values(
        "Count",
        ascending=False,
    ).reset_index(drop=True)
    clean_verbs = verb_counts.loc[
        ~verb_counts["Word"].isin(stopverbs)
    ].reset_index(drop=True)
    return clean_nouns, clean_verbs


def save_figure_with_fallback(fig, output_path, **savefig_kwargs):
    """Save normally, or use an _updated name when Windows locks the target."""
    output_path = Path(output_path)
    try:
        fig.savefig(output_path, **savefig_kwargs)
        return output_path
    except PermissionError:
        fallback_path = output_path.with_name(
            f"{output_path.stem}_updated{output_path.suffix}"
        )
        fig.savefig(fallback_path, **savefig_kwargs)
        return fallback_path


def main():
    """Generate both publication PDFs without requiring another code file."""
    clean_nouns, clean_verbs = load_and_clean_data()
    output_dir = Path(__file__).resolve().parent

    fig_nouns = figure_words_with_categories(
        clean_nouns,
        word_col="Word",
        count_col="Count",
        translation_dict=korean_wholenouns_to_english,
        category_map=noun_category_map,
        top_n=TOP_N,
        proportion=True,
        font_path=FONT_PATH,
        font_size=FONT_SIZE,
    )
    noun_output = save_figure_with_fallback(
        fig_nouns,
        output_dir / "nouns_categories_alluvial.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    noun_png_output = save_figure_with_fallback(
        fig_nouns,
        output_dir / "nouns_categories_alluvial.png",
        bbox_inches="tight",
        facecolor="white",
        dpi=PNG_DPI,
    )
    plt.close(fig_nouns)

    fig_verbs = figure_words_with_categories(
        clean_verbs,
        word_col="Word",
        count_col="Count",
        translation_dict=korean_verbs_to_english,
        category_map=verb_category_map,
        top_n=TOP_N,
        proportion=True,
        font_path=FONT_PATH,
        font_size=FONT_SIZE,
    )
    verb_output = save_figure_with_fallback(
        fig_verbs,
        output_dir / "verbs_categories_alluvial.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    verb_png_output = save_figure_with_fallback(
        fig_verbs,
        output_dir / "verbs_categories_alluvial.png",
        bbox_inches="tight",
        facecolor="white",
        dpi=PNG_DPI,
    )
    plt.close(fig_verbs)

    print(f"Saved: {noun_output}")
    print(f"Saved: {noun_png_output}")
    print(f"Saved: {verb_output}")
    print(f"Saved: {verb_png_output}")


if __name__ == "__main__":
    main()
