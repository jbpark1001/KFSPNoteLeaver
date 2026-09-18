"""
The restricted source data are not distributed with this repository.
"""

from __future__ import annotations

import argparse
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "restricted" / "kfsp_withnotelabels.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "derived" / "cleaned_demographic_data.xlsx"
DEFAULT_DIAGNOSTICS = PROJECT_ROOT / "derived" / "cleaning_diagnostics.csv"
DEFAULT_METADATA = PROJECT_ROOT / "derived" / "cleaning_metadata.pkl"

NUMERIC_UNKNOWN_CODES = {88, 999, 9999, 99999, 999999}
TEXT_UNKNOWN_TOKENS = {
    "na",
    "n/a",
    "none",
    "null",
    "missing",
    "unknown",
    "dk",
    "dont know",
    "don't know",
    "무응답",
    "모름",
    "미상",
    "없음",
    "결측",
    "모르겠",
}

JOB_PB = [
    "WORKLOAD",
    "REDUCE_SALARY",
    "COMPETIOTION",
    "UNEMPLOYMENT",
    "ALINATION",
    "ALINAT_ETC",
]
ECONOMY_PB = [
    "REDUCE_INCOME",
    "INCRE_SPENDING",
    "BANKRUPT",
    "DEBT",
    "POVERTY",
    "INOCCUPATION",
    "ECONOMY_ETC",
]
FAMILY_PB = [
    "FAM_SPOUSE",
    "FAM_PARENT",
    "FAM_GRANDPARENT",
    "FAM_CHILD",
    "FAM_SIBSHIP",
    "FAM_RELATIVE",
    "FAM_DEATH",
    "FAM_ISOLATED",
    "FAM_ETC",
]
RELATION_PB = [
    "RELAT_LOVER",
    "RELAT_FRIEND",
    "RELAT_JOB",
    "RELAT_ISOL",
    "RELAT_OTHERS",
    "RELAT_ETC",
]
PERSON_CONCERNED = [
    "PC_PARENT",
    "PC_GRANDPARENT",
    "PC_CHILD",
    "PC_SIBSHIP",
    "PC_RELATIVE",
    "PC_LOVER",
    "PC_FRIEND",
    "PC_STRANGER",
    "PC_ETC",
]
COHABITATION_EX = [
    "SPOUSE",
    "PARENT",
    "GRANDPARENT",
    "CHILD",
    "SIBSHIP",
    "RELATIVE",
    "LOVER",
    "FRIEND",
    "STRANGER",
    "ETC",
]
NP_SX = [
    "PSYCHOSIS",
    "MANIC_SX",
    "DEPRESSION_SX",
    "ANXIETY_SX",
    "ACUTE_SX",
    "SOMATIZATION_SX",
    "INSOMNIA_SX",
    "COGDECLINE_SX",
    "ALCOHOL_SX",
    "SUBSTANCE_SX",
    "BEHAVADD_SX",
    "CHILDADOLSX_A",
    "NP_ETC",
]
NP_DXEX = [
    "SPR",
    "BIPOLAR_D",
    "DEPRESSIVE_D",
    "ANXIETY_D",
    "ADJUSTMENT_D",
    "SOMATICSX_D",
    "SLEEP_D",
    "DEMENTIA",
    "ALCOHOLUSE_D",
    "SUBSTANCEUSE_D",
    "BEHAV_ADD",
    "CHILDADOLSX_B",
    "NP_DXETC",
]
NP_THERAPYEX = [
    "THERAPY_DK",
    "NP_OUTPATIENT",
    "NP_INPATIENT",
    "NONNP_THERAPY",
    "COUNSEL_CENTER",
    "THERAPYETC",
]

WARNING_SIGN_COLUMNS = [
    "WARNSPEAK1",
    "WARNSPEAK2",
    "WARNSPEAK3",
    "WARNSPEAK4",
    "WARNBEHAV_DK",
    "WARNBEHAV1",
    "WARNBEHAV2",
    "WARNBEHAV3",
    "WARNBEHAV4",
    "WARNEMOTION_DK",
    "WARNEMOTION1",
    "WARNEMOTION2",
    "WARNEMOTION3",
    "WARNEMOTION4",
]

GROUP_SUICIDE_COLUMNS = [
    "G_SPOUSE",
    "G_PARENT",
    "G_GRANDPARENT",
    "G_CHILD",
    "G_SIBSHIP",
    "G_RELATIVE",
    "G_LOVER",
    "G_FRIEND",
    "G_FORSUICIDE",
    "G_ETC",
]

MURDER_SUICIDE_COLUMNS = [
    "M_SPOUSE",
    "M_PARENT",
    "M_GRANDPARENT",
    "M_CHILD",
    "M_SIBSHIP",
    "M_RELATIVE",
    "M_LOVER",
    "M_FRIEND",
    "M_FORSUICIDE",
    "M_ETC",
]

COLUMNS_TO_DROP = [
    "birthday",
    "DK",
    "AGE",
    "SUICIDE_DATE",
    "FIND_HM",
    "FIND_DATE",
    "FIND_HOMEPLACE",
    "NOTE_CONTENT1",
    "NOTE_CONTENT2",
    "NOTE_CONTENT3",
    "NOTE_CONTENT4",
    "NOTE_CONTENT5",
    "NOTE_CONTENT6",
    "NOTE_CONTENTDTL",
    "SUICIDE_HABTP",
    "NOTE_RECDK",
    "NOTE_PAPER",
    "NOTE_MEMOPAPER",
    "NOTE_ONLINE",
    "NOTE_PHONETEXT",
    "NOTE_RECETC",
    "RRN_SIDO",
    "RRN_SIGOONGU",
    "RRN_DONG_ADM",
    "HOME_SIGOONGU",
    "HOME_DONG_ADM",
    "FIND_SIGOONGU",
    "FIND_DONG_ADM",
    "NoteExist",
    "G_DK",
    "GROUP_COGCOS",
    "M_DK",
    "METHOD_DRUG",
    "SLEEPINGPILL",
    "PAINKILLER",
    "PRESCRIB_DRUG",
    "DRUG_ETC",
    "METHOD_CHEMICAL",
    "AGROCHEMICAL",
    "INSECTICIDE",
    "HERBICIDE",
    "RODENTICIDE",
    "AGRO_ETC",
    "ASPHYXIA",
    "HANGING",
    "GAS_ASPHYXIA",
    "DROWNINGINWATER",
    "ASP_ETC",
    "FALLDOWN",
    "SELF_INJURY",
    "PIERCE_TOOL",
    "JUMP_VEHICLE",
    "SELF_BURNING",
    "INJURY_WTGUN",
    "INJURY_ETC",
    "METHOD_ETC",
    "METHOD_DK",
    "ALINAT_DK",
    "ECONOMY_DK",
    "FAM_DK",
    "RELAT_DK",
    "NP_DXDK",
    "NP_DK",
    "THERAPY_DK",
    "PC_DK",
    "WARNSPEAK_DK",
    "WARNSIGN_TP",
    "WARNBEHAV_DK",
    "WARNEMOTION_DK",
    "GROUP_SUICIDE",
    "S_AFTERMURDER",
    "PERSONCONCERNED",
    "NOTE_DK",
]

ADDITIONAL_DROPS = [
    "HAB_TP",
    "HOME_SIDO",
    "FIND_SIDO",
    "PERSONCONCERNED_None",
] + PERSON_CONCERNED


def normalize_text(value: Any) -> Any:
    """Normalize text only for missing-value matching."""
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value).strip().lower()
        return re.sub(r"\s+", " ", normalized)
    return value


def unknown_mask(
    series: pd.Series, include_nan_as_unknown: bool = True
) -> tuple[pd.Series, pd.Series]:
    """Return the original analysis' unknown mask and numeric conversion."""
    normalized = series.map(normalize_text)
    original_nan = series.isna()
    numeric = pd.to_numeric(normalized, errors="coerce")
    numeric_unknown = numeric.isin(NUMERIC_UNKNOWN_CODES)
    text_unknown = normalized.map(
        lambda value: isinstance(value, str)
        and (value in TEXT_UNKNOWN_TOKENS or value == "")
    )
    mask = numeric_unknown | text_unknown
    if include_nan_as_unknown:
        mask |= original_nan
    return mask, numeric


def is_binary_12(series: pd.Series) -> bool:
    """Return True when all observed, non-unknown values are coded 1 or 2."""
    mask, numeric = unknown_mask(series)
    valid = numeric[~mask]
    return not valid.empty and set(valid.unique()).issubset({1, 2})


def recode_binary(series: pd.Series) -> pd.Series:
    """Apply the original 1 -> 1, 2 -> 0, unknown -> -1 rule."""
    mask, numeric = unknown_mask(series)
    output = pd.Series(-1, index=series.index, dtype="Int64")
    valid = ~mask
    output.loc[valid & numeric.eq(1)] = 1
    output.loc[valid & numeric.eq(2)] = 0
    return output


def factorize_by_frequency(series: pd.Series) -> tuple[pd.Series, dict[int, Any]]:
    """Code valid levels 0..K-1 in descending observed-frequency order."""
    mask, _ = unknown_mask(series)
    valid = series[~mask]
    levels = valid.value_counts().index.tolist()
    forward_mapping = {value: index for index, value in enumerate(levels)}
    output = pd.Series(-1, index=series.index, dtype="Int64")
    output.loc[~mask] = valid.map(forward_mapping).astype("Int64")
    reverse_mapping = {index: value for value, index in forward_mapping.items()}
    return output, reverse_mapping


def clean_and_diagnose(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Categorize every column and record its missingness/coding metadata."""
    cleaned = df.copy()
    metadata: list[dict[str, Any]] = []

    for column in cleaned.columns:
        source = cleaned[column]
        if is_binary_12(source):
            recoded = recode_binary(source)
            coding_type = "binary_1->1_2->0_unknown->-1"
            mapping: dict[int, Any] = {1: "Yes", 0: "No", -1: "Unknown"}
        else:
            recoded, mapping = factorize_by_frequency(source)
            coding_type = "factorized_0..K-1_unknown=-1"

        cleaned[column] = recoded.astype("category")
        unknown_count = int(recoded.eq(-1).sum())
        metadata.append(
            {
                "column": column,
                "type": coding_type,
                "n": len(cleaned),
                "n_unique": cleaned[column].nunique(dropna=False),
                "unknown_count": unknown_count,
                "unknown_pct": unknown_count / len(cleaned) if len(cleaned) else 0.0,
                "value_counts": recoded.value_counts(dropna=False).to_dict(),
                "mapping": mapping,
            }
        )

    diagnostics = (
        pd.DataFrame(metadata)
        .sort_values("unknown_pct", ascending=False)
        .reset_index(drop=True)
    )
    diagnostics["mostly_unknown"] = diagnostics["unknown_pct"].ge(0.5)
    return cleaned, diagnostics


def require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    """Fail early with a readable list when the restricted input schema differs."""
    missing = sorted(columns.difference(df.columns))
    if missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise KeyError(
            f"The source dataframe is missing {len(missing)} required column(s): "
            f"{preview}{suffix}"
        )


def reorganize_demographic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact preprocessing logic used for the manuscript analysis."""
    # PERSONCONCERNED_None is created below; every other dropped column must
    # already exist in the restricted source dataframe.
    required = set(COLUMNS_TO_DROP + ADDITIONAL_DROPS)
    required.discard("PERSONCONCERNED_None")
    required.update(
        {
            "NOTE_EX",
            "COHABITATION_EX",
            "GROUP_SUICIDE",
            "S_AFTERMURDER",
            "PERSONCONCERNED",
            "NP_THERAPYEX",
            "JOB_PB",
            "ECONOMY_PB",
            "FAMILY_PB",
            "RELATION_PB",
            "NP_SX",
            "NP_DXEX",
            "EMPLOYMENT_TP",
            "JOB_TP",
            "ALINAT_DK",
            "ECONOMY_DK",
            "FAM_DK",
            "RELAT_DK",
            "NP_DK",
            "NP_DXDK",
            "THERAPY_DK",
            "PC_DK",
        }
    )
    required.update(WARNING_SIGN_COLUMNS)
    required.update(GROUP_SUICIDE_COLUMNS)
    required.update(MURDER_SUICIDE_COLUMNS)
    required.update(JOB_PB + ECONOMY_PB + FAMILY_PB + RELATION_PB)
    required.update(COHABITATION_EX + NP_SX + NP_DXEX + NP_THERAPYEX)
    required.update(f"TIME{index}" for index in range(1, 13))
    require_columns(df, required)

    work = df.copy()

    work["COHABITATION_EX_None"] = work["COHABITATION_EX"].eq(2).astype(int)
    work["GROUP_SUICIDE_None"] = work["GROUP_SUICIDE"].eq(2).astype(int)
    work["S_AFTERMURDER_None"] = work["S_AFTERMURDER"].eq(2).astype(int)
    work["PERSONCONCERNED_None"] = work["PERSONCONCERNED"].eq(2).astype(int)

    corrections = [
        ("NP_THERAPYEX", "THERAPY_DK", "THERAPYETC"),
        ("COHABITATION_EX", "DK", "ETC"),
        ("JOB_PB", "ALINAT_DK", "ALINAT_ETC"),
        ("ECONOMY_PB", "ECONOMY_DK", "ECONOMY_ETC"),
        ("FAMILY_PB", "FAM_DK", "FAM_ETC"),
        ("RELATION_PB", "RELAT_DK", "RELAT_ETC"),
        ("NP_SX", "NP_DK", "NP_ETC"),
        ("NP_DXEX", "NP_DXDK", "NP_DXETC"),
        ("PERSONCONCERNED", "PC_DK", "PC_ETC"),
    ]
    for base, unknown, other in corrections:
        work.loc[work[base].eq(1) & work[unknown].eq(999), other] = 1

    work.loc[work["WARNSIGN_TP"].eq(2), WARNING_SIGN_COLUMNS] = 2

    work["G_DK"] = work["G_DK"].mask(work["G_DK"].eq(999), 1).fillna(2)
    group_mask = work["G_DK"].eq(1) | work["GROUP_SUICIDE"].eq(2)
    work.loc[group_mask, GROUP_SUICIDE_COLUMNS] = 2
    work.loc[work["G_DK"].eq(1), "G_ETC"] = 1

    work["M_DK"] = work["M_DK"].mask(work["M_DK"].eq(999), 1).fillna(2)
    murder_mask = work["M_DK"].eq(1) | work["S_AFTERMURDER"].eq(2)
    work.loc[murder_mask, MURDER_SUICIDE_COLUMNS] = 2
    work.loc[work["M_DK"].eq(1), "M_ETC"] = 1

    grouped_subcolumns = [
        (JOB_PB, "JOB_PB"),
        (ECONOMY_PB, "ECONOMY_PB"),
        (FAMILY_PB, "FAMILY_PB"),
        (RELATION_PB, "RELATION_PB"),
        (COHABITATION_EX, "COHABITATION_EX"),
        (NP_SX, "NP_SX"),
        (NP_DXEX, "NP_DXEX"),
        (NP_THERAPYEX, "NP_THERAPYEX"),
    ]
    for subcolumns, base in grouped_subcolumns:
        work.loc[work[base].eq(2), subcolumns] = 2

    warning_signs = [
        "WARNSPEAK1",
        "WARNSPEAK2",
        "WARNSPEAK3",
        "WARNSPEAK4",
        "WARNBEHAV1",
        "WARNBEHAV2",
        "WARNBEHAV3",
        "WARNBEHAV4",
        "WARNEMOTION1",
        "WARNEMOTION2",
        "WARNEMOTION3",
        "WARNEMOTION4",
    ]
    for sign, timing in zip(warning_signs, (f"TIME{i}" for i in range(1, 13))):
        work.loc[work[sign].eq(2), timing] = 2

    non_economic = work["EMPLOYMENT_TP"].eq(5) | work["EMPLOYMENT_TP"].eq(7)
    work.loc[non_economic, "JOB_TP"] = 0

    work = work.drop(columns=COLUMNS_TO_DROP)
    work = work.drop(columns=ADDITIONAL_DROPS)
    return work.loc[work["NOTE_EX"].ne(88)].copy()


def load_source_dataframe(path: Path) -> pd.DataFrame:
    """Load the authorized pickle and verify that it contains one dataframe."""
    if not path.exists():
        raise FileNotFoundError(
            f"Restricted source data not found: {path}\n"
            "See data/README.md for placement and privacy guidance."
        )
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, pd.DataFrame):
        raise TypeError("The source pickle must contain a pandas DataFrame.")
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_source_dataframe(args.input)
    reorganized = reorganize_demographic_data(source)
    cleaned, diagnostics = clean_and_diagnose(reorganized)

    for path in (args.output, args.diagnostics, args.metadata):
        path.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_excel(args.output, index=False)
    diagnostics.to_csv(args.diagnostics, index=False)
    with args.metadata.open("wb") as handle:
        pickle.dump({"diagnostics": diagnostics}, handle)

    print(f"Saved cleaned dataframe: {args.output} ({len(cleaned):,} rows)")
    print(f"Saved cleaning diagnostics: {args.diagnostics}")


if __name__ == "__main__":
    main()
