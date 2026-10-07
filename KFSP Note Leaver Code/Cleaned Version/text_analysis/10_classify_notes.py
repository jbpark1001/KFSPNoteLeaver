from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project import ROOT, input_path, output_path, load_frame
import pandas as pd
from raw_detector import SummarizationDetector, run_all_stability_checks
import argparse
parser = argparse.ArgumentParser(description="Fit the supplied weak-supervision detector; preserve source row IDs.")
parser.add_argument("--stability", action="store_true")
parser.add_argument("--weak-holdout", action="store_true", help="Train a separate validation pipeline without test-set leakage")
args = parser.parse_args()
notes = load_frame(input_path("eligible_notes"))
if not notes.index.is_unique:
    raise ValueError("Input note index must be unique for sentence linkage.")
notes = notes.loc[notes["NOTE_CONTENTDTL"].notna()].copy()
# Original manual workflow mutates this field before the classifier cell runs.
# Apply the same normalization here, before weak labeling and TF-IDF fitting.
notes["NOTE_CONTENTDTL"] = notes["NOTE_CONTENTDTL"].str.replace("엄마아빠", "엄마 아빠", regex=False)
det = SummarizationDetector(text_col="NOTE_CONTENTDTL", min_rule_conf=0.34, use_length_feature=False)
det.fit(notes, validate_weak_holdout=args.weak_holdout)
scored = det.score_dataframe(notes, threshold=0.5)
scored.index = notes.index
scored.to_pickle(ROOT / "derived/scored_notes.pkl")
scored["pred_label"].value_counts().to_csv(output_path("tables/classification_counts.csv"))
if hasattr(det,'weak_holdout_report_'):
    import json
    output_path('tables/classifier_weak_holdout.json').write_text(json.dumps(det.weak_holdout_report_,indent=2),encoding='utf-8')
if args.stability:
    results = run_all_stability_checks(det, notes, text_col="NOTE_CONTENTDTL", threshold=0.5, run_retrain_bootstrap=True)
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_path("tables/classifier_" + key + ".csv"), index=False)
