"""Project-local paths; importing never loads study data."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def input_path(key):
    profile = os.environ.get("MANUSCRIPT_INPUT_PROFILE", "saved")
    filename = "paths.rebuilt.json" if profile == "rebuilt" else "paths.json"
    config = json.loads((ROOT / "config" / filename).read_text(encoding="utf-8"))
    value = Path(config[key]).expanduser()
    return value if value.is_absolute() else ROOT / value

def output_path(relative):
    base = ROOT / "outputs"
    if os.environ.get("MANUSCRIPT_INPUT_PROFILE", "saved") == "rebuilt":
        base = base / "rebuilt"
    value = base / relative
    value.parent.mkdir(parents=True, exist_ok=True)
    return value

def load_frame(path):
    import pandas as pd
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}. Update config/paths.json.")
    if path.suffix.lower() == ".xlsx": return pd.read_excel(path)
    if path.suffix.lower() == ".csv": return pd.read_csv(path)
    return pd.read_pickle(path)
