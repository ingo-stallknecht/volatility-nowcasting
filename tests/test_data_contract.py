import json
from pathlib import Path
import pandas as pd

DATA = Path("data")

def test_demo_files_exist():
    for name in ["feat.csv", "oof.csv", "y.csv", "meta.json"]:
        assert (DATA / name).exists(), f"Missing data/{name}. Export from notebook and commit it."

def test_feat_has_date_column():
    df = pd.read_csv(DATA / "feat.csv")
    assert "date" in df.columns, "data/feat.csv must include a 'date' column for the app."
    # basic parsability
    pd.to_datetime(df["date"])

def test_oof_and_y_shapes():
    oof = pd.read_csv(DATA / "oof.csv")
    y   = pd.read_csv(DATA / "y.csv")
    assert "oof" in oof.columns, "data/oof.csv must have an 'oof' column."
    assert "y" in y.columns,     "data/y.csv must have a 'y' column."
    # equal length is expected for line charts
    assert len(oof) == len(y), "oof and y must have the same number of rows."

def test_meta_schema():
    meta = json.loads((DATA / "meta.json").read_text())
    assert "H" in meta and isinstance(meta["H"], int), "meta.json must contain integer H."
    assert "log1p" in meta and isinstance(meta["log1p"], bool), "meta.json must contain boolean log1p."
