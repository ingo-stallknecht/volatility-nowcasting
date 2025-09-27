# app.py — Volatility Nowcasting dashboard (Streamlit)
# ---------------------------------------------------
# Expects four small files in repo ./data:
#   - feat.csv   (must contain a 'date' column; case-insensitive)
#   - oof.csv    (column 'oof' with out-of-fold predictions of log(1+RV_H) or RV_H)
#   - y.csv      (column 'y' with truth in the same space as 'oof')
#   - meta.json  ({"H": 5, "log1p": true})
#
# If these files are missing on Streamlit Cloud, the app will fall back to a small
# synthetic demo dataset in /tmp to stay interactive rather than crashing.

import os, json, math
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------
# Page & constants
# -----------------------------
st.set_page_config(page_title="Volatility Nowcasting — S&P 500", layout="wide")
TRADING_DAYS_PER_YEAR = 252

# -----------------------------
# Robust data loading
# -----------------------------
REQUIRED_FILES = ["feat.csv", "oof.csv", "y.csv", "meta.json"]


def listdir_safe(path: str):
    try:
        return os.listdir(path)
    except Exception:
        return []


def have_all_files(base: str) -> bool:
    return all(os.path.exists(os.path.join(base, f)) for f in REQUIRED_FILES)


def ensure_demo_data(base: str):
    """Create a tiny synthetic demo dataset so the app stays usable."""
    os.makedirs(base, exist_ok=True)
    dates = pd.bdate_range("2023-01-03", periods=260)
    rng = np.random.default_rng(42)
    # "Truth" in log(1+RV_H) space (toy)
    y = rng.normal(0.03, 0.015, size=len(dates)).clip(-0.02, 0.20)
    # Noisy predictions
    o = (y + rng.normal(0.0, 0.01, size=len(dates))).clip(-0.05, 0.25)
    pd.DataFrame({"date": dates}).to_csv(os.path.join(base, "feat.csv"), index=False)
    pd.DataFrame({"oof": o}).to_csv(os.path.join(base, "oof.csv"), index=False)
    pd.DataFrame({"y": y}).to_csv(os.path.join(base, "y.csv"), index=False)
    with open(os.path.join(base, "meta.json"), "w") as f:
        json.dump({"H": 5, "log1p": True}, f)


def detect_data_dir() -> Tuple[str, bool]:
    """
    Prefer repo ./data on Streamlit Cloud. If missing/incomplete, fall back to /tmp/vol_demo
    and populate a tiny demo dataset. Returns (data_dir, is_demo).
    """
    repo_data = "data"
    if os.path.isdir(repo_data) and have_all_files(repo_data):
        return repo_data, False
    # Fall back to a temp demo dataset
    tmp_demo = "/tmp/vol_demo"
    if not have_all_files(tmp_demo):
        ensure_demo_data(tmp_demo)
    return tmp_demo, True


DATA_DIR, IS_DEMO = detect_data_dir()


def load_csv(name: str) -> pd.DataFrame:
    p = os.path.join(DATA_DIR, name)
    if name == "feat.csv":
        df = pd.read_csv(p)
        # allow 'Date'/'DATE'/etc.
        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        if date_col is None:
            st.error("`feat.csv` must include a 'date' column.")
            st.stop()
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            st.error("Some dates in feat.csv could not be parsed.")
            st.stop()
        return df
    return pd.read_csv(p)


# -----------------------------
# Load artifacts
# -----------------------------
feat = load_csv("feat.csv")  # contains 'date'
oof = pd.read_csv(os.path.join(DATA_DIR, "oof.csv"))["oof"].astype(float).values
y = pd.read_csv(os.path.join(DATA_DIR, "y.csv"))["y"].astype(float).values
with open(os.path.join(DATA_DIR, "meta.json")) as fh:
    meta = json.load(fh)
H = int(meta.get("H", 5))
LOG1P_TARGET = bool(meta.get("log1p", True))

# Safety: align lengths defensively
n = min(len(feat), len(oof), len(y))
feat = feat.iloc[:n].reset_index(drop=True)
oof = oof[:n]
y = y[:n]

# -----------------------------
# Sidebar diagnostics
# -----------------------------
with st.sidebar:
    st.markdown("#### Data diagnostics")
    st.write("**Source:**", "demo (/tmp/vol_demo)" if IS_DEMO else "repo ./data")
    st.write("CWD:", os.getcwd())
    st.write("DATA_DIR:", os.path.abspath(DATA_DIR))
    st.write("Files:", listdir_safe(DATA_DIR))
    st.caption("If you see the demo source, commit the four data files under ./data in your repo.")

# -----------------------------
# Header & description
# -----------------------------
st.title("Volatility Nowcasting — S&P 500")

st.markdown(
    """
**What this dashboard shows**

We estimate **near-term market risk** by predicting the next **H** trading days’ **realized variance (RV)** of the S&P 500.
The model is trained on **prices** (returns, realized vols, ranges) plus **daily-pooled news** (FinBERT sentiment & embeddings).
Training uses **log(1+RV_H)** for stability, and we convert predictions back to **volatility (σ)** so results are readable in percent.

**How to read the charts**

1) **OOF — truth vs prediction (log(1+RV_H))**  
   Out-of-fold (OOF) predictions mean **no look-ahead leakage**.

2) **Derived volatility: annualized (%)**  
   Convert RV to volatility:  
   σ_daily = √(RV_H / H), and **σ_annual ≈ σ_daily × √252**.

3) **Rolling metrics (63d)**  
   - **IC (Information Coefficient)** = Spearman rank correlation.  
     *Answers:* “Does the model get the **ordering** of higher vs lower risk right?”  
     Range −1…+1 (higher is better).
   - **R² (Coefficient of Determination)** = squared Pearson correlation.  
     *Answers:* “How much **variation in magnitude** does the model explain?”  
     Range 0…1 (higher is better).
"""
)

# -----------------------------
# Plot window selector
# -----------------------------
dmin, dmax = feat["date"].min(), feat["date"].max()
default_start = max(dmin, dmax - pd.Timedelta(days=2 * 365))
win = st.slider(
    "Plot window",
    min_value=dmin.to_pydatetime(),
    max_value=dmax.to_pydatetime(),
    value=(default_start.to_pydatetime(), dmax.to_pydatetime()),
)
mask = (feat["date"] >= pd.Timestamp(win[0])) & (feat["date"] <= pd.Timestamp(win[1]))
dates = feat.loc[mask, "date"]

# Build aligned series (full → slice by mask)
y_full = pd.Series(y, index=feat.index).astype(float)
oof_full = pd.Series(oof, index=feat.index).astype(float)

# -----------------------------
# Left column: truth/pred + annual vol
# -----------------------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("OOF — truth vs prediction (log(1+RV_H))")
    y_plot = y_full.loc[mask].to_numpy()
    oof_plot = oof_full.loc[mask].to_numpy()
    df_plot = pd.DataFrame({"date": dates, "truth": y_plot, "pred": oof_plot}).set_index("date")
    st.line_chart(df_plot)

    st.subheader("Derived volatility: annualized (%)")
    # Convert from log(1+RV_H) → RV_H if needed
    rv_true = np.expm1(y_plot) if LOG1P_TARGET else y_plot
    rv_pred = np.expm1(oof_plot) if LOG1P_TARGET else oof_plot
    # Guard and convert to vol
    rv_true = np.maximum(rv_true, 0.0)
    rv_pred = np.maximum(rv_pred, 0.0)
    sigma_daily_true = np.sqrt(rv_true / max(H, 1))
    sigma_daily_pred = np.sqrt(rv_pred / max(H, 1))
    ann_true = sigma_daily_true * np.sqrt(TRADING_DAYS_PER_YEAR)
    ann_pred = sigma_daily_pred * np.sqrt(TRADING_DAYS_PER_YEAR)
    st.line_chart(pd.DataFrame({"σ_annual_true": ann_true, "σ_annual_pred": ann_pred}, index=dates))

# -----------------------------
# Right column: rolling IC & R² (same axis)
# -----------------------------
def roll_r2(y_arr, p_arr, w):
    y_arr = np.asarray(y_arr, float)
    p_arr = np.asarray(p_arr, float)
    out = np.full(len(y_arr), np.nan, float)
    need = max(5, int(0.8 * w))
    for i in range(w - 1, len(y_arr)):
        ys = y_arr[i - w + 1 : i + 1]
        ps = p_arr[i - w + 1 : i + 1]
        m = np.isfinite(ys) & np.isfinite(ps)
        if m.sum() < need:
            continue
        ys = ys[m]
        ps = ps[m]
        if np.var(ys) <= 1e-12 or np.var(ps) <= 1e-12:
            out[i] = 0.0
            continue
        r = np.corrcoef(ys, ps)[0, 1]
        out[i] = float(r * r)
    return out


def roll_ic(y_arr, p_arr, w):
    y_arr = np.asarray(y_arr, float)
    p_arr = np.asarray(p_arr, float)
    out = np.full(len(y_arr), np.nan, float)
    need = max(5, int(0.8 * w))
    for i in range(w - 1, len(y_arr)):
        ys = y_arr[i - w + 1 : i + 1]
        ps = p_arr[i - w + 1 : i + 1]
        m = np.isfinite(ys) & np.isfinite(ps)
        if m.sum() < need:
            continue
        out[i] = pd.Series(ys[m]).corr(pd.Series(ps[m]), method="spearman")
    return out


with col2:
    st.subheader("Rolling metrics (63d)")
    win_n = 63
    r2_full = roll_r2(y_full.values, oof_full.values, win_n)
    ic_full = roll_ic(y_full.values, oof_full.values, win_n)
    r2_roll = np.asarray(r2_full)[mask.values]
    ic_roll = np.asarray(ic_full)[mask.values]

    df_metrics = pd.DataFrame({"date": dates.to_numpy(), "IC": ic_roll, "R2": r2_roll})
    df_long = df_metrics.melt("date", var_name="metric", value_name="value")
    # draw gaps for NaN
    df_long.loc[~np.isfinite(df_long["value"]), "value"] = None

    chart = (
        alt.Chart(df_long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Rolling IC & R² (63d)", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["IC", "R2"], range=["#1f77b4", "#ff7f0e"])),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("metric:N"), alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=230)
    )
    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Latest forecast card
# -----------------------------
last_valid_idx = int(pd.Series(oof_full).last_valid_index())
last_pred_log = float(oof_full.iloc[last_valid_idx])
rvH_pred = math.expm1(last_pred_log) if LOG1P_TARGET else last_pred_log
rvH_pred = max(rvH_pred, 0.0)
sigma_daily = (rvH_pred / max(H, 1)) ** 0.5
sigma_annual = sigma_daily * (TRADING_DAYS_PER_YEAR ** 0.5)

st.markdown("### Latest next-H-day forecast")
st.metric(label=f"Annualized σ (H={H})", value=f"{sigma_annual:.2%}")
st.caption(f"RV_H≈{rvH_pred:.6f} • σ_daily≈{sigma_daily:.2%}")

st.markdown("---")
st.caption(
    "OOF = out-of-fold predictions from expanding time splits with embargo (no look-ahead leakage). "
    "Annualization uses √252 trading days."
)
