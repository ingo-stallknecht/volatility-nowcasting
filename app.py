# app.py — Volatility Nowcasting demo (Streamlit)
# Reads precomputed CSVs (feat.csv, oof.csv, y.csv, meta.json) and visualizes results.

import os, json, math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Volatility Nowcasting — S&P 500", layout="wide")

# --------------------------
# Config & data location
# --------------------------
TRADING_DAYS_PER_YEAR = 252  # finance standard for annualization

# Prefer an env var (e.g., set DATA_DIR=data/demo on Streamlit Cloud), else fall back in order:
DATA_DIR = os.getenv("DATA_DIR")
if not DATA_DIR:
    # 1) repo layout (for Streamlit Cloud / local runs)
    if os.path.isdir("data/demo"):
        DATA_DIR = "data/demo"
    # 2) Colab launcher writes to /tmp/vol_app
    elif os.path.isdir("/tmp/vol_app"):
        DATA_DIR = "/tmp/vol_app"
    else:
        DATA_DIR = "."  # last resort; will error nicely if files missing

def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV that contains a 'date' column."""
    p = os.path.join(DATA_DIR, name)
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

# --------------------------
# Load artifacts exported by your notebook
# --------------------------
feat = load_csv("feat.csv")  # must contain a 'date' column
oof = pd.read_csv(os.path.join(DATA_DIR, "oof.csv"))["oof"].astype(float).values
y = pd.read_csv(os.path.join(DATA_DIR, "y.csv"))["y"].astype(float).values
with open(os.path.join(DATA_DIR, "meta.json")) as f:
    meta = json.load(f)

H = int(meta.get("H", 5))
log_target = bool(meta.get("log1p", True))

# --------------------------
# Intro / Description
# --------------------------
st.title(f"Volatility Nowcasting — S&P 500 (H={H})")

st.markdown(
    """
**What this dashboard shows**

We estimate **near-term market risk** by predicting the next **H** trading days’ **realized variance (RV)** of the S&P 500.
The model uses **prices** (returns/vols/ranges) and **daily pooled news** signals (FinBERT sentiment + embeddings).
Training is done on **log(1 + RV)** for stability, then we convert back to **volatility (σ)** so results are readable in percent.

**How to read the charts**

1) **OOF — truth vs prediction (log(1+RV_H))**  
   Compares actual target vs model prediction in the same (log) space. *OOF* = out-of-fold, i.e., no look-ahead leakage.

2) **Derived volatility: daily & annualized (%)**  
   Converts RV to σ so you can read risk in percent:  
   σ_daily = √(RV_H / H) and σ_annual ≈ σ_daily × √252.

3) **Rolling metrics (63d)** — two complementary views of skill  
   - **IC (Information Coefficient)** = **Spearman rank correlation** between truth and prediction within each window.  
     Answers: “Does the model get the **ordering** of higher vs lower risk periods right?” (−1..+1; higher is better)
   - **R2 (coefficient of determination)** = **squared Pearson correlation** in the same window.  
     Answers: “How much of the **variation in magnitude** does the model explain?” ([0..1]; higher is better)  
   In short, **IC** measures **ranking skill**, **R2** measures **fit to levels**.
"""
)

# --------------------------
# Window selector
# --------------------------
dmin, dmax = feat["date"].min(), feat["date"].max()
start_default = max(dmin, dmax - pd.Timedelta(days=2 * 365))
s = st.slider(
    "Plot window",
    min_value=dmin.to_pydatetime(),
    max_value=dmax.to_pydatetime(),
    value=(start_default.to_pydatetime(), dmax.to_pydatetime()),
)
mask = (feat["date"] >= pd.Timestamp(s[0])) & (feat["date"] <= pd.Timestamp(s[1]))
dates = feat.loc[mask, "date"]

# Series aligned to full frame (then we slice)
y_full = pd.Series(y, index=feat.index).astype(float)
oof_full = pd.Series(oof, index=feat.index).astype(float)

# --------------------------
# Left column: truth vs pred + volatility
# --------------------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    y_plot = y_full.loc[mask].to_numpy()
    oof_plot = oof_full.loc[mask].to_numpy()

    st.subheader("OOF — truth vs prediction (log(1+RV_H))")
    df_plot = pd.DataFrame({"date": dates, "truth": y_plot, "pred": oof_plot}).set_index("date")
    st.line_chart(df_plot)

    st.subheader("Derived volatility: daily & annualized (%)")
    rv_true = np.expm1(y_plot) if log_target else y_plot
    rv_pred = np.expm1(oof_plot) if log_target else oof_plot
    sigma_true = np.sqrt(np.maximum(rv_true, 0.0) / H)
    sigma_pred = np.sqrt(np.maximum(rv_pred, 0.0) / H)
    ann_true = sigma_true * (TRADING_DAYS_PER_YEAR ** 0.5)
    ann_pred = sigma_pred * (TRADING_DAYS_PER_YEAR ** 0.5)
    st.line_chart(pd.DataFrame({"σ_ann_true": ann_true, "σ_ann_pred": ann_pred}, index=dates))

# --------------------------
# Right column: robust rolling metrics (IC & R2 on same y-axis)
# --------------------------
def roll_r2(y_arr, p_arr, w: int):
    """Robust rolling R² via corr^2 with variance guards; returns 0 if variance ~ 0."""
    y = np.asarray(y_arr, float)
    p = np.asarray(p_arr, float)
    out = np.full(len(y), np.nan, float)
    need = max(5, int(0.8 * w))
    for i in range(w - 1, len(y)):
        ys = y[i - w + 1 : i + 1]
        ps = p[i - w + 1 : i + 1]
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

def roll_ic(y_arr, p_arr, w: int):
    """Robust rolling Spearman rank correlation (IC)."""
    y = np.asarray(y_arr, float)
    p = np.asarray(p_arr, float)
    out = np.full(len(y), np.nan, float)
    need = max(5, int(0.8 * w))
    for i in range(w - 1, len(y)):
        ys = y[i - w + 1 : i + 1]
        ps = p[i - w + 1 : i + 1]
        m = np.isfinite(ys) & np.isfinite(ps)
        if m.sum() < need:
            continue
        out[i] = pd.Series(ys[m]).corr(pd.Series(ps[m]), method="spearman")
    return out

with col2:
    st.subheader("Rolling metrics (63d)")
    win = 63
    # compute on full series, then slice to the window (avoids empty early windows)
    r2_full = roll_r2(y_full.values, oof_full.values, win)
    ic_full = roll_ic(y_full.values, oof_full.values, win)
    r2_roll = np.asarray(r2_full)[mask.values]
    ic_roll = np.asarray(ic_full)[mask.values]

    df_metrics = pd.DataFrame({"date": dates.to_numpy(), "IC": ic_roll, "R2": r2_roll})
    df_long = df_metrics.melt("date", var_name="metric", value_name="value")
    df_long.loc[~np.isfinite(df_long["value"]), "value"] = None  # gaps instead of dropping

    chart = (
        alt.Chart(df_long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Rolling IC & R2 (63d)", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["IC", "R2"], range=["#1f77b4", "#ff7f0e"])),
        )
        .properties(height=230)
    )
    st.altair_chart(chart, use_container_width=True)

# --------------------------
# Latest forecast card
# --------------------------
last_idx = int(pd.Series(oof_full).last_valid_index())
last_pred = float(oof_full.iloc[last_idx])
rvH_pred = math.expm1(last_pred) if log_target else last_pred
sigma_daily = (max(rvH_pred, 0.0) / H) ** 0.5
sigma_ann = sigma_daily * (TRADING_DAYS_PER_YEAR ** 0.5)

st.markdown("### Latest next-H-day forecast")
st.metric(label=f"Annualized σ (H={H})", value=f"{sigma_ann:.2%}")
st.caption(f"RV_H≈{rvH_pred:.6f} • σ_daily≈{sigma_daily:.2%}")

st.markdown("---")
st.caption("OOF = out-of-fold predictions on expanding time splits with embargo (no look-ahead leakage).")
