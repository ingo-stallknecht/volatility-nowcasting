import numpy as np

def roll_r2(y, p, w):
    y = np.asarray(y, float); p = np.asarray(p, float)
    out = np.full(len(y), np.nan, float); need = max(5, int(0.8*w))
    for i in range(w-1, len(y)):
        ys, ps = y[i-w+1:i+1], p[i-w+1:i+1]
        m = np.isfinite(ys) & np.isfinite(ps)
        if m.sum() < need: continue
        ys, ps = ys[m], ps[m]
        if np.var(ys) <= 1e-12 or np.var(ps) <= 1e-12:
            out[i] = 0.0; continue
        r = np.corrcoef(ys, ps)[0,1]
        out[i] = float(r*r)
    return out

def test_roll_r2_increases_with_better_fit():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    p_bad = rng.normal(size=300)
    p_good = y + rng.normal(scale=0.3, size=300)
    r2_bad = roll_r2(y, p_bad, 63)
    r2_good = roll_r2(y, p_good, 63)
    # compare last valid values (end-of-window)
    rb = r2_bad[~np.isnan(r2_bad)][-1]
    rg = r2_good[~np.isnan(r2_good)][-1]
    assert rg >= rb, "R2 should be higher for better predictions."
