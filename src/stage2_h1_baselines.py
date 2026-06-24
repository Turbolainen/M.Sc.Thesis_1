"""
Stage 2 — H1 baseline regressions
  Reg 1: Y1 ~ e_V + e_L  (zone FE + calendar dummies, DK-SE)
  Reg 2: Y2 ~ e_V + e_L  (same)
"""

import pandas as pd
import numpy as np
from linearmodels import PanelOLS

# ---------------------------------------------------------------------------
# Load enriched panel
# ---------------------------------------------------------------------------
df = pd.read_pickle("data/processed/panel_enriched.pkl")

# linearmodels needs entity as the *first* level of the MultiIndex
# df is already (zone, timestamp_cet)

# ---------------------------------------------------------------------------
# Calendar dummies helper
# ---------------------------------------------------------------------------

def make_calendar_dummies(data):
    """Return DataFrame of hour, dow, month dummies (drop-one each)."""
    parts = []
    for col, prefix, n in [("hour", "hr", 24), ("dayofweek", "dow", 7), ("month", "mon", 12)]:
        d = pd.get_dummies(data[col], prefix=prefix, drop_first=True, dtype=float)
        parts.append(d)
    return pd.concat(parts, axis=1)


def run_h1(dep_var: str, label: str):
    """Run one H1 baseline regression and return the result object."""
    cols = [dep_var, "e_V", "e_L", "hour", "dayofweek", "month"]
    sub = df[cols].dropna()
    print(f"\n  {label}: N = {len(sub):,}  (dropped {len(df) - len(sub):,} NaN rows)")

    cal = make_calendar_dummies(sub)
    exog = pd.concat([sub[["e_V", "e_L"]], cal], axis=1)
    exog.insert(0, "const", 1.0)

    mod = PanelOLS(
        dependent=sub[[dep_var]],
        exog=exog,
        entity_effects=True,
        time_effects=False,
    )
    res = mod.fit(
        cov_type="kernel",
        kernel="bartlett",
        bandwidth=24,
    )
    return res, len(sub)


print("=" * 60)
print("STAGE 2 — H1 BASELINES")
print("=" * 60)

res_y1, n_y1 = run_h1("Y1", "Y1 ~ e_V + e_L")
res_y2, n_y2 = run_h1("Y2", "Y2 ~ e_V + e_L")

# ---------------------------------------------------------------------------
# Print key coefficients
# ---------------------------------------------------------------------------

def show_coefs(res, label, n):
    params = res.params
    se     = res.std_errors
    tstat  = res.tstats
    pval   = res.pvalues
    r2     = res.rsquared_within

    print(f"\n{'='*60}")
    print(f"  {label}   N={n:,}   Within-R²={r2:.4f}")
    print(f"{'='*60}")
    print(f"  {'Variable':<15} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
    print(f"  {'-'*53}")
    for v in ["e_V", "e_L"]:
        stars = "***" if pval[v] < 0.01 else "**" if pval[v] < 0.05 else "*" if pval[v] < 0.10 else ""
        print(f"  {v:<15} {params[v]:>10.4f} {se[v]:>10.4f} {tstat[v]:>8.3f} {pval[v]:>8.4f}  {stars}")

show_coefs(res_y1, "Y1 ~ e_V + e_L", n_y1)
show_coefs(res_y2, "Y2 ~ e_V + e_L", n_y2)

# ---------------------------------------------------------------------------
# Save results for Stage 4
# ---------------------------------------------------------------------------
import pickle
with open("data/processed/stage2_results.pkl", "wb") as f:
    pickle.dump({"res_y1": res_y1, "n_y1": n_y1,
                 "res_y2": res_y2, "n_y2": n_y2}, f)
print("\nSaved data/processed/stage2_results.pkl")
