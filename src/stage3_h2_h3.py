"""
Stage 3 — H2 leakage + H3 zone heterogeneity regressions
  Reg 3: Y2 ~ e_V_pre + e_V_post + e_L
  Reg 4: Y1 ~ e_V:SE1 + e_V:SE2 + e_V:SE3 + e_V:SE4 + e_L  + Wald test
  Reg 5: Y2 ~ e_V:SE1 + e_V:SE2 + e_V:SE3 + e_V:SE4 + e_L  + Wald test
"""

import pickle
import numpy as np
import pandas as pd
from linearmodels import PanelOLS

df = pd.read_pickle("data/processed/panel_enriched.pkl")
ZONES = ["SE1", "SE2", "SE3", "SE4"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_calendar_dummies(data):
    parts = []
    for col, prefix in [("hour", "hr"), ("dayofweek", "dow"), ("month", "mon")]:
        d = pd.get_dummies(data[col], prefix=prefix, drop_first=True, dtype=float)
        parts.append(d)
    return pd.concat(parts, axis=1)


def fit_panel(dep, exog_df):
    mod = PanelOLS(
        dependent=dep,
        exog=exog_df,
        entity_effects=True,
        time_effects=False,
    )
    return mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=24)


def show_coefs(res, label, n, vars_of_interest):
    params  = res.params
    se      = res.std_errors
    tstat   = res.tstats
    pval    = res.pvalues
    r2      = res.rsquared_within
    print(f"\n{'='*65}")
    print(f"  {label}   N={n:,}   Within-R²={r2:.4f}")
    print(f"{'='*65}")
    print(f"  {'Variable':<20} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
    print(f"  {'-'*58}")
    for v in vars_of_interest:
        stars = "***" if pval[v]<0.01 else "**" if pval[v]<0.05 else "*" if pval[v]<0.10 else ""
        print(f"  {v:<20} {params[v]:>10.4f} {se[v]:>10.4f} {tstat[v]:>8.3f} {pval[v]:>8.4f}  {stars}")


def wald_equality(res, param_names):
    """
    Test H0: all coefficients in param_names are equal.
    Uses J-1 linear restrictions: beta[0]=beta[1], beta[0]=beta[2], ...
    Returns (W_stat, p_value, df_num).
    """
    all_params = list(res.params.index)
    k = len(all_params)
    J = len(param_names)
    idx = [all_params.index(p) for p in param_names]

    # Contrast matrix: (J-1) × k
    R = np.zeros((J - 1, k))
    for i in range(J - 1):
        R[i, idx[0]]   =  1.0
        R[i, idx[i+1]] = -1.0

    wt = res.wald_test(R)
    return float(wt.stat), float(wt.pval), J - 1


# ---------------------------------------------------------------------------
# REG 3: H2 Leakage  Y2 ~ e_V_pre + e_V_post + e_L
# ---------------------------------------------------------------------------
print("=" * 65)
print("STAGE 3 — H2 LEAKAGE + H3 ZONE HETEROGENEITY")
print("=" * 65)

cols3 = ["Y2", "e_V_pre", "e_V_post", "e_L", "hour", "dayofweek", "month"]
sub3  = df[cols3].dropna()
n3    = len(sub3)
print(f"\nReg 3  Y2 ~ e_V_pre + e_V_post + e_L   N={n3:,}  (dropped {len(df)-n3:,})")

cal3   = make_calendar_dummies(sub3)
exog3  = pd.concat([sub3[["e_V_pre", "e_V_post", "e_L"]], cal3], axis=1)
exog3.insert(0, "const", 1.0)
res3   = fit_panel(sub3[["Y2"]], exog3)
show_coefs(res3, "Y2 ~ e_V_pre + e_V_post + e_L", n3,
           ["e_V_pre", "e_V_post", "e_L"])


# ---------------------------------------------------------------------------
# REG 4 & 5: H3 Zone heterogeneity  Y ~ zone×e_V + e_L
# ---------------------------------------------------------------------------

def run_h3(dep_var, label):
    cols = [dep_var, "e_V", "e_L", "hour", "dayofweek", "month"]
    sub  = df[cols].dropna()
    n    = len(sub)
    print(f"\nReg  {label}   N={n:,}  (dropped {len(df)-n:,})")

    # Zone indicator interactions: e_V × 1{zone==z}  for each z
    entity = sub.index.get_level_values("zone")
    inter_cols = {}
    for z in ZONES:
        col_name = f"eV_{z}"
        inter_cols[col_name] = sub["e_V"] * (entity == z).astype(float)

    inter_df = pd.DataFrame(inter_cols, index=sub.index)
    cal      = make_calendar_dummies(sub)
    exog     = pd.concat([inter_df, sub[["e_L"]], cal], axis=1)
    exog.insert(0, "const", 1.0)

    res  = fit_panel(sub[[dep_var]], exog)
    iv   = [f"eV_{z}" for z in ZONES]
    show_coefs(res, f"{dep_var} ~ zone×e_V + e_L", n, iv + ["e_L"])

    # Wald test H0: β_SE1 = β_SE2 = β_SE3 = β_SE4
    W, pW, df_num = wald_equality(res, iv)
    print(f"\n  Wald test H0: β_SE1=β_SE2=β_SE3=β_SE4  →  W={W:.3f}  p={pW:.4f}  (df={df_num})")

    return res, n, W, pW

res4, n4, W4, pW4 = run_h3("Y1", "H3-Y1")
res5, n5, W5, pW5 = run_h3("Y2", "H3-Y2")

# ---------------------------------------------------------------------------
# Save for Stage 4
# ---------------------------------------------------------------------------
with open("data/processed/stage3_results.pkl", "wb") as f:
    pickle.dump({
        "res3": res3, "n3": n3,
        "res4": res4, "n4": n4, "W4": W4, "pW4": pW4,
        "res5": res5, "n5": n5, "W5": W5, "pW5": pW5,
    }, f)
print("\nSaved data/processed/stage3_results.pkl")
