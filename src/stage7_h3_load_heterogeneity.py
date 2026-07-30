"""
Stage 7 — H3 extended: zone-specific load forecast error.

Reg 4/5 in stage3_h2_h3.py interact e_V with zone but leave e_L pooled,
implicitly assuming the load-error response is homogeneous across zones.
That assumption is not tested. This stage adds two new regressions that
interact *both* regressors with zone, leaving no pooled term for either:

  Reg 6: Y1 ~ zone×e_V + zone×e_L   (+ Wald tests, both regressors)
  Reg 7: Y2 ~ zone×e_V + zone×e_L   (same)

Does not modify stage3_h2_h3.py or its outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels import PanelOLS

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_pickle("data/processed/panel_enriched.pkl")
ZONES = ["SE1", "SE2", "SE3", "SE4"]


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
    params, se, tstat, pval, r2 = res.params, res.std_errors, res.tstats, res.pvalues, res.rsquared_within
    print(f"\n{'='*65}")
    print(f"  {label}   N={n:,}   Within-R²={r2:.4f}")
    print(f"{'='*65}")
    print(f"  {'Variable':<20} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
    print(f"  {'-'*58}")
    for v in vars_of_interest:
        stars = "***" if pval[v]<0.01 else "**" if pval[v]<0.05 else "*" if pval[v]<0.10 else ""
        print(f"  {v:<20} {params[v]:>10.4f} {se[v]:>10.4f} {tstat[v]:>8.3f} {pval[v]:>8.4f}  {stars}")


def wald_equality(res, param_names):
    """Test H0: all coefficients in param_names are equal."""
    all_params = list(res.params.index)
    k = len(all_params)
    J = len(param_names)
    idx = [all_params.index(p) for p in param_names]

    R = np.zeros((J - 1, k))
    for i in range(J - 1):
        R[i, idx[0]]   =  1.0
        R[i, idx[i+1]] = -1.0

    wt = res.wald_test(R)
    return float(wt.stat), float(wt.pval), J - 1


# ---------------------------------------------------------------------------
# REG 6 & 7: H3 extended  Y ~ zone×e_V + zone×e_L
# ---------------------------------------------------------------------------

def run_h3_ext(dep_var, label):
    cols = [dep_var, "e_V", "e_L", "hour", "dayofweek", "month"]
    sub  = df[cols].dropna()
    n    = len(sub)
    print(f"\nReg  {label}   N={n:,}  (dropped {len(df)-n:,})")

    entity = sub.index.get_level_values("zone")
    inter_cols = {}
    for z in ZONES:
        mask = (entity == z).astype(float)
        inter_cols[f"eV_{z}"] = sub["e_V"] * mask
        inter_cols[f"eL_{z}"] = sub["e_L"] * mask

    inter_df = pd.DataFrame(inter_cols, index=sub.index)
    cal      = make_calendar_dummies(sub)
    exog     = pd.concat([inter_df, cal], axis=1)
    exog.insert(0, "const", 1.0)

    res  = fit_panel(sub[[dep_var]], exog)
    iv_V = [f"eV_{z}" for z in ZONES]
    iv_L = [f"eL_{z}" for z in ZONES]
    show_coefs(res, f"{dep_var} ~ zone×e_V + zone×e_L", n, iv_V + iv_L)

    W_V, pW_V, dfV = wald_equality(res, iv_V)
    W_L, pW_L, dfL = wald_equality(res, iv_L)
    print(f"\n  Wald test H0: e_V equal across zones  →  W={W_V:.3f}  p={pW_V:.4f}  (df={dfV})")
    print(f"  Wald test H0: e_L equal across zones  →  W={W_L:.3f}  p={pW_L:.4f}  (df={dfL})")

    return res, n, W_V, pW_V, W_L, pW_L


print("=" * 65)
print("STAGE 7 — H3 EXTENDED: ZONE-SPECIFIC LOAD FORECAST ERROR")
print("=" * 65)

res6, n6, W6_V, pW6_V, W6_L, pW6_L = run_h3_ext("Y1", "H3ext-Y1")
res7, n7, W7_V, pW7_V, W7_L, pW7_L = run_h3_ext("Y2", "H3ext-Y2")

import pickle
with open("data/processed/stage7_results.pkl", "wb") as f:
    pickle.dump({
        "res6": res6, "n6": n6, "W6_V": W6_V, "pW6_V": pW6_V, "W6_L": W6_L, "pW6_L": pW6_L,
        "res7": res7, "n7": n7, "W7_V": W7_V, "pW7_V": pW7_V, "W7_L": W7_L, "pW7_L": pW7_L,
    }, f)
print("\nSaved data/processed/stage7_results.pkl")


# ===========================================================================
# LaTeX table (same style as table_h3.tex)
# ===========================================================================

def stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def fmt_coef(b, se, p):
    st = stars(p)
    lp = "" if b < 0 else "\\phantom{-}"
    coef_str = f"${lp}{b:.4f}^{{{st}}}$" if st else f"${lp}{b:.4f}$"
    return coef_str, f"$({se:.4f})$"


def extract(res, var):
    return {
        "coef": float(res.params[var]),
        "se":   float(res.std_errors[var]),
        "t":    float(res.tstats[var]),
        "p":    float(res.pvalues[var]),
    }


def reg_row(label, results, last=False):
    pairs     = [fmt_coef(r["coef"], r["se"], r["p"]) for r in results]
    coef_line = " & ".join(b for b, _ in pairs)
    se_line   = " & ".join(s for _, s in pairs)
    end       = "\\\\\n" if last else "\\\\[0.4ex]\n"
    return f"{label} & {coef_line} \\\\\n & {se_line} {end}"


def fmt_wald(W, p):
    st = stars(p)
    return f"${W:.3f}^{{{st}}}$" if st else f"${W:.3f}$"


def tablenotes(text):
    return (
        "\\begin{tablenotes}[flushleft]\n"
        "\\footnotesize\n"
        "\\item \\textit{Notes:} " + text + "\n"
        "\\end{tablenotes}\n"
    )


PREAMBLE = (
    "\\documentclass{article}\n"
    "\\usepackage{booktabs,threeparttable,tabularx}\n"
    "\\begin{document}\n\n"
)
POSTAMBLE = "\n\\end{document}\n"

BASE_NOTES = (
    "Driscoll--Kraay standard errors (Bartlett kernel, bandwidth~24) in parentheses. "
    "Calendar controls: hour-of-day, day-of-week, and month fixed effects. "
    "$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
)

FE_2COL = "Zone FE & Yes & Yes \\\\\nCalendar Controls & Yes & Yes \\\\\n"

tex_r7 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Zone Heterogeneity: Zone-Specific Effects of Renewable and Load Forecast Errors}\n"
    "\\label{tab:r7_h3_loadzone}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xcc}\n"
    "\\toprule\n"
    " & (1) & (2) \\\\\n"
    "\\cmidrule(r){2-2}\\cmidrule(l){3-3}\n"
    " & $Y_1$ & $Y_2$ \\\\\n"
    "\\midrule\n"
)

for zone in ZONES:
    r6 = extract(res6, f"eV_{zone}")
    r7 = extract(res7, f"eV_{zone}")
    tex_r7 += reg_row(f"Renewable error $\\times$ {zone}", [r6, r7])

for zone in ZONES:
    r6 = extract(res6, f"eL_{zone}")
    r7 = extract(res7, f"eL_{zone}")
    last = (zone == ZONES[-1])
    tex_r7 += reg_row(f"Load error $\\times$ {zone}", [r6, r7], last=last)

tex_r7 += (
    "\\midrule\n"
    f"Within $R^2$ & {res6.rsquared_within:.3f} & {res7.rsquared_within:.3f} \\\\\n"
    f"Observations & {n6:,} & {n7:,} \\\\\n"
    + FE_2COL
    + f"Wald $\\chi^2(3)$, renewable error & {fmt_wald(W6_V, pW6_V)} & {fmt_wald(W7_V, pW7_V)} \\\\\n"
    + f"Wald $\\chi^2(3)$, load error & {fmt_wald(W6_L, pW6_L)} & {fmt_wald(W7_L, pW7_L)} \\\\\n"
    + "\\bottomrule\n"
    "\\end{tabularx}\n"
    + tablenotes(
        "Zone-specific interactions $\\varepsilon_V \\times \\mathrm{SE}z$ and "
        "$\\varepsilon_L \\times \\mathrm{SE}z$, $z \\in \\{1,2,3,4\\}$; no pooled "
        "$\\varepsilon_V$ or $\\varepsilon_L$ term. "
        + BASE_NOTES + " "
        "Wald tests: $H_0\\colon \\beta_{\\mathrm{SE1}}=\\cdots=\\beta_{\\mathrm{SE4}}$ "
        "for each regressor separately, $\\chi^2(3)$."
    )
    + "\\end{threeparttable}\n"
    "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table_r7_h3_loadzone.tex").write_text(tex_r7)
print("Wrote table_r7_h3_loadzone.tex")
