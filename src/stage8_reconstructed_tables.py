"""
Stage 8 — Table 4 and Table 5: full timing-correct specification.

Table 4 extends Table 3 (Y2 ~ e_V_post + e_L) by adding back the pre-gate
component, decomposed rather than pooled:

  Table 4: Y2 ~ e_V_pre + e_V_post + e_L   (zone FE, calendar dummies, DK-SE)

Table 5 is the zone-heterogeneity counterpart of Table 2 and Table 3,
Y1 and Y2 reported side by side in one table — each spread's zone-specific
response to the forecast-error component actually available when that
spread is realized, together with the zone-specific load-error response:

  Column (1): Y1 ~ zone×e_V_pre  + zone×e_L   (+ Wald tests, both regressors)
  Column (2): Y2 ~ zone×e_V_post + zone×e_L   (same)
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


print("=" * 65)
print("STAGE 8 — TABLE 4, TABLE 5, TABLE 6")
print("=" * 65)

# ---------------------------------------------------------------------------
# TABLE 4: Y2 ~ e_V_pre + e_V_post + e_L
# ---------------------------------------------------------------------------

cols4 = ["Y2", "e_V_pre", "e_V_post", "e_L", "hour", "dayofweek", "month"]
sub4  = df[cols4].dropna()
n4    = len(sub4)
print(f"\nTable 4  Y2 ~ e_V_pre + e_V_post + e_L   N={n4:,}  (dropped {len(df)-n4:,})")

cal4  = make_calendar_dummies(sub4)
exog4 = pd.concat([sub4[["e_V_pre", "e_V_post", "e_L"]], cal4], axis=1)
exog4.insert(0, "const", 1.0)
res4  = fit_panel(sub4[["Y2"]], exog4)
show_coefs(res4, "Y2 ~ e_V_pre + e_V_post + e_L", n4, ["e_V_pre", "e_V_post", "e_L"])

# ---------------------------------------------------------------------------
# TABLE 5: Y1 ~ zone×e_V_pre + zone×e_L
# ---------------------------------------------------------------------------

def run_zone_het(dep_var, var, label):
    cols = [dep_var, var, "e_L", "hour", "dayofweek", "month"]
    sub  = df[cols].dropna()
    n    = len(sub)
    print(f"\nTable  {label}   N={n:,}  (dropped {len(df)-n:,})")

    entity = sub.index.get_level_values("zone")
    inter_cols = {}
    for z in ZONES:
        mask = (entity == z).astype(float)
        inter_cols[f"eV_{z}"] = sub[var] * mask
        inter_cols[f"eL_{z}"] = sub["e_L"] * mask

    inter_df = pd.DataFrame(inter_cols, index=sub.index)
    cal      = make_calendar_dummies(sub)
    exog     = pd.concat([inter_df, cal], axis=1)
    exog.insert(0, "const", 1.0)

    res  = fit_panel(sub[[dep_var]], exog)
    iv_V = [f"eV_{z}" for z in ZONES]
    iv_L = [f"eL_{z}" for z in ZONES]
    show_coefs(res, f"{dep_var} ~ zone×{var} + zone×e_L", n, iv_V + iv_L)

    W_V, pW_V, dfV = wald_equality(res, iv_V)
    W_L, pW_L, dfL = wald_equality(res, iv_L)
    print(f"\n  Wald test H0: {var} equal across zones  →  W={W_V:.3f}  p={pW_V:.4f}  (df={dfV})")
    print(f"  Wald test H0: e_L equal across zones  →  W={W_L:.3f}  p={pW_L:.4f}  (df={dfL})")

    return res, n, W_V, pW_V, W_L, pW_L


res5, n5, W5_V, pW5_V, W5_L, pW5_L = run_zone_het("Y1", "e_V_pre",  "5 (Y1)")
res6, n6, W6_V, pW6_V, W6_L, pW6_L = run_zone_het("Y2", "e_V_post", "6 (Y2)")

import pickle
with open("data/processed/stage8_results.pkl", "wb") as f:
    pickle.dump({
        "res4": res4, "n4": n4,
        "res5": res5, "n5": n5, "W5_V": W5_V, "pW5_V": pW5_V, "W5_L": W5_L, "pW5_L": pW5_L,
        "res6": res6, "n6": n6, "W6_V": W6_V, "pW6_V": pW6_V, "W6_L": W6_L, "pW6_L": pW6_L,
    }, f)
print("\nSaved data/processed/stage8_results.pkl")


# ===========================================================================
# LaTeX tables
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


def reg_row(label, result, last=False):
    b, s = fmt_coef(result["coef"], result["se"], result["p"])
    end  = "\\\\\n" if last else "\\\\[0.4ex]\n"
    return f"{label} & {b} \\\\\n & {s} {end}"


def reg_row_2col(label, d1, d2, last=False):
    b1, s1 = fmt_coef(d1["coef"], d1["se"], d1["p"])
    b2, s2 = fmt_coef(d2["coef"], d2["se"], d2["p"])
    end = "\\\\\n" if last else "\\\\[0.4ex]\n"
    return (f"{label} & {b1} & {b2} \\\\\n"
            f" & {s1} & {s2} {end}")


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

FE_1COL = "Zone FE & Yes \\\\\nCalendar Controls & Yes \\\\\n"
FE_2COL = "Zone FE & Yes & Yes \\\\\nCalendar Controls & Yes & Yes \\\\\n"

# ---------------------------------------------------------------------------
# Table 4 tex
# ---------------------------------------------------------------------------

pre4  = extract(res4, "e_V_pre")
post4 = extract(res4, "e_V_post")
eL4   = extract(res4, "e_L")

tex4 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Intraday--Balancing Spread: Pre- and Post-Gate Renewable Forecast Errors}\n"
    "\\label{tab:4}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xc}\n"
    "\\toprule\n"
    " & $Y_2$ \\\\\n"
    "\\midrule\n"
)
tex4 += reg_row("Pre-gate renewable error ($\\varepsilon_V^{\\mathrm{pre}}$)",  pre4)
tex4 += reg_row("Post-gate renewable error ($\\varepsilon_V^{\\mathrm{post}}$)", post4)
tex4 += reg_row("Load forecast error ($\\varepsilon_L$)", eL4, last=True)
tex4 += (
    "\\midrule\n"
    f"Within $R^2$ & {res4.rsquared_within:.3f} \\\\\n"
    f"Observations & {n4:,} \\\\\n"
    + FE_1COL
    + "\\bottomrule\n"
    "\\end{tabularx}\n"
    + tablenotes(BASE_NOTES)
    + "\\end{threeparttable}\n"
    "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table4.tex").write_text(tex4)
print("Wrote table4.tex")

# ---------------------------------------------------------------------------
# Table 5 tex (Y1 and Y2 side by side: zone×e_V_pre / zone×e_V_post + zone×e_L)
# ---------------------------------------------------------------------------

tex5 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Zone Heterogeneity: Day-Ahead--Intraday and Intraday--Balancing Spreads}\n"
    "\\label{tab:5}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xcc}\n"
    "\\toprule\n"
    " & (1) & (2) \\\\\n"
    "\\cmidrule(r){2-2}\\cmidrule(l){3-3}\n"
    " & $Y_1$ & $Y_2$ \\\\\n"
    "\\midrule\n"
)
for zone in ZONES:
    tex5 += reg_row_2col(f"Renewable error $\\times$ {zone}",
                          extract(res5, f"eV_{zone}"), extract(res6, f"eV_{zone}"))
for i, zone in enumerate(ZONES):
    last = (i == len(ZONES) - 1)
    tex5 += reg_row_2col(f"Load error $\\times$ {zone}",
                          extract(res5, f"eL_{zone}"), extract(res6, f"eL_{zone}"), last=last)
tex5 += (
    "\\midrule\n"
    f"Within $R^2$ & {res5.rsquared_within:.3f} & {res6.rsquared_within:.3f} \\\\\n"
    f"Observations & {n5:,} & {n6:,} \\\\\n"
    + FE_2COL
    + f"Wald $\\chi^2(3)$, renewable error & {fmt_wald(W5_V, pW5_V)} & {fmt_wald(W6_V, pW6_V)} \\\\\n"
    + f"Wald $\\chi^2(3)$, load error & {fmt_wald(W5_L, pW5_L)} & {fmt_wald(W6_L, pW6_L)} \\\\\n"
    + "\\bottomrule\n"
    "\\end{tabularx}\n"
    + tablenotes(
        "Column (1) uses the pre-gate renewable error component "
        "($\\varepsilon_V^{\\mathrm{pre}}$); column (2) uses the post-gate "
        "component ($\\varepsilon_V^{\\mathrm{post}}$). "
        + BASE_NOTES
    )
    + "\\end{threeparttable}\n"
    "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table5.tex").write_text(tex5)
print("Wrote table5.tex")
