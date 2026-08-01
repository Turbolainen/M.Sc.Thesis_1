"""
Stage 6 — Table 2 & Table 3: timing-corrected baseline specifications.

The forecast error for each spread must be restricted to the information
available at the time that spread is realized:

  Table 2: Y1 ~ e_V_pre  + e_L   (DA -> ID revision only; zone FE, calendar
                                   dummies, DK-SE)
  Table 3: Y2 ~ e_V_post + e_L   (ID -> actual revision only; same)

e_L is left unchanged in both (load has no separate ID-forecast series,
so it cannot be split into pre/post components).
"""

from pathlib import Path

import pandas as pd
from linearmodels import PanelOLS

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_pickle("data/processed/panel_enriched.pkl")


def make_calendar_dummies(data):
    parts = []
    for col, prefix in [("hour", "hr"), ("dayofweek", "dow"), ("month", "mon")]:
        d = pd.get_dummies(data[col], prefix=prefix, drop_first=True, dtype=float)
        parts.append(d)
    return pd.concat(parts, axis=1)


def run_reg(dep_var: str, var: str, label: str):
    cols = [dep_var, var, "e_L", "hour", "dayofweek", "month"]
    sub = df[cols].dropna()
    print(f"\n  {label}: N = {len(sub):,}  (dropped {len(df) - len(sub):,} NaN rows)")

    cal = make_calendar_dummies(sub)
    exog = pd.concat([sub[[var, "e_L"]], cal], axis=1)
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
print("STAGE 6 — TABLE 2 & TABLE 3: TIMING-CORRECTED SPECIFICATIONS")
print("=" * 60)

res_y1c, n_y1c = run_reg("Y1", "e_V_pre",  "Y1 ~ e_V_pre + e_L")
res_y2c, n_y2c = run_reg("Y2", "e_V_post", "Y2 ~ e_V_post + e_L")


def show_coefs(res, label, n, var):
    params = res.params
    se     = res.std_errors
    tstat  = res.tstats
    pval   = res.pvalues
    r2     = res.rsquared_within

    print(f"\n{'='*60}")
    print(f"  {label}   N={n:,}   Within-R^2={r2:.4f}")
    print(f"{'='*60}")
    print(f"  {'Variable':<15} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
    print(f"  {'-'*53}")
    for v in [var, "e_L"]:
        stars = "***" if pval[v] < 0.01 else "**" if pval[v] < 0.05 else "*" if pval[v] < 0.10 else ""
        print(f"  {v:<15} {params[v]:>10.4f} {se[v]:>10.4f} {tstat[v]:>8.3f} {pval[v]:>8.4f}  {stars}")


show_coefs(res_y1c, "Y1 ~ e_V_pre + e_L",  n_y1c, "e_V_pre")
show_coefs(res_y2c, "Y2 ~ e_V_post + e_L", n_y2c, "e_V_post")

import pickle
with open("data/processed/stage6_results.pkl", "wb") as f:
    pickle.dump({"res_y1c": res_y1c, "n_y1c": n_y1c,
                 "res_y2c": res_y2c, "n_y2c": n_y2c}, f)
print("\nSaved data/processed/stage6_results.pkl")


# ===========================================================================
# LaTeX tables: Table 2, Table 3
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

e_V_pre_y1c = extract(res_y1c, "e_V_pre")
e_L_y1c     = extract(res_y1c, "e_L")

tex2 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Day-Ahead--Intraday Spread: Pre-Gate Renewable Forecast Error}\n"
    "\\label{tab:2}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xc}\n"
    "\\toprule\n"
    " & $Y_1$ \\\\\n"
    "\\midrule\n"
)
tex2 += reg_row("Pre-gate renewable error ($\\varepsilon_V^{\\mathrm{pre}}$)", e_V_pre_y1c)
tex2 += reg_row("Load forecast error ($\\varepsilon_L$)", e_L_y1c, last=True)
tex2 += (
    "\\midrule\n"
    f"Within $R^2$ & {res_y1c.rsquared_within:.3f} \\\\\n"
    f"Observations & {n_y1c:,} \\\\\n"
    + FE_1COL
    + "\\bottomrule\n"
    "\\end{tabularx}\n"
    + tablenotes(
        "$\\varepsilon_V^{\\mathrm{pre}}$: renewable forecast error before intraday "
        "gate closure (MW). $\\varepsilon_L$: load forecast error (MW). "
        + BASE_NOTES
    )
    + "\\end{threeparttable}\n"
    "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table2.tex").write_text(tex2)
print("Wrote table2.tex")

e_V_post_y2c = extract(res_y2c, "e_V_post")
e_L_y2c      = extract(res_y2c, "e_L")

tex3 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Intraday--Balancing Spread: Post-Gate Renewable Forecast Error}\n"
    "\\label{tab:3}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xc}\n"
    "\\toprule\n"
    " & $Y_2$ \\\\\n"
    "\\midrule\n"
)
tex3 += reg_row("Post-gate renewable error ($\\varepsilon_V^{\\mathrm{post}}$)", e_V_post_y2c)
tex3 += reg_row("Load forecast error ($\\varepsilon_L$)", e_L_y2c, last=True)
tex3 += (
    "\\midrule\n"
    f"Within $R^2$ & {res_y2c.rsquared_within:.3f} \\\\\n"
    f"Observations & {n_y2c:,} \\\\\n"
    + FE_1COL
    + "\\bottomrule\n"
    "\\end{tabularx}\n"
    + tablenotes(
        "$\\varepsilon_V^{\\mathrm{post}}$: renewable forecast error after intraday "
        "gate closure (MW). $\\varepsilon_L$: load forecast error (MW). "
        + BASE_NOTES
    )
    + "\\end{threeparttable}\n"
    "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table3.tex").write_text(tex3)
print("Wrote table3.tex")
