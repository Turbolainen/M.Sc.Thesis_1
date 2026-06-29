"""
Stage 4 — LaTeX tables and master results file.

Outputs (all in results/):
  table_descriptives.tex
  table_h1.tex
  table_h2.tex
  table_h3.tex
  results_master.txt   (human-readable concatenation)
  results_master.json  (machine-readable: all coefs, SE, t, p, R², N, Wald)

Preamble requirements: \\usepackage{booktabs,threeparttable}
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load all stage results
# ---------------------------------------------------------------------------
with open("data/processed/stage2_results.pkl", "rb") as f:
    s2 = pickle.load(f)
with open("data/processed/stage3_results.pkl", "rb") as f:
    s3 = pickle.load(f)

df_panel = pd.read_pickle("data/processed/panel_enriched.pkl")

res_y1  = s2["res_y1"];  n_y1  = s2["n_y1"]
res_y2  = s2["res_y2"];  n_y2  = s2["n_y2"]
res3    = s3["res3"];    n3    = s3["n3"]
res4    = s3["res4"];    n4    = s3["n4"];  W4 = s3["W4"];  pW4 = s3["pW4"]
res5    = s3["res5"];    n5    = s3["n5"];  W5 = s3["W5"];  pW5 = s3["pW5"]

ZONES = ["SE1", "SE2", "SE3", "SE4"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def fmt_coef(b, se, p):
    """Coefficient with superscript stars; \\phantom{-} aligns positive values."""
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
    """One variable block: coefficient row + SE row.

    results: list of extract() dicts, one per data column.
    last:    omit inter-variable spacing after SE row (use before \\midrule).
    """
    pairs     = [fmt_coef(r["coef"], r["se"], r["p"]) for r in results]
    coef_line = " & ".join(b for b, _ in pairs)
    se_line   = " & ".join(s for _, s in pairs)
    end       = "\\\\\n" if last else "\\\\[0.4ex]\n"
    return f"{label} & {coef_line} \\\\\n & {se_line} {end}"


def tablenotes(text):
    return (
        "\\begin{tablenotes}[flushleft]\n"
        "\\footnotesize\n"
        "\\item \\textit{Notes:} " + text + "\n"
        "\\end{tablenotes}\n"
    )


BASE_NOTES = (
    "Driscoll--Kraay standard errors (Bartlett kernel, bandwidth~24) in parentheses. "
    "Calendar controls: hour-of-day, day-of-week, and month fixed effects. "
    "$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
)

FE_2COL = "Zone FE & Yes & Yes \\\\\nCalendar Controls & Yes & Yes \\\\\n"
FE_1COL = "Zone FE & Yes \\\\\nCalendar Controls & Yes \\\\\n"


# ---------------------------------------------------------------------------
# Table 1 — Descriptive statistics by zone
# ---------------------------------------------------------------------------

VARS = ["Y1", "Y2", "e_V", "e_V_pre", "e_V_post", "e_L"]
VAR_LABELS = {
    "Y1":       r"$Y_1$",
    "Y2":       r"$Y_2$",
    "e_V":      r"$\varepsilon_V$",
    "e_V_pre":  r"$\varepsilon_V^{\mathrm{pre}}$",
    "e_V_post": r"$\varepsilon_V^{\mathrm{post}}$",
    "e_L":      r"$\varepsilon_L$",
}
ZONE_NAMES = {
    "SE1": "SE1 --- Northern Sweden",
    "SE2": "SE2 --- Central-Northern Sweden",
    "SE3": "SE3 --- Central-Southern Sweden",
    "SE4": "SE4 --- Southern Sweden",
}

rows_desc = []
for zone in ZONES:
    sub = df_panel.xs(zone, level="zone")[VARS]
    for var in VARS:
        s = sub[var].dropna()
        rows_desc.append({
            "Zone":     zone,
            "Variable": VAR_LABELS[var],
            "Mean":     s.mean(),
            "SD":       s.std(),
            "Min":      s.min(),
            "Max":      s.max(),
            "N":        len(s),
        })

desc_df = pd.DataFrame(rows_desc)

PREAMBLE = (
    "\\documentclass{article}\n"
    "\\usepackage{booktabs,threeparttable,tabularx}\n"
    "\\begin{document}\n\n"
)
POSTAMBLE = "\n\\end{document}\n"

tex_desc = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Descriptive Statistics by Bidding Zone}\n"
    "\\label{tab:descriptives}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xrrrrr}\n"
    "\\toprule\n"
    "Variable & Mean & SD & Min & Max & $N$ \\\\\n"
    "\\midrule\n"
)

current_zone = None
for _, r in desc_df.iterrows():
    if r["Zone"] != current_zone:
        if current_zone is not None:
            tex_desc += "\\addlinespace\n"
        current_zone = r["Zone"]
        tex_desc += (
            f"\\multicolumn{{6}}{{l}}{{\\textit{{{ZONE_NAMES[current_zone]}}}}}"
            " \\\\\n"
            "\\addlinespace[0.3ex]\n"
        )
    tex_desc += (
        f"{r['Variable']} & "
        f"{r['Mean']:8.2f} & {r['SD']:8.2f} & "
        f"{r['Min']:10.2f} & {r['Max']:10.2f} & "
        f"{int(r['N']):,} \\\\\n"
    )

tex_desc += (
    "\\bottomrule\n"
    "\\end{tabularx}\n"
    + tablenotes(
        "All price variables in EUR/MWh; forecast error variables in MW. "
        "Sample: 2021-12-01 to 2025-03-17, hourly observations, bidding zones SE1--SE4."
    )
    + "\\end{threeparttable}\n"
    "\\end{table}\n"
    + POSTAMBLE
)

(RESULTS_DIR / "table_descriptives.tex").write_text(tex_desc)
print("Wrote table_descriptives.tex")


# ---------------------------------------------------------------------------
# Table 2 — H1 Baselines (Y1 and Y2 side by side)
# ---------------------------------------------------------------------------

e_V_y1 = extract(res_y1, "e_V");  e_L_y1 = extract(res_y1, "e_L")
e_V_y2 = extract(res_y2, "e_V");  e_L_y2 = extract(res_y2, "e_L")

tex_h1 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Baseline Effect of Renewable Forecast Errors on Electricity Price Spreads}\n"
    "\\label{tab:h1}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xcc}\n"
    "\\toprule\n"
    " & (1) & (2) \\\\\n"
    "\\cmidrule(r){2-2}\\cmidrule(l){3-3}\n"
    " & $Y_1$ & $Y_2$ \\\\\n"
    "\\midrule\n"
)
tex_h1 += reg_row("Renewable forecast error ($\\varepsilon_V$)", [e_V_y1, e_V_y2])
tex_h1 += reg_row("Load forecast error ($\\varepsilon_L$)", [e_L_y1, e_L_y2], last=True)
tex_h1 += (
    "\\midrule\n"
    f"Within $R^2$ & {res_y1.rsquared_within:.3f} & {res_y2.rsquared_within:.3f} \\\\\n"
    f"Observations & {n_y1:,} & {n_y2:,} \\\\\n"
    + FE_2COL
    + "\\bottomrule\n"
    "\\end{tabularx}\n"
    + tablenotes(
        "$\\varepsilon_V$: total renewable (wind and solar) forecast error (MW). "
        "$\\varepsilon_L$: load forecast error (MW). "
        + BASE_NOTES
    )
    + "\\end{threeparttable}\n"
    "\\end{table}\n"
    + POSTAMBLE
)

(RESULTS_DIR / "table_h1.tex").write_text(tex_h1)
print("Wrote table_h1.tex")


# ---------------------------------------------------------------------------
# Table 3 — H2 Leakage
# ---------------------------------------------------------------------------

pre  = extract(res3, "e_V_pre")
post = extract(res3, "e_V_post")
eL3  = extract(res3, "e_L")

tex_h2 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Information Leakage: Pre- vs.\\ Post-Gate Renewable Forecast Errors}\n"
    "\\label{tab:h2}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xc}\n"
    "\\toprule\n"
    " & $Y_2$ \\\\\n"
    "\\midrule\n"
)
tex_h2 += reg_row("Pre-gate renewable error ($\\varepsilon_V^{\\mathrm{pre}}$)",  [pre])
tex_h2 += reg_row("Post-gate renewable error ($\\varepsilon_V^{\\mathrm{post}}$)", [post])
tex_h2 += reg_row("Load forecast error ($\\varepsilon_L$)", [eL3], last=True)
tex_h2 += (
    "\\midrule\n"
    f"Within $R^2$ & {res3.rsquared_within:.3f} \\\\\n"
    f"Observations & {n3:,} \\\\\n"
    + FE_1COL
    + "\\bottomrule\n"
    "\\end{tabularx}\n"
    + tablenotes(
        "$\\varepsilon_V^{\\mathrm{pre}}$ ($\\varepsilon_V^{\\mathrm{post}}$): "
        "renewable forecast error before (after) intraday gate closure (MW). "
        "$\\varepsilon_L$: load forecast error (MW). "
        + BASE_NOTES
    )
    + "\\end{threeparttable}\n"
    "\\end{table}\n"
    + POSTAMBLE
)

(RESULTS_DIR / "table_h2.tex").write_text(tex_h2)
print("Wrote table_h2.tex")


# ---------------------------------------------------------------------------
# Table 4 — H3 Zone Heterogeneity (Y1 and Y2 side by side)
# ---------------------------------------------------------------------------

eL4 = extract(res4, "e_L")
eL5 = extract(res5, "e_L")

tex_h3 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Zone Heterogeneity: Zone-Specific Effects of Renewable Forecast Errors}\n"
    "\\label{tab:h3}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xcc}\n"
    "\\toprule\n"
    " & (1) & (2) \\\\\n"
    "\\cmidrule(r){2-2}\\cmidrule(l){3-3}\n"
    " & $Y_1$ & $Y_2$ \\\\\n"
    "\\midrule\n"
)

for zone in ZONES:
    r4 = extract(res4, f"eV_{zone}")
    r5 = extract(res5, f"eV_{zone}")
    tex_h3 += reg_row(f"Renewable error $\\times$ {zone}", [r4, r5])

tex_h3 += reg_row("Load forecast error ($\\varepsilon_L$)", [eL4, eL5], last=True)

def fmt_wald(W, p):
    st = stars(p)
    return f"${W:.3f}^{{{st}}}$" if st else f"${W:.3f}$"

tex_h3 += (
    "\\midrule\n"
    f"Within $R^2$ & {res4.rsquared_within:.3f} & {res5.rsquared_within:.3f} \\\\\n"
    f"Observations & {n4:,} & {n5:,} \\\\\n"
    + FE_2COL
    + f"Wald $\\chi^2(3)$ & {fmt_wald(W4, pW4)} & {fmt_wald(W5, pW5)} \\\\\n"
    + "\\bottomrule\n"
    "\\end{tabularx}\n"
    + tablenotes(
        "Zone-specific interactions $\\varepsilon_V \\times \\mathrm{SE}z$, "
        "$z \\in \\{1,2,3,4\\}$; no pooled $\\varepsilon_V$ term. "
        "$\\varepsilon_L$: load forecast error (MW). "
        + BASE_NOTES + " "
        "Wald test: $H_0\\colon \\beta_{\\mathrm{SE1}}=\\cdots=\\beta_{\\mathrm{SE4}}$, "
        "$\\chi^2(3)$."
    )
    + "\\end{threeparttable}\n"
    "\\end{table}\n"
    + POSTAMBLE
)

(RESULTS_DIR / "table_h3.tex").write_text(tex_h3)
print("Wrote table_h3.tex")


# ---------------------------------------------------------------------------
# Master plain-text file
# ---------------------------------------------------------------------------

master_txt = "\n".join([
    "=" * 70,
    "MASTER RESULTS FILE — Sequential Price Formation, SE Electricity Markets",
    "=" * 70,
    "",
    (RESULTS_DIR / "table_descriptives.tex").read_text(),
    "",
    (RESULTS_DIR / "table_h1.tex").read_text(),
    "",
    (RESULTS_DIR / "table_h2.tex").read_text(),
    "",
    (RESULTS_DIR / "table_h3.tex").read_text(),
])
(RESULTS_DIR / "results_master.txt").write_text(master_txt)
print("Wrote results_master.txt")


# ---------------------------------------------------------------------------
# Master JSON — every number
# ---------------------------------------------------------------------------

def res_to_dict(res, vars_of_interest, n, r2):
    out = {"n": n, "within_r2": r2, "coefficients": {}}
    for v in vars_of_interest:
        out["coefficients"][v] = {
            "coef": float(res.params[v]),
            "se":   float(res.std_errors[v]),
            "t":    float(res.tstats[v]),
            "p":    float(res.pvalues[v]),
        }
    return out

master_json = {
    "h1_y1": res_to_dict(res_y1, ["e_V", "e_L"], n_y1, float(res_y1.rsquared_within)),
    "h1_y2": res_to_dict(res_y2, ["e_V", "e_L"], n_y2, float(res_y2.rsquared_within)),
    "h2_y2": res_to_dict(res3, ["e_V_pre", "e_V_post", "e_L"], n3, float(res3.rsquared_within)),
    "h3_y1": {
        **res_to_dict(res4, [f"eV_{z}" for z in ZONES] + ["e_L"], n4, float(res4.rsquared_within)),
        "wald": {"stat": W4, "p": pW4, "df": 3},
    },
    "h3_y2": {
        **res_to_dict(res5, [f"eV_{z}" for z in ZONES] + ["e_L"], n5, float(res5.rsquared_within)),
        "wald": {"stat": W5, "p": pW5, "df": 3},
    },
}

(RESULTS_DIR / "results_master.json").write_text(
    json.dumps(master_json, indent=2)
)
print("Wrote results_master.json")

# ---------------------------------------------------------------------------
# Quick sanity print
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("JSON CONTENTS SUMMARY")
print("=" * 70)
for spec, vals in master_json.items():
    print(f"\n[{spec}]  N={vals['n']:,}  within-R²={vals['within_r2']:.4f}")
    for var, est in vals["coefficients"].items():
        st = stars(est["p"])
        print(f"  {var:<15}  {est['coef']:>9.4f}  ({est['se']:.4f})  t={est['t']:>7.3f}  p={est['p']:.4f}  {st}")
    if "wald" in vals:
        w = vals["wald"]
        print(f"  Wald χ²({w['df']})={w['stat']:.3f}  p={w['p']:.4f}  {stars(w['p'])}")
