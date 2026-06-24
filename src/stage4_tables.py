"""
Stage 4 — LaTeX tables and master results file.

Outputs (all in results/):
  table_descriptives.tex
  table_h1.tex
  table_h2.tex
  table_h3.tex
  results_master.txt   (human-readable concatenation)
  results_master.json  (machine-readable: all coefs, SE, t, p, R², N, Wald)
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
    """Return (coef_str, se_str) with significance stars on coef."""
    return f"{b:.4f}{stars(p)}", f"({se:.4f})"


def extract(res, var):
    return {
        "coef": float(res.params[var]),
        "se":   float(res.std_errors[var]),
        "t":    float(res.tstats[var]),
        "p":    float(res.pvalues[var]),
    }


# ---------------------------------------------------------------------------
# Table 1 — Descriptive statistics by zone
# ---------------------------------------------------------------------------

VARS = ["Y1", "Y2", "e_V", "e_V_pre", "e_V_post", "e_L"]
VAR_LABELS = {
    "Y1":       r"$Y_1 = P^{ID} - P^{DA}$",
    "Y2":       r"$Y_2 = P^{B}  - P^{ID}$",
    "e_V":      r"$\varepsilon_V$ (total)",
    "e_V_pre":  r"$\varepsilon_V^{pre}$",
    "e_V_post": r"$\varepsilon_V^{post}$",
    "e_L":      r"$\varepsilon_L$",
}

rows_desc = []
for zone in ZONES:
    sub = df_panel.xs(zone, level="zone")[VARS]
    for var in VARS:
        s = sub[var].dropna()
        rows_desc.append({
            "Zone": zone,
            "Variable": VAR_LABELS[var],
            "Mean": s.mean(),
            "SD":   s.std(),
            "Min":  s.min(),
            "Max":  s.max(),
            "N":    len(s),
        })

desc_df = pd.DataFrame(rows_desc)

tex_desc = r"""\begin{table}[htbp]
\centering
\caption{Descriptive Statistics by Bidding Zone}
\label{tab:descriptives}
\begin{tabular}{llrrrr}
\toprule
Zone & Variable & Mean & SD & Min & Max \\
\midrule
"""

current_zone = None
for _, r in desc_df.iterrows():
    if r["Zone"] != current_zone:
        if current_zone is not None:
            tex_desc += r"\addlinespace" + "\n"
        current_zone = r["Zone"]
        zone_label = r["Zone"]
    else:
        zone_label = ""
    tex_desc += (
        f"{zone_label} & {r['Variable']} & "
        f"{r['Mean']:>8.2f} & {r['SD']:>8.2f} & "
        f"{r['Min']:>10.2f} & {r['Max']:>10.2f} \\\\\n"
    )

tex_desc += r"""\bottomrule
\multicolumn{6}{l}{\footnotesize All price variables in EUR/MWh; generation variables in MW.} \\
\multicolumn{6}{l}{\footnotesize Sample: 2021-12-01 -- 2025-03-17, hourly, SE1--SE4.} \\
\end{tabular}
\end{table}
"""

(RESULTS_DIR / "table_descriptives.tex").write_text(tex_desc)
print("Wrote table_descriptives.tex")


# ---------------------------------------------------------------------------
# Table 2 — H1 Baselines (Y1 and Y2 side by side)
# ---------------------------------------------------------------------------

def h1_row(var, label, r1, r2):
    b1, s1 = fmt_coef(r1["coef"], r1["se"], r1["p"])
    b2, s2 = fmt_coef(r2["coef"], r2["se"], r2["p"])
    return (
        f"{label} & {b1} & {b2} \\\\\n"
        f"         & {s1} & {s2} \\\\\n"
    )

e_V_y1 = extract(res_y1, "e_V");  e_L_y1 = extract(res_y1, "e_L")
e_V_y2 = extract(res_y2, "e_V");  e_L_y2 = extract(res_y2, "e_L")

tex_h1 = r"""\begin{table}[htbp]
\centering
\caption{H1 Baseline Regressions: Effect of Renewable Forecast Errors on Price Spreads}
\label{tab:h1}
\begin{tabular}{lcc}
\toprule
 & $Y_1 = P^{ID} - P^{DA}$ & $Y_2 = P^{B} - P^{ID}$ \\
\midrule
"""
tex_h1 += h1_row("e_V", r"$\varepsilon_V$", e_V_y1, e_V_y2)
tex_h1 += r"\addlinespace" + "\n"
tex_h1 += h1_row("e_L", r"$\varepsilon_L$", e_L_y1, e_L_y2)
tex_h1 += r"""\midrule
Within $R^2$ & """ + f"{res_y1.rsquared_within:.4f} & {res_y2.rsquared_within:.4f}" + r""" \\
$N$          & """ + f"{n_y1:,} & {n_y2:,}" + r""" \\
\bottomrule
\multicolumn{3}{l}{\footnotesize Zone fixed effects and calendar dummies (hour, day-of-week, month) included.} \\
\multicolumn{3}{l}{\footnotesize Driscoll--Kraay standard errors (Bartlett kernel, bandwidth~24) in parentheses.} \\
\multicolumn{3}{l}{\footnotesize $^{***}p<0.01$,\ $^{**}p<0.05$,\ $^{*}p<0.10$.} \\
\end{tabular}
\end{table}
"""

(RESULTS_DIR / "table_h1.tex").write_text(tex_h1)
print("Wrote table_h1.tex")


# ---------------------------------------------------------------------------
# Table 3 — H2 Leakage
# ---------------------------------------------------------------------------

pre  = extract(res3, "e_V_pre")
post = extract(res3, "e_V_post")
eL3  = extract(res3, "e_L")

def h2_row(label, r):
    b, s = fmt_coef(r["coef"], r["se"], r["p"])
    return f"{label} & {b} \\\\\n         & {s} \\\\\n"

tex_h2 = r"""\begin{table}[htbp]
\centering
\caption{H2 Leakage Regression: Pre- vs.\ Post-Gate Forecast Errors on $Y_2$}
\label{tab:h2}
\begin{tabular}{lc}
\toprule
 & $Y_2 = P^{B} - P^{ID}$ \\
\midrule
"""
tex_h2 += h2_row(r"$\varepsilon_V^{pre}$",  pre)
tex_h2 += r"\addlinespace" + "\n"
tex_h2 += h2_row(r"$\varepsilon_V^{post}$", post)
tex_h2 += r"\addlinespace" + "\n"
tex_h2 += h2_row(r"$\varepsilon_L$",        eL3)
tex_h2 += r"""\midrule
Within $R^2$ & """ + f"{res3.rsquared_within:.4f}" + r""" \\
$N$          & """ + f"{n3:,}" + r""" \\
\bottomrule
\multicolumn{2}{l}{\footnotesize Zone fixed effects and calendar dummies included.} \\
\multicolumn{2}{l}{\footnotesize Driscoll--Kraay standard errors (Bartlett kernel, bandwidth~24) in parentheses.} \\
\multicolumn{2}{l}{\footnotesize $^{***}p<0.01$,\ $^{**}p<0.05$,\ $^{*}p<0.10$.} \\
\end{tabular}
\end{table}
"""

(RESULTS_DIR / "table_h2.tex").write_text(tex_h2)
print("Wrote table_h2.tex")


# ---------------------------------------------------------------------------
# Table 4 — H3 Zone Heterogeneity (Y1 and Y2 side by side)
# ---------------------------------------------------------------------------

def h3_row(zone, r4, r5):
    b4, s4 = fmt_coef(r4["coef"], r4["se"], r4["p"])
    b5, s5 = fmt_coef(r5["coef"], r5["se"], r5["p"])
    return (
        f"$\\varepsilon_V \\times$ {zone} & {b4} & {b5} \\\\\n"
        f"                         & {s4} & {s5} \\\\\n"
    )

eL4 = extract(res4, "e_L")
eL5 = extract(res5, "e_L")

tex_h3 = r"""\begin{table}[htbp]
\centering
\caption{H3 Zone Heterogeneity: Zone-Specific Effects of Renewable Forecast Errors}
\label{tab:h3}
\begin{tabular}{lcc}
\toprule
 & $Y_1 = P^{ID} - P^{DA}$ & $Y_2 = P^{B} - P^{ID}$ \\
\midrule
"""
for zone in ZONES:
    r4 = extract(res4, f"eV_{zone}")
    r5 = extract(res5, f"eV_{zone}")
    tex_h3 += h3_row(zone, r4, r5)
    tex_h3 += r"\addlinespace" + "\n"

b4, s4 = fmt_coef(eL4["coef"], eL4["se"], eL4["p"])
b5, s5 = fmt_coef(eL5["coef"], eL5["se"], eL5["p"])
tex_h3 += f"$\\varepsilon_L$ & {b4} & {b5} \\\\\n"
tex_h3 += f"               & {s4} & {s5} \\\\\n"

# Wald p-value formatting
def wald_str(W, p):
    st = stars(p)
    return f"$W={W:.3f}$, $p={p:.4f}${st}"

tex_h3 += r"""\midrule
Within $R^2$            & """ + f"{res4.rsquared_within:.4f} & {res5.rsquared_within:.4f}" + r""" \\
$N$                     & """ + f"{n4:,} & {n5:,}" + r""" \\
Wald test (equality)    & """ + wald_str(W4, pW4) + " & " + wald_str(W5, pW5) + r""" \\
\bottomrule
\multicolumn{3}{l}{\footnotesize Zone fixed effects and calendar dummies included. No standalone $\varepsilon_V$ term} \\
\multicolumn{3}{l}{\footnotesize (zone interactions are exhaustive). Driscoll--Kraay SE (Bartlett, bw~24) in parentheses.} \\
\multicolumn{3}{l}{\footnotesize Wald test: $H_0\colon \beta_{SE1}=\beta_{SE2}=\beta_{SE3}=\beta_{SE4}$, $\chi^2(3)$.} \\
\multicolumn{3}{l}{\footnotesize $^{***}p<0.01$,\ $^{**}p<0.05$,\ $^{*}p<0.10$.} \\
\end{tabular}
\end{table}
"""

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
# Quick sanity print of JSON
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
