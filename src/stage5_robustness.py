"""
Stage 5 — Appendix robustness tables.

Table R1 : Crisis-period exclusion  (22 Sep 2021 – 3 Dec 2022;
           data begin 1 Dec 2021, so effective window is 1 Dec 2021 – 3 Dec 2022)
Table R2 : High cross-border flow exclusion
           (top decile of |net_export_mw| per zone; no threshold pre-defined in codebase)
Table R3 : Levin-Lin-Chu (2002) panel unit root test on Y1 and Y2
Table R4 : DK lag-length sensitivity — bandwidths 12, 24 (baseline), 48 hours

Flags (†): coefficient deviates > 20 % from Table 2 baseline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from scipy import stats

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = pd.read_pickle("data/processed/panel_enriched.pkl")
ZONES = ["SE1", "SE2", "SE3", "SE4"]

# ---------------------------------------------------------------------------
# Baseline coefficients for 20 % flag check (Table 2 / stage2 results)
# ---------------------------------------------------------------------------
BASELINE = {
    ("Y1", "e_V"): -0.0173,
    ("Y1", "e_L"):  0.0074,
    ("Y2", "e_V"): -0.0022,
    ("Y2", "e_L"):  0.0205,
}

def is_flagged(dep, var, coef):
    base = BASELINE.get((dep, var))
    if base is None:
        return False
    return abs(coef - base) / abs(base) > 0.20


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fmt_coef(b, se, p, dep=None, var=None):
    """Coefficient cell + SE cell; adds dagger if >20 % from baseline."""
    st  = stars(p)
    dag = "\\dagger" if (dep and var and is_flagged(dep, var, b)) else ""
    lp  = "" if b < 0 else "\\phantom{-}"
    sup = st + dag
    if sup:
        coef_str = f"${lp}{b:.4f}^{{{sup}}}$"
    else:
        coef_str = f"${lp}{b:.4f}$"
    return coef_str, f"$({se:.4f})$"


def make_cal(data):
    parts = []
    for col, pfx in [("hour", "hr"), ("dayofweek", "dow"), ("month", "mon")]:
        parts.append(
            pd.get_dummies(data[col], prefix=pfx, drop_first=True, dtype=float)
        )
    return pd.concat(parts, axis=1)


def fit_panel(dep, regs, data, bw=24):
    sub  = data[[dep] + regs + ["hour", "dayofweek", "month"]].dropna()
    cal  = make_cal(sub)
    exog = pd.concat([sub[regs], cal], axis=1)
    exog.insert(0, "const", 1.0)
    res  = PanelOLS(sub[[dep]], exog, entity_effects=True).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=bw
    )
    return res, len(sub)


def extract(res, var):
    return dict(coef=float(res.params[var]),
                se=float(res.std_errors[var]),
                p=float(res.pvalues[var]))


# ---------------------------------------------------------------------------
# LaTeX building blocks
# ---------------------------------------------------------------------------

PREAMBLE = (
    "\\documentclass{article}\n"
    "\\usepackage{booktabs,threeparttable,tabularx}\n"
    "\\begin{document}\n\n"
)
POSTAMBLE = "\n\\end{document}\n"

DK_NOTE = (
    "Driscoll--Kraay standard errors (Bartlett kernel, bandwidth {bw}~h) in parentheses. "
    "Calendar controls: hour-of-day, day-of-week, and month fixed effects. "
    "$^{{***}}p<0.01$, $^{{**}}p<0.05$, $^{{*}}p<0.10$. "
    "$^{{\\dagger}}$ coefficient deviates ${{>}}20\\%$ from Table~2 baseline."
)

def tablenotes(text):
    return (
        "\\begin{tablenotes}[flushleft]\n"
        "\\footnotesize\n"
        "\\item \\textit{Notes:} " + text + "\n"
        "\\end{tablenotes}\n"
    )


def reg_row_2col(label, d1, dep1, d2, dep2, var, last=False):
    b1, s1 = fmt_coef(d1["coef"], d1["se"], d1["p"], dep1, var)
    b2, s2 = fmt_coef(d2["coef"], d2["se"], d2["p"], dep2, var)
    end = "\\\\\n" if last else "\\\\[0.4ex]\n"
    return (f"{label} & {b1} & {b2} \\\\\n"
            f" & {s1} & {s2} {end}")


def reg_row_ncol(label, entries, last=False):
    """entries: list of (coef_str, se_str)."""
    coef_line = " & ".join(b for b, _ in entries)
    se_line   = " & ".join(s for _, s in entries)
    end = "\\\\\n" if last else "\\\\[0.4ex]\n"
    return f"{label} & {coef_line} \\\\\n & {se_line} {end}"


BASE_2COL_HEADER = (
    "\\toprule\n"
    " & (1) & (2) \\\\\n"
    "\\cmidrule(r){2-2}\\cmidrule(l){3-3}\n"
    " & $Y_1$ & $Y_2$ \\\\\n"
    "\\midrule\n"
)

FE_2COL = "Zone FE & Yes & Yes \\\\\nCalendar Controls & Yes & Yes \\\\\n"

REGRESSORS = ["e_V", "e_L"]


# ===========================================================================
# TABLE R1 — Crisis-period exclusion
# ===========================================================================
print("=" * 60)
print("TABLE R1: Crisis exclusion")

CRISIS_START = pd.Timestamp("2021-09-22", tz="Europe/Stockholm")
CRISIS_END   = pd.Timestamp("2022-12-03 23:59:59", tz="Europe/Stockholm")

ts      = df.index.get_level_values("timestamp_cet")
excl    = (ts >= CRISIS_START) & (ts <= CRISIS_END)
df_r1   = df[~excl]
n_excl  = excl.sum()
print(f"  Excluded {n_excl:,} obs  ({n_excl / len(df) * 100:.1f}% of full sample)")
print(f"  Effective window in data: {ts[excl].min()} → {ts[excl].max()}")

res_r1y1, n_r1y1 = fit_panel("Y1", REGRESSORS, df_r1)
res_r1y2, n_r1y2 = fit_panel("Y2", REGRESSORS, df_r1)

eV1 = extract(res_r1y1, "e_V");  eL1 = extract(res_r1y1, "e_L")
eV2 = extract(res_r1y2, "e_V");  eL2 = extract(res_r1y2, "e_L")

print(f"  Y1: e_V={eV1['coef']:.4f}  e_L={eL1['coef']:.4f}")
print(f"  Y2: e_V={eV2['coef']:.4f}  e_L={eL2['coef']:.4f}")

tex_r1 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
      "\\centering\n"
      "\\caption{Robustness: Crisis-Period Exclusion (Dec.~2021 -- Dec.~2022)}\n"
      "\\label{tab:r1_crisis}\n"
      "\\begin{threeparttable}\n"
      "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xcc}\n"
    + BASE_2COL_HEADER
    + reg_row_2col("Renewable forecast error ($\\varepsilon_V$)",
                   eV1, "Y1", eV2, "Y2", "e_V")
    + reg_row_2col("Load forecast error ($\\varepsilon_L$)",
                   eL1, "Y1", eL2, "Y2", "e_L", last=True)
    + (f"\\midrule\n"
       f"Within $R^2$ & {res_r1y1.rsquared_within:.3f} & {res_r1y2.rsquared_within:.3f} \\\\\n"
       f"Observations & {n_r1y1:,} & {n_r1y2:,} \\\\\n"
       + FE_2COL
       + "\\bottomrule\n"
         "\\end{tabularx}\n")
    + tablenotes(
        "Sample excludes the energy-crisis period 22~Sep.~2021--3~Dec.~2022 "
        "(data begin 1~Dec.~2021, so the effective exclusion window is "
        "1~Dec.~2021--3~Dec.~2022). "
        + DK_NOTE.format(bw=24)
    )
    + "\\end{threeparttable}\n"
      "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table_r1_crisis.tex").write_text(tex_r1)
print("  → table_r1_crisis.tex")


# ===========================================================================
# TABLE R2 — High cross-border flow exclusion
# ===========================================================================
print("\nTABLE R2: High-flow exclusion")

# No threshold pre-defined in codebase; use zone-specific 90th percentile of |net_export_mw|
abs_flow = df["net_export_mw"].abs()
q90      = abs_flow.groupby(level="zone").transform(lambda x: x.quantile(0.90))
hi_flow  = abs_flow >= q90
df_r2    = df[~hi_flow]

print("  Zone 90th-pct |net_export_mw| (MW):")
for z in ZONES:
    v = abs_flow.xs(z, level="zone").quantile(0.90)
    print(f"    {z}: {v:.1f} MW")
print(f"  Excluded {hi_flow.sum():,} obs  ({hi_flow.mean()*100:.1f}% of full sample)")

res_r2y1, n_r2y1 = fit_panel("Y1", REGRESSORS, df_r2)
res_r2y2, n_r2y2 = fit_panel("Y2", REGRESSORS, df_r2)

eV1r2 = extract(res_r2y1, "e_V");  eL1r2 = extract(res_r2y1, "e_L")
eV2r2 = extract(res_r2y2, "e_V");  eL2r2 = extract(res_r2y2, "e_L")

print(f"  Y1: e_V={eV1r2['coef']:.4f}  e_L={eL1r2['coef']:.4f}")
print(f"  Y2: e_V={eV2r2['coef']:.4f}  e_L={eL2r2['coef']:.4f}")

tex_r2 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
      "\\centering\n"
      "\\caption{Robustness: Excluding High Cross-Border Flow Hours}\n"
      "\\label{tab:r2_flow}\n"
      "\\begin{threeparttable}\n"
      "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xcc}\n"
    + BASE_2COL_HEADER
    + reg_row_2col("Renewable forecast error ($\\varepsilon_V$)",
                   eV1r2, "Y1", eV2r2, "Y2", "e_V")
    + reg_row_2col("Load forecast error ($\\varepsilon_L$)",
                   eL1r2, "Y1", eL2r2, "Y2", "e_L", last=True)
    + (f"\\midrule\n"
       f"Within $R^2$ & {res_r2y1.rsquared_within:.3f} & {res_r2y2.rsquared_within:.3f} \\\\\n"
       f"Observations & {n_r2y1:,} & {n_r2y2:,} \\\\\n"
       + FE_2COL
       + "\\bottomrule\n"
         "\\end{tabularx}\n")
    + tablenotes(
        "High cross-border flow hours excluded: zone-level observations where "
        "$|\\text{net export}|$ exceeds the zone-specific 90th percentile "
        "(no threshold is pre-defined in the codebase; the top-decile cut-off "
        "is applied separately per zone). "
        + DK_NOTE.format(bw=24)
    )
    + "\\end{threeparttable}\n"
      "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table_r2_flow.tex").write_text(tex_r2)
print("  → table_r2_flow.tex")


# ===========================================================================
# TABLE R3 — Levin-Lin-Chu (2002) panel unit root test
# ===========================================================================
print("\nTABLE R3: Levin-Lin-Chu panel unit root test")


def llc_test(df_panel, var):
    """
    Levin, Lin & Chu (2002) three-step pooled panel unit root test.

    H0: all panels have a unit root  (δ = 0)
    H1: all panels are stationary    (δ < 0, common AR coefficient)

    Step 1: For each zone i, AIC-select lag p_i over {0,...,min(24, T^{1/3})}.
            Run two auxiliary OLS regressions (excluding y_{i,t-1}):
              (a) Δy_it  on {1, Δy_{i,t-1}, ..., Δy_{i,t-p}} → residual ẽ_it
              (b) y_{i,t-1} on {1, Δy_{i,t-1}, ..., Δy_{i,t-p}} → residual f̃_it
            Normalise both by σ̂_i (std error of regression (a)).

    Step 2: Pool across zones; OLS of ẽ on f̃ → δ̂, t_δ.

    Step 3: LLC finite-sample adjustment.  With T̄ ≈ 7 000 the tabulated
            correction factors (LLC 2002, Table 2, model with individual means)
            approach their N(0,1) limit, so the unadjusted t_δ is referred
            directly to N(0,1).
    """
    zones = df_panel.index.get_level_values("zone").unique()
    N     = len(zones)
    all_e, all_f, T_eff, lags_chosen = [], [], [], []

    for zone in zones:
        y = (df_panel
             .xs(zone, level="zone")[var]
             .sort_index()
             .dropna())
        T = len(y)
        dy = y.diff().dropna()

        # --- AIC lag selection ---
        best_aic, best_p = np.inf, 0
        max_p = min(24, int(T ** (1 / 3)))

        for p_cand in range(0, max_p + 1):
            lag_d = {f"dl{j}": dy.shift(j) for j in range(1, p_cand + 1)}
            tmp   = pd.DataFrame({"dy": dy, "ylg": y.shift(1), **lag_d}).dropna()
            if len(tmp) < p_cand + 5:
                continue
            Xm = np.column_stack([np.ones(len(tmp)),
                                   tmp["ylg"].values,
                                   *[tmp[f"dl{j}"].values
                                     for j in range(1, p_cand + 1)]])
            ym  = tmp["dy"].values
            b   = np.linalg.lstsq(Xm, ym, rcond=None)[0]
            rr  = ym - Xm @ b
            aic = len(ym) * np.log(rr @ rr / len(ym)) + 2 * Xm.shape[1]
            if aic < best_aic:
                best_aic, best_p = aic, p_cand

        p = best_p
        lags_chosen.append(p)

        # --- Build auxiliary regression data ---
        lag_d = {f"dl{j}": dy.shift(j) for j in range(1, p + 1)}
        data  = pd.DataFrame({"dy": dy, "ylg": y.shift(1), **lag_d}).dropna()
        T_i   = len(data)
        T_eff.append(T_i)

        # Auxiliary regressor X: constant + p lag-diffs (no y_{t-1})
        if p > 0:
            Xaux = np.column_stack([np.ones(T_i),
                                    *[data[f"dl{j}"].values
                                      for j in range(1, p + 1)]])
        else:
            Xaux = np.ones((T_i, 1))

        dy_vec  = data["dy"].values
        ylg_vec = data["ylg"].values

        # (a) regress Δy on Xaux
        b_e   = np.linalg.lstsq(Xaux, dy_vec, rcond=None)[0]
        e_hat = dy_vec - Xaux @ b_e
        sigma = np.sqrt(e_hat @ e_hat / (T_i - Xaux.shape[1]))

        # (b) regress y_{t-1} on Xaux
        b_f   = np.linalg.lstsq(Xaux, ylg_vec, rcond=None)[0]
        f_hat = ylg_vec - Xaux @ b_f

        all_e.append(e_hat / sigma)
        all_f.append(f_hat / sigma)

    # --- Pooled OLS ---
    e_pool = np.concatenate(all_e)
    f_pool = np.concatenate(all_f)

    delta  = (f_pool @ e_pool) / (f_pool @ f_pool)
    resid  = e_pool - delta * f_pool
    s2     = (resid @ resid) / (len(resid) - N)
    t_stat = delta / np.sqrt(s2 / (f_pool @ f_pool))

    # Left-tailed N(0,1) p-value
    p_val  = stats.norm.cdf(t_stat)

    return {
        "stat":  t_stat,
        "pval":  p_val,
        "lags":  int(np.median(lags_chosen)),
        "T_bar": int(np.mean(T_eff)),
    }


llc_y1 = llc_test(df, "Y1")
llc_y2 = llc_test(df, "Y2")
print(f"  Y1: t={llc_y1['stat']:.3f}  p={llc_y1['pval']:.4f}  lags={llc_y1['lags']}  T_bar={llc_y1['T_bar']:,}")
print(f"  Y2: t={llc_y2['stat']:.3f}  p={llc_y2['pval']:.4f}  lags={llc_y2['lags']}  T_bar={llc_y2['T_bar']:,}")


def pval_cell(p):
    if p < 0.001: return "$<0.001$"
    return f"${p:.3f}$"


tex_r3 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
      "\\centering\n"
      "\\caption{Panel Unit Root Test: Levin--Lin--Chu (2002)}\n"
      "\\label{tab:r3_unitroot}\n"
      "\\begin{threeparttable}\n"
      "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xccc}\n"
      "\\toprule\n"
      "Variable & LLC statistic & $p$-value & Lags ($\\bar{p}$) \\\\\n"
      "\\midrule\n"
    + f"$Y_1$ & ${llc_y1['stat']:.3f}$ & {pval_cell(llc_y1['pval'])} & {llc_y1['lags']} \\\\\n"
    + f"$Y_2$ & ${llc_y2['stat']:.3f}$ & {pval_cell(llc_y2['pval'])} & {llc_y2['lags']} \\\\\n"
    + "\\bottomrule\n"
      "\\end{tabularx}\n"
    + tablenotes(
        "Levin, Lin and Chu (2002) pooled panel unit root test. "
        "$H_0$: all panels contain a unit root; $H_1$: all panels are stationary. "
        "Model includes individual intercepts (entity demeaning via auxiliary OLS). "
        "Lag order selected by AIC over "
        "$\\bar{p}\\in\\{0,\\ldots,\\min(24,\\lfloor T^{1/3}\\rfloor)\\}$ per zone; "
        "$\\bar{p}$ reports the median across zones. "
        f"$\\bar{{T}}\\approx{llc_y1['T_bar']:,}$ usable observations per zone; "
        "at this sample size the LLC finite-sample correction factors "
        "(Levin et al.\\ 2002, Table~2) are negligible "
        "and the statistic is referred to $N(0,1)$ (left-tailed)."
    )
    + "\\end{threeparttable}\n"
      "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table_r3_unitroot.tex").write_text(tex_r3)
print("  → table_r3_unitroot.tex")


# ===========================================================================
# TABLE R4 — DK lag-length sensitivity
# ===========================================================================
print("\nTABLE R4: DK lag-length sensitivity")

BWS = [12, 24, 48]
lag_res = {}
for bw in BWS:
    ry1, ny1 = fit_panel("Y1", REGRESSORS, df, bw=bw)
    ry2, ny2 = fit_panel("Y2", REGRESSORS, df, bw=bw)
    lag_res[bw] = {
        "Y1": {"res": ry1, "n": ny1},
        "Y2": {"res": ry2, "n": ny2},
    }
    print(f"  bw={bw:2d}  Y1: e_V={ry1.params['e_V']:.4f}  e_L={ry1.params['e_L']:.4f}  "
          f"Y2: e_V={ry2.params['e_V']:.4f}  e_L={ry2.params['e_L']:.4f}")


def lag_entries(var, dep):
    out = []
    for bw in BWS:
        d = extract(lag_res[bw][dep]["res"], var)
        out.append(fmt_coef(d["coef"], d["se"], d["p"], dep, var))
    return out


# Panel A = Y1, Panel B = Y2
n_cols = len(BWS)
col_spec = ">{\\raggedright\\arraybackslash}X" + "c" * n_cols

tex_r4 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
      "\\centering\n"
      "\\caption{Robustness: Driscoll--Kraay Lag-Length Sensitivity}\n"
      "\\label{tab:r4_lag}\n"
      "\\begin{threeparttable}\n"
    + f"\\begin{{tabularx}}{{\\linewidth}}{{{col_spec}}}\n"
      "\\toprule\n"
      " & (1) & (2) & (3) \\\\\n"
      "\\cmidrule(r){2-2}\\cmidrule(lr){3-3}\\cmidrule(l){4-4}\n"
      " & BW = 12 & BW = 24 & BW = 48 \\\\\n"
      "\\midrule\n"
      f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textit{{Panel A: $Y_1$ (DA--intraday spread)}}}}"
      " \\\\\n"
      "\\addlinespace[0.3ex]\n"
    + reg_row_ncol("Renewable forecast error ($\\varepsilon_V$)",
                   lag_entries("e_V", "Y1"))
    + reg_row_ncol("Load forecast error ($\\varepsilon_L$)",
                   lag_entries("e_L", "Y1"), last=True)
    + ("Within $R^2$ & "
       + " & ".join(f"{lag_res[bw]['Y1']['res'].rsquared_within:.3f}" for bw in BWS)
       + " \\\\\n"
       "Observations & "
       + " & ".join(f"{lag_res[bw]['Y1']['n']:,}" for bw in BWS)
       + " \\\\\n"
       "\\addlinespace\n"
       f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textit{{Panel B: $Y_2$ (intraday--balancing spread)}}}}"
       " \\\\\n"
       "\\addlinespace[0.3ex]\n")
    + reg_row_ncol("Renewable forecast error ($\\varepsilon_V$)",
                   lag_entries("e_V", "Y2"))
    + reg_row_ncol("Load forecast error ($\\varepsilon_L$)",
                   lag_entries("e_L", "Y2"), last=True)
    + ("Within $R^2$ & "
       + " & ".join(f"{lag_res[bw]['Y2']['res'].rsquared_within:.3f}" for bw in BWS)
       + " \\\\\n"
       "Observations & "
       + " & ".join(f"{lag_res[bw]['Y2']['n']:,}" for bw in BWS)
       + " \\\\\n"
       f"Zone FE & " + " & ".join(["Yes"] * n_cols) + " \\\\\n"
       f"Calendar Controls & " + " & ".join(["Yes"] * n_cols) + " \\\\\n"
       "\\bottomrule\n"
       "\\end{tabularx}\n")
    + tablenotes(
        "Columns vary only the Driscoll--Kraay Bartlett kernel bandwidth (BW, hours). "
        "Column~(2) reproduces Table~2 (BW~$=24$). "
        "$\\varepsilon_V$: total renewable (wind and solar) forecast error (MW). "
        "$\\varepsilon_L$: load forecast error (MW). "
        + DK_NOTE.format(bw="varies")
    )
    + "\\end{threeparttable}\n"
      "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table_r4_lag.tex").write_text(tex_r4)
print("  → table_r4_lag.tex")


# ===========================================================================
# Flag summary
# ===========================================================================
print("\n" + "=" * 65)
print("FLAG SUMMARY  (>20 % deviation from Table 2 baseline)")
print("=" * 65)
print(f"  {'Table':<22} {'Dep':<4} {'Var':<5}  {'Base':>8}  {'New':>8}  {'Δ%':>7}  Status")
print(f"  {'-'*63}")

checks = [
    ("R1 Crisis excl.",   "Y1", "e_V", eV1["coef"]),
    ("R1 Crisis excl.",   "Y1", "e_L", eL1["coef"]),
    ("R1 Crisis excl.",   "Y2", "e_V", eV2["coef"]),
    ("R1 Crisis excl.",   "Y2", "e_L", eL2["coef"]),
    ("R2 High-flow excl.","Y1", "e_V", eV1r2["coef"]),
    ("R2 High-flow excl.","Y1", "e_L", eL1r2["coef"]),
    ("R2 High-flow excl.","Y2", "e_V", eV2r2["coef"]),
    ("R2 High-flow excl.","Y2", "e_L", eL2r2["coef"]),
]
for bw in [12, 48]:
    for dep in ["Y1", "Y2"]:
        for var in ["e_V", "e_L"]:
            c = float(lag_res[bw][dep]["res"].params[var])
            checks.append((f"R4 BW={bw}", dep, var, c))

for tbl, dep, var, coef in checks:
    base = BASELINE[(dep, var)]
    pct  = (coef - base) / abs(base) * 100
    flag = "*** FLAGGED ***" if abs(pct) > 20 else "ok"
    print(f"  {tbl:<22} {dep:<4} {var:<5}  {base:>8.4f}  {coef:>8.4f}  {pct:>+6.1f}%  {flag}")

print("\nDone. Four tables written to results/.")
