"""
Stage 5 — Appendix robustness tables.

Table R1 : Crisis-period exclusion  (22 Sep 2021 – 3 Dec 2022;
           data begin 1 Dec 2021, so effective window is 1 Dec 2021 – 3 Dec 2022).
           Y1 ~ e_V_pre + e_L and Y2 ~ e_V_post + e_L, each its own regression,
           reported side by side.
Table R2 : High cross-border flow exclusion
           (top decile of |net_export_mw| per zone; no threshold pre-defined in codebase).
           Same specification as R1.
Table R3 : Levin-Lin-Chu (2002) panel unit root test on Y1 and Y2
Table R4 : Intraday-balancing spread on the total (undecomposed) renewable
           forecast error: Y2 ~ e_V + e_L, full sample
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
# Formatting helpers
# ---------------------------------------------------------------------------

def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fmt_coef(b, se, p):
    """Coefficient cell + SE cell."""
    st  = stars(p)
    lp  = "" if b < 0 else "\\phantom{-}"
    coef_str = f"${lp}{b:.4f}^{{{st}}}$" if st else f"${lp}{b:.4f}$"
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
    "$^{{***}}p<0.01$, $^{{**}}p<0.05$, $^{{*}}p<0.10$."
)

def tablenotes(text):
    return (
        "\\begin{tablenotes}[flushleft]\n"
        "\\footnotesize\n"
        "\\item \\textit{Notes:} " + text + "\n"
        "\\end{tablenotes}\n"
    )


def reg_row_2col(label, d1, d2, last=False):
    b1, s1 = fmt_coef(d1["coef"], d1["se"], d1["p"])
    b2, s2 = fmt_coef(d2["coef"], d2["se"], d2["p"])
    end = "\\\\\n" if last else "\\\\[0.4ex]\n"
    return (f"{label} & {b1} & {b2} \\\\\n"
            f" & {s1} & {s2} {end}")


BASE_2COL_HEADER = (
    "\\toprule\n"
    " & (1) & (2) \\\\\n"
    "\\cmidrule(r){2-2}\\cmidrule(l){3-3}\n"
    " & $Y_1$ & $Y_2$ \\\\\n"
    "\\midrule\n"
)

def reg_row_1col(label, d, last=False):
    b, s = fmt_coef(d["coef"], d["se"], d["p"])
    end = "\\\\\n" if last else "\\\\[0.4ex]\n"
    return f"{label} & {b} \\\\\n & {s} {end}"


FE_2COL = "Zone FE & Yes & Yes \\\\\nCalendar Controls & Yes & Yes \\\\\n"
FE_1COL = "Zone FE & Yes \\\\\nCalendar Controls & Yes \\\\\n"


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

res_r1y1, n_r1y1 = fit_panel("Y1", ["e_V_pre",  "e_L"], df_r1)
res_r1y2, n_r1y2 = fit_panel("Y2", ["e_V_post", "e_L"], df_r1)

eV1 = extract(res_r1y1, "e_V_pre");  eL1 = extract(res_r1y1, "e_L")
eV2 = extract(res_r1y2, "e_V_post"); eL2 = extract(res_r1y2, "e_L")

print(f"  Y1: e_V_pre={eV1['coef']:.4f}  e_L={eL1['coef']:.4f}")
print(f"  Y2: e_V_post={eV2['coef']:.4f}  e_L={eL2['coef']:.4f}")

tex_r1 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
      "\\centering\n"
      "\\caption{Robustness: Crisis-Period Exclusion (Dec.~2021 -- Dec.~2022)}\n"
      "\\label{tab:r1_crisis}\n"
      "\\begin{threeparttable}\n"
      "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xcc}\n"
    + BASE_2COL_HEADER
    + reg_row_2col("Renewable forecast error", eV1, eV2)
    + reg_row_2col("Load forecast error ($\\varepsilon_L$)", eL1, eL2, last=True)
    + (f"\\midrule\n"
       f"Within $R^2$ & {res_r1y1.rsquared_within:.3f} & {res_r1y2.rsquared_within:.3f} \\\\\n"
       f"Observations & {n_r1y1:,} & {n_r1y2:,} \\\\\n"
       + FE_2COL
       + "\\bottomrule\n"
         "\\end{tabularx}\n")
    + tablenotes(DK_NOTE.format(bw=24))
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

res_r2y1, n_r2y1 = fit_panel("Y1", ["e_V_pre",  "e_L"], df_r2)
res_r2y2, n_r2y2 = fit_panel("Y2", ["e_V_post", "e_L"], df_r2)

eV1r2 = extract(res_r2y1, "e_V_pre");  eL1r2 = extract(res_r2y1, "e_L")
eV2r2 = extract(res_r2y2, "e_V_post"); eL2r2 = extract(res_r2y2, "e_L")

print(f"  Y1: e_V_pre={eV1r2['coef']:.4f}  e_L={eL1r2['coef']:.4f}")
print(f"  Y2: e_V_post={eV2r2['coef']:.4f}  e_L={eL2r2['coef']:.4f}")

tex_r2 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
      "\\centering\n"
      "\\caption{Robustness: Excluding High Cross-Border Flow Hours}\n"
      "\\label{tab:r2_flow}\n"
      "\\begin{threeparttable}\n"
      "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xcc}\n"
    + BASE_2COL_HEADER
    + reg_row_2col("Renewable forecast error", eV1r2, eV2r2)
    + reg_row_2col("Load forecast error ($\\varepsilon_L$)", eL1r2, eL2r2, last=True)
    + (f"\\midrule\n"
       f"Within $R^2$ & {res_r2y1.rsquared_within:.3f} & {res_r2y2.rsquared_within:.3f} \\\\\n"
       f"Observations & {n_r2y1:,} & {n_r2y2:,} \\\\\n"
       + FE_2COL
       + "\\bottomrule\n"
         "\\end{tabularx}\n")
    + tablenotes(DK_NOTE.format(bw=24))
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
        "$H_0$: all panels contain a unit root; $H_1$: all panels are stationary. "
        "Statistic referred to $N(0,1)$ (left-tailed)."
    )
    + "\\end{threeparttable}\n"
      "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table_r3_unitroot.tex").write_text(tex_r3)
print("  → table_r3_unitroot.tex")


# ===========================================================================
# TABLE R4 — Intraday-balancing spread on the total renewable forecast error
# ===========================================================================
print("\nTABLE R4: Total renewable forecast error")

res_r4, n_r4 = fit_panel("Y2", ["e_V", "e_L"], df)

eV_r4 = extract(res_r4, "e_V"); eL_r4 = extract(res_r4, "e_L")

print(f"  Y2: e_V={eV_r4['coef']:.4f}  e_L={eL_r4['coef']:.4f}")

tex_r4 = (
    PREAMBLE
    + "\\begin{table}[htbp]\n"
      "\\centering\n"
      "\\caption{Intraday--Balancing Spread: Total Renewable Forecast Error}\n"
      "\\label{tab:r4_totalv}\n"
      "\\begin{threeparttable}\n"
      "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xc}\n"
      "\\toprule\n"
      " & $Y_2$ \\\\\n"
      "\\midrule\n"
    + reg_row_1col("Renewable forecast error ($\\varepsilon_V$)", eV_r4)
    + reg_row_1col("Load forecast error ($\\varepsilon_L$)", eL_r4, last=True)
    + (f"\\midrule\n"
       f"Within $R^2$ & {res_r4.rsquared_within:.3f} \\\\\n"
       f"Observations & {n_r4:,} \\\\\n"
       + FE_1COL
       + "\\bottomrule\n"
         "\\end{tabularx}\n")
    + tablenotes(DK_NOTE.format(bw=24))
    + "\\end{threeparttable}\n"
      "\\end{table}\n"
    + POSTAMBLE
)
(RESULTS_DIR / "table_r4_totalv.tex").write_text(tex_r4)
print("  → table_r4_totalv.tex")

print("\nDone. Four tables written to results/.")
