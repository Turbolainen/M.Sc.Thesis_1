"""
Stage 4 — Descriptive statistics table.

Output (in results/):
  table_descriptives.tex

The H1/H2/H3 regression tables previously produced here (table_h1.tex,
table_h2.tex, table_h3.tex) used a mis-timed forecast-error specification
for Y1 (see stage6/stage8) and have been superseded by table2.tex through
table6.tex.

Preamble requirements: \\usepackage{booktabs,threeparttable}
"""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df_panel = pd.read_pickle("data/processed/panel_enriched.pkl")

ZONES = ["SE1", "SE2", "SE3", "SE4"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tablenotes(text):
    return (
        "\\begin{tablenotes}[flushleft]\n"
        "\\footnotesize\n"
        "\\item \\textit{Notes:} " + text + "\n"
        "\\end{tablenotes}\n"
    )


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
