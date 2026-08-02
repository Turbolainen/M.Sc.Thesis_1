"""
Combine all results/*.tex tables into one paste-ready file.

Each table file is a standalone compilable document (own \\documentclass,
\\begin{document}/\\end{document}). This strips that wrapper and
concatenates just the \\begin{table}...\\end{table} bodies, in table-number
order, so the result can be pasted directly into the thesis document.

Output: results/combined.tex
"""

from pathlib import Path

RESULTS_DIR = Path("results")

TABLES = [
    ("table_descriptives.tex", "Table 1 — Descriptive Statistics"),
    ("table2.tex",             "Table 2 — Day-Ahead–Intraday Spread"),
    ("table3.tex",             "Table 3 — Intraday–Balancing Spread"),
    ("table4.tex",             "Table 4 — Intraday–Balancing Spread, Pre/Post Decomposition"),
    ("table5.tex",             "Table 5 — Zone Heterogeneity, Day-Ahead–Intraday Spread"),
    ("table6.tex",             "Table 6 — Zone Heterogeneity, Intraday–Balancing Spread"),
    ("table_r1_crisis.tex",    "Table R1 — Robustness: Crisis-Period Exclusion"),
    ("table_r2_flow.tex",      "Table R2 — Robustness: High Cross-Border Flow Exclusion"),
    ("table_r3_unitroot.tex",  "Table R3 — Panel Unit Root Test"),
    ("table_r4_totalv.tex",    "Table R4 — Intraday–Balancing Spread, Total Renewable Forecast Error"),
]


def extract_table_body(text: str) -> str:
    start = text.index("\\begin{table}")
    end = text.index("\\end{table}") + len("\\end{table}")
    return text[start:end]


sections = []
for filename, heading in TABLES:
    path = RESULTS_DIR / filename
    body = extract_table_body(path.read_text())
    sections.append(f"% {'=' * 70}\n% {heading}  ({filename})\n% {'=' * 70}\n{body}\n")

(RESULTS_DIR / "combined.tex").write_text("\n".join(sections))
print(f"Wrote combined.tex ({len(sections)} tables)")
