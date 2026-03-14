"""
Central configuration: API credentials, paths, and analysis parameters.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES        = ROOT / "output" / "figures"
TABLES         = ROOT / "output" / "tables"

# ── ENTSO-E API ────────────────────────────────────────────────────────────────
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

# ── Analysis parameters ────────────────────────────────────────────────────────
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

# Bidding zones / country codes (ENTSO-E EIC codes)
COUNTRY_CODES = {
    "Finland":  "10YFI-1--------U",
    "Sweden":   "10YSE-1--------K",
    "Norway":   "10YNO-0--------C",
    "Germany":  "10Y1001A1001A83F",
    "Denmark":  "10Y1001A1001A65H",
}
