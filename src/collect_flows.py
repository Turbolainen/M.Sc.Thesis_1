"""
Pull cross-border physical flows from ENTSO-E for all corridors connected
to Swedish bidding zones (SE1–SE4) and save raw per-corridor CSVs to
data/raw/flow_{corridor}.csv.

Net exports per zone are computed in build_panel.py.

Run: ENTSOE_API_KEY=... python src/collect_flows.py
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ["ENTSOE_API_KEY"]
client  = EntsoePandasClient(api_key=API_KEY)

TZ           = "Europe/Stockholm"
YEARS        = list(range(2021, 2026))
GLOBAL_START = pd.Timestamp("2021-12-01", tz=TZ)
GLOBAL_END   = pd.Timestamp("2025-03-17 23:59", tz=TZ)

RAW_DIR  = Path(__file__).resolve().parents[1] / "data" / "raw"
LOG_FILE = RAW_DIR / "data_pull_log.txt"
RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Corridors — (from_eic, to_eic)
# Both directions pulled so net export per zone can be computed
# ---------------------------------------------------------------------------

CORRIDORS = {
    # Internal SE zone borders
    "se1_se2": ("10Y1001A1001A44P", "10Y1001A1001A45N"),
    "se2_se1": ("10Y1001A1001A45N", "10Y1001A1001A44P"),
    "se2_se3": ("10Y1001A1001A45N", "10Y1001A1001A46L"),
    "se3_se2": ("10Y1001A1001A46L", "10Y1001A1001A45N"),
    "se3_se4": ("10Y1001A1001A46L", "10Y1001A1001A47J"),
    "se4_se3": ("10Y1001A1001A47J", "10Y1001A1001A46L"),
    # SE1 external
    "se1_fi":  ("10Y1001A1001A44P", "10YFI-1--------U"),
    "fi_se1":  ("10YFI-1--------U", "10Y1001A1001A44P"),
    "se1_no4": ("10Y1001A1001A44P", "10YNO-4--------9"),
    "no4_se1": ("10YNO-4--------9", "10Y1001A1001A44P"),
    # SE2 external
    "se2_no3": ("10Y1001A1001A45N", "10YNO-3--------J"),
    "no3_se2": ("10YNO-3--------J", "10Y1001A1001A45N"),
    # SE3 external
    "se3_no1": ("10Y1001A1001A46L", "10YNO-1--------2"),
    "no1_se3": ("10YNO-1--------2", "10Y1001A1001A46L"),
    "se3_dk1": ("10Y1001A1001A46L", "10YDK-1--------W"),
    "dk1_se3": ("10YDK-1--------W", "10Y1001A1001A46L"),
    # SE4 external
    "se4_dk2": ("10Y1001A1001A47J", "10YDK-2--------M"),
    "dk2_se4": ("10YDK-2--------M", "10Y1001A1001A47J"),
    "se4_de":  ("10Y1001A1001A47J", "10Y1001A1001A82H"),
    "de_se4":  ("10Y1001A1001A82H", "10Y1001A1001A47J"),
    "se4_pl":  ("10Y1001A1001A47J", "10YPL-AREA-----S"),
    "pl_se4":  ("10YPL-AREA-----S", "10Y1001A1001A47J"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def year_range(year: int):
    start = max(pd.Timestamp(f"{year}-01-01", tz=TZ), GLOBAL_START)
    end   = min(pd.Timestamp(f"{year}-12-31 23:59", tz=TZ), GLOBAL_END)
    return start, end


def pull_with_retry(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.warning("  First attempt failed: %s — retrying in 5 s …", exc)
        time.sleep(5)
        return fn(*args, **kwargs)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log.info("Starting cross-border flow collection — %s", datetime.now().isoformat())

failures = []

for corridor, (frm, to) in CORRIDORS.items():
    csv_path = RAW_DIR / f"flow_{corridor}.csv"

    if csv_path.exists():
        log.info("SKIP  flow_%s.csv  (already exists)", corridor)
        continue

    yearly = []
    for year in YEARS:
        start, end = year_range(year)
        label = f"flow_{corridor}_{year}"
        try:
            raw = pull_with_retry(client.query_crossborder_flows, frm, to, start=start, end=end)
            raw = raw.to_frame(name="flow_mw")
            raw.index = raw.index.tz_convert("UTC")
            raw.index.name = "timestamp_utc"
            yearly.append(raw)
            log.info("OK    %s", label)
        except Exception as exc:
            log.error("FAIL  %s  →  %s", label, exc)
            failures.append(label)
        finally:
            time.sleep(1)

    if yearly:
        combined = pd.concat(yearly).sort_index()
        combined.to_csv(csv_path)
        log.info("SAVED flow_%s.csv  (%d rows)", corridor, len(combined))

# Summary
print("\n" + "=" * 60)
if failures:
    print(f"Failed pulls ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
else:
    print("All corridors collected successfully.")
print("=" * 60)
