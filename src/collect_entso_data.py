"""
ENTSO-E data collection script for M.Sc. thesis on sequential price formation
in Swedish electricity markets.

Pulls data for SE1–SE4, 2021-12-01 to 2025-12-31, saves one CSV per variable per zone.
Run: python src/collect_entso_data.py
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ["ENTSOE_API_KEY"]

ZONES = {
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
}

YEARS = list(range(2021, 2026))  # 2021 … 2025 (partial)
GLOBAL_START = pd.Timestamp("2021-12-01", tz="Europe/Stockholm")
GLOBAL_END   = pd.Timestamp("2025-03-17 23:59", tz="Europe/Stockholm")  # ISP switches to 15-min on 2025-03-18
TZ = "Europe/Stockholm"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
LOG_FILE = RAW_DIR / "data_pull_log.txt"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging setup — writes to both console and log file
# ---------------------------------------------------------------------------

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
# Helpers
# ---------------------------------------------------------------------------

client = EntsoePandasClient(api_key=API_KEY)


def year_range(year: int):
    """Return (start, end) Timestamps for a year in Stockholm TZ, clamped to sample window."""
    start = max(pd.Timestamp(f"{year}-01-01", tz=TZ), GLOBAL_START)
    end   = min(pd.Timestamp(f"{year}-12-31 23:59", tz=TZ), GLOBAL_END)
    return start, end


def pull_with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retry once after 5 s on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.warning("  First attempt failed: %s — retrying in 5 s …", exc)
        time.sleep(5)
        return fn(*args, **kwargs)  # let caller catch if this also fails


def to_utc_frame(series_or_frame) -> pd.DataFrame:
    """Convert index to UTC and wrap in a DataFrame with a labelled column."""
    if isinstance(series_or_frame, pd.Series):
        df = series_or_frame.to_frame(name="value")
    else:
        df = series_or_frame.copy()
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp_utc"
    return df


# ---------------------------------------------------------------------------
# Pull functions — one per data type
# ---------------------------------------------------------------------------

def pull_da_prices(zone_code: str, year: int) -> pd.DataFrame:
    start, end = year_range(year)
    raw = pull_with_retry(client.query_day_ahead_prices, zone_code, start=start, end=end)
    return to_utc_frame(raw)


def pull_imbalance_prices(zone_code: str, year: int) -> pd.DataFrame:
    start, end = year_range(year)
    raw = pull_with_retry(client.query_imbalance_prices, zone_code, start=start, end=end)
    return to_utc_frame(raw)


def pull_wind_da_forecast(zone_code: str, year: int) -> pd.DataFrame:
    start, end = year_range(year)
    raw = pull_with_retry(
        client.query_wind_and_solar_forecast, zone_code, start=start, end=end, psr_type="B19"
    )
    return to_utc_frame(raw)


def pull_solar_da_forecast(zone_code: str, year: int) -> pd.DataFrame:
    start, end = year_range(year)
    raw = pull_with_retry(
        client.query_wind_and_solar_forecast, zone_code, start=start, end=end, psr_type="B16"
    )
    return to_utc_frame(raw)


def pull_wind_actual(zone_code: str, year: int) -> pd.DataFrame:
    start, end = year_range(year)
    raw = pull_with_retry(
        client.query_generation, zone_code, start=start, end=end, psr_type="B19"
    )
    return to_utc_frame(raw)


def pull_solar_actual(zone_code: str, year: int) -> pd.DataFrame:
    start, end = year_range(year)
    raw = pull_with_retry(
        client.query_generation, zone_code, start=start, end=end, psr_type="B16"
    )
    return to_utc_frame(raw)


def pull_load_forecast(zone_code: str, year: int) -> pd.DataFrame:
    start, end = year_range(year)
    raw = pull_with_retry(client.query_load_forecast, zone_code, start=start, end=end)
    return to_utc_frame(raw)


def pull_load_actual(zone_code: str, year: int) -> pd.DataFrame:
    start, end = year_range(year)
    raw = pull_with_retry(client.query_load, zone_code, start=start, end=end)
    return to_utc_frame(raw)


def pull_installed_capacity(zone_code: str, _year: int) -> pd.DataFrame:
    # query_installed_generation_capacity returns NoMatchingDataError for SE zones;
    # _per_unit works and covers the full period in one call — year arg is ignored here.
    start = pd.Timestamp("2020-01-01", tz=TZ)
    end   = pd.Timestamp("2025-12-31 23:59", tz=TZ)
    raw = pull_with_retry(
        client.query_installed_generation_capacity_per_unit, zone_code, start=start, end=end
    )
    # result is unit-indexed, not time-indexed — return as-is
    return raw


def pull_generation_actual(zone_code: str, year: int) -> pd.DataFrame:
    start, end = year_range(year)
    raw = pull_with_retry(client.query_generation, zone_code, start=start, end=end)
    return to_utc_frame(raw)


def pull_generation_forecast(zone_code: str, year: int) -> pd.DataFrame:
    start, end = year_range(year)
    raw = pull_with_retry(client.query_generation_forecast, zone_code, start=start, end=end)
    return to_utc_frame(raw)


# ---------------------------------------------------------------------------
# Dispatch table  {filename_stem: pull_function}
# ---------------------------------------------------------------------------

DATASETS = [
    ("da_prices",            pull_da_prices),
    ("imbalance_prices",     pull_imbalance_prices),
    ("wind_da_forecast",     pull_wind_da_forecast),
    ("solar_da_forecast",    pull_solar_da_forecast),
    ("wind_actual",          pull_wind_actual),
    ("solar_actual",         pull_solar_actual),
    ("load_forecast",        pull_load_forecast),
    ("load_actual",          pull_load_actual),
    ("generation_actual",    pull_generation_actual),
    ("generation_forecast",  pull_generation_forecast),
    ("installed_capacity",   pull_installed_capacity),
]

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def collect_all():
    successes = []
    failures  = []

    for zone_name, zone_code in ZONES.items():
        for stem, pull_fn in DATASETS:
            csv_path = RAW_DIR / f"{stem}_{zone_name}.csv"

            # --- skip if file already exists (safe to rerun) ---
            if csv_path.exists():
                log.info("SKIP  %s  (already exists)", csv_path.name)
                successes.append(str(csv_path))
                continue

            # installed_capacity pulls the full range in one call; all others loop per year
            if stem == "installed_capacity":
                label = f"{stem}_{zone_name}"
                try:
                    df = pull_fn(zone_code, 0)
                    df.to_csv(csv_path)
                    log.info("OK    %s  (%d rows)", label, len(df))
                    successes.append(str(csv_path))
                except Exception as exc:
                    log.error("FAIL  %s  →  %s", label, exc)
                    failures.append(label)
                finally:
                    time.sleep(1)
                continue

            yearly_frames = []
            for year in YEARS:
                label = f"{stem}_{zone_name}_{year}"
                try:
                    df = pull_fn(zone_code, year)
                    yearly_frames.append(df)
                    log.info("OK    %s", label)
                except Exception as exc:
                    log.error("FAIL  %s  →  %s", label, exc)
                    failures.append(label)
                finally:
                    time.sleep(1)   # respect API rate limit

            if yearly_frames:
                combined = pd.concat(yearly_frames)
                combined.sort_index(inplace=True)
                combined.to_csv(csv_path)
                log.info("SAVED %s  (%d rows)", csv_path.name, len(combined))
                successes.append(str(csv_path))
            else:
                log.error("NO DATA  %s_%s — nothing saved", stem, zone_name)
                failures.append(f"{stem}_{zone_name}")

    return successes, failures


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(successes: list[str], failures: list[str]):
    print("\n" + "=" * 70)
    print("DATA PULL SUMMARY")
    print("=" * 70)

    print(f"\nSuccessfully created / already present: {len(successes)} files")
    for path_str in sorted(successes):
        p = Path(path_str)
        if p.exists():
            size_kb = p.stat().st_size / 1024
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                date_min = df.index.min()
                date_max = df.index.max()
                coverage = f"{date_min}  →  {date_max}"
            except Exception:
                coverage = "(could not parse dates)"
            print(f"  {p.name:<45}  {size_kb:>8.1f} KB   {coverage}")

    if failures:
        print(f"\nFailed pulls ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nNo failures.")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting ENTSO-E data collection  —  %s", datetime.now().isoformat())
    successes, failures = collect_all()
    print_summary(successes, failures)
    log.info("Done.")
