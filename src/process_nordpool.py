"""
Parse Nord Pool intraday hourly statistics files and save one CSV per zone
to data/raw/intraday_{zone}.csv, trimmed to the sample window.

Source files: data/NORDPOOL/IntradayHourlyStatistics_{year}_SE1,SE2,SE3,SE4_None.csv
Run: python src/process_nordpool.py
"""

from pathlib import Path
import pandas as pd

NORDPOOL_DIR = Path(__file__).resolve().parents[1] / "data" / "NORDPOOL"
RAW_DIR      = Path(__file__).resolve().parents[1] / "data" / "raw"

CUTOFF_START = pd.Timestamp("2021-12-01", tz="UTC")
CUTOFF_END   = pd.Timestamp("2025-03-17 23:59", tz="UTC")

ZONES = ["SE1", "SE2", "SE3", "SE4"]

# Clean column name map (zone prefix stripped, then this applied)
COL_RENAME = {
    "High Price (EUR/MWh)":            "id_high_price_eur",
    "Low Price (EUR/MWh)":             "id_low_price_eur",
    "Open Price (EUR/MWh)":            "id_open_price_eur",
    "Close Price (EUR/MWh)":           "id_close_price_eur",
    "Average Price (EUR/MWh)":         "id_avg_price_eur",
    "Average Price Last 3 H (EUR/MWh)":"id_avg_price_last3h_eur",
    "Average Price Last 1 H (EUR/MWh)":"id_avg_price_last1h_eur",
    "Volume (MW)":                     "id_volume_mw",
    "Buy Volume (MW)":                 "id_buy_volume_mw",
    "Sell Volume (MW)":                "id_sell_volume_mw",
    "Open Trade Time (CET)":           "id_open_trade_time",
    "Close Trade Time (CET)":          "id_close_trade_time",
}

# ---------------------------------------------------------------------------
# Load and parse all yearly files
# ---------------------------------------------------------------------------

all_frames = {z: [] for z in ZONES}

files = sorted(NORDPOOL_DIR.glob("IntradayHourlyStatistics_*.csv"))
print(f"Found {len(files)} Nord Pool files\n")

for fpath in files:
    print(f"  Reading {fpath.name} …")
    df = pd.read_csv(fpath, sep=";", parse_dates=["Delivery Start (CET)"])

    # Parse CET timestamps and convert to UTC
    df["timestamp_utc"] = (
        pd.to_datetime(df["Delivery Start (CET)"], dayfirst=True)
        .dt.tz_localize("Europe/Paris", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    df = df.set_index("timestamp_utc").drop(columns=["Delivery Start (CET)", "Delivery End (CET)"])

    for zone in ZONES:
        # Extract this zone's columns and strip the zone prefix
        zone_cols = {c: c[len(zone) + 1:] for c in df.columns if c.startswith(zone + " ")}
        z_df = df[list(zone_cols.keys())].rename(columns=zone_cols)
        # Apply clean column names; anything not in the map gets snake_cased with id_ prefix
        z_df = z_df.rename(columns={
            c: COL_RENAME.get(c, "id_" + c.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_"))
            for c in z_df.columns
        })
        all_frames[zone].append(z_df)

# ---------------------------------------------------------------------------
# Concatenate, trim, save
# ---------------------------------------------------------------------------

print()
for zone in ZONES:
    combined = pd.concat(all_frames[zone]).sort_index()
    combined = combined[(combined.index >= CUTOFF_START) & (combined.index <= CUTOFF_END)]

    out_path = RAW_DIR / f"intraday_{zone}.csv"
    combined.to_csv(out_path)
    print(f"  Saved intraday_{zone}.csv  —  {len(combined)} rows  |  {combined.index.min()} → {combined.index.max()}")
    print(f"    Columns: {combined.columns.tolist()}")

print("\nDone.")
