"""
Two jobs in one script:

1. TRIM — remove rows before 2021-12-01 UTC from all time-series CSVs in data/raw/
2. BUILD — merge all trimmed CSVs into a single panel-data master file at
           data/processed/panel_data.csv
           Columns: timestamp_utc, zone, da_price, imbalance_price_long,
                    imbalance_price_short, wind_da_forecast_mw, solar_da_forecast_mw,
                    wind_actual_mw, solar_actual_mw, load_forecast_mw, load_actual_mw

Run: python src/build_panel.py
"""

from pathlib import Path
import pandas as pd

RAW_DIR       = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_START = pd.Timestamp("2021-12-01", tz="UTC")
CUTOFF_END   = pd.Timestamp("2025-03-17 23:59", tz="UTC")  # ISP switches to 15-min on 2025-03-18

ZONES = ["SE1", "SE2", "SE3", "SE4"]

# Map from CSV filename stem → (column(s) in file, column name(s) in panel)
TIMESERIES_MAP = {
    "da_prices":         (["value"],            ["da_price"]),
    "imbalance_prices":  (["Long", "Short"],     ["imbalance_price_long", "imbalance_price_short"]),
    "wind_da_forecast":  (["Wind Onshore"],      ["wind_da_forecast_mw"]),
    "solar_da_forecast": (["Solar"],             ["solar_da_forecast_mw"]),
    "wind_actual":       (["Wind Onshore"],      ["wind_actual_mw"]),
    "solar_actual":      (["Solar"],             ["solar_actual_mw"]),
    "load_forecast":     (["Forecasted Load"],   ["load_forecast_mw"]),
    "load_actual":       (["Actual Load"],       ["load_actual_mw"]),
}

# ---------------------------------------------------------------------------
# 1. TRIM
# ---------------------------------------------------------------------------

print("=" * 60)
print("STEP 1 — Trimming CSVs to >= 2021-12-01 UTC")
print("=" * 60)

for stem in TIMESERIES_MAP:
    for zone in ZONES:
        path = RAW_DIR / f"{stem}_{zone}.csv"
        if not path.exists():
            print(f"  MISSING  {path.name}")
            continue

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)

        before = len(df)
        df = df[(df.index >= CUTOFF_START) & (df.index <= CUTOFF_END)]
        after  = len(df)

        df.to_csv(path)
        removed = before - after
        print(f"  {path.name:<45}  removed {removed:>5} rows  →  {after} remain")

print()

# ---------------------------------------------------------------------------
# 2. BUILD PANEL
# ---------------------------------------------------------------------------

print("=" * 60)
print("STEP 2 — Building panel_data.csv")
print("=" * 60)

zone_frames = []

for zone in ZONES:
    dfs = []
    for stem, (src_cols, dst_cols) in TIMESERIES_MAP.items():
        path = RAW_DIR / f"{stem}_{zone}.csv"
        if not path.exists():
            print(f"  WARNING  {path.name} not found — skipping for {zone}")
            continue

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[(df.index >= CUTOFF_START) & (df.index <= CUTOFF_END)]
        df = df[src_cols].copy()
        df.columns = dst_cols
        dfs.append(df)

    if not dfs:
        print(f"  WARNING  no data loaded for {zone}")
        continue

    merged = dfs[0].join(dfs[1:], how="outer")
    merged.insert(0, "zone", zone)
    zone_frames.append(merged)
    print(f"  {zone}  {len(merged)} rows  ×  {merged.shape[1]-1} variables")

panel = pd.concat(zone_frames)
panel.index.name = "timestamp_utc"
panel.sort_values(["timestamp_utc", "zone"], inplace=True)

out_path = PROCESSED_DIR / "panel_data.csv"
panel.to_csv(out_path)

size_mb = out_path.stat().st_size / 1024 / 1024
print()
print(f"Saved  {out_path.name}")
print(f"  Shape   : {panel.shape[0]:,} rows  ×  {panel.shape[1]} columns")
print(f"  Size    : {size_mb:.1f} MB")
print(f"  Columns : {panel.columns.tolist()}")
print(f"  From    : {panel.index.min()}")
print(f"  To      : {panel.index.max()}")
print(f"  Zones   : {panel['zone'].unique().tolist()}")
