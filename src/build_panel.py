"""
Two jobs in one script:

1. TRIM — enforce sample window (2021-12-01 → 2025-03-17 UTC) on all
          time-series CSVs in data/raw/
2. BUILD — merge all trimmed CSVs into a single panel-data master file at
           data/processed/panel_data.csv

Fixed columns per zone:
    da_price, imbalance_price_long, imbalance_price_short,
    wind_da_forecast_mw, solar_da_forecast_mw,
    wind_actual_mw, solar_actual_mw,
    load_forecast_mw, load_actual_mw,
    gen_forecast_total_mw,
    gen_actual_{type}_mw  (all generation types reported for that zone),
    id_*  (Nord Pool intraday statistics)

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

# Fixed-column time series: stem → (source cols, panel cols)
TIMESERIES_MAP = {
    "da_prices":          (["value"],           ["da_price"]),
    "imbalance_prices":   (["Long", "Short"],    ["imbalance_price_long", "imbalance_price_short"]),
    "wind_da_forecast":   (["Wind Onshore"],     ["wind_da_forecast_mw"]),
    "solar_da_forecast":  (["Solar"],            ["solar_da_forecast_mw"]),
    "wind_actual":        (["Wind Onshore"],     ["wind_actual_mw"]),
    "solar_actual":       (["Solar"],            ["solar_actual_mw"]),
    "load_forecast":      (["Forecasted Load"],  ["load_forecast_mw"]),
    "load_actual":        (["Actual Load"],      ["load_actual_mw"]),
}

# Generation types already covered by dedicated columns — skip to avoid duplication
GEN_SKIP = {"Wind Onshore", "Solar"}


def load_trimmed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[(df.index >= CUTOFF_START) & (df.index <= CUTOFF_END)]


def snake(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# 1. TRIM all raw CSVs to sample window
# ---------------------------------------------------------------------------

print("=" * 60)
print("STEP 1 — Trimming CSVs to sample window")
print("=" * 60)

# Fixed time-series + generation + intraday
stems_to_trim = list(TIMESERIES_MAP.keys()) + ["generation_actual", "generation_forecast", "intraday"]

for stem in stems_to_trim:
    for zone in ZONES:
        path = RAW_DIR / f"{stem}_{zone}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        before = len(df)
        df = df[(df.index >= CUTOFF_START) & (df.index <= CUTOFF_END)]
        df.to_csv(path)
        print(f"  {path.name:<45}  {before:>6} → {len(df)} rows")

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

    # --- fixed-column time series ---
    for stem, (src_cols, dst_cols) in TIMESERIES_MAP.items():
        path = RAW_DIR / f"{stem}_{zone}.csv"
        if not path.exists():
            print(f"  WARNING  {path.name} not found")
            continue
        df = load_trimmed(path)[src_cols].copy()
        df.columns = dst_cols
        dfs.append(df)

    # --- generation forecast (single series) ---
    gf_path = RAW_DIR / f"generation_forecast_{zone}.csv"
    if gf_path.exists():
        df = load_trimmed(gf_path)
        df.columns = ["gen_forecast_total_mw"]
        dfs.append(df)

    # --- generation actual (multi-column, skip wind/solar already in panel) ---
    ga_path = RAW_DIR / f"generation_actual_{zone}.csv"
    if ga_path.exists():
        df = load_trimmed(ga_path)
        # Drop MultiIndex level if present (entsoe sometimes returns Actual Aggregated / Actual Consumption)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs("Actual Aggregated", axis=1, level=1)
        df = df[[c for c in df.columns if c not in GEN_SKIP]]
        df.columns = [f"gen_actual_{snake(c)}_mw" for c in df.columns]
        dfs.append(df)

    # --- Nord Pool intraday ---
    id_path = RAW_DIR / f"intraday_{zone}.csv"
    if id_path.exists():
        df = load_trimmed(id_path)
        dfs.append(df)

    if not dfs:
        print(f"  WARNING  no data for {zone}")
        continue

    merged = dfs[0].join(dfs[1:], how="outer")
    merged.insert(0, "zone", zone)
    zone_frames.append(merged)
    print(f"  {zone}  {len(merged):>6} rows  ×  {merged.shape[1]-1} variables")

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
