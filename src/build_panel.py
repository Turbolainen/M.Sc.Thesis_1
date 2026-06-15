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
    "wind_id_forecast":   (["Wind Onshore"],     ["wind_id_forecast_mw"]),
    "solar_id_forecast":  (["Solar"],            ["solar_id_forecast_mw"]),
    "wind_actual":        (["Wind Onshore"],     ["wind_actual_mw"]),
    "solar_actual":       (["Solar"],            ["solar_actual_mw"]),
    "load_forecast":      (["Forecasted Load"],  ["load_forecast_mw"]),
    "load_actual":        (["Actual Load"],      ["load_actual_mw"]),
}

# Generation types already covered by dedicated columns — skip to avoid duplication
GEN_SKIP = {"Wind Onshore", "Solar"}

# Cross-border flow corridors per zone: positive = export out of zone
# Each entry: (corridor_stem, sign)  +1 = outflow from this zone, -1 = inflow
ZONE_CORRIDORS = {
    "SE1": [
        ("se1_se2", +1), ("se2_se1", -1),
        ("se1_fi",  +1), ("fi_se1",  -1),
        ("se1_no4", +1), ("no4_se1", -1),
    ],
    "SE2": [
        ("se2_se1", +1), ("se1_se2", -1),
        ("se2_se3", +1), ("se3_se2", -1),
        ("se2_no3", +1), ("no3_se2", -1),
    ],
    "SE3": [
        ("se3_se2", +1), ("se2_se3", -1),
        ("se3_se4", +1), ("se4_se3", -1),
        ("se3_no1", +1), ("no1_se3", -1),
        ("se3_dk1", +1), ("dk1_se3", -1),
    ],
    "SE4": [
        ("se4_se3", +1), ("se3_se4", -1),
        ("se4_dk2", +1), ("dk2_se4", -1),
        ("se4_de",  +1), ("de_se4",  -1),
        ("se4_pl",  +1), ("pl_se4",  -1),
    ],
}


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
flow_stems = [f"flow_{c}" for c in sum([[(s, s2) for s, _ in v for s2 in [s]] for v in ZONE_CORRIDORS.values()], [])]
all_corridors = list({c for v in ZONE_CORRIDORS.values() for c, _ in v})
stems_to_trim = list(TIMESERIES_MAP.keys()) + ["generation_actual", "generation_forecast", "intraday"] + [f"flow_{c}" for c in all_corridors]

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

    # --- Cross-border net exports ---
    flow_series = []
    for corridor, sign in ZONE_CORRIDORS.get(zone, []):
        fp = RAW_DIR / f"flow_{corridor}.csv"
        if fp.exists():
            s = load_trimmed(fp)["flow_mw"] * sign
            flow_series.append(s)
    if flow_series:
        net = pd.concat(flow_series, axis=1).sum(axis=1).to_frame("net_export_mw")
        dfs.append(net)

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

# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("ALIGNMENT CHECK")
print("=" * 60)

rows_per_zone = panel.groupby("zone").size()
print(f"  Rows per zone:\n{rows_per_zone.to_string()}")

expected_hours = int((pd.Timestamp("2025-03-17 23:00", tz="UTC") -
                      pd.Timestamp("2021-12-01 00:00", tz="UTC")).total_seconds() / 3600) + 1
print(f"\n  Expected hours in window: {expected_hours}")

ts_per_zone = panel.groupby("zone").apply(lambda x: x.index.nunique())
print(f"  Unique timestamps per zone:\n{ts_per_zone.to_string()}")

dups = panel.groupby(["zone", panel.index]).size()
max_dup = dups.max()
print(f"\n  Max duplicate (zone, timestamp) pairs: {max_dup}  {'✓' if max_dup == 1 else '✗ DUPLICATES FOUND'}")

common_ts = set(panel[panel.zone == "SE1"].index)
for z in ["SE2", "SE3", "SE4"]:
    common_ts &= set(panel[panel.zone == z].index)
print(f"  Timestamps common to all 4 zones: {len(common_ts)}")

# ---------------------------------------------------------------------------
# 3. POST-PROCESS — make analysis-ready
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("STEP 3 — Post-processing for panel analysis")
print("=" * 60)

# 3a. Convert trade times (CET strings) to numeric lead hours before delivery
for raw_col, new_col in [
    ("id_open_trade_time",  "id_lead_open_h"),
    ("id_close_trade_time", "id_lead_close_h"),
]:
    trade_utc = (
        pd.to_datetime(panel[raw_col], format="%d.%m.%Y %H:%M:%S", errors="coerce")
        .dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    delivery_utc = panel.index.tz_convert("UTC")
    panel[new_col] = (delivery_utc - trade_utc).dt.total_seconds() / 3600
    panel.drop(columns=[raw_col], inplace=True)
print("  Trade times → lead hours (id_lead_open_h, id_lead_close_h)")

# 3b. Drop low-quality columns
#     id_avg_price_last1h_eur: only present in 2025 Nord Pool files → 99.99% missing
#     gen_actual_marine_mw:    reporting artefact for SE3 only → 23.75% missing
DROP_COLS = ["id_avg_price_last1h_eur", "gen_actual_marine_mw"]
dropped = [c for c in DROP_COLS if c in panel.columns]
panel.drop(columns=dropped, inplace=True)
print(f"  Dropped low-quality columns: {dropped}")

# 3c. Fill structural NaN generation columns with 0
#     A column is structural-zero for a zone if it is entirely NaN for that zone
gen_cols = [c for c in panel.columns if c.startswith("gen_actual_")]
fills = 0
for col in gen_cols:
    mask = panel.groupby("zone")[col].transform(lambda s: s.isna().all())
    panel.loc[mask, col] = 0.0
    fills += int(mask.sum())
print(f"  Structural NaN fills (generation): {fills} cells set to 0")

# 3c. Convert UTC index → Europe/Stockholm so timestamps match ENTSO-E website display
local_index = panel.index.tz_convert("Europe/Stockholm")
panel.index = local_index
panel.index.name = "timestamp_cet"
print("  Index converted UTC → Europe/Stockholm (CET/CEST)")

# 3d. Set (zone, timestamp_cet) MultiIndex — entity first, as expected by linearmodels
panel = panel.drop(columns=["zone"]).set_index(
    pd.MultiIndex.from_arrays(
        [panel["zone"], panel.index],
        names=["zone", "timestamp_cet"],
    )
)
print("  MultiIndex set: (zone, timestamp_cet)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_path = PROCESSED_DIR / "panel_data.csv"
panel.to_csv(out_path)

size_mb = out_path.stat().st_size / 1024 / 1024
print()
print(f"Saved  {out_path.name}")
print(f"  Shape   : {panel.shape[0]:,} rows  ×  {panel.shape[1]} columns")
print(f"  Size    : {size_mb:.1f} MB")
print(f"  Columns : {panel.columns.tolist()}")
print(f"  Index   : {panel.index.names}")
print(f"  From    : {panel.index.get_level_values('timestamp_cet').min()}")
print(f"  To      : {panel.index.get_level_values('timestamp_cet').max()}")
print(f"  Zones   : {panel.index.get_level_values('zone').unique().tolist()}")
