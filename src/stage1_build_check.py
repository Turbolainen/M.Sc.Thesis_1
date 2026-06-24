"""
Stage 1 — build panel variables, run consistency checks, show descriptives.
"""

import pandas as pd

DATA = "data/processed/panel_data.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, parse_dates=["timestamp_cet"])
df["timestamp_cet"] = pd.to_datetime(df["timestamp_cet"], utc=True).dt.tz_convert("Europe/Stockholm")

# Panel is already trimmed; just drop any stray rows before the start
START = pd.Timestamp("2021-12-01", tz="Europe/Stockholm")
df = df[df["timestamp_cet"] >= START].copy()

df = df.set_index(["zone", "timestamp_cet"]).sort_index()
print(f"Panel shape: {df.shape}  (expected 115,488 × 28)")

# ---------------------------------------------------------------------------
# Construct variables
# ---------------------------------------------------------------------------
df["P_DA"] = df["da_price"]
df["P_ID"] = df["id_avg_price_last3h_eur"]
df["P_B"]  = df["imbalance_price_long"]

df["Y1"] = df["P_ID"] - df["P_DA"]
df["Y2"] = df["P_B"]  - df["P_ID"]

df["e_V"]      = (df["wind_actual_mw"]    - df["wind_da_forecast_mw"]) \
               + (df["solar_actual_mw"]   - df["solar_da_forecast_mw"])
df["e_V_pre"]  = (df["wind_id_forecast_mw"]  - df["wind_da_forecast_mw"]) \
               + (df["solar_id_forecast_mw"]  - df["solar_da_forecast_mw"])
df["e_V_post"] = (df["wind_actual_mw"]    - df["wind_id_forecast_mw"]) \
               + (df["solar_actual_mw"]   - df["solar_id_forecast_mw"])
df["e_L"]      = df["load_actual_mw"] - df["load_forecast_mw"]

# Calendar dummies (derived from time level of index)
ts = df.index.get_level_values("timestamp_cet")
df["hour"]       = ts.hour
df["dayofweek"]  = ts.dayofweek
df["month"]      = ts.month

# ---------------------------------------------------------------------------
# Build check: e_V == e_V_pre + e_V_post
# ---------------------------------------------------------------------------
residual = (df["e_V"] - (df["e_V_pre"] + df["e_V_post"])).abs()
max_resid = residual.max()
print(f"\nIdentity check  e_V == e_V_pre + e_V_post")
print(f"  Max absolute deviation: {max_resid:.2e}  {'✓ PASS' if max_resid < 1e-6 else '✗ FAIL'}")

# ---------------------------------------------------------------------------
# Missing-value summary
# ---------------------------------------------------------------------------
print("\nMissing values (NaN counts):")
for col in ["Y1", "Y2", "e_V", "e_V_pre", "e_V_post", "e_L", "P_DA", "P_ID", "P_B"]:
    n_miss = df[col].isna().sum()
    pct    = 100 * n_miss / len(df)
    print(f"  {col:<12}  {n_miss:>6}  ({pct:.2f}%)")

# ---------------------------------------------------------------------------
# Descriptives by zone
# ---------------------------------------------------------------------------
VARS = ["Y1", "Y2", "e_V", "e_V_pre", "e_V_post", "e_L"]

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS BY ZONE")
print("="*70)

for zone in ["SE1", "SE2", "SE3", "SE4"]:
    sub = df.xs(zone, level="zone")[VARS]
    print(f"\n--- {zone} ---")
    desc = sub.describe().loc[["mean", "std", "min", "max"]].T
    desc.columns = ["mean", "sd", "min", "max"]
    print(desc.to_string(float_format=lambda x: f"{x:10.3f}"))

# ---------------------------------------------------------------------------
# Pooled descriptives (for table)
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("POOLED DESCRIPTIVE STATISTICS")
print("="*70)
pooled = df[VARS].describe().loc[["mean", "std", "min", "max"]].T
pooled.columns = ["mean", "sd", "min", "max"]
print(pooled.to_string(float_format=lambda x: f"{x:10.3f}"))

# ---------------------------------------------------------------------------
# Save enriched panel for later stages
# ---------------------------------------------------------------------------
df.to_pickle("data/processed/panel_enriched.pkl")
print("\nSaved data/processed/panel_enriched.pkl")
