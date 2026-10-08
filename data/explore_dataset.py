"""
explore_dataset.py – Quick inspection of the Delhi Traffic Density Dataset.

Shows:
  - Column names and data types
  - Time range per file
  - Density statistics (mean, min, max) per channel
  - Mapped flow rates per intersection

Run:
    python data/explore_dataset.py
"""

import sys
from pathlib import Path

# project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.dataset_loader import DatasetLoader, LANE_TO_TL_MAP, ROAD_CAPACITY_VPH

DATA_DIR = Path(__file__).parent / "DelhiTrafficDensityDataset"
TL_IDS = ["TL00", "TL01", "TL10", "TL11"]

def main():
    loader = DatasetLoader(data_dir=DATA_DIR, tl_ids=TL_IDS)

    print("=" * 65)
    print(f"  Delhi Traffic Density Dataset")
    print(f"  {loader.n_episodes} daily files found")
    print("=" * 65)

    # ----- Inspect first file -----
    df = loader.load_episode(0)
    print(f"\nFile:    {loader.episode_names[0]}.csv")
    print(f"Rows:    {len(df)}")
    t0 = pd.to_datetime(df['EpochTime'].iloc[0],  unit='s')
    t1 = pd.to_datetime(df['EpochTime'].iloc[-1], unit='s')
    print(f"Time:    {t0.strftime('%Y-%m-%d %H:%M:%S')}  →  {t1.strftime('%H:%M:%S')}  (UTC)")
    print(f"Step:    {df['EpochTime'].diff().dropna().mode()[0]:.0f} second(s)\n")

    # ----- Density stats -----
    q_cols = [c for c in df.columns if c.startswith("QueueDensity")]
    s_cols = [c for c in df.columns if c.startswith("StopDensity")]

    print(f"{'Channel':<20}  {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
    print("  " + "-" * 55)
    for col in q_cols + s_cols:
        print(f"  {col:<18}  {df[col].mean():8.4f}  {df[col].min():8.4f}  "
              f"{df[col].max():8.4f}  {df[col].std():8.4f}")

    # ----- Mapped flow rates -----
    print("\n── Flow rates for first episode (first 5 minutes) ────────")
    flows = loader.get_flow_rates(episode_index=0, window_start=0, window_steps=300)
    print(f"  {'Intersection':<12}  {'Flow (veh/h)':>14}")
    print("  " + "-" * 28)
    for tid, fv in flows.items():
        print(f"  {tid:<12}  {fv:14.1f}")

    # ----- ARIMA warmup sample -----
    print("\n── ARIMA warmup sample (first 5 steps of TL00) ──────────")
    warmup = loader.get_arima_warmup(episode_index=0, n_steps=5)
    for step_val in warmup["TL00"]:
        print(f"  inflow = {step_val:.4f} veh/step")

    # ----- All episode date summary -----
    print("\n── All episodes ──────────────────────────────────────────")
    for i, name in enumerate(loader.episode_names):
        df_ep = loader.load_episode(i)
        t = pd.to_datetime(df_ep['EpochTime'].iloc[0], unit='s')
        print(f"  [{i:2d}] {name:<8}  {t.strftime('%Y-%m-%d %H:%M')} UTC  "
              f"  {len(df_ep):>6} rows")

    print("\n[OK] Dataset is compatible with the project.")
    print("     Set  dataset.enabled: true  in config/config.yaml to activate.\n")

if __name__ == "__main__":
    main()
