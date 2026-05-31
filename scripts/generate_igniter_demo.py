#!/usr/bin/env python3
"""
Generate a self-contained igniter hot-fire demo test for HDA Qt Single Test Analysis.

Output layout (under --out):
  IGN-HF-DEMO-001/
    metadata.json
    raw_data/data.csv

Quick test:
  1. python scripts/generate_igniter_demo.py
  2. python -m hda
  3. Single Test Analysis → Browse CSV → sample_data/igniter_demo/IGN-HF-DEMO-001/raw_data/data.csv
  4. Config: Igniter C1 - Hot Fire Standard
  5. Preprocess (enable channel mapping) → Steady State window 2.0–4.0 s
  6. Analyze tab → Pull from steady window → Run igniter analysis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_csv(duration_s: float = 5.0, hz: float = 100.0) -> pd.DataFrame:
    """Synthetic igniter trace using MTB_IGN_HF_Config channel IDs."""
    n = int(duration_s * hz) + 1
    t_ms = np.linspace(0, duration_s * 1000, n)
    t_s = t_ms / 1000.0
    rng = np.random.default_rng(42)

    def ramp(steady_val: float, t0: float = 2.0, t1: float = 4.0, noise: float = 0.01):
        env = np.clip((t_s - 0.5) / 0.3, 0, 1) * np.clip((4.5 - t_s) / 0.3, 0, 1)
        in_steady = (t_s >= t0) & (t_s <= t1)
        base = steady_val * env
        jitter = np.where(
            in_steady,
            rng.normal(0, steady_val * noise, n),
            0.0,
        )
        return base + jitter

    df = pd.DataFrame(
        {
            "timestamp": t_ms.astype(int),
            "1007": ramp(20.0),       # IG-PT-01 chamber pressure [bar]
            "10004": ramp(30.0),      # OX-FM-01 N2O mass flow [g/s]
            "10003": ramp(12.0),      # FU-FM-01 ethanol mass flow [g/s]
            "10002": ramp(55.0),      # FU-PT-01 fuel upstream [bar]
            "10009": ramp(85.0, noise=0.02),  # LC-01 thrust [N] approximate
            "10001": np.zeros(n),     # unused channel
        }
    )
    return df


def demo_metadata() -> dict:
    return {
        "test_id": "IGN-HF-DEMO-001",
        "test_type": "hot_fire",
        "test_date": "2026-05-27",
        "operator": "Demo Generator",
        "part": "IGN-CP01",
        "serial_num": "SN-DEMO-001",
        "propellant_combination": "N2O/Ethanol",
        "geometry": {
            "throat_diameter_mm": 4.4,
            "throat_area_mm2": math.pi / 4.0 * 4.4 ** 2,
            "expansion_ratio": 1.5,
        },
        "igniter_hardware": {
            "throat_diameter_mm": 4.4,
            "cd_n2o": 0.77,
            "cd_eth": 0.77,
            "t_n2o_c": 20.0,
            "t_eth_c": 20.0,
            "eta_cstar_target": 0.85,
            "d_n2o_orifice_mm": 0.6,
            "d_eth_orifice_mm": 0.5,
            "p_n2o_upstream_bar": 55.0,
        },
        "sensor_roles": {
            "chamber_pressure": "IG-PT-01",
            "mass_flow_ox": "OX-FM-01",
            "mass_flow_fuel": "FU-FM-01",
            "upstream_pressure": "FU-PT-01",
            "thrust": "LC-01",
        },
        "test_conditions": {
            "target_pc_bar": 20.0,
            "target_of_ratio": 2.5,
            "steady_window_s": [2.0, 4.0],
        },
        "notes": [
            "Demo igniter hot fire for HDA Qt Single Test Analysis",
            "Expected steady averages: Pc≈20 bar, ṁ_N2O≈30 g/s, ṁ_EtOH≈12 g/s",
        ],
    }


def write_demo(out_dir: Path) -> Path:
    test_dir = out_dir / "IGN-HF-DEMO-001"
    raw_dir = test_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "data.csv"
    meta_path = test_dir / "metadata.json"

    df = generate_demo_csv()
    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(demo_metadata(), indent=2), encoding="utf-8")
    return test_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate igniter hot-fire demo test data")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sample_data" / "igniter_demo",
        help="Output directory for demo test folder",
    )
    args = parser.parse_args()
    test_dir = write_demo(args.out)
    print(f"Wrote demo test to {test_dir}")
    print(f"  CSV: {test_dir / 'raw_data' / 'data.csv'}")
    print(f"  Metadata: {test_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
