"""
03_chokepoint_stats.py
Computes intensity around key chokepoints by windowed reads on the massive raster.
Outputs:
- Tables/generated/chokepoint_intensity.csv
- Figures/generated/chokepoint_ranking.png
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config.yaml"
TIF = ROOT / "Data" / "shipdensity_global.tif"

TAB_OUT = ROOT / "Tables" / "generated" / "chokepoint_intensity.csv"
FIG_OUT = ROOT / "Figures" / "generated" / "chokepoint_ranking.png"

def main():
    if not CFG.exists():
        raise FileNotFoundError(f"Missing {CFG}")
    if not TIF.exists():
        raise FileNotFoundError(f"Missing {TIF}")

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    chokepoints = cfg.get("chokepoints", {})

    rows = []
    with rasterio.open(TIF) as src:
        for name, bbox in chokepoints.items():
            lon_min, lon_max, lat_min, lat_max = bbox
            win = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
            arr = src.read(1, window=win, masked=True)
            data = np.ma.filled(arr, 0.0).astype("float64")
            total = float(data.sum())
            mean = float(data.mean())
            p99 = float(np.quantile(data.ravel(), 0.99)) if data.size else float("nan")
            rows.append({
                "chokepoint": name,
                "lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max,
                "sum_intensity": total,
                "mean_intensity": mean,
                "p99_cell": p99,
                "n_cells": int(data.size)
            })

    df = pd.DataFrame(rows).sort_values("sum_intensity", ascending=False)
    df.to_csv(TAB_OUT, index=False)

    # plot
    plt.figure(figsize=(10, 5))
    plt.bar(df["chokepoint"], np.log1p(df["sum_intensity"]))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("log(1 + summed AIS positions)")
    plt.title("Chokepoint Activity Proxy from AIS Density Raster")
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=200)
    plt.close()

    print(f"Wrote {TAB_OUT}")
    print(f"Wrote {FIG_OUT}")

if __name__ == "__main__":
    main()
