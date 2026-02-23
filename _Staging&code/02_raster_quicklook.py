"""
02_raster_quicklook.py
Creates memory-safe "quicklook" figures from a massive GeoTIFF (shipdensity_global.tif).
Outputs:
- Figures/generated/global_quicklook.png
- Tables/generated/raster_summary.csv

Notes:
- This script prefers built-in overviews (.ovr) if present.
- It reads a low-resolution version using out_shape to avoid loading the full raster.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
FIG_OUT = ROOT / "Figures" / "generated" / "global_quicklook.png"
TAB_OUT = ROOT / "Tables" / "generated" / "raster_summary.csv"

TIF = DATA / "shipdensity_global.tif"

def main():
    if not TIF.exists():
        raise FileNotFoundError(f"Missing {TIF}. Put the big GeoTIFF in ./Data/")

    with rasterio.open(TIF) as src:
        # Aggressive downsample target (tune if needed)
        out_h, out_w = 1800, 3600
        arr = src.read(1, out_shape=(out_h, out_w), masked=True, resampling=rasterio.enums.Resampling.average)
        arr = np.ma.filled(arr.astype("float64"), np.nan)

        summary = {
            "crs": str(src.crs),
            "bounds_left": src.bounds.left,
            "bounds_right": src.bounds.right,
            "bounds_bottom": src.bounds.bottom,
            "bounds_top": src.bounds.top,
            "nodata": src.nodata,
            "sample_shape_h": out_h,
            "sample_shape_w": out_w,
            "nan_share": float(np.isnan(arr).mean()),
            "min": float(np.nanmin(arr)),
            "p50": float(np.nanpercentile(arr, 50)),
            "p90": float(np.nanpercentile(arr, 90)),
            "p99": float(np.nanpercentile(arr, 99)),
            "max": float(np.nanmax(arr)),
        }
        pd.DataFrame([summary]).to_csv(TAB_OUT, index=False)

        # Plot with log scaling
        plot = np.log1p(arr)
        plt.figure(figsize=(12, 6))
        plt.imshow(plot, origin="upper", aspect="auto",
                   extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
        plt.title("Global AIS Ship Density (log1p quicklook)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(label="log(1 + AIS positions)")
        plt.tight_layout()
        plt.savefig(FIG_OUT, dpi=200)
        plt.close()

    print(f"Wrote {FIG_OUT}")
    print(f"Wrote {TAB_OUT}")

if __name__ == "__main__":
    main()
