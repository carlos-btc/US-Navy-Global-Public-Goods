"""
08_Appendix_figures.py
Generate all appendix figures:

Appendix A: Chokepoint Bounding Box Maps
  - appendix_all_bboxes_world.png     (world map with all 6 bounding boxes)
  - appendix_bbox_[name].png x 6      (individual zoomed maps)

Appendix B: Chokepoint Traffic Density Close-ups
  - appendix_density_[name].png x 6   (AIS density overlay for each chokepoint)

Appendix C: Chokepoint Descriptive Statistics
  - appendix_stats_[name].png x 6     (4-panel stats for each chokepoint)

Note: Danish Straits and Cape of Good Hope are excluded
(Cape is a waypoint, Danish is removed from analysis).
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.windows import from_bounds

import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
TIFF = ROOT / "Data" / "shipdensity_global.tif"
CHOKE_TAB = ROOT / "Tables" / "generated" / "chokepoint_intensity.csv"
OUTDIR = ROOT / "Figures" / "generated"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Increased base font sizes globally
plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 15})


def load_config():
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_ais_window(lon_min, lon_max, lat_min, lat_max, max_pixels=800):
    """Read a windowed portion of the AIS raster, downsampled."""
    with rasterio.open(TIFF) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        h = max(1, int(window.height))
        w = max(1, int(window.width))
        scale = max(1, max(h, w) // max_pixels)
        out_h = max(1, h // scale)
        out_w = max(1, w // scale)
        data = src.read(1, window=window, out_shape=(out_h, out_w))
        data = data.astype("float64")
        data[data <= 0] = np.nan
        extent = [lon_min, lon_max, lat_min, lat_max]
        return data, extent


NICE_NAMES = {
    "Suez_Canal": "Suez Canal",
    "Bab_el_Mandeb": "Bab el-Mandeb",
    "Strait_of_Malacca": "Strait of Malacca",
    "Panama_Canal": "Panama Canal",
    "Bosporus": "Bosporus",
    "Gibraltar": "Strait of Gibraltar",
}

# Padding for zoomed views
ZOOM_PAD = {
    "Suez_Canal": 1.0,
    "Bab_el_Mandeb": 1.5,
    "Strait_of_Malacca": 2.0,
    "Panama_Canal": 1.0,
    "Bosporus": 0.5,
    "Gibraltar": 1.0,
}


# ========================================================================
# APPENDIX A: ALL BOUNDING BOXES ON WORLD MAP
# ========================================================================
def make_all_bboxes_world():
    print("  Creating appendix_all_bboxes_world.png ...")
    cfg = load_config()
    chokepoints = cfg["chokepoints"]

    fig, ax = plt.subplots(1, 1, figsize=(16, 9),
                           subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0ece3", edgecolor="gray", linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor="#d4e7f6")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, len(chokepoints)))

    for (name, bbox), color in zip(chokepoints.items(), colors):
        lon_min, lon_max, lat_min, lat_max = bbox
        rect = mpatches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(rect)
        cx = (lon_min + lon_max) / 2
        cy = lat_max + 1.5
        ax.text(cx, cy, NICE_NAMES.get(name, name),
                fontsize=8, ha="center", fontweight="bold",
                color=color, transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.7, edgecolor="none"))

    ax.set_title("All Six Chokepoint Bounding Boxes",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTDIR / "appendix_all_bboxes_world.png", dpi=200, bbox_inches="tight")
    plt.close()


# ========================================================================
# APPENDIX A: INDIVIDUAL BOUNDING BOX MAPS
# ========================================================================
def make_individual_bbox_maps():
    cfg = load_config()
    chokepoints = cfg["chokepoints"]

    for name, bbox in chokepoints.items():
        print(f"  Creating appendix_bbox_{name}.png ...")
        lon_min, lon_max, lat_min, lat_max = bbox
        pad = ZOOM_PAD.get(name, 1.5)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                               subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent([lon_min - pad, lon_max + pad,
                       lat_min - pad, lat_max + pad], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#e8e4d8", edgecolor="gray", linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor="#d4e7f6")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")

        try:
            data, extent = read_ais_window(lon_min - pad, lon_max + pad,
                                            lat_min - pad, lat_max + pad, max_pixels=600)
            log_data = np.log1p(data)
            ax.imshow(log_data, extent=extent, origin="upper",
                      cmap="inferno", alpha=0.5, transform=ccrs.PlateCarree(),
                      interpolation="bilinear")
        except Exception:
            pass

        rect = mpatches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=2.5, edgecolor="cyan", facecolor="none", linestyle="--",
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(rect)

        ax.text(lon_min, lat_min - 0.1, f"({lon_min}, {lat_min})",
                fontsize=8, ha="left", va="top", color="cyan",
                transform=ccrs.PlateCarree())
        ax.text(lon_max, lat_max + 0.1, f"({lon_max}, {lat_max})",
                fontsize=8, ha="right", va="bottom", color="cyan",
                transform=ccrs.PlateCarree())

        ax.set_title(f"Bounding Box: {NICE_NAMES.get(name, name)}\n"
                     f"[{lon_min}, {lon_max}] x [{lat_min}, {lat_max}]",
                     fontsize=14, fontweight="bold")
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        plt.tight_layout()
        plt.savefig(OUTDIR / f"appendix_bbox_{name}.png", dpi=200, bbox_inches="tight")
        plt.close()


# ========================================================================
# APPENDIX B: DENSITY CLOSE-UPS
# ========================================================================
def make_density_closeups():
    cfg = load_config()
    chokepoints = cfg["chokepoints"]

    for name, bbox in chokepoints.items():
        print(f"  Creating appendix_density_{name}.png ...")
        lon_min, lon_max, lat_min, lat_max = bbox
        pad = ZOOM_PAD.get(name, 1.0) * 0.5

        try:
            data, extent = read_ais_window(lon_min - pad, lon_max + pad,
                                            lat_min - pad, lat_max + pad, max_pixels=800)
        except Exception as e:
            print(f"    Skipping {name}: {e}")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                               subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent([lon_min - pad, lon_max + pad,
                       lat_min - pad, lat_max + pad], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#e8e4d8", edgecolor="gray", linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        log_data = np.log1p(data)
        im = ax.imshow(log_data, extent=[lon_min - pad, lon_max + pad,
                                          lat_min - pad, lat_max + pad],
                       origin="upper", cmap="hot", alpha=0.9,
                       transform=ccrs.PlateCarree(), interpolation="bilinear")
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("log(1 + AIS position count)", fontsize=11)

        rect = mpatches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=2, edgecolor="cyan", facecolor="none", linestyle="--",
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(rect)

        valid = data.flatten()
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            ax.text(0.02, 0.02,
                    f"Cells: {len(valid):,}\n"
                    f"Max: {np.max(valid):,.0f}\n"
                    f"Mean: {np.mean(valid):,.0f}\n"
                    f"P99: {np.percentile(valid, 99):,.0f}",
                    fontsize=10, transform=ax.transAxes, va="bottom",
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(f"AIS Ship Density: {NICE_NAMES.get(name, name)}",
                     fontsize=14, fontweight="bold")
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        plt.tight_layout()
        plt.savefig(OUTDIR / f"appendix_density_{name}.png", dpi=200, bbox_inches="tight")
        plt.close()


# ========================================================================
# APPENDIX C: PER-CHOKEPOINT DESCRIPTIVE STATISTICS
# ========================================================================
def make_stats_panels():
    cfg = load_config()
    chokepoints = cfg["chokepoints"]
    choke_df = pd.read_csv(CHOKE_TAB) if CHOKE_TAB.exists() else None

    for name, bbox in chokepoints.items():
        print(f"  Creating appendix_stats_{name}.png ...")
        lon_min, lon_max, lat_min, lat_max = bbox

        try:
            data, extent = read_ais_window(lon_min, lon_max, lat_min, lat_max, max_pixels=600)
        except Exception:
            continue

        flat = data.flatten()
        valid = flat[~np.isnan(flat)]
        if len(valid) == 0:
            continue

        # Use larger figure and give panel (d) more room
        fig = plt.figure(figsize=(16, 13))
        # Custom grid: top row normal, bottom row gives (d) extra width
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.3, wspace=0.3)

        # Panel A: Spatial intensity
        ax = fig.add_subplot(gs[0, 0])
        log_data = np.log1p(data)
        im = ax.imshow(log_data, cmap="hot", aspect="auto", interpolation="bilinear")
        ax.set_title("(a) Spatial Intensity", fontsize=18, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8, label="log(1 + count)")
        ax.tick_params(labelsize=13)

        # Panel B: Histogram
        ax = fig.add_subplot(gs[0, 1])
        log_valid = np.log1p(valid)
        ax.hist(log_valid, bins=60, color="#2c7bb6", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(log_valid), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {np.mean(valid):,.0f}")
        ax.axvline(np.log1p(np.percentile(valid, 99)), color="orange",
                   linestyle="--", linewidth=2,
                   label=f"P99: {np.percentile(valid, 99):,.0f}")
        ax.set_xlabel("log(1 + count)", fontsize=15)
        ax.set_ylabel("Cells", fontsize=15)
        ax.set_title("(b) Cell Distribution", fontsize=18, fontweight="bold")
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=13)

        # Panel C: Cumulative distribution
        ax = fig.add_subplot(gs[1, 0])
        sorted_vals = np.sort(valid)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(np.log1p(sorted_vals), cdf, color="#e74c3c", linewidth=1.5)
        ax.set_xlabel("log(1 + AIS count)", fontsize=15)
        ax.set_ylabel("Cumulative fraction", fontsize=15)
        ax.set_title("(c) Cumulative Distribution", fontsize=18, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.axhline(0.99, color="gray", linestyle=":", linewidth=1, label="P99")
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=13)

        # Panel D: Summary stats (LARGER TEXT)
        ax = fig.add_subplot(gs[1, 1])
        ax.axis("off")
        nice = NICE_NAMES.get(name, name)
        stats_text = (
            f"{nice}\n"
            f"{'─' * 44}\n"
            f"Bounding box: [{lon_min}, {lon_max}] x [{lat_min}, {lat_max}]\n"
            f"Grid cells:   {len(valid):,}\n"
            f"Total:        {np.sum(valid):,.0f}\n"
            f"Mean:         {np.mean(valid):,.1f}\n"
            f"Median:       {np.median(valid):,.1f}\n"
            f"Std:          {np.std(valid):,.1f}\n"
            f"Min:          {np.min(valid):,.0f}\n"
            f"P10:          {np.percentile(valid, 10):,.0f}\n"
            f"P50:          {np.percentile(valid, 50):,.0f}\n"
            f"P90:          {np.percentile(valid, 90):,.0f}\n"
            f"P99:          {np.percentile(valid, 99):,.0f}\n"
            f"Max:          {np.max(valid):,.0f}"
        )
        ax.text(0.05, 0.95, stats_text, fontsize=22, family="monospace",
                va="top", ha="left", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                          edgecolor="gray"))
        ax.set_title("(d) Summary Statistics", fontsize=18, fontweight="bold")

        fig.suptitle(f"Chokepoint Analysis: {nice}",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.savefig(OUTDIR / f"appendix_stats_{name}.png", dpi=200, bbox_inches="tight")
        plt.close()


# ========================================================================
# MAIN
# ========================================================================
def main():
    print("=" * 60)
    print("GENERATING APPENDIX FIGURES")
    print("=" * 60)

    print("\n--- Appendix A: Bounding Box Maps ---")
    make_all_bboxes_world()
    make_individual_bbox_maps()

    print("\n--- Appendix B: Density Close-ups ---")
    make_density_closeups()

    print("\n--- Appendix C: Descriptive Statistics ---")
    make_stats_panels()

    print("\n" + "=" * 60)
    print("ALL APPENDIX FIGURES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
