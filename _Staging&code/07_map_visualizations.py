"""
07_map_visualizations.py
Generate all map-based figures for the main paper body.

Uses cartopy for world maps with AIS density overlays, network graphs,
bounding boxes, and scenario illustrations.

Network: 78 nodes (6 chokepoints, 16 waypoints), ~155 edges.
Edge format supports [a, b] or [a, b, cost_mult].

Outputs (in Figures/generated/):
  - bounding_box_example.png
  - network_world_map.png
  - edge_cost_diagram.png
  - security_shifter_comparison.png
  - congestion_calibration.png
  - route_choice_before_after.png
  - suez_scenario_3panel.png
  - suez_intensity_detail.png
  - suez_density_stats.png
  - closure_impact_map.png
"""

from pathlib import Path
import yaml
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import rasterio
from rasterio.windows import from_bounds
import networkx as nx

import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
TIFF = ROOT / "Data" / "shipdensity_global.tif"
CHOKE_TAB = ROOT / "Tables" / "generated" / "chokepoint_intensity.csv"
SCENARIO_TAB = ROOT / "Tables" / "generated" / "scenario_deltas.csv"
SCENARIO_STATS = ROOT / "Tables" / "generated" / "scenario_summary_stats.csv"
VULN_TAB = ROOT / "Tables" / "generated" / "port_vulnerability.csv"
OUTDIR = ROOT / "Figures" / "generated"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Increased base font sizes globally
plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14})


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


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))


def build_quick_graph(cfg, remove_node=None, extra_mult=None):
    """Build a lightweight NetworkX graph for route computations."""
    G = nx.Graph()
    for name, info in cfg["nodes"].items():
        lon, lat = info["coords"]
        G.add_node(name, lat=lat, lon=lon, kind=info["type"],
                   behind=info.get("behind", None))
    for edge in cfg["edges"]:
        a, b = edge[0], edge[1]
        base_cm = float(edge[2]) if len(edge) > 2 else 1.0
        if a not in G or b not in G:
            continue
        la, loa = G.nodes[a]["lat"], G.nodes[a]["lon"]
        lb, lob = G.nodes[b]["lat"], G.nodes[b]["lon"]
        dist = haversine(la, loa, lb, lob)
        mult = base_cm
        if extra_mult:
            if a in extra_mult:
                mult *= extra_mult[a]
            if b in extra_mult:
                mult *= extra_mult[b]
        G.add_edge(a, b, weight=dist * mult)
    if remove_node and remove_node in G:
        G.remove_node(remove_node)
    return G


def draw_path_segments(ax, path, nodes, color="#e74c3c", linewidth=3, alpha=0.9):
    """Draw a path as individual line segments through waypoints (PlateCarree)."""
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if a in nodes and b in nodes:
            lon_a, lat_a = nodes[a]["coords"]
            lon_b, lat_b = nodes[b]["coords"]
            ax.plot([lon_a, lon_b], [lat_a, lat_b],
                    color=color, linewidth=linewidth, alpha=alpha,
                    transform=ccrs.Geodetic(), zorder=4)


# ========================================================================
# FIGURE 1: Bounding Box Example (Suez Canal)
# ========================================================================
# def make_bounding_box_example():
#     print("  Creating bounding_box_example.png ...")
#     cfg = load_config()
#     bbox = cfg["chokepoints"]["Suez_Canal"]
#     lon_min, lon_max, lat_min, lat_max = bbox

#     pad = 2.0
#     data, extent = read_ais_window(lon_min - pad, lon_max + pad,
#                                     lat_min - pad, lat_max + pad)

#     fig, ax = plt.subplots(1, 1, figsize=(10, 8),
#                            subplot_kw={"projection": ccrs.PlateCarree()})
#     ax.set_extent([lon_min - pad, lon_max + pad,
#                    lat_min - pad, lat_max + pad], ccrs.PlateCarree())
#     ax.add_feature(cfeature.LAND, facecolor="#e8e4d8", edgecolor="gray", linewidth=0.5)
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
#     ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")

#     log_data = np.log1p(data)
#     im = ax.imshow(log_data, extent=extent, origin="upper",
#                    cmap="inferno", alpha=0.85, transform=ccrs.PlateCarree(),
#                    interpolation="bilinear")
#     cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
#     cbar.set_label("log(1 + AIS position count)", fontsize=11)

#     rect = mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
#                                linewidth=2.5, edgecolor="cyan", facecolor="none",
#                                linestyle="--", transform=ccrs.PlateCarree())
#     ax.add_patch(rect)

#     ax.text(lon_min + (lon_max - lon_min) / 2, lat_max + 0.15,
#             "Suez Canal\nBounding Box",
#             ha="center", va="bottom", fontsize=12, fontweight="bold",
#             color="cyan", transform=ccrs.PlateCarree(),
#             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

#     ax.set_title("AIS Ship Density with Chokepoint Bounding Box:\nSuez Canal",
#                  fontsize=14, fontweight="bold")
#     ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

#     plt.tight_layout()
#     plt.savefig(OUTDIR / "bounding_box_example.png", dpi=200, bbox_inches="tight")
#     plt.close()


def make_bounding_box_example():
    print("  Creating bounding_box_example.png ...")
    cfg = load_config()
    bbox = cfg["chokepoints"]["Suez_Canal"]
    lon_min, lon_max, lat_min, lat_max = bbox

    pad = 2.0
    data, extent = read_ais_window(lon_min - pad, lon_max + pad,
                                    lat_min - pad, lat_max + pad)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_min - pad, lon_max + pad,
                   lat_min - pad, lat_max + pad], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#e8e4d8", edgecolor="gray", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")

    log_data = np.log1p(data)

    # --- swap the color scale (reverse the existing colormap) ---
    im = ax.imshow(
        log_data, extent=extent, origin="upper",
        cmap="inferno_r",  # swapped scale vs. inferno
        alpha=0.85, transform=ccrs.PlateCarree(),
        interpolation="bilinear"
    )

    # Slightly more padding so the bar sits a bit farther from the axes
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("log(1 + AIS position count)", fontsize=11)

    rect = mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                               linewidth=2.5, edgecolor="cyan", facecolor="none",
                               linestyle="--", transform=ccrs.PlateCarree())
    ax.add_patch(rect)

    ax.text(lon_min + (lon_max - lon_min) / 2, lat_max + 0.15,
            "Suez Canal\nBounding Box",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="cyan", transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    ax.set_title("AIS Ship Density with Chokepoint Bounding Box:\nSuez Canal",
                 fontsize=14, fontweight="bold")

    # --- prevent label overlap with the colorbar on the right ---
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.right_labels = False   # key fix: removes labels that overlap the colorbar
    gl.top_labels = False     # optional but usually cleaner; remove if you want top labels

    plt.tight_layout()
    plt.savefig(OUTDIR / "bounding_box_example.png", dpi=200, bbox_inches="tight")
    plt.close()



# ========================================================================
# FIGURE 2: Network World Map
# ========================================================================
def make_network_world_map():
    print("  Creating network_world_map.png ...")
    cfg = load_config()
    nodes = cfg["nodes"]
    edges = cfg["edges"]

    n_ports = sum(1 for n in nodes.values() if n["type"] == "port")
    n_choke = sum(1 for n in nodes.values() if n["type"] == "chokepoint")
    n_wp = sum(1 for n in nodes.values() if n["type"] == "waypoint")
    n_edges = len(edges)

    fig, ax = plt.subplots(1, 1, figsize=(18, 10),
                           subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0ece3", edgecolor="gray", linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor="#d4e7f6")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)

    # Draw edges
    for edge in edges:
        a, b = edge[0], edge[1]
        base_cm = float(edge[2]) if len(edge) > 2 else 1.0
        if a in nodes and b in nodes:
            lon_a, lat_a = nodes[a]["coords"]
            lon_b, lat_b = nodes[b]["coords"]
            color = "#cc6633" if base_cm > 1.5 else "#4a90d9"
            style = "--" if base_cm > 1.5 else "-"
            lw = 0.6 if base_cm > 1.5 else 0.9
            ax.plot([lon_a, lon_b], [lat_a, lat_b],
                    color=color, linewidth=lw, alpha=0.5, linestyle=style,
                    transform=ccrs.Geodetic())

    # Draw nodes
    for name, info in nodes.items():
        lon, lat = info["coords"]
        if info["type"] == "chokepoint":
            ax.plot(lon, lat, "D", color="red", markersize=9, markeredgecolor="white",
                    markeredgewidth=0.8, transform=ccrs.PlateCarree(), zorder=5)
            ax.text(lon + 1.5, lat + 1.5, name.replace("_", " "),
                    fontsize=14, fontweight="bold", color="red", #changed from 7 to 14
                    transform=ccrs.PlateCarree(), zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              alpha=0.9, edgecolor="none")) #changed from 0.7 to 0.9
        elif info["type"] == "port":
            ax.plot(lon, lat, "o", color="#2c7bb6", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.3,
                    transform=ccrs.PlateCarree(), zorder=4)
        else:  # waypoint
            ax.plot(lon, lat, "s", color="gray", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.3,
                    transform=ccrs.PlateCarree(), zorder=3)

    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
               markersize=8, label=f"Chokepoint ({n_choke})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c7bb6",
               markersize=6, label=f"Port ({n_ports})"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=5, label=f"Waypoint ({n_wp})"),
        Line2D([0], [0], color="#4a90d9", linewidth=1.5, label="Maritime route"),
        Line2D([0], [0], color="#cc6633", linewidth=1, linestyle="--",
               label="Overland bypass"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10,
              framealpha=0.9)

    ax.set_title(f"Maritime Transport Network: {len(nodes)} Nodes, {n_edges} Edges",
                 fontsize=15, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(OUTDIR / "network_world_map.png", dpi=200, bbox_inches="tight")
    plt.close()


# ========================================================================
# FIGURE 3: Edge Cost Diagram (Suez-Red Sea corridor)
# ========================================================================

#changed fontsize from 8 to 12 and some other parameters
#changed linewidth from 2.5 to 3.5
#changed markersize from 10 to 12
#changed markersize from 8 to 10
#changed markersize from 8 to 10
def make_edge_cost_diagram():
    print("  Creating edge_cost_diagram.png ...")
    cfg = load_config()
    nodes = cfg["nodes"]

    corridor_nodes = ["Port_Said", "Suez_Canal", "Jeddah", "Bab_el_Mandeb",
                      "Aden_Junction", "Djibouti", "Mumbai"]

    lons = [nodes[n]["coords"][0] for n in corridor_nodes if n in nodes]
    lats = [nodes[n]["coords"][1] for n in corridor_nodes if n in nodes]
    pad = 3.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 8),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([min(lons) - pad, max(lons) + pad,
                   min(lats) - pad, max(lats) + pad], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f0ece3", edgecolor="gray", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="#d4e7f6")

    try:
        data, extent = read_ais_window(min(lons) - pad, max(lons) + pad,
                                        min(lats) - pad, max(lats) + pad, max_pixels=600)
        log_data = np.log1p(data)
        ax.imshow(log_data, extent=extent, origin="upper",
                  cmap="inferno", alpha=0.4, transform=ccrs.PlateCarree())
    except Exception:
        pass

    edges_to_draw = [
        ("Port_Said", "Suez_Canal"),
        ("Suez_Canal", "Bab_el_Mandeb"),
        ("Suez_Canal", "Jeddah"),
        ("Jeddah", "Bab_el_Mandeb"),
        ("Bab_el_Mandeb", "Aden_Junction"),
        ("Aden_Junction", "Djibouti"),
        ("Aden_Junction", "Mumbai"),
    ]

    for a, b in edges_to_draw:
        if a in nodes and b in nodes:
            lon_a, lat_a = nodes[a]["coords"]
            lon_b, lat_b = nodes[b]["coords"]
            dist = haversine(lat_a, lon_a, lat_b, lon_b)
            ax.plot([lon_a, lon_b], [lat_a, lat_b],
                    color="#e74c3c", linewidth=3.5, alpha=0.8,
                    transform=ccrs.PlateCarree())
            mid_lon = (lon_a + lon_b) / 2
            mid_lat = (lat_a + lat_b) / 2
            ax.text(mid_lon, mid_lat, f"{dist:.0f} km",
                    fontsize=12, ha="center", va="center",
                    transform=ccrs.PlateCarree(),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow",
                              alpha=0.8, edgecolor="none"))

    for n in corridor_nodes:
        if n in nodes:
            lon, lat = nodes[n]["coords"]
            ntype = nodes[n]["type"]
            if ntype == "chokepoint":
                ax.plot(lon, lat, "D", color="red", markersize=12,
                        markeredgecolor="white", markeredgewidth=1,
                        transform=ccrs.PlateCarree(), zorder=5)
            elif ntype == "waypoint":
                ax.plot(lon, lat, "s", color="orange", markersize=10,
                        markeredgecolor="white", markeredgewidth=1,
                        transform=ccrs.PlateCarree(), zorder=5)
            else:
                ax.plot(lon, lat, "o", color="#2c7bb6", markersize=10,
                        markeredgecolor="white", markeredgewidth=1,
                        transform=ccrs.PlateCarree(), zorder=5)
            ax.text(lon + 0.5, lat + 0.5, n.replace("_", " "),
                    fontsize=9, fontweight="bold",
                    transform=ccrs.PlateCarree(), zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.8, edgecolor="gray"))

    ax.set_title("Edge Cost Illustration: Suez--Red Sea Corridor\n"
                 r"$t_e = \mathrm{dist}_e \times m_e \times \Xi_e^{\lambda}$",
                 fontsize=14, fontweight="bold")
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTDIR / "edge_cost_diagram.png", dpi=200, bbox_inches="tight")
    plt.close()


# ========================================================================
# FIGURE 4: Security Cost Shifter Before/After
# ========================================================================
#changed fontsize from 12 to 16
#changed fontsize from 8 to 12
#changed fontsize from 8 to 12
#changed linewidth from 3.0 to 5.0


def make_security_shifter_comparison():
    print("  Creating security_shifter_comparison.png ...")
    cfg = load_config()
    nodes = cfg["nodes"]

    center_lon, center_lat = 43.3, 12.6
    pad = 5.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             subplot_kw={"projection": ccrs.PlateCarree()})

    for idx, (ax, title, risk_mult) in enumerate(zip(
        axes,
        ["Baseline (High Security)", "Risk Scenario (+100% Premium)"],
        [1.0, 2.0]
    )):
        ax.set_extent([center_lon - pad - 5, center_lon + pad + 15,
                       center_lat - pad - 3, center_lat + pad + 8], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f0ece3", edgecolor="gray", linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor="#d4e7f6")

        corridor = [
            ("Suez_Canal", "Bab_el_Mandeb"),
            ("Bab_el_Mandeb", "Aden_Junction"),
            ("Aden_Junction", "Mumbai"),
            ("Aden_Junction", "Djibouti"),
            ("Aden_Junction", "Mombasa"),
        ]
        for a, b in corridor:
            if a in nodes and b in nodes:
                lon_a, lat_a = nodes[a]["coords"]
                lon_b, lat_b = nodes[b]["coords"]
                dist = haversine(lat_a, lon_a, lat_b, lon_b)
                cost = dist * risk_mult if (a == "Bab_el_Mandeb" or b == "Bab_el_Mandeb") else dist
                width = 5.0 if (a == "Bab_el_Mandeb" or b == "Bab_el_Mandeb") else 1.5
                color = "#e74c3c" if risk_mult > 1 and (a == "Bab_el_Mandeb" or b == "Bab_el_Mandeb") else "#2c7bb6"
                ax.plot([lon_a, lon_b], [lat_a, lat_b],
                        color=color, linewidth=width, alpha=0.8,
                        transform=ccrs.PlateCarree())
                mid_lon = (lon_a + lon_b) / 2
                mid_lat = (lat_a + lat_b) / 2
                ax.text(mid_lon + 0.3, mid_lat - 0.3, f"{cost:.0f}",
                        fontsize=14, ha="center",
                        transform=ccrs.PlateCarree(),
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="yellow" if risk_mult == 1 else "#ffcccc",
                                  alpha=0.8, edgecolor="none"))

        for n in ["Suez_Canal", "Bab_el_Mandeb", "Aden_Junction", "Djibouti", "Mumbai", "Mombasa"]:
            if n in nodes:
                lon, lat = nodes[n]["coords"]
                marker = "D" if nodes[n]["type"] == "chokepoint" else ("s" if nodes[n]["type"] == "waypoint" else "o")
                color = "red" if nodes[n]["type"] == "chokepoint" else ("#2c7bb6" if nodes[n]["type"] == "port" else "orange")
                ax.plot(lon, lat, marker, color=color, markersize=12,
                        markeredgecolor="white", markeredgewidth=0.8,
                        transform=ccrs.PlateCarree(), zorder=5)
                ax.text(lon + 0.5, lat + 0.5, n.replace("_", " "),
                        fontsize=12, fontweight="bold",
                        transform=ccrs.PlateCarree(),
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  alpha=0.8, edgecolor="none"))

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3)

    fig.suptitle("Security Cost Shifter: Impact of Risk Premium on Edge Costs\n"
                 r"$m_e = m_e^{\mathrm{base}} \times (1 + \delta_{\mathrm{risk}} \times \mathrm{Risk}_e)$",
                 fontsize=20, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTDIR / "security_shifter_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


# ========================================================================
# FIGURE 5: Congestion Calibration
# ========================================================================
#changed fontsize from 12 to 16
#changed fontsize from 8 to 12
#changed fontsize from 8 to 12
#changed linewidth from 3.0 to 5.0
#changed markersize from 10 to 12
#changed markersize from 8 to 10
#changed markersize from 8 to 10
def make_congestion_calibration():
    print("  Creating congestion_calibration.png ...")
    if not CHOKE_TAB.exists():
        print("    Skipping: chokepoint_intensity.csv not found")
        return

    df = pd.read_csv(CHOKE_TAB)
    df = df.sort_values("sum_intensity", ascending=False)

    x = np.log1p(df["sum_intensity"].values.astype(float))
    lo, hi = np.nanmin(x), np.nanmax(x)
    z = (x - lo) / (hi - lo + 1e-9)
    mult = 1.0 + 0.5 * z
    df["congestion_mult"] = mult
    df["log_intensity"] = x

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    names = [c.replace("_", " ") for c in df["chokepoint"]]
    ax1.barh(names, df["log_intensity"], color="#2c7bb6", edgecolor="white")
    ax1.set_xlabel("log(1 + Sum AIS Intensity)", fontsize=12)
    ax1.set_title("AIS Traffic Intensity by Chokepoint", fontsize=16, fontweight="bold")
    ax1.invert_yaxis()
    ax1.tick_params(labelsize=16)
    for i, v in enumerate(df["log_intensity"]):
        ax1.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=15)

    ax2 = axes[1]
    ax2.scatter(df["log_intensity"], df["congestion_mult"], s=120, c="#e74c3c",
                edgecolor="white", zorder=5)
    for _, row in df.iterrows():
        ax2.annotate(row["chokepoint"].replace("_", " "),
                     (row["log_intensity"], row["congestion_mult"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=12)

    x_range = np.linspace(lo, hi, 100)
    y_range = 1.0 + 0.5 * (x_range - lo) / (hi - lo + 1e-9)
    ax2.plot(x_range, y_range, "k--", alpha=0.5, label=r"$m = 1 + 0.5 \cdot z$")
    ax2.set_xlabel("log(1 + Sum AIS Intensity)", fontsize=12)
    ax2.set_ylabel("Congestion Multiplier $m_b$", fontsize=12)
    ax2.set_title("Congestion Calibration:\nAIS Density to Cost Multiplier", fontsize=13,
                  fontweight="bold")
    ax2.legend(fontsize=16)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0.95, 1.55)
    ax2.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(OUTDIR / "congestion_calibration.png", dpi=200, bbox_inches="tight")
    plt.close()


# ========================================================================
# FIGURE 6: Route Choice Before/After (uses waypoint-based paths)
# ========================================================================
#changed fontsize from 12 to 16
#changed fontsize from 8 to 12
#changed fontsize from 8 to 12
#changed linewidth from 3.0 to 5.0
#changed markersize from 10 to 12
#changed markersize from 8 to 10
#changed markersize from 8 to 10


def make_route_choice_before_after():
    print("  Creating route_choice_before_after.png ...")
    cfg = load_config()
    nodes = cfg["nodes"]

    G_base = build_quick_graph(cfg)
    G_shock = build_quick_graph(cfg, remove_node="Suez_Canal")

    try:
        path_base = nx.shortest_path(G_base, "Shanghai", "Rotterdam", weight="weight")
        path_shock = nx.shortest_path(G_shock, "Shanghai", "Rotterdam", weight="weight")
        cost_base = nx.shortest_path_length(G_base, "Shanghai", "Rotterdam", weight="weight")
        cost_shock = nx.shortest_path_length(G_shock, "Shanghai", "Rotterdam", weight="weight")
    except nx.NetworkXNoPath:
        print("    No path found, skipping")
        return

    print(f"    Baseline path: {' -> '.join(path_base)}")
    print(f"    Rerouted path: {' -> '.join(path_shock)}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             subplot_kw={"projection": ccrs.Robinson()})

    for ax, path, title, cost in zip(
        axes,
        [path_base, path_shock],
        ["Baseline Route", "Suez Closure: Rerouted via Cape"],
        [cost_base, cost_shock]
    ):
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="#f0ece3", edgecolor="gray", linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor="#d4e7f6")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)

        # Draw all edges faintly
        for edge in cfg["edges"]:
            a, b = edge[0], edge[1]
            if a in nodes and b in nodes:
                lon_a, lat_a = nodes[a]["coords"]
                lon_b, lat_b = nodes[b]["coords"]
                ax.plot([lon_a, lon_b], [lat_a, lat_b],
                        color="gray", linewidth=0.3, alpha=0.2,
                        transform=ccrs.Geodetic())

        # Draw the chosen path in bold (segment by segment through waypoints)
        draw_path_segments(ax, path, nodes, color="#e74c3c", linewidth=3, alpha=0.9)

        # Mark origin and destination
        for n, label in [("Shanghai", "Shanghai"), ("Rotterdam", "Rotterdam")]:
            lon, lat = nodes[n]["coords"]
            ax.plot(lon, lat, "*", color="gold", markersize=20,
                    markeredgecolor="black", markeredgewidth=0.8,
                    transform=ccrs.PlateCarree(), zorder=10)
            ax.text(lon + 3, lat + 3, label, fontsize=15, fontweight="bold",
                    transform=ccrs.PlateCarree(),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.8, edgecolor="none"))

        # Mark chokepoints and waypoints on path
        for n in path:
            if n in nodes:
                ntype = nodes[n]["type"]
                lon, lat = nodes[n]["coords"]
                if ntype == "chokepoint":
                    ax.plot(lon, lat, "D", color="red", markersize=9,
                            markeredgecolor="white", markeredgewidth=0.5,
                            transform=ccrs.PlateCarree(), zorder=8)
                elif ntype == "waypoint":
                    ax.plot(lon, lat, "s", color="orange", markersize=8,
                            markeredgecolor="white", markeredgewidth=0.5,
                            transform=ccrs.PlateCarree(), zorder=7)

        ax.set_title(f"{title}\nCost: {cost:,.0f} km-equiv.", fontsize=20, fontweight="bold")

    delta = cost_shock - cost_base
    fig.suptitle(f"Route Choice: Shanghai to Rotterdam\n"
                 f"Suez closure adds {delta:,.0f} km ({delta/cost_base*100:.1f}%) to route cost",
                 fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTDIR / "route_choice_before_after.png", dpi=200, bbox_inches="tight")
    plt.close()


# ========================================================================
# FIGURE 7: Suez Canal 3-Panel Scenario Illustration
# ========================================================================
#changed fontsize from 12 to 16
#changed fontsize from 8 to 12
#changed fontsize from 8 to 12
#changed linewidth from 3.0 to 5.0
#changed markersize from 10 to 12
#changed markersize from 8 to 10
#changed markersize from 8 to 10

def make_suez_scenario_3panel():
    print("  Creating suez_scenario_3panel.png ...")
    cfg = load_config()
    nodes = cfg["nodes"]

    scenarios = [
        ("Full Closure", "Suez_Canal", None),
        ("Partial Degradation (3x)", None, {"Suez_Canal": 3.0}),
        ("Risk Premium (+100%)", None, {"Suez_Canal": 2.0}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                             subplot_kw={"projection": ccrs.PlateCarree()})

    region = [25, 60, 5, 38]
    gulf_ports = ["Port_Said", "Jeddah", "Djibouti", "Piraeus"]

    for ax, (title, remove, extra) in zip(axes, scenarios):
        ax.set_extent(region, ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f0ece3", edgecolor="gray", linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor="#d4e7f6")

        G = build_quick_graph(cfg, remove_node=remove, extra_mult=extra)

        # Draw edges in region
        for edge in cfg["edges"]:
            a, b = edge[0], edge[1]
            base_cm = float(edge[2]) if len(edge) > 2 else 1.0
            if a in nodes and b in nodes:
                lon_a, lat_a = nodes[a]["coords"]
                lon_b, lat_b = nodes[b]["coords"]
                if (region[0] - 5 <= lon_a <= region[1] + 5 and
                    region[0] - 5 <= lon_b <= region[1] + 5):
                    if G.has_node(a) and G.has_node(b) and G.has_edge(a, b):
                        color = "#2c7bb6"
                        width = 3.5
                        style = "-"
                        if base_cm > 1.5:
                            color = "#cc6633"
                            style = "--"
                        if extra and ("Suez_Canal" in [a, b]):
                            color = "#e74c3c"
                            width = 4.5
                        ax.plot([lon_a, lon_b], [lat_a, lat_b],
                                color=color, linewidth=width, alpha=0.7,
                                linestyle=style, transform=ccrs.PlateCarree())
                    else:
                        ax.plot([lon_a, lon_b], [lat_a, lat_b],
                                color="gray", linewidth=0.5, alpha=0.3,
                                linestyle="--", transform=ccrs.PlateCarree())

        # Draw Suez Canal
        if "Suez_Canal" in nodes:
            lon, lat = nodes["Suez_Canal"]["coords"]
            if remove == "Suez_Canal":
                ax.plot(lon, lat, "X", color="red", markersize=25,
                        markeredgecolor="white", markeredgewidth=1.5,
                        transform=ccrs.PlateCarree(), zorder=10)
            else:
                ax.plot(lon, lat, "D", color="red", markersize=15,
                        markeredgecolor="white", markeredgewidth=1,
                        transform=ccrs.PlateCarree(), zorder=10)

        # Draw nearby ports
        for p in gulf_ports:
            if p in nodes:
                lon, lat = nodes[p]["coords"]
                ax.plot(lon, lat, "o", color="#2c7bb6", markersize=12,
                        markeredgecolor="white", markeredgewidth=0.5,
                        transform=ccrs.PlateCarree(), zorder=8)
                ax.text(lon + 0.3, lat + 0.3, p.replace("_", " "),
                        fontsize=7, transform=ccrs.PlateCarree())

        # Status text
        if remove:
            status = "REROUTED VIA BYPASS"
            status_color = "#d7191c"
        elif extra:
            mult_val = extra.get("Suez_Canal", 1.0)
            status = f"Costs x{mult_val:.0f}"
            status_color = "#e74c3c"
        else:
            status = "Normal"
            status_color = "#2c7bb6"

        ax.text(0.5, 0.02, status, ha="center", va="bottom",
                fontsize=22, fontweight="bold", color="white",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=status_color, alpha=0.8))

        ax.set_title(title, fontsize=23, fontweight="bold")
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3)

    fig.suptitle("Suez Canal: Three Disruption Scenarios",
                 fontsize=25, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTDIR / "suez_scenario_3panel.png", dpi=200, bbox_inches="tight")
    plt.close()


# ========================================================================
# FIGURE 8: Suez Canal AIS Intensity Detail
# ========================================================================
# def make_suez_intensity_detail():
#     print("  Creating suez_intensity_detail.png ...")
#     cfg = load_config()
#     bbox = cfg["chokepoints"]["Suez_Canal"]
#     lon_min, lon_max, lat_min, lat_max = bbox

#     pad = 0.5
#     data, extent = read_ais_window(lon_min - pad, lon_max + pad,
#                                     lat_min - pad, lat_max + pad, max_pixels=600)

#     fig, ax = plt.subplots(1, 1, figsize=(10, 8),
#                            subplot_kw={"projection": ccrs.PlateCarree()})
#     ax.set_extent([lon_min - pad, lon_max + pad,
#                    lat_min - pad, lat_max + pad], ccrs.PlateCarree())
#     ax.add_feature(cfeature.LAND, facecolor="#e8e4d8", edgecolor="gray", linewidth=0.5)
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

#     log_data = np.log1p(data)
#     im = ax.imshow(log_data, extent=extent, origin="upper",
#                    cmap="hot", alpha=0.9, transform=ccrs.PlateCarree(),
#                    interpolation="bilinear")
#     cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
#     cbar.set_label("log(1 + AIS position count)", fontsize=11)

#     if not np.all(np.isnan(data)):
#         flat = data.flatten()
#         valid = flat[~np.isnan(flat)]
#         if len(valid) > 0:
#             p99 = np.percentile(valid, 99)
#             ax.text(0.02, 0.02, f"P99 intensity: {p99:,.0f}\nMax: {np.nanmax(data):,.0f}",
#                     fontsize=10, transform=ax.transAxes, va="bottom",
#                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

#     ax.set_title("AIS Ship Density: Suez Canal (Detail)",
#                  fontsize=14, fontweight="bold")
#     ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

#     plt.tight_layout()
#     plt.savefig(OUTDIR / "suez_intensity_detail.png", dpi=200, bbox_inches="tight")
#     plt.close()

def make_suez_intensity_detail():
    print("  Creating suez_intensity_detail.png ...")
    cfg = load_config()
    bbox = cfg["chokepoints"]["Suez_Canal"]
    lon_min, lon_max, lat_min, lat_max = bbox

    pad = 0.5
    data, extent = read_ais_window(lon_min - pad, lon_max + pad,
                                    lat_min - pad, lat_max + pad, max_pixels=600)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_min - pad, lon_max + pad,
                   lat_min - pad, lat_max + pad], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#e8e4d8", edgecolor="gray", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    log_data = np.log1p(data)

    # --- swap the color scale (reverse the existing colormap) ---
    im = ax.imshow(
        log_data, extent=extent, origin="upper",
        cmap="hot_r",  # swapped scale vs. hot
        alpha=0.9, transform=ccrs.PlateCarree(),
        interpolation="bilinear"
    )

    # a bit more padding so the colorbar sits farther right
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("log(1 + AIS position count)", fontsize=11)

    if not np.all(np.isnan(data)):
        flat = data.flatten()
        valid = flat[~np.isnan(flat)]
        if len(valid) > 0:
            p99 = np.percentile(valid, 99)
            ax.text(0.02, 0.02, f"P99 intensity: {p99:,.0f}\nMax: {np.nanmax(data):,.0f}",
                    fontsize=10, transform=ax.transAxes, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_title("AIS Ship Density: Suez Canal (Detail)",
                 fontsize=14, fontweight="bold")

    # --- prevent lat/lon labels from overlapping the colorbar on the right ---
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False  # keeps things clean; remove if you want top labels

    plt.tight_layout()
    plt.savefig(OUTDIR / "suez_intensity_detail.png", dpi=200, bbox_inches="tight")
    plt.close()

# ========================================================================
# FIGURE 9: Suez Canal Density Stats (4-panel)
# ========================================================================
def make_suez_density_stats():
    print("  Creating suez_density_stats.png ...")
    cfg = load_config()
    bbox = cfg["chokepoints"]["Suez_Canal"]
    lon_min, lon_max, lat_min, lat_max = bbox

    data, extent = read_ais_window(lon_min, lon_max, lat_min, lat_max, max_pixels=600)
    flat = data.flatten()
    valid = flat[~np.isnan(flat)]

    if len(valid) == 0:
        print("    No valid data, skipping")
        return

    choke_df = pd.read_csv(CHOKE_TAB) if CHOKE_TAB.exists() else None

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))

    # # Panel A: Spatial intensity map
    # ax = axes[0, 0]
    # log_data = np.log1p(data)
    # im = ax.imshow(log_data, cmap="hot", aspect="auto", interpolation="bilinear")
    # ax.set_title("(a) Spatial Intensity Map", fontsize=16, fontweight="bold")
    # cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    # cbar.set_label("log(1 + count)", fontsize=14)
    # ax.set_xlabel("Longitude", fontsize=24)
    # ax.set_ylabel("Latitude", fontsize=24)
    # ax.tick_params(labelsize=9)

    # Panel A: Spatial intensity map
    ax = axes[0, 0]

    log_data = np.log1p(data)

    # --- Improve contrast using percentile scaling ---
    vmin = np.percentile(log_data[~np.isnan(log_data)], 5)
    vmax = np.percentile(log_data[~np.isnan(log_data)], 99)

    im = ax.imshow(
        log_data,
        cmap="hot_r",
        aspect="auto",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title("(a) Spatial Intensity Map", fontsize=16, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log(1 + count)", fontsize=24)

    ax.set_xlabel("Longitude", fontsize=24)
    ax.set_ylabel("Latitude", fontsize=24)
    ax.tick_params(labelsize=16)

    # Panel B: Cell distribution histogram
    ax = axes[0, 1]
    log_valid = np.log1p(valid)
    ax.hist(log_valid, bins=80, color="#2c7bb6", edgecolor="white", alpha=0.8)
    ax.axvline(np.log1p(np.mean(valid)), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(valid):,.0f}")
    ax.axvline(np.log1p(np.percentile(valid, 99)), color="orange", linestyle="--", linewidth=2,
               label=f"P99: {np.percentile(valid, 99):,.0f}")
    ax.set_xlabel("log(1 + AIS count per cell)", fontsize=14)
    ax.set_ylabel("Number of cells", fontsize=14)
    ax.set_title("(b) Cell-Level Intensity Distribution", fontsize=16, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    # Panel C: Mean vs P99 comparison across chokepoints
    ax = axes[1, 0]
    if choke_df is not None:
        choke_sorted = choke_df.sort_values("mean_intensity", ascending=True)
        names = [c.replace("_", " ") for c in choke_sorted["chokepoint"]]
        y = range(len(names))
        ax.barh(y, choke_sorted["mean_intensity"], height=0.4, label="Mean",
                color="#2c7bb6", alpha=0.8)
        ax.barh([yi + 0.4 for yi in y], choke_sorted["p99_cell"], height=0.4,
                label="P99", color="#e74c3c", alpha=0.8)
        ax.set_yticks([yi + 0.2 for yi in y])
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("AIS Position Count", fontsize=14)
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        # Highlight Suez Canal
        if "Suez_Canal" in list(choke_sorted["chokepoint"]):
            suez_idx = list(choke_sorted["chokepoint"]).index("Suez_Canal")
            ax.barh(suez_idx, choke_sorted.iloc[suez_idx]["mean_intensity"],
                    height=0.4, color="gold", edgecolor="red", linewidth=2)
    ax.set_title("(c) Mean vs P99 Across Chokepoints", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=9)

    # Panel D: Summary statistics (LARGER)
    ax = axes[1, 1]
    ax.axis("off")
    stats_text = (
        f"Suez Canal\n"
        f"Descriptive Statistics\n"
        f"{'─' * 44}\n"
        f"Bounding box:     [{lon_min}, {lon_max}] x [{lat_min}, {lat_max}]\n"
        f"Grid cells:       {len(valid):,}\n"
        f"Total intensity:  {np.sum(valid):,.0f}\n"
        f"Mean per cell:    {np.mean(valid):,.1f}\n"
        f"Median per cell:  {np.median(valid):,.1f}\n"
        f"Std dev:          {np.std(valid):,.1f}\n"
        f"P10:              {np.percentile(valid, 10):,.0f}\n"
        f"P50:              {np.percentile(valid, 50):,.0f}\n"
        f"P90:              {np.percentile(valid, 90):,.0f}\n"
        f"P99:              {np.percentile(valid, 99):,.0f}\n"
        f"Max:              {np.max(valid):,.0f}\n"
        f"Cells > 0:        {np.sum(valid > 0):,} ({np.sum(valid > 0)/len(valid)*100:.1f}%)"
    )
    ax.text(0.05, 0.95, stats_text, fontsize=18, family="monospace",
            va="top", ha="left", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                      edgecolor="gray"))
    ax.set_title("(d) Summary Statistics", fontsize=16, fontweight="bold")

    fig.suptitle("AIS Density Analysis: Suez Canal",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTDIR / "suez_density_stats.png", dpi=200, bbox_inches="tight")
    plt.close()


# ========================================================================
# FIGURE 10: Closure Impact World Map
# ========================================================================
def make_closure_impact_map():
    print("  Creating closure_impact_map.png ...")
    cfg = load_config()
    nodes = cfg["nodes"]

    if not VULN_TAB.exists():
        print("    Skipping: port_vulnerability.csv not found")
        return

    vuln_df = pd.read_csv(VULN_TAB)

    top_chokes = ["Panama_Canal", "Gibraltar", "Suez_Canal"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                             subplot_kw={"projection": ccrs.Robinson()})

    for ax, choke in zip(axes, top_chokes):
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="#f0ece3", edgecolor="gray", linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor="#d4e7f6")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)

        sub = vuln_df[vuln_df["chokepoint"] == choke]
        port_impact = dict(zip(sub["port"], sub["mean_pct_increase"]))
        port_disconn = dict(zip(sub["port"], sub["pairs_disconnected"]))

        max_val = max(port_impact.values()) if port_impact else 1
        if max_val == 0:
            max_val = 1

        for port, info in nodes.items():
            if info["type"] != "port":
                continue
            lon, lat = info["coords"]
            impact = port_impact.get(port, 0)
            disconn = port_disconn.get(port, 0)

            if disconn > 0:
                color = "black"
                size = 60
            elif impact > 0:
                norm_val = min(impact / max_val, 1.0)
                color = plt.cm.YlOrRd(norm_val)
                size = 20 + 40 * norm_val
            else:
                color = "#aaaaaa"
                size = 10

            ax.scatter(lon, lat, c=[color], s=size, edgecolors="white",
                       linewidths=0.3, transform=ccrs.PlateCarree(), zorder=5)

        if choke in nodes:
            lon, lat = nodes[choke]["coords"]
            ax.plot(lon, lat, "X", color="red", markersize=15,
                    markeredgecolor="white", markeredgewidth=1.5,
                    transform=ccrs.PlateCarree(), zorder=10)

        ax.set_title(f"{choke.replace('_', ' ')} Closure", fontsize=13, fontweight="bold")

    fig.suptitle("Port Vulnerability Under Chokepoint Closure\n"
                 "(color = % cost increase; black = disconnected; X = closed chokepoint)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTDIR / "closure_impact_map.png", dpi=200, bbox_inches="tight")
    plt.close()


# ========================================================================
# MAIN
# ========================================================================
def main():
    print("=" * 60)
    print("GENERATING MAP VISUALIZATIONS")
    print("=" * 60)

    make_bounding_box_example()
    make_network_world_map()
    make_edge_cost_diagram()
    make_security_shifter_comparison()
    make_congestion_calibration()
    make_route_choice_before_after()
    make_suez_scenario_3panel()
    make_suez_intensity_detail()
    make_suez_density_stats()
    make_closure_impact_map()

    print("\n" + "=" * 60)
    print("ALL MAP VISUALIZATIONS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
