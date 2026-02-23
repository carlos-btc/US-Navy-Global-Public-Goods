"""
04_scenario_engine.py
Maritime transport network scenario engine (70-node expanded network).

Reads node/edge definitions from config.yaml.
Computes all port-to-port shortest paths (1,540 OD pairs from 56 ports).
Runs full-closure scenarios for each of the 7 chokepoints.

Edge format supports optional cost multiplier: [a, b] or [a, b, cost_mult].

Outputs:
- Tables/generated/scenario_deltas.csv
- Tables/generated/scenario_summary_stats.csv
- Figures/generated/scenario_summary.png
- Figures/generated/scenario_example.png
"""

from pathlib import Path
import math
import yaml
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
CHOKE_TAB = ROOT / "Tables" / "generated" / "chokepoint_intensity.csv"

OUT_TAB = ROOT / "Tables" / "generated" / "scenario_deltas.csv"
OUT_STATS = ROOT / "Tables" / "generated" / "scenario_summary_stats.csv"
FIG_SUM = ROOT / "Figures" / "generated" / "scenario_summary.png"
FIG_EX = ROOT / "Figures" / "generated" / "scenario_example.png"

for p in [OUT_TAB, OUT_STATS, FIG_SUM, FIG_EX]:
    p.parent.mkdir(parents=True, exist_ok=True)


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_config():
    """Load network definition from config.yaml."""
    with open(CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_chokepoint_factors():
    """Load congestion multipliers from AIS intensity data."""
    factors = {}
    if CHOKE_TAB.exists():
        df = pd.read_csv(CHOKE_TAB)
        x = np.log1p(df["sum_intensity"].values.astype(float))
        if np.isfinite(x).any():
            lo, hi = np.nanmin(x), np.nanmax(x)
            z = (x - lo) / (hi - lo + 1e-9)
            mult = 1.0 + 0.5 * z
            for name, m in zip(df["chokepoint"], mult):
                factors[name] = float(m)
    return factors


def build_graph(cfg, choke_factor, extra_multipliers=None):
    """Build the maritime network graph from config.yaml.

    Parameters
    ----------
    cfg : dict
        Parsed config.yaml with 'nodes' and 'edges'.
    choke_factor : dict
        AIS-based congestion multipliers per chokepoint.
    extra_multipliers : dict or None
        Additional cost multipliers on edges touching specific nodes
        (used for partial degradation scenarios).

    Returns
    -------
    G : nx.Graph
    ports : list of str
    chokepoint_names : list of str
    """
    if extra_multipliers is None:
        extra_multipliers = {}

    nodes = cfg["nodes"]
    edges = cfg["edges"]

    G = nx.Graph()

    # Add nodes
    for name, info in nodes.items():
        lon, lat = info["coords"]
        G.add_node(name, lat=lat, lon=lon,
                   kind=info["type"],
                   behind=info.get("behind", None))

    # Add edges
    for edge in edges:
        a, b = edge[0], edge[1]
        # Optional cost multiplier (3rd element)
        base_cost_mult = float(edge[2]) if len(edge) > 2 else 1.0

        if a not in G or b not in G:
            continue
        la = G.nodes[a]["lat"]
        loa = G.nodes[a]["lon"]
        lb = G.nodes[b]["lat"]
        lob = G.nodes[b]["lon"]
        dist = haversine(la, loa, lb, lob)

        mult = base_cost_mult
        # Apply AIS-based congestion factor for chokepoint-adjacent edges
        if a in choke_factor:
            mult *= choke_factor[a]
        if b in choke_factor:
            mult *= choke_factor[b]
        # Apply extra scenario multipliers
        if a in extra_multipliers:
            mult *= extra_multipliers[a]
        if b in extra_multipliers:
            mult *= extra_multipliers[b]

        G.add_edge(a, b, weight=dist * mult, dist_km=dist, mult=mult,
                   base_cost_mult=base_cost_mult)

    # Collect port and chokepoint names
    ports = sorted([n for n, d in G.nodes(data=True) if d["kind"] == "port"])
    chokepoint_names = sorted([n for n, d in G.nodes(data=True) if d["kind"] == "chokepoint"])

    return G, ports, chokepoint_names


def compute_all_od(G, ports):
    """Compute shortest-path costs and paths for all port-to-port pairs."""
    costs = {}
    paths = {}
    for i, a in enumerate(ports):
        for b in ports[i + 1:]:
            try:
                length = nx.shortest_path_length(G, a, b, weight="weight")
                path = nx.shortest_path(G, a, b, weight="weight")
                costs[(a, b)] = length
                paths[(a, b)] = path
            except nx.NetworkXNoPath:
                costs[(a, b)] = float("inf")
                paths[(a, b)] = []
    return costs, paths


def run_full_closure(cfg, choke_factor):
    """Run full-closure scenarios for each chokepoint."""
    G0, ports, chokepoints = build_graph(cfg, choke_factor)
    n_pairs = len(ports) * (len(ports) - 1) // 2

    print(f"  Network: {G0.number_of_nodes()} nodes, {G0.number_of_edges()} edges")
    print(f"  Ports: {len(ports)}, Chokepoints: {len(chokepoints)}")
    print(f"  OD pairs: {n_pairs}")

    # Baseline
    print("  Computing baseline shortest paths...")
    base_costs, base_paths = compute_all_od(G0, ports)

    # For each chokepoint, remove it and recompute
    all_rows = []
    summary_rows = []

    for choke in chokepoints:
        print(f"  Shock: removing {choke}...")
        G = G0.copy()
        if choke in G:
            G.remove_node(choke)

        shocked_costs, shocked_paths = compute_all_od(G, ports)

        n_affected = 0
        n_disconnected = 0
        deltas = []

        for (a, b), bc in base_costs.items():
            sc = shocked_costs.get((a, b), float("inf"))

            if np.isinf(sc) and np.isfinite(bc):
                delta = float("inf")
                pct = float("inf")
                n_disconnected += 1
                n_affected += 1
                rerouted = True
            elif sc > bc + 0.01:
                delta = sc - bc
                pct = (delta / bc * 100) if bc > 0 else 0.0
                n_affected += 1
                rerouted = shocked_paths.get((a, b), []) != base_paths.get((a, b), [])
                deltas.append(delta)
            else:
                delta = 0.0
                pct = 0.0
                rerouted = False
                deltas.append(0.0)

            # Check if either port is behind this chokepoint
            a_behind = G0.nodes[a].get("behind", None) == choke
            b_behind = G0.nodes[b].get("behind", None) == choke

            all_rows.append({
                "shock": choke,
                "origin": a,
                "destination": b,
                "pair": f"{a}__{b}",
                "base_cost": bc,
                "shock_cost": sc,
                "delta_cost": delta if np.isfinite(delta) else float("inf"),
                "pct_increase": pct if np.isfinite(pct) else float("inf"),
                "rerouted": rerouted,
                "disconnected": np.isinf(sc) and np.isfinite(bc),
                "port_behind_choke": a_behind or b_behind,
            })

        finite_deltas = [d for d in deltas if np.isfinite(d) and d > 0]
        summary_rows.append({
            "chokepoint": choke,
            "mean_delta_km": np.mean(finite_deltas) if finite_deltas else 0.0,
            "median_delta_km": np.median(finite_deltas) if finite_deltas else 0.0,
            "max_delta_km": np.max(finite_deltas) if finite_deltas else 0.0,
            "pairs_affected": n_affected,
            "pairs_disconnected": n_disconnected,
            "total_pairs": n_pairs,
            "frac_affected": n_affected / n_pairs if n_pairs > 0 else 0.0,
        })

    return pd.DataFrame(all_rows), pd.DataFrame(summary_rows), base_costs, base_paths, ports, chokepoints


def main():
    print("=" * 60)
    print("FULL CLOSURE SCENARIO ENGINE (70-node network)")
    print("=" * 60)

    cfg = load_config()
    choke_factor = load_chokepoint_factors()

    df, summary, base_costs, base_paths, ports, chokepoints = run_full_closure(cfg, choke_factor)

    # Save outputs
    df.to_csv(OUT_TAB, index=False)
    print(f"\n  Wrote {OUT_TAB}")

    summary = summary.sort_values("mean_delta_km", ascending=False)
    summary.to_csv(OUT_STATS, index=False)
    print(f"  Wrote {OUT_STATS}")

    # Print summary
    print("\n  Full closure impact ranking:")
    for _, row in summary.iterrows():
        name = row["chokepoint"].replace("_", " ")
        print(f"    {name:25s}  mean delta={row['mean_delta_km']:8.0f} km  "
              f"affected={row['pairs_affected']:4d}/{row['total_pairs']}  "
              f"disconnected={row['pairs_disconnected']:3d}")

    # ---- FIGURE: Summary bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Mean delta cost
    ax1 = axes[0]
    names = [c.replace("_", " ") for c in summary["chokepoint"]]
    ax1.barh(names, summary["mean_delta_km"], color="#2c7bb6", edgecolor="white")
    ax1.set_xlabel("Mean rerouting cost increase (km-equiv.)", fontsize=12)
    ax1.set_title("Chokepoint Closure Impact:\nMean Cost Increase", fontsize=14, fontweight="bold")
    ax1.invert_yaxis()
    ax1.tick_params(labelsize=10)
    for i, v in enumerate(summary["mean_delta_km"]):
        if v > 0:
            ax1.text(v + 20, i, f"{v:,.0f}", va="center", fontsize=9)

    # Right: Fraction affected / disconnected
    ax2 = axes[1]
    affected = summary["frac_affected"].values * 100
    disconnected = (summary["pairs_disconnected"] / summary["total_pairs"]).values * 100
    ax2.barh(names, affected, color="#fdae61", edgecolor="white", label="Affected (rerouted)")
    ax2.barh(names, disconnected, color="#d7191c", edgecolor="white", label="Disconnected")
    ax2.set_xlabel("% of port pairs", fontsize=12)
    ax2.set_title("Chokepoint Closure Impact:\nPairs Affected & Disconnected", fontsize=14, fontweight="bold")
    ax2.invert_yaxis()
    ax2.tick_params(labelsize=10)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG_SUM, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {FIG_SUM}")

    # ---- FIGURE: Example shock (Strait of Hormuz) ----
    example_shock = "Strait_of_Hormuz"
    sub = df[df["shock"] == example_shock].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan)

    # Show top 20 most affected pairs
    top = sub.dropna(subset=["delta_cost"]).nlargest(20, "delta_cost")

    if len(top) > 0:
        fig, ax = plt.subplots(figsize=(13, 7))
        pair_labels = [f"{r['origin'].replace('_',' ')} - {r['destination'].replace('_',' ')}"
                       for _, r in top.iterrows()]
        colors = ["#d7191c" if r["disconnected"] else "#2c7bb6" for _, r in top.iterrows()]
        ax.barh(pair_labels, top["delta_cost"], color=colors, edgecolor="white")
        ax.set_xlabel("Rerouting cost increase (km-equiv.)", fontsize=12)
        ax.set_title("Strait of Hormuz Closure: Top 20 Most Affected Port Pairs",
                     fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.tick_params(labelsize=10)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#d7191c", label="Disconnected"),
                          Patch(facecolor="#2c7bb6", label="Rerouted (via bypass)")]
        ax.legend(handles=legend_elements, fontsize=10)

        plt.tight_layout()
        plt.savefig(FIG_EX, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Wrote {FIG_EX}")

    # Show affected/disconnected pairs
    rerouted = sub[sub["rerouted"] == True]
    disconn = sub[sub["disconnected"] == True]
    print(f"\n  Strait of Hormuz closure:")
    print(f"    Rerouted pairs: {len(rerouted)}")
    print(f"    Disconnected pairs: {len(disconn)}")
    if len(rerouted) > 0:
        finite_costs = rerouted.dropna(subset=["delta_cost"])
        if len(finite_costs) > 0:
            print(f"    Mean rerouting cost: {finite_costs['delta_cost'].mean():,.0f} km")
            print(f"    Max rerouting cost: {finite_costs['delta_cost'].max():,.0f} km")

    print("\n" + "=" * 60)
    print("FULL CLOSURE SCENARIOS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
