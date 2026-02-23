"""
04b_enhanced_scenarios.py
Enhanced maritime network scenario engine (70-node network):

1) PARTIAL DEGRADATION: Multiply edge costs by alpha for each chokepoint.
2) PORT VULNERABILITY: Per-port cost increase for each chokepoint closure.
3) SECURITY RISK SCENARIOS: Apply risk premiums to individual chokepoints.

Edge format supports optional cost multiplier: [a, b] or [a, b, cost_mult].

Outputs:
- Tables/generated/partial_degradation_results.csv
- Figures/generated/partial_degradation_surface.png
- Tables/generated/port_vulnerability.csv
- Figures/generated/port_vulnerability_heatmap.png
- Tables/generated/security_scenario_results.csv
- Figures/generated/security_scenario_comparison.png
- Tables/generated/chokepoint_density_table.tex
- Tables/generated/scenario_deltas_table.tex
- Tables/generated/cost_per_day_table.tex
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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
CHOKE_TAB = ROOT / "Tables" / "generated" / "chokepoint_intensity.csv"
BASELINE_DELTAS = ROOT / "Tables" / "generated" / "scenario_deltas.csv"
BASELINE_STATS = ROOT / "Tables" / "generated" / "scenario_summary_stats.csv"

# Output paths
OUT_PARTIAL_CSV = ROOT / "Tables" / "generated" / "partial_degradation_results.csv"
OUT_PARTIAL_FIG = ROOT / "Figures" / "generated" / "partial_degradation_surface.png"
OUT_VULN_CSV = ROOT / "Tables" / "generated" / "port_vulnerability.csv"
OUT_VULN_FIG = ROOT / "Figures" / "generated" / "port_vulnerability_heatmap.png"
OUT_SECURITY_CSV = ROOT / "Tables" / "generated" / "security_scenario_results.csv"
OUT_SECURITY_FIG = ROOT / "Figures" / "generated" / "security_scenario_comparison.png"
OUT_CHOKE_TEX = ROOT / "Tables" / "generated" / "chokepoint_density_table.tex"
OUT_DELTAS_TEX = ROOT / "Tables" / "generated" / "scenario_deltas_table.tex"
OUT_COST_TEX = ROOT / "Tables" / "generated" / "cost_per_day_table.tex"

for p in [OUT_PARTIAL_CSV, OUT_PARTIAL_FIG, OUT_VULN_CSV, OUT_VULN_FIG,
          OUT_SECURITY_CSV, OUT_SECURITY_FIG, OUT_CHOKE_TEX, OUT_DELTAS_TEX,
          OUT_COST_TEX]:
    p.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_config():
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chokepoint_factors():
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
    if extra_multipliers is None:
        extra_multipliers = {}
    nodes = cfg["nodes"]
    edges = cfg["edges"]
    G = nx.Graph()
    for name, info in nodes.items():
        lon, lat = info["coords"]
        G.add_node(name, lat=lat, lon=lon, kind=info["type"],
                   behind=info.get("behind", None))
    for edge in edges:
        a, b = edge[0], edge[1]
        base_cost_mult = float(edge[2]) if len(edge) > 2 else 1.0
        if a not in G or b not in G:
            continue
        la, loa = G.nodes[a]["lat"], G.nodes[a]["lon"]
        lb, lob = G.nodes[b]["lat"], G.nodes[b]["lon"]
        dist = haversine(la, loa, lb, lob)
        mult = base_cost_mult
        if a in choke_factor:
            mult *= choke_factor[a]
        if b in choke_factor:
            mult *= choke_factor[b]
        if a in extra_multipliers:
            mult *= extra_multipliers[a]
        if b in extra_multipliers:
            mult *= extra_multipliers[b]
        G.add_edge(a, b, weight=dist * mult, dist_km=dist, mult=mult,
                   base_cost_mult=base_cost_mult)
    ports = sorted([n for n, d in G.nodes(data=True) if d["kind"] == "port"])
    chokepoints = sorted([n for n, d in G.nodes(data=True) if d["kind"] == "chokepoint"])
    return G, ports, chokepoints


def compute_all_od_costs(G, ports):
    costs = {}
    for i, a in enumerate(ports):
        for b in ports[i + 1:]:
            try:
                costs[(a, b)] = nx.shortest_path_length(G, a, b, weight="weight")
            except nx.NetworkXNoPath:
                costs[(a, b)] = float("inf")
    return costs


# ---------------------------------------------------------------------------
# Scenario 1: PARTIAL DEGRADATION
# ---------------------------------------------------------------------------

def run_partial_degradation():
    print("=" * 60)
    print("SCENARIO: PARTIAL DEGRADATION (70-node network)")
    print("=" * 60)

    alphas = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    cfg = load_config()
    choke_factor = load_chokepoint_factors()

    G0, ports, chokepoints = build_graph(cfg, choke_factor)
    base_costs = compute_all_od_costs(G0, ports)
    n_pairs = len(base_costs)

    print(f"  Network: {G0.number_of_nodes()} nodes, {G0.number_of_edges()} edges")
    print(f"  Ports: {len(ports)}, Chokepoints: {len(chokepoints)}, OD pairs: {n_pairs}")
    print(f"  Alphas: {alphas}")

    rows = []
    for choke in chokepoints:
        print(f"  Degrading: {choke}...")
        for alpha in alphas:
            extra = {choke: alpha}
            G = build_graph(cfg, choke_factor, extra_multipliers=extra)[0]
            shocked = compute_all_od_costs(G, ports)

            for (a, b), bc in base_costs.items():
                sc = shocked.get((a, b), float("inf"))
                delta = sc - bc if np.isfinite(sc) else float("inf")
                pct = (delta / bc * 100) if bc > 0 and np.isfinite(delta) else float("nan")
                rows.append({
                    "chokepoint": choke,
                    "alpha": alpha,
                    "origin": a,
                    "destination": b,
                    "pair": f"{a}__{b}",
                    "base_cost": bc,
                    "degraded_cost": sc,
                    "delta_cost": delta,
                    "pct_increase": pct,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PARTIAL_CSV, index=False)
    print(f"  Wrote {OUT_PARTIAL_CSV}")

    # Build mean pct increase matrix
    heatmap_data = (
        df.replace([np.inf, -np.inf], np.nan)
        .groupby(["chokepoint", "alpha"], as_index=False)["pct_increase"]
        .mean()
    )

    # ---- 3D SURFACE PLOT ----
    pivot = heatmap_data.pivot(index="chokepoint", columns="alpha", values="pct_increase")
    pivot = pivot.fillna(0)

    # Sort by max impact
    pivot["max_impact"] = pivot.max(axis=1)
    pivot = pivot.sort_values("max_impact", ascending=False).drop(columns="max_impact")

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    choke_names = list(pivot.index)
    alpha_vals = [float(c) for c in pivot.columns]
    X, Y = np.meshgrid(range(len(alpha_vals)), range(len(choke_names)))
    Z = pivot.values

    norm = mcolors.Normalize(vmin=0, vmax=max(Z.max(), 1))
    colors = plt.cm.YlOrRd(norm(Z))

    ax.plot_surface(X, Y, Z, facecolors=colors, edgecolor="gray",
                    linewidth=0.3, alpha=0.9, shade=True)

    ax.set_xticks(range(len(alpha_vals)))
    ax.set_xticklabels([f"{a:.1f}x" for a in alpha_vals], fontsize=9, rotation=20)
    ax.set_yticks(range(len(choke_names)))
    ax.set_yticklabels([c.replace("_", " ") for c in choke_names], fontsize=9)
    ax.set_zlabel("Mean % cost increase", fontsize=11)
    ax.set_xlabel("Degradation multiplier", fontsize=11, labelpad=10)
    ax.set_ylabel("Chokepoint", fontsize=11, labelpad=10)
    ax.set_title("Partial Degradation: Mean Trade Cost Increase\nby Chokepoint and Severity",
                 fontsize=14, fontweight="bold", pad=20)
    ax.view_init(elev=25, azim=-55)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label("Mean % cost increase", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT_PARTIAL_FIG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {OUT_PARTIAL_FIG}")

    # Print summary
    print("\n  Mean % cost increase (alpha=5.0):")
    if 5.0 in pivot.columns:
        for choke in choke_names:
            val = pivot.loc[choke, 5.0]
            print(f"    {choke.replace('_', ' '):25s}  {val:.2f}%")

    return df, pivot


# ---------------------------------------------------------------------------
# PORT VULNERABILITY ANALYSIS
# ---------------------------------------------------------------------------

def compute_port_vulnerability():
    print("\n" + "=" * 60)
    print("PORT VULNERABILITY ANALYSIS")
    print("=" * 60)

    cfg = load_config()
    choke_factor = load_chokepoint_factors()
    G0, ports, chokepoints = build_graph(cfg, choke_factor)
    base_costs = compute_all_od_costs(G0, ports)

    vuln_rows = []
    for choke in chokepoints:
        print(f"  Closure: {choke}...")
        G = G0.copy()
        if choke in G:
            G.remove_node(choke)
        shocked = compute_all_od_costs(G, ports)

        for port in ports:
            port_deltas = []
            port_disconn = 0
            port_total = 0
            for (a, b), bc in base_costs.items():
                if a == port or b == port:
                    port_total += 1
                    sc = shocked.get((a, b), float("inf"))
                    if np.isinf(sc) and np.isfinite(bc):
                        port_disconn += 1
                    elif sc > bc + 0.01 and np.isfinite(sc):
                        port_deltas.append((sc - bc) / bc * 100)

            mean_pct = np.mean(port_deltas) if port_deltas else 0.0
            vuln_rows.append({
                "port": port,
                "chokepoint": choke,
                "mean_pct_increase": mean_pct,
                "pairs_disconnected": port_disconn,
                "pairs_total": port_total,
                "is_behind": G0.nodes[port].get("behind", None) == choke,
            })

    vuln_df = pd.DataFrame(vuln_rows)
    vuln_df.to_csv(OUT_VULN_CSV, index=False)
    print(f"  Wrote {OUT_VULN_CSV}")

    # ---- HEATMAP: Port vulnerability ----
    pivot = vuln_df.pivot(index="port", columns="chokepoint", values="mean_pct_increase").fillna(0)

    pivot["max_vuln"] = pivot.max(axis=1)
    pivot = pivot.sort_values("max_vuln", ascending=False).drop(columns="max_vuln")

    col_order = pivot.sum(axis=0).sort_values(ascending=False).index
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(16, 22))
    data = pivot.values
    masked = np.ma.masked_where(data == 0, data)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="white")
    vmax = np.percentile(data[data > 0], 95) if (data > 0).any() else 1

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, label="Mean % cost increase for port")
    cbar.set_label("Mean % cost increase for port", fontsize=14)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in pivot.columns],
                       fontsize=14, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([p.replace("_", " ") for p in pivot.index], fontsize=13)

    ax.set_title("Port Vulnerability to Chokepoint Closures\n"
                 "(Mean % cost increase across all routes from/to port)",
                 fontsize=18, fontweight="bold")

    # Mark disconnected ports with X
    for _, row in vuln_df[vuln_df["pairs_disconnected"] > 0].iterrows():
        port_idx = list(pivot.index).index(row["port"]) if row["port"] in pivot.index else None
        choke_idx = list(pivot.columns).index(row["chokepoint"]) if row["chokepoint"] in pivot.columns else None
        if port_idx is not None and choke_idx is not None:
            ax.text(choke_idx, port_idx, "X", ha="center", va="center",
                    fontsize=8, color="black", fontweight="bold")

    fig.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(OUT_VULN_FIG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {OUT_VULN_FIG}")

    return vuln_df


# ---------------------------------------------------------------------------
# Scenario 2: SECURITY RISK SCENARIOS (pure scenario analysis)
# ---------------------------------------------------------------------------

def run_security_scenarios():
    print("\n" + "=" * 60)
    print("SCENARIO: SECURITY RISK PREMIUM (per-chokepoint)")
    print("=" * 60)

    risk_premiums = [0.10, 0.20, 0.50, 1.00, 2.00]
    cfg = load_config()
    choke_factor = load_chokepoint_factors()
    G0, ports, chokepoints = build_graph(cfg, choke_factor)
    base_costs = compute_all_od_costs(G0, ports)

    rows = []
    for choke in chokepoints:
        print(f"  Security risk scenarios for: {choke}...")
        for premium in risk_premiums:
            extra = {choke: 1.0 + premium}
            G = build_graph(cfg, choke_factor, extra_multipliers=extra)[0]
            shocked = compute_all_od_costs(G, ports)

            for (a, b), bc in base_costs.items():
                sc = shocked.get((a, b), float("inf"))
                delta = sc - bc if np.isfinite(sc) else float("inf")
                pct = (delta / bc * 100) if bc > 0 and np.isfinite(delta) else float("nan")
                rows.append({
                    "chokepoint": choke,
                    "risk_premium": premium,
                    "pair": f"{a}__{b}",
                    "base_cost": bc,
                    "security_cost": sc,
                    "delta_cost": delta,
                    "pct_increase": pct,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_SECURITY_CSV, index=False)
    print(f"  Wrote {OUT_SECURITY_CSV}")

    # ---- FIGURE: Per-chokepoint security impact ----
    agg = (
        df.replace([np.inf, -np.inf], np.nan)
        .groupby(["chokepoint", "risk_premium"], as_index=False)["pct_increase"]
        .mean()
    )
    pivot = agg.pivot(index="chokepoint", columns="risk_premium", values="pct_increase").fillna(0)
    pivot["max_impact"] = pivot.max(axis=1)
    pivot = pivot.sort_values("max_impact", ascending=False).drop(columns="max_impact")

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(pivot.index))
    width = 0.15
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(pivot.columns)))

    for i, (col, color) in enumerate(zip(pivot.columns, colors)):
        offset = (i - len(pivot.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pivot[col], width, label=f"+{int(col*100)}% risk",
               color=color, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in pivot.index],
                       fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("Mean % cost increase across all port pairs", fontsize=12)
    ax.set_title("Security Risk Scenario: Per-Chokepoint Impact\n"
                 "(Risk premium applied to single chokepoint)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(OUT_SECURITY_FIG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {OUT_SECURITY_FIG}")

    return df


# ---------------------------------------------------------------------------
# LaTeX Table Generation
# ---------------------------------------------------------------------------

def generate_latex_tables():
    print("\n" + "=" * 60)
    print("GENERATING LaTeX TABLE FRAGMENTS")
    print("=" * 60)

    # ---- Table 1: Chokepoint Intensity ----
    if CHOKE_TAB.exists():
        choke_df = pd.read_csv(CHOKE_TAB)
        choke_df = choke_df.sort_values("sum_intensity", ascending=False)

        def fmt_intensity(val):
            if val >= 1e12:
                return f"{val / 1e12:.2f}T"
            elif val >= 1e9:
                return f"{val / 1e9:.2f}B"
            elif val >= 1e6:
                return f"{val / 1e6:.2f}M"
            else:
                return f"{val:,.0f}"

        def fmt_mean(val):
            if val >= 1e6:
                return f"{val / 1e6:.2f}M"
            elif val >= 1e3:
                return f"{val / 1e3:.1f}K"
            else:
                return f"{val:,.0f}"

        tex_lines = []
        tex_lines.append(r"% Auto-generated by 04b_enhanced_scenarios.py")
        tex_lines.append(r"\begin{table}[htbp]")
        tex_lines.append(r"\centering")
        tex_lines.append(r"\caption{Chokepoint traffic intensity from AIS raster data. "
                         r"Sum intensity reflects total vessel traffic density within "
                         r"each chokepoint bounding box.}")
        tex_lines.append(r"\label{tab:chokepoint_intensity}")
        tex_lines.append(r"\begin{tabular}{lrrrrr}")
        tex_lines.append(r"\toprule")
        tex_lines.append(r"Chokepoint & Sum Intensity & Mean Intensity & P99 Cell & "
                         r"Grid Cells & Rank \\")
        tex_lines.append(r"\midrule")

        for rank, (_, row) in enumerate(choke_df.iterrows(), 1):
            name = row["chokepoint"].replace("_", " ")
            si = fmt_intensity(row["sum_intensity"])
            mi = fmt_mean(row["mean_intensity"])
            p99 = fmt_mean(row["p99_cell"])
            nc = f"{int(row['n_cells']):,}"
            tex_lines.append(f"  {name} & {si} & {mi} & {p99} & {nc} & {rank} \\\\")

        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        tex_lines.append(r"\end{table}")

        with open(OUT_CHOKE_TEX, "w") as f:
            f.write("\n".join(tex_lines) + "\n")
        print(f"  Wrote {OUT_CHOKE_TEX}")

    # ---- Table 2: Scenario Deltas (from 04_scenario_engine.py) ----
    if BASELINE_STATS.exists():
        stats = pd.read_csv(BASELINE_STATS)
        stats = stats.sort_values("mean_delta_km", ascending=False)

        tex2 = []
        tex2.append(r"% Auto-generated by 04b_enhanced_scenarios.py")
        tex2.append(r"\begin{table}[htbp]")
        tex2.append(r"\centering")
        tex2.append(r"\caption{Full closure scenario results: impact of removing each "
                    r"chokepoint on shortest-path costs across all port-to-port pairs. "
                    r"Mean and max $\Delta$ report the rerouting cost for pairs that "
                    r"find alternative routes via bypass edges.}")
        tex2.append(r"\label{tab:scenario_deltas}")
        tex2.append(r"\begin{tabular}{lrrrrr}")
        tex2.append(r"\toprule")
        tex2.append(r"Chokepoint Removed & \multicolumn{1}{c}{Mean $\Delta$} & \multicolumn{1}{c}{Max $\Delta$} & \multicolumn{1}{c}{Affected} & \multicolumn{1}{c}{Disconn.} & \multicolumn{1}{c}{Total} \\")
        tex2.append(r"& \multicolumn{1}{c}{(km-equiv.)} & \multicolumn{1}{c}{(km-equiv.)} & \multicolumn{1}{c}{Pairs} & \multicolumn{1}{c}{Pairs} & \multicolumn{1}{c}{Pairs} \\")
        tex2.append(r"\midrule")

        for _, row in stats.iterrows():
            name = row["chokepoint"].replace("_", " ")
            md = f"{row['mean_delta_km']:,.0f}" if row["mean_delta_km"] > 0 else "0"
            mx = f"{row['max_delta_km']:,.0f}" if row["max_delta_km"] > 0 else "0"
            na = f"{int(row['pairs_affected'])}"
            nd = f"{int(row['pairs_disconnected'])}"
            tp = f"{int(row['total_pairs'])}"
            tex2.append(f"  {name} & {md} & {mx} & {na} & {nd} & {tp} \\\\")

        tex2.append(r"\bottomrule")
        tex2.append(r"\end{tabular}")
        tex2.append(r"\end{table}")

        with open(OUT_DELTAS_TEX, "w") as f:
            f.write("\n".join(tex2) + "\n")
        print(f"  Wrote {OUT_DELTAS_TEX}")


# ---------------------------------------------------------------------------
# Cost-Per-Day Table Generation
# ---------------------------------------------------------------------------

def generate_cost_per_day_table():
    print("\n" + "=" * 60)
    print("GENERATING COST-PER-DAY TABLE")
    print("=" * 60)

    if not BASELINE_STATS.exists():
        print(f"  Skipping: {BASELINE_STATS} not found.")
        return

    stats = pd.read_csv(BASELINE_STATS)

    # Annual transit estimates (UNCTAD, Suez Canal Authority, Panama Canal Authority)
    transit_counts = {
        "Panama_Canal": 13500,
        "Gibraltar": 80000,
        "Suez_Canal": 19500,
        "Bab_el_Mandeb": 25000,
        "Bosporus": 43000,
        "Strait_of_Malacca": 85000,
    }

    cost_per_km = 75  # mid estimate (USD)

    rows = []
    for _, row in stats.iterrows():
        choke = row["chokepoint"]
        if choke not in transit_counts:
            continue
        annual = transit_counts[choke]
        daily_vessels = annual / 365.0
        mean_delta = row["mean_delta_km"]

        daily_cost_mid = daily_vessels * mean_delta * cost_per_km
        daily_cost_low = daily_vessels * mean_delta * 50
        daily_cost_high = daily_vessels * mean_delta * 100

        rows.append({
            "chokepoint": choke,
            "daily_vessels": daily_vessels,
            "mean_delta_km": mean_delta,
            "cost_low": daily_cost_low,
            "cost_mid": daily_cost_mid,
            "cost_high": daily_cost_high,
        })

    cost_df = pd.DataFrame(rows)
    cost_df = cost_df.sort_values("cost_mid", ascending=False)

    def fmt_cost(val):
        if val >= 1e9:
            return f"\\${val / 1e9:.1f}B"
        elif val >= 1e6:
            return f"\\${val / 1e6:.1f}M"
        elif val >= 1e3:
            return f"\\${val / 1e3:.1f}K"
        else:
            return f"\\${val:,.0f}"

    tex = []
    tex.append(r"% Auto-generated by 04b_enhanced_scenarios.py")
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{Estimated daily rerouting cost of chokepoint closure. "
               r"Low/Mid/High estimates use \$50/\$75/\$100 per additional km "
               r"respectively, reflecting fuel, charter time, and crew costs. "
               r"Daily vessel counts from UNCTAD, Suez Canal Authority, and "
               r"Panama Canal Authority statistics.}")
    tex.append(r"\label{tab:cost_per_day}")
    tex.append(r"\begin{tabular}{lrrrrrr}")
    tex.append(r"\toprule")
    tex.append(r"Chokepoint & Daily Vessels & Mean Detour (km) & "
               r"Cost/Day (Low) & Cost/Day (Mid) & Cost/Day (High) \\")
    tex.append(r"\midrule")

    for _, row in cost_df.iterrows():
        name = row["chokepoint"].replace("_", " ")
        dv = f"{row['daily_vessels']:.0f}"
        md = f"{row['mean_delta_km']:,.0f}"
        cl = fmt_cost(row["cost_low"])
        cm = fmt_cost(row["cost_mid"])
        ch = fmt_cost(row["cost_high"])
        tex.append(f"  {name} & {dv} & {md} & {cl} & {cm} & {ch} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    with open(OUT_COST_TEX, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"  Wrote {OUT_COST_TEX}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    partial_df, pivot = run_partial_degradation()
    vuln_df = compute_port_vulnerability()
    security_df = run_security_scenarios()
    generate_latex_tables()
    generate_cost_per_day_table()

    print("\n" + "=" * 60)
    print("ALL ENHANCED SCENARIOS COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    for p in [OUT_PARTIAL_CSV, OUT_PARTIAL_FIG, OUT_VULN_CSV, OUT_VULN_FIG,
              OUT_SECURITY_CSV, OUT_SECURITY_FIG, OUT_CHOKE_TEX, OUT_DELTAS_TEX,
              OUT_COST_TEX]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
