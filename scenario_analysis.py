"""
scenario_analysis.py
────────────────────
Generates grouped bar charts comparing algorithm performance across
three weight scenarios — matching the style in the reference image.

Scenarios (x-axis groups):
  • Strict Distance   — only raw length matters (all factors = 1.0)
  • Traffic Avoidance — traffic slider maxed (avoid busy roads)
  • Maximum Safety    — safety slider maxed (prefer safe roads)

Metrics (2×2 grid, one algorithm per row pair):
  • Time Complexity   — nodes expanded (proxy for time)
  • Memory Complexity — peak nodes in memory (tracemalloc)

The 3 bars per group represent the 3 route diversity profiles
(fast-road path / quiet-road path / balanced path) that exist in
the search subgraph, so you can see how each scenario affects
each candidate route independently.

Usage:
    python scenario_analysis.py

Output:
    scenario_output/scenario_<AlgoName>.png   — one 2×2 grid per algorithm
    scenario_output/scenario_all_algos.png    — all algorithms in one figure
"""

import os
import time
import tracemalloc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
import osmnx as ox

from map_loader import load_graph, build_search_graph, HIGHWAY_TYPES
from comparison import ALGORITHMS
from metrics import build_comparison_record

# ── Configuration ─────────────────────────────────────────────────────────────

OUT_DIR      = 'scenario_output'
SEED         = 42
CENTER       = (23.7766, 90.4227)
DIST_M       = 15000

# Three landmark pairs — each pair is one "route" (bar colour)
# Route A: Dhaka University → Uttara
# Route B: Motijheel → Mirpur
# Route C: Old Dhaka → Bashundhara
ROUTE_PAIRS = [
    ((23.7279, 90.3960), (23.8759, 90.3795), 'DU → Uttara'),
    ((23.7230, 90.4170), (23.7938, 90.3584), 'Motijheel → Mirpur'),
    ((23.7104, 90.4074), (23.8223, 90.4317), 'Old Dhaka → Bashundhara'),
]

# Three weight scenarios — each is a dict of slider values
SCENARIOS = {
    'Strict Distance':   {'traffic': 1.0, 'safety': 1.0, 'pothole': 1.0, 'road_age': 1.0, 'turn': 1.0},
    'Traffic Avoidance': {'traffic': 3.0, 'safety': 1.0, 'pothole': 1.0, 'road_age': 1.0, 'turn': 1.0},
    'Maximum Safety':    {'traffic': 1.0, 'safety': 3.0, 'pothole': 1.0, 'road_age': 1.0, 'turn': 1.0},
}

# Bar colours for the 3 routes — purple / blue / green (matches reference image)
BAR_COLORS = ['#7b52ab', '#4a90d9', '#4caf7d']

# Which algorithms to show (pick the most interesting ones)
SHOW_ALGOS = ['A*', 'Greedy', 'BFS', 'UCS', 'IDS', 'DFS']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nearest(G, lat, lon):
    return ox.distance.nearest_nodes(G, X=lon, Y=lat)


def _apply_weights(G, sliders):
    G = G.copy()
    for u, v, key, data in G.edges(keys=True, data=True):
        lkm = data.get('length', 100) / 1000.0
        t   = max(data.get('traffic_factor',  1.4), 1e-6)
        s   = max(data.get('safety_factor',   0.8), 1e-6)
        p   = max(data.get('pothole_factor',  1.3), 1e-6)
        ra  = max(data.get('road_age_factor', 1.1), 1e-6)
        tc  = max(data.get('turn_complexity', 0.7), 1e-6)
        G[u][v][key]['custom_weight'] = round(
            lkm
            * (t  ** sliders['traffic'])
            * (p  ** sliders['pothole'])
            * (ra ** sliders['road_age'])
            * (tc ** sliders['turn'])
            / (s  ** sliders['safety']),
            6
        )
    return G


def _measure(module, G, start, goal):
    """Return (nodes_expanded, peak_nodes_in_memory)."""
    # nodes expanded
    result = module.search(G, start, goal, 'custom_weight')
    n_exp  = result.nodes_expanded

    # peak memory measured as peak live objects (tracemalloc)
    tracemalloc.start()
    module.search(G, start, goal, 'custom_weight')
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Convert bytes → approximate node count (each node ~200 bytes in heap)
    peak_nodes = max(1, peak_bytes // 200)

    return n_exp, peak_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_data(osm_G):
    """
    Returns nested dict:
      data[algo_name][scenario_name][route_idx] = (nodes_expanded, peak_nodes)
    """
    algo_map = {name: mod for name, mod, _ in ALGORITHMS}

    # Pre-build subgraphs for each route pair
    print("Building search subgraphs for each route...")
    subgraphs = []
    node_pairs = []
    for (s_lat, s_lon), (g_lat, g_lon), label in ROUTE_PAIRS:
        start = _nearest(osm_G, s_lat, s_lon)
        goal  = _nearest(osm_G, g_lat, g_lon)
        chosen = [
            {'id': start, 'lat': s_lat, 'lon': s_lon, 'label': 'START'},
            {'id': goal,  'lat': g_lat, 'lon': g_lon, 'label': 'GOAL'},
        ]
        sg = build_search_graph(osm_G, chosen, seed=SEED)
        subgraphs.append(sg)
        node_pairs.append((start, goal, label))
        print(f"  {label}: {sg.number_of_nodes()} nodes, {sg.number_of_edges()} edges")

    data = {algo: {sc: [] for sc in SCENARIOS} for algo in SHOW_ALGOS}

    for sc_name, sliders in SCENARIOS.items():
        print(f"\nScenario: {sc_name}")
        for route_idx, (sg, (start, goal, label)) in enumerate(zip(subgraphs, node_pairs)):
            G_w = _apply_weights(sg, sliders)
            for algo_name in SHOW_ALGOS:
                mod = algo_map.get(algo_name)
                if mod is None:
                    data[algo_name][sc_name].append((0, 0))
                    continue
                try:
                    n_exp, peak = _measure(mod, G_w, start, goal)
                except Exception as e:
                    print(f"    {algo_name} on {label}: ERROR {e}")
                    n_exp, peak = 0, 0
                data[algo_name][sc_name].append((n_exp, peak))
                print(f"  {algo_name:20s} | {label:25s} | exp={n_exp:4d}  peak={peak:6d}")

    return data, node_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _grouped_bars(ax, scenario_names, route_values, route_labels,
                  title, ylabel, bar_colors):
    """
    Draw grouped bar chart.
    scenario_names : list of x-axis group labels
    route_values   : list of lists — route_values[route_idx][scenario_idx]
    """
    n_groups  = len(scenario_names)
    n_bars    = len(route_values)
    width     = 0.22
    x         = np.arange(n_groups)

    for i, (vals, color, label) in enumerate(zip(route_values, bar_colors, route_labels)):
        offset = (i - n_bars / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, color=color,
                        label=label, edgecolor='white', linewidth=0.4)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + max(max(v) for v in route_values) * 0.015,
                        str(int(h)),
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.yaxis.grid(True, color='#eeeeee', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')


def plot_single_algo(algo_name, algo_data, node_pairs, out_path):
    """2×2 grid: [time | memory] × [top row title | bottom row title]."""
    sc_names    = list(SCENARIOS.keys())
    route_labels = [label for _, _, label in node_pairs]

    # Reorganise: route_values[route_idx] = [val_for_sc0, val_for_sc1, val_for_sc2]
    time_by_route   = [[algo_data[sc][ri][0] for sc in sc_names] for ri in range(len(node_pairs))]
    memory_by_route = [[algo_data[sc][ri][1] for sc in sc_names] for ri in range(len(node_pairs))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'{algo_name} — Scenario Analysis', fontsize=13, fontweight='bold', y=1.02)

    _grouped_bars(axes[0], sc_names, time_by_route, route_labels,
                  f'{algo_name} Time Complexity (Nodes Generated)',
                  'Nodes Expanded', BAR_COLORS)

    _grouped_bars(axes[1], sc_names, memory_by_route, route_labels,
                  f'{algo_name} Memory Complexity (Peak Nodes in Memory)',
                  'Peak Nodes', BAR_COLORS)

    # Shared legend
    handles = [mpatches.Patch(color=BAR_COLORS[i], label=route_labels[i])
               for i in range(len(route_labels))]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_all_algos(data, node_pairs, out_path):
    """
    One large figure: rows = algorithms, cols = [time, memory].
    Matches the reference image layout (2×2 per algo pair).
    """
    sc_names     = list(SCENARIOS.keys())
    route_labels = [label for _, _, label in node_pairs]
    n_algos      = len(SHOW_ALGOS)

    # Arrange as pairs of rows: every 2 algorithms share a 2×2 block
    n_rows = n_algos        # one row per algo
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4.5 * n_rows))
    fig.patch.set_facecolor('white')
    fig.suptitle('Algorithm Scenario Analysis — Dhaka Road Network',
                 fontsize=15, fontweight='bold', y=1.005)

    for row, algo_name in enumerate(SHOW_ALGOS):
        algo_data    = data[algo_name]
        time_by_route   = [[algo_data[sc][ri][0] for sc in sc_names] for ri in range(len(node_pairs))]
        memory_by_route = [[algo_data[sc][ri][1] for sc in sc_names] for ri in range(len(node_pairs))]

        _grouped_bars(axes[row, 0], sc_names, time_by_route, route_labels,
                      f'{algo_name} Time Complexity (Nodes Generated)',
                      'Nodes Expanded', BAR_COLORS)

        _grouped_bars(axes[row, 1], sc_names, memory_by_route, route_labels,
                      f'{algo_name} Memory Complexity (Peak Nodes in Memory)',
                      'Peak Nodes', BAR_COLORS)

    # Shared legend at bottom
    handles = [mpatches.Patch(color=BAR_COLORS[i], label=route_labels[i])
               for i in range(len(route_labels))]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.012))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading Dhaka OSM graph...")
    osm_G = load_graph(center=CENTER, dist=DIST_M, seed=SEED)
    print(f"  {osm_G.number_of_nodes():,} nodes, {osm_G.number_of_edges():,} edges\n")

    data, node_pairs = collect_data(osm_G)

    print("\nGenerating plots...")

    # One plot per algorithm
    for algo_name in SHOW_ALGOS:
        safe = algo_name.replace('*', '_star').replace(' ', '_')
        plot_single_algo(
            algo_name, data[algo_name], node_pairs,
            os.path.join(OUT_DIR, f'scenario_{safe}.png')
        )

    # Combined all-algorithms plot
    plot_all_algos(data, node_pairs,
                   os.path.join(OUT_DIR, 'scenario_all_algos.png'))

    print(f"\nDone. All outputs saved to '{OUT_DIR}/'")


if __name__ == '__main__':
    main()
