"""
dhaka_fullmap.py
────────────────
Standalone script — no web app, no manual interaction.

Loads the full Dhaka OSM road network, picks landmark START/GOAL nodes,
runs all 6 search algorithms, and saves one expansion-map PNG per algorithm.

Weight matrix used (project's own road metrics from map_loader.py):
  • Traffic Intensity  — traffic_factor   (penalty: higher = more congested)
  • Road Quality       — road_age_factor  (penalty: higher = older/worse road)
  • Safety Index       — safety_factor    (benefit:  higher = safer, divides cost)
  • Turn Complexity    — turn_complexity  (penalty: higher = complex intersections)
  • Pothole factor     — EXCLUDED

Formula:
  custom_weight = length_km * traffic_factor * road_age_factor
                  * turn_complexity / safety_factor

Usage:
    python dhaka_fullmap.py

Output:
    fullmap_output/expansion_<AlgoName>.png   — one per algorithm
    fullmap_output/summary_comparison.png     — bar charts side-by-side
"""

import os
import time
import tracemalloc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import osmnx as ox

from map_loader import load_graph
from comparison import ALGORITHMS
from metrics import build_comparison_record

# ── Configuration ─────────────────────────────────────────────────────────────

OUT_DIR   = 'fullmap_output'
SEED      = 42
CENTER    = (23.7766, 90.4227)   # Dhaka city centre
DIST_M    = 15000                # 15 km radius — covers greater Dhaka

# Well-known landmark coordinates (snapped to nearest OSM node at runtime)
# START: Dhaka University area
# GOAL:  Uttara (north Dhaka)
START_LATLON = (23.7279, 90.3960)
GOAL_LATLON  = (23.8759, 90.3795)

# ── POI amenity tags ──────────────────────────────────────────────────────────
SAFETY_TAGS = {'police', 'hospital', 'fire_station', 'clinic', 'pharmacy'}
HAZARD_TAGS = {'bus_stop', 'marketplace', 'crossing', 'traffic_signals', 'fuel'}

COLORS = {
    'BFS':    '#1f77b4',
    'DFS':    '#9467bd',
    'IDS':    '#17becf',
    'UCS':    '#2ca02c',
    'A*':     '#d62728',
    'Greedy': '#ff7f0e',
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_node(G, lat, lon):
    return ox.distance.nearest_nodes(G, X=lon, Y=lat)


def _node_pos(G):
    return {n: (d['x'], d['y']) for n, d in G.nodes(data=True)}


def _collect_pois(G, lon_min, lon_max, lat_min, lat_max):
    safety, hazard = [], []
    for nid, data in G.nodes(data=True):
        lon, lat = data.get('x'), data.get('y')
        if lon is None or lat is None:
            continue
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        tag = str(data.get('amenity', '') or data.get('highway', '') or '')
        if tag in SAFETY_TAGS:
            safety.append((lon, lat))
        elif tag in HAZARD_TAGS:
            hazard.append((lon, lat))
    return safety, hazard


def _apply_weights(G):
    """
    Compute custom_weight using the project's own road metric matrix.

    Included factors (from map_loader._ROAD_PROFILES):
      traffic_factor   — Traffic Intensity  (penalty)
      road_age_factor  — Road Quality       (penalty: older = worse)
      safety_factor    — Safety Index       (benefit: divides cost)
      turn_complexity  — Turn Complexity    (penalty)

    Excluded:
      pothole_factor   — not used in this standalone run

    Formula:
      custom_weight = length_km
                      * traffic_factor
                      * road_age_factor
                      * turn_complexity
                      / safety_factor
    """
    G = G.copy()
    for u, v, key, data in G.edges(keys=True, data=True):
        lkm = data.get('length', 100) / 1000.0
        t   = max(data.get('traffic_factor',  1.4), 1e-6)   # Traffic Intensity
        s   = max(data.get('safety_factor',   0.8), 1e-6)   # Safety Index
        ra  = max(data.get('road_age_factor', 1.1), 1e-6)   # Road Quality
        tc  = max(data.get('turn_complexity', 0.7), 1e-6)   # Turn Complexity
        # pothole_factor intentionally excluded
        G[u][v][key]['custom_weight'] = round(
            lkm * t * ra * tc / s, 6
        )
    return G


def _run_algorithm(module, G, start, goal, name):
    """Run one algorithm, measure time + memory, return (record, time_ms, mem_kb)."""
    t0 = time.perf_counter()
    result = module.search(G, start, goal, 'custom_weight')
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Memory (second run under tracemalloc)
    tracemalloc.start()
    module.search(G, start, goal, 'custom_weight')
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_kb = peak // 1024

    record = build_comparison_record(name, result, G)
    record['expansion_log'] = result.expansion_log
    record['path']          = result.path
    return record, elapsed_ms, mem_kb


# ─────────────────────────────────────────────────────────────────────────────
# Expansion map (one per algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_expansion_map(G, pos, record, start, goal,
                        safety_pois, hazard_pois,
                        lon_min, lon_max, lat_min, lat_max,
                        out_path):
    algo      = record['algorithm']
    expansion = record.get('expansion_log', [])
    path      = record.get('path', [])
    n_exp     = record['nodes_expanded']
    cost      = record.get('path_cost')
    hops      = record.get('hop_count', 0)

    fig, ax = plt.subplots(figsize=(13, 13))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')

    # ── Layer 1: Full road network (grey) ─────────────────────────────────────
    for u, v in G.edges():
        pu, pv = pos.get(u), pos.get(v)
        if pu is None or pv is None:
            continue
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]],
                color='#cccccc', linewidth=0.35, zorder=1, alpha=0.65)

    # ── Layer 2: Expanded nodes coloured by expansion order ───────────────────
    if expansion:
        n = len(expansion)
        exp_lons, exp_lats, exp_c = [], [], []
        for order, nid in enumerate(expansion):
            p = pos.get(nid)
            if p:
                exp_lons.append(p[0])
                exp_lats.append(p[1])
                exp_c.append(order / max(n - 1, 1))

        sc = ax.scatter(exp_lons, exp_lats, c=exp_c, cmap='viridis',
                        s=10, zorder=3, alpha=0.85, linewidths=0,
                        vmin=0, vmax=1)

        cbar = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.01)
        cbar.set_label('Expansion order (early → late)', fontsize=9)
        ticks = [0, 0.25, 0.5, 0.75, 1.0]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(int(t * (n - 1))) for t in ticks])

    # ── Layer 3: Found path — thick red outline ───────────────────────────────
    if len(path) >= 2:
        px = [pos[nid][0] for nid in path if nid in pos]
        py = [pos[nid][1] for nid in path if nid in pos]
        if len(px) >= 2:
            ax.plot(px, py, color='white',   linewidth=7, zorder=5, solid_capstyle='round')
            ax.plot(px, py, color='#e74c3c', linewidth=4, zorder=6, solid_capstyle='round')

    # ── Layer 4: Safety POI markers only ─────────────────────────────────────
    if safety_pois:
        ax.scatter([p[0] for p in safety_pois], [p[1] for p in safety_pois],
                   marker='+', s=40, color='#2980b9', linewidths=1.2, zorder=7, alpha=0.8)

    # ── Layer 5: Start / Goal ─────────────────────────────────────────────────
    for nid, marker, color, sz in [
        (start, '*', '#27ae60', 260),
        (goal,  'o', '#27ae60', 160),
    ]:
        p = pos.get(nid)
        if p:
            ax.scatter(*p, s=sz, color=color, marker=marker,
                       edgecolors='white', linewidths=1.5, zorder=10)

    # ── Axes / title / legend ─────────────────────────────────────────────────
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    cost_str = f"{cost:,.3f}" if cost is not None else 'N/A'
    ax.set_title(
        f"{algo}   (expanded={n_exp:,},  hops={hops},  cost={cost_str})",
        fontsize=12, fontweight='bold', pad=12
    )
    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude',  fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')

    legend_items = [
        plt.Line2D([0], [0], marker='*', color='#27ae60', linestyle='None',
                   markersize=11, label='Start'),
        plt.Line2D([0], [0], marker='o', color='#27ae60', linestyle='None',
                   markersize=8,  label='Goal'),
        mpatches.Patch(color='#e74c3c', label='Found path'),
    ]
    if safety_pois:
        legend_items.append(
            plt.Line2D([0], [0], marker='+', color='#2980b9', linestyle='None',
                       markersize=8, markeredgewidth=1.5,
                       label='Safety POI (police/hospital/fire)')
        )
    ax.legend(handles=legend_items, loc='lower left', fontsize=8, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison graph — all algorithms side by side
# ─────────────────────────────────────────────────────────────────────────────

def _draw_comparison(records, times_ms, mems_kb, out_path):
    """
    3×2 comparison figure showing all 6 algorithms across 6 metrics:
      Row 0: Nodes Expanded | Execution Time (ms)
      Row 1: Peak Memory (KB) | Path Cost
      Row 2: Hop Count | Optimality + Type table
    """
    names    = [r['algorithm'] for r in records]
    colors   = [COLORS.get(n, '#888') for n in names]
    expanded = [r['nodes_expanded'] for r in records]
    costs    = [r['path_cost'] if r['path_cost'] is not None else 0 for r in records]
    hops     = [r['hop_count'] for r in records]
    informed = {'A*', 'Greedy'}

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        'Full Dhaka City — Algorithm Comparison\n'
        'Weight: Traffic Intensity × Road Quality × Turn Complexity / Safety Index',
        fontsize=13, fontweight='bold', y=1.01
    )

    bar_datasets = [
        (axes[0, 0], expanded,  'Nodes Expanded',     'Nodes',      '{:.0f}'),
        (axes[0, 1], times_ms,  'Execution Time',     'ms',         '{:.0f}'),
        (axes[1, 0], mems_kb,   'Peak Memory',        'KB',         '{:.0f}'),
        (axes[1, 1], costs,     'Path Cost',          'cost units', '{:.3f}'),
        (axes[2, 0], hops,      'Hop Count',          'edges',      '{:.0f}'),
    ]

    for ax, values, title, ylabel, fmt in bar_datasets:
        ax.set_facecolor('white')
        bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.5)
        max_v = max(values) if max(values) > 0 else 1
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + max_v * 0.01,
                        fmt.format(h), ha='center', va='bottom', fontsize=8)
        # Shade informed vs uninformed
        for i, name in enumerate(names):
            fc = (0.16, 0.50, 0.73, 0.07) if name in informed else (0.15, 0.68, 0.38, 0.05)
            ax.axvspan(i - 0.5, i + 0.5, facecolor=fc, zorder=0)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.yaxis.grid(True, color='#eeeeee', linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
        # Informed/uninformed legend
        ax.legend(handles=[
            mpatches.Patch(color=(0.16, 0.50, 0.73, 0.25), label='Informed'),
            mpatches.Patch(color=(0.15, 0.68, 0.38, 0.20), label='Uninformed'),
        ], fontsize=7, loc='upper right', framealpha=0.8)

    # [2,1] Results table
    ax = axes[2, 1]
    ax.axis('off')
    table_data = [
        [r['algorithm'],
         'Informed' if r['algorithm'] in informed else 'Uninformed',
         '★ Yes' if r.get('is_optimal') else 'No',
         f"{r['path_cost']:.4f}" if r['path_cost'] else 'N/A',
         str(r['nodes_expanded']),
         str(r['hop_count'])]
        for r in records
    ]
    col_labels = ['Algorithm', 'Type', 'Optimal', 'Cost', 'Expanded', 'Hops']
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2980b9')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i, row in enumerate(table_data, 1):
        bg = '#ddeeff' if row[1] == 'Informed' else '#eeffee'
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
            tbl[i, j].set_text_props(color='#222')
    ax.set_title('Results Summary', fontsize=11, fontweight='bold', pad=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load full Dhaka graph ──────────────────────────────────────────────
    print("Loading full Dhaka OSM graph...")
    G_full = load_graph(center=CENTER, dist=DIST_M, seed=SEED)
    print(f"  {G_full.number_of_nodes():,} nodes, {G_full.number_of_edges():,} edges")

    # ── 2. Snap landmarks to nearest OSM nodes ────────────────────────────────
    start = _nearest_node(G_full, *START_LATLON)
    goal  = _nearest_node(G_full, *GOAL_LATLON)
    print(f"  START node: {start}  ({G_full.nodes[start]['y']:.4f}, {G_full.nodes[start]['x']:.4f})")
    print(f"  GOAL  node: {goal}   ({G_full.nodes[goal]['y']:.4f}, {G_full.nodes[goal]['x']:.4f})")

    # ── 3. Apply project weight matrix (traffic, road quality, safety, turn) ──
    print("Applying weight matrix (traffic × road_age × turn / safety)...")
    G = _apply_weights(G_full)

    # ── 4. Build position dict and bounding box ───────────────────────────────
    pos = _node_pos(G)
    all_lons = [p[0] for p in pos.values()]
    all_lats = [p[1] for p in pos.values()]
    pad_lon  = (max(all_lons) - min(all_lons)) * 0.02
    pad_lat  = (max(all_lats) - min(all_lats)) * 0.02
    lon_min, lon_max = min(all_lons) - pad_lon, max(all_lons) + pad_lon
    lat_min, lat_max = min(all_lats) - pad_lat, max(all_lats) + pad_lat

    # ── 5. Collect POIs ───────────────────────────────────────────────────────
    print("Collecting POIs...")
    safety_pois, hazard_pois = _collect_pois(G, lon_min, lon_max, lat_min, lat_max)
    print(f"  Safety POIs: {len(safety_pois)},  Hazard POIs: {len(hazard_pois)}")

    # ── 6. Run all algorithms ─────────────────────────────────────────────────
    print("\nRunning algorithms on full Dhaka graph...")
    records, times_ms, mems_kb = [], [], []

    for name, module, _ in ALGORITHMS:
        print(f"  {name}...", end=' ', flush=True)
        try:
            rec, t_ms, m_kb = _run_algorithm(module, G, start, goal, name)
            # Mark optimal
            try:
                true_cost = nx.shortest_path_length(G, start, goal, weight='custom_weight')
                rec['is_optimal'] = (
                    rec['path_cost'] is not None and
                    abs(rec['path_cost'] - true_cost) < 1e-4
                )
            except Exception:
                rec['is_optimal'] = False
            print(f"expanded={rec['nodes_expanded']:,}  cost={rec['path_cost']}  {t_ms:.0f}ms")
        except Exception as e:
            print(f"ERROR: {e}")
            rec = {'algorithm': name, 'path_cost': None, 'hop_count': 0,
                   'nodes_expanded': 0, 'path': [], 'expansion_log': [],
                   'is_optimal': False}
            t_ms, m_kb = 0, 0

        records.append(rec)
        times_ms.append(t_ms)
        mems_kb.append(m_kb)

    # ── 7. Generate expansion maps ────────────────────────────────────────────
    print("\nGenerating expansion maps...")
    for rec in records:
        safe_name = rec['algorithm'].replace('*', '_star').replace(' ', '_')
        out_path  = os.path.join(OUT_DIR, f"expansion_{safe_name}.png")
        _draw_expansion_map(
            G, pos, rec, start, goal,
            safety_pois, hazard_pois,
            lon_min, lon_max, lat_min, lat_max,
            out_path
        )

    # ── 8. Comparison chart ───────────────────────────────────────────────────
    print("\nGenerating comparison chart...")
    _draw_comparison(records, times_ms, mems_kb,
                     os.path.join(OUT_DIR, 'comparison_all_algorithms.png'))

    print(f"\nDone. All outputs saved to '{OUT_DIR}/'")


if __name__ == '__main__':
    main()
