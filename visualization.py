"""
visualization.py
Generates matplotlib figures for:
  1. Path map       — algorithm paths on a real OSM tile basemap
  2. Complexity     — nodes expanded, execution time, peak memory, theory table
                      all on a clean white background
"""
import time
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import networkx as nx
import contextily as ctx
from pyproj import Transformer

from comparison import ALGORITHMS

# ── Algorithm colours (same as dashboard) ────────────────────────────────────
COLORS = {
    'BFS':              '#1f77b4',
    'DFS':              '#9467bd',
    'IDS':              '#17becf',
    'UCS':              '#2ca02c',
    'A*':               '#d62728',
    'Greedy':           '#ff7f0e',
    'IDA*':             '#e377c2',
    'Bidirectional A*': '#bcbd22',
}

# Highway type → display style on the map (width, colour, zorder)
_HW_STYLE = {
    'motorway':     (3.5, '#e8a020', 4),
    'trunk':        (3.0, '#e8c020', 4),
    'primary':      (2.5, '#e06030', 3),
    'secondary':    (2.0, '#3070c0', 3),
    'tertiary':     (1.5, '#507090', 2),
    'residential':  (1.0, '#708090', 2),
    'service':      (0.8, '#909090', 2),
    'unclassified': (0.8, '#a0a0a0', 2),
}
_HW_DEFAULT = (0.8, '#b0b0b0', 2)

_INFORMED = {'A*', 'Greedy', 'IDA*', 'Bidirectional A*'}

_TIME_COMPLEXITY = {
    'BFS': 'O(b^d)', 'DFS': 'O(b^m)', 'IDS': 'O(b^d)',
    'UCS': 'O(b^(1+C*/ε))', 'A*': 'O(b^d)', 'Greedy': 'O(b^m)',
    'IDA*': 'O(b^d)', 'Bidirectional A*': 'O(b^(d/2))',
}
_SPACE_COMPLEXITY = {
    'BFS': 'O(b^d)', 'DFS': 'O(bm)', 'IDS': 'O(bd)',
    'UCS': 'O(b^(1+C*/ε))', 'A*': 'O(b^d)', 'Greedy': 'O(b^m)',
    'IDA*': 'O(bd)', 'Bidirectional A*': 'O(b^(d/2))',
}

# WGS-84 → Web Mercator (EPSG:3857) for contextily
_to_mercator = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node_positions_wgs(G: nx.MultiDiGraph) -> dict:
    """Return {node_id: (lon, lat)} in WGS-84."""
    return {n: (d['x'], d['y']) for n, d in G.nodes(data=True)}


def _to_merc(lon: float, lat: float):
    return _to_mercator.transform(lon, lat)


def _measure_time(module, G, start, goal, weight) -> float:
    t0 = time.perf_counter()
    try:
        module.search(G, start, goal, weight)
    except Exception:
        pass
    return (time.perf_counter() - t0) * 1000


def _measure_memory(module, G, start, goal, weight) -> int:
    tracemalloc.start()
    try:
        module.search(G, start, goal, weight)
    except Exception:
        pass
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak // 1024


# ─────────────────────────────────────────────────────────────────────────────
# Path map  (OSM tile basemap + road overlay + algorithm paths)
# ─────────────────────────────────────────────────────────────────────────────

def generate_path_map(osm_G: nx.MultiDiGraph,
                      search_G: nx.MultiDiGraph,
                      weighted_G: nx.MultiDiGraph,
                      records: list,
                      start: int, goal: int,
                      chosen_nodes: list,
                      out_path: str = 'output_path_map.png') -> str:
    """
    Render algorithm paths on a real OSM tile basemap (contextily).

    Layers bottom → top:
      1. OSM tile basemap  (CartoDB Positron — clean light style)
      2. Full OSM road network clipped to bbox — coloured by highway type
      3. Search-subgraph corridor — slightly thicker highlight
      4. Algorithm paths — each in its dashboard colour with glow
      5. Chosen node markers (START / waypoints / GOAL)
    All coordinates are projected to Web Mercator (EPSG:3857) for contextily.
    """
    osm_pos    = _node_positions_wgs(osm_G)
    search_pos = _node_positions_wgs(search_G)

    # ── Bounding box in WGS-84 then Mercator ─────────────────────────────────
    lons = [p[0] for p in search_pos.values()]
    lats = [p[1] for p in search_pos.values()]
    pad_lon = max((max(lons) - min(lons)) * 0.22, 0.008)
    pad_lat = max((max(lats) - min(lats)) * 0.22, 0.008)
    lon_min, lon_max = min(lons) - pad_lon, max(lons) + pad_lon
    lat_min, lat_max = min(lats) - pad_lat, max(lats) + pad_lat

    x_min, y_min = _to_merc(lon_min, lat_min)
    x_max, y_max = _to_merc(lon_max, lat_max)

    # Helper: convert WGS-84 node position to Mercator
    def merc(pos_wgs):
        return _to_merc(pos_wgs[0], pos_wgs[1])

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ── Layer 1: OSM tile basemap ─────────────────────────────────────────────
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    try:
        ctx.add_basemap(ax, crs='EPSG:3857',
                        source=ctx.providers.CartoDB.Positron,
                        zoom='auto', attribution_size=6)
    except Exception:
        # Fallback if tiles unavailable (offline)
        ax.set_facecolor('#f5f5f0')

    # ── Layer 2: Full OSM road network clipped to bbox ───────────────────────
    for u, v, data in osm_G.edges(data=True):
        pu = osm_pos.get(u)
        pv = osm_pos.get(v)
        if pu is None or pv is None:
            continue
        if not (lon_min <= pu[0] <= lon_max and lat_min <= pu[1] <= lat_max):
            continue
        hw = data.get('highway', 'unclassified')
        if isinstance(hw, list):
            hw = hw[0]
        hw = hw.replace('_link', '').strip()
        lw, color, zo = _HW_STYLE.get(hw, _HW_DEFAULT)
        mx0, my0 = merc(pu)
        mx1, my1 = merc(pv)
        ax.plot([mx0, mx1], [my0, my1],
                color=color, linewidth=lw * 0.6,
                alpha=0.45, zorder=zo, solid_capstyle='round')

    # ── Layer 3: Search subgraph corridor — thicker highlight ─────────────────
    for u, v, data in search_G.edges(data=True):
        pu = search_pos.get(u)
        pv = search_pos.get(v)
        if pu is None or pv is None:
            continue
        hw = data.get('highway_type', 'unclassified')
        lw, color, zo = _HW_STYLE.get(hw, _HW_DEFAULT)
        mx0, my0 = merc(pu)
        mx1, my1 = merc(pv)
        # White halo then coloured line for clear visibility
        ax.plot([mx0, mx1], [my0, my1],
                color='white', linewidth=lw * 1.8 + 1.5,
                alpha=0.6, zorder=zo + 2, solid_capstyle='round')
        ax.plot([mx0, mx1], [my0, my1],
                color=color, linewidth=lw * 1.4,
                alpha=0.85, zorder=zo + 3, solid_capstyle='round')

    # ── Layer 4: Algorithm paths ──────────────────────────────────────────────
    valid_records = [r for r in records if len(r.get('path', [])) >= 2]
    n_valid = len(valid_records)
    legend_handles = []

    for idx, rec in enumerate(valid_records):
        name  = rec['algorithm']
        path  = rec['path']
        color = COLORS.get(name, '#333333')

        coords = [merc(search_pos[n]) for n in path if n in search_pos]
        if len(coords) < 2:
            continue

        # Tiny lateral offset so overlapping paths remain distinguishable
        offset = (idx - n_valid / 2) * 8   # metres in Mercator
        xs = [c[0] + offset for c in coords]
        ys = [c[1] + offset for c in coords]

        # Glow: thick semi-transparent stroke behind the main line
        ax.plot(xs, ys, color=color, linewidth=7, alpha=0.18,
                zorder=10 + idx, solid_capstyle='round')
        ax.plot(xs, ys, color=color, linewidth=2.8, alpha=0.95,
                zorder=11 + idx, solid_capstyle='round')

        cost_str = f"{rec['path_cost']:.3f}" if rec['path_cost'] is not None else 'N/A'
        opt_mark = ' ★' if rec.get('is_optimal') else ''
        legend_handles.append(mpatches.Patch(
            color=color,
            label=f"{name}{opt_mark}   cost={cost_str}   hops={rec['hop_count']}   exp={rec['nodes_expanded']}"
        ))

    # ── Layer 5: Chosen node markers ─────────────────────────────────────────
    _NODE_STYLE = {
        'START': ('#27ae60', '*', 400, 11),
        'GOAL':  ('#e74c3c', 'D', 260, 10),
    }
    for cn in chosen_nodes:
        nid   = cn['id']
        label = cn.get('label', '')
        wgs   = search_pos.get(nid) or osm_pos.get(nid)
        if wgs is None:
            continue
        mx, my = merc(wgs)

        if label in _NODE_STYLE:
            color, marker, size, fsize = _NODE_STYLE[label]
        else:
            color, marker, size, fsize = '#2980b9', 'o', 180, 9

        ax.scatter(mx, my, s=size, color=color, marker=marker,
                   edgecolors='white', linewidths=1.2, zorder=20)
        ax.annotate(
            label, (mx, my),
            textcoords='offset points', xytext=(10, 8),
            fontsize=fsize, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, alpha=0.9, linewidth=1.2)
        )

    # ── Road type legend (bottom-right) ──────────────────────────────────────
    hw_legend = [
        mpatches.Patch(color='#e8a020', label='Motorway / Trunk'),
        mpatches.Patch(color='#e06030', label='Primary'),
        mpatches.Patch(color='#3070c0', label='Secondary'),
        mpatches.Patch(color='#507090', label='Tertiary / Residential'),
    ]
    road_leg = ax.legend(handles=hw_legend, loc='lower right',
                         fontsize=8, framealpha=0.92,
                         title='Road type', title_fontsize=8)
    ax.add_artist(road_leg)

    # ── Algorithm legend (lower-left) ────────────────────────────────────────
    ax.legend(handles=legend_handles, loc='lower left',
              fontsize=8, framealpha=0.92,
              title='Algorithms  (★ = optimal)', title_fontsize=8)

    # ── Axes labels (convert Mercator ticks back to lat/lon for readability) ──
    ax.set_title('Algorithm Paths on Road Network', fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Longitude', fontsize=9, color='#444')
    ax.set_ylabel('Latitude',  fontsize=9, color='#444')

    def _merc_to_lon(x, pos=None):
        lon, _ = Transformer.from_crs('EPSG:3857', 'EPSG:4326',
                                       always_xy=True).transform(x, 0)
        return f'{lon:.4f}°'

    def _merc_to_lat(y, pos=None):
        _, lat = Transformer.from_crs('EPSG:3857', 'EPSG:4326',
                                       always_xy=True).transform(0, y)
        return f'{lat:.4f}°'

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_merc_to_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_merc_to_lat))
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Complexity graphs  (white background)
# ─────────────────────────────────────────────────────────────────────────────

def generate_complexity_graphs(G: nx.MultiDiGraph,
                                start: int, goal: int,
                                records: list,
                                out_path: str = 'output_complexity.png') -> str:
    """
    2×2 white-background figure:
      [0,0] Nodes expanded
      [0,1] Execution time (ms)
      [1,0] Peak memory (KB)
      [1,1] Theoretical complexity table
    """
    names    = [r['algorithm'] for r in records]
    colors   = [COLORS.get(n, '#888888') for n in names]
    expanded = [r['nodes_expanded'] for r in records]

    weight   = 'custom_weight'
    algo_map = {name: mod for name, mod, _ in ALGORITHMS}

    times_ms, mem_kb = [], []
    for name in names:
        mod = algo_map.get(name)
        if mod:
            times_ms.append(_measure_time(mod, G, start, goal, weight))
            mem_kb.append(_measure_memory(mod, G, start, goal, weight))
        else:
            times_ms.append(0)
            mem_kb.append(0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor('white')
    for ax in axes.flat:
        ax.set_facecolor('white')

    bar_kw = dict(edgecolor='white', linewidth=0.6)

    # [0,0] Nodes expanded
    _bar_chart(axes[0, 0], names, expanded, colors,
               'Nodes Expanded  (time proxy)', 'Algorithm', 'Nodes Expanded', bar_kw)

    # [0,1] Execution time
    _bar_chart(axes[0, 1], names, times_ms, colors,
               'Execution Time (ms)', 'Algorithm', 'Time (ms)', bar_kw, fmt='{:.1f}')

    # [1,0] Peak memory
    _bar_chart(axes[1, 0], names, mem_kb, colors,
               'Peak Memory Usage (KB)', 'Algorithm', 'Memory (KB)', bar_kw)

    # [1,1] Theoretical complexity table
    _complexity_table(axes[1, 1], names)

    fig.suptitle('Algorithm Complexity Analysis', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar_chart(ax, names, values, colors, title, xlabel, ylabel, bar_kw, fmt='{:.0f}'):
    bars = ax.bar(names, values, color=colors, **bar_kw)

    # Value labels on top of each bar
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + max(values) * 0.01,
                fmt.format(h),
                ha='center', va='bottom', fontsize=8, color='#333')

    # Shade informed vs uninformed columns
    for i, name in enumerate(names):
        fc = (0.16, 0.50, 0.73, 0.07) if name in _INFORMED else (0.15, 0.68, 0.38, 0.05)
        ax.axvspan(i - 0.5, i + 0.5, facecolor=fc, zorder=0)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color='#555')
    ax.set_ylabel(ylabel, fontsize=9, color='#555')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')

    # Category legend (informed / uninformed)
    informed_patch   = mpatches.Patch(color=(0.16, 0.50, 0.73, 0.25), label='Informed')
    uninformed_patch = mpatches.Patch(color=(0.15, 0.68, 0.38, 0.20), label='Uninformed')
    ax.legend(handles=[informed_patch, uninformed_patch],
              fontsize=7, loc='upper right', framealpha=0.8)


def _complexity_table(ax, names):
    ax.axis('off')
    table_data = [
        [n,
         'Informed' if n in _INFORMED else 'Uninformed',
         _TIME_COMPLEXITY.get(n, '—'),
         _SPACE_COMPLEXITY.get(n, '—')]
        for n in names
    ]
    col_labels = ['Algorithm', 'Type', 'Time Complexity', 'Space Complexity']
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.1, 1.8)

    # Header row
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor('#2980b9')
        cell.set_text_props(color='white', fontweight='bold')

    # Data rows — alternate shading, highlight informed
    for i, row in enumerate(table_data, start=1):
        is_informed = row[1] == 'Informed'
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            if is_informed:
                cell.set_facecolor('#ddeeff' if i % 2 == 0 else '#eef5ff')
            else:
                cell.set_facecolor('#eeffee' if i % 2 == 0 else '#f5fff5')
            cell.set_text_props(color='#222222')

    ax.set_title('Theoretical Complexity', fontsize=11, fontweight='bold', pad=10)
