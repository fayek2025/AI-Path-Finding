import json
import matplotlib.pyplot as plt
import networkx as nx

from models import SearchResult
from metrics import build_comparison_record

import algorithms.bfs as bfs
import algorithms.dfs as dfs
import algorithms.ids as ids
import algorithms.ucs as ucs
import algorithms.astar as astar
import algorithms.greedy as greedy
import algorithms.idastar as idastar
import algorithms.bidirectional_astar as bi_astar

ALGORITHMS = [
    ('BFS',               bfs,      None),
    ('DFS',               dfs,      None),
    ('IDS',               ids,      None),
    ('UCS',               ucs,      'custom_weight'),
    ('A*',                astar,    'custom_weight'),
    ('Greedy',            greedy,   'custom_weight'),
    ('IDA*',              idastar,  'custom_weight'),
    ('Bidirectional A*',  bi_astar, 'custom_weight'),
]


def run_all(G: nx.MultiDiGraph, start: int, goal: int,
            waypoints: list = None) -> list:
    """
    Run all 8 algorithms directly on G (already a small custom graph).
    If waypoints provided, stitches segment results together.
    """
    stops = [start] + (waypoints or []) + [goal]
    records = []

    for name, module, weight in ALGORITHMS:
        print(f"  Running {name}...", end=' ', flush=True)
        try:
            full_path      = []
            total_expanded = 0
            full_log       = []
            full_parents   = {}

            for i in range(len(stops) - 1):
                result = module.search(G, stops[i], stops[i + 1], weight)
                if not result.path:
                    full_path = []
                    break
                seg = result.path if i == 0 else result.path[1:]
                full_path      += seg
                total_expanded += result.nodes_expanded
                full_log       += result.expansion_log
                full_parents.update(result.parent_map)

            from models import SearchResult
            merged = SearchResult(
                path=full_path,
                nodes_expanded=total_expanded,
                expansion_log=full_log,
                parent_map=full_parents,
            )
            record = build_comparison_record(name, merged, G)
            status = (f"{record['hop_count']} hops, "
                      f"cost={record['path_cost']}, "
                      f"expanded={record['nodes_expanded']}")
        except Exception as e:
            from models import SearchResult
            merged = SearchResult(path=[], nodes_expanded=0)
            record = build_comparison_record(name, merged, G)
            record['error'] = str(e)
            status = f"ERROR: {e}"

        records.append(record)
        print(status)
    return records


def export_json(records: list, filepath: str = 'results.json') -> None:
    """Write comparison records to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Results saved to {filepath}")


def plot_comparison(records: list) -> None:
    """Render a bar chart comparing algorithms on cost, hops, and nodes expanded."""
    valid = [r for r in records if r.get('path_cost') is not None]
    if not valid:
        print("No valid paths to plot.")
        return

    names = [r['algorithm'] for r in valid]
    costs = [r['path_cost'] for r in valid]
    hops = [r['hop_count'] for r in valid]
    expanded = [r['nodes_expanded'] for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Algorithm Comparison — DU → Bashundhara', fontsize=14)

    _bar(axes[0], names, costs, 'Total Path Cost (custom_weight)', 'steelblue')
    _bar(axes[1], names, hops, 'Hop Count (edges)', 'darkorange')
    _bar(axes[2], names, expanded, 'Nodes Expanded', 'seagreen')

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    print("Comparison chart saved to comparison.png")
    plt.show()


def _bar(ax, names, values, title, color):
    bars = ax.bar(names, values, color=color, edgecolor='black')
    ax.set_title(title)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=7)
