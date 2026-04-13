import networkx as nx
from models import SearchResult


def compute_path_cost(G: nx.MultiDiGraph, path: list, weight: str = 'custom_weight') -> float:
    """Sum of edge weights along the path. Returns 0 for empty or single-node paths."""
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G[u][v]
        total += min(d.get(weight, 1.0) for d in edge_data.values())
    return total


def compute_hop_count(path: list) -> int:
    """Number of edges in the path (nodes - 1)."""
    return max(0, len(path) - 1)


def build_comparison_record(name: str, result: SearchResult,
                             G: nx.MultiDiGraph, weight: str = 'custom_weight') -> dict:
    """Build a standardized comparison dict for one algorithm run."""
    return {
        'algorithm': name,
        'path_cost': round(compute_path_cost(G, result.path, weight), 4) if result.path else None,
        'hop_count': compute_hop_count(result.path),
        'nodes_expanded': result.nodes_expanded,
        'path': result.path,
    }
