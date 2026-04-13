import networkx as nx
from heuristic import haversine


def compute_g(path: list, G: nx.MultiDiGraph, weight: str = 'custom_weight') -> float:
    """
    g(n): cumulative cost from start to the last node in path.
    Sums the edge weights along the given path.
    """
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        # MultiDiGraph may have parallel edges; take the minimum weight edge
        edge_data = G[u][v]
        min_w = min(data.get(weight, 1.0) for data in edge_data.values())
        total += min_w
    return total


def compute_h(G: nx.MultiDiGraph, node: int, goal: int) -> float:
    """
    h(n): haversine straight-line distance in meters from node to goal.
    Admissible and consistent — never overestimates actual road cost.
    """
    return haversine(G, node, goal)


def compute_f(path: list, G: nx.MultiDiGraph, weight: str, goal: int) -> float:
    """
    f(n) = g(n) + h(n)
    Used by A*, IDA*, Greedy, and Bidirectional A* to prioritize nodes.
    """
    return compute_g(path, G, weight) + compute_h(G, path[-1], goal)
