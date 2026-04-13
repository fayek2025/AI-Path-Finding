import math
from models import SearchResult
from heuristic import haversine


def search(G, start: int, goal: int, weight: str = 'custom_weight') -> SearchResult:
    """
    IDA* (Iterative Deepening A*).
    Memory-efficient A*: runs DFS with an f-cost threshold, increasing the
    threshold each iteration to the minimum f-value that exceeded the previous limit.
    Optimal with an admissible heuristic.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    threshold = haversine(G, start, goal)
    expansion_log = []
    parent_map = {}
    total_expanded = [0]

    while True:
        result, new_threshold = _dls(
            G, goal, weight,
            path=[start], g=0.0, threshold=threshold,
            expansion_log=expansion_log, parent_map=parent_map,
            total_expanded=total_expanded,
        )
        if result is not None:
            return SearchResult(
                path=result,
                nodes_expanded=total_expanded[0],
                expansion_log=expansion_log,
                parent_map=parent_map,
            )
        if new_threshold == math.inf:
            return SearchResult(path=[], nodes_expanded=total_expanded[0],
                                expansion_log=expansion_log, parent_map=parent_map)
        threshold = new_threshold


def _dls(G, goal, weight, path, g, threshold, expansion_log, parent_map, total_expanded):
    """Depth-limited search used internally by IDA*."""
    node = path[-1]
    f = g + haversine(G, node, goal)

    if f > threshold:
        return None, f

    total_expanded[0] += 1
    expansion_log.append(node)

    if node == goal:
        return path, threshold

    minimum = math.inf
    for neighbor in G.successors(node):
        if neighbor in path:   # avoid cycles within current path
            continue
        edge_data = G[node][neighbor]
        edge_cost = min(d.get(weight, 1.0) for d in edge_data.values())
        parent_map[neighbor] = node
        result, t = _dls(
            G, goal, weight,
            path + [neighbor], g + edge_cost, threshold,
            expansion_log, parent_map, total_expanded,
        )
        if result is not None:
            return result, t
        minimum = min(minimum, t)

    return None, minimum
