import math
import time
from models import SearchResult
from heuristic import haversine

# Hard limits so IDA* never hangs on large graphs
_MAX_SECONDS  = 30
_MAX_EXPANDED = 500_000


def search(G, start: int, goal: int, weight: str = 'custom_weight') -> SearchResult:
    """
    IDA* (Iterative Deepening A*).
    Memory-efficient A*: DFS with an f-cost threshold that increases each
    iteration to the minimum f-value that exceeded the previous limit.
    Optimal with an admissible heuristic.

    Hard limits (_MAX_SECONDS, _MAX_EXPANDED) prevent hanging on large graphs.
    Cycle detection uses a set (O(1)) instead of scanning the path list.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    threshold      = haversine(G, start, goal)
    expansion_log  = []
    parent_map     = {}
    total_expanded = [0]
    deadline       = time.perf_counter() + _MAX_SECONDS

    while True:
        # Abort if we've hit global limits
        if time.perf_counter() > deadline:
            return SearchResult(path=[], nodes_expanded=total_expanded[0],
                                expansion_log=expansion_log, parent_map=parent_map)
        if total_expanded[0] >= _MAX_EXPANDED:
            return SearchResult(path=[], nodes_expanded=total_expanded[0],
                                expansion_log=expansion_log, parent_map=parent_map)

        result, new_threshold = _dls(
            G, goal, weight,
            path=[start], visited={start}, g=0.0, threshold=threshold,
            expansion_log=expansion_log, parent_map=parent_map,
            total_expanded=total_expanded, deadline=deadline,
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


def _dls(G, goal, weight, path, visited, g, threshold,
         expansion_log, parent_map, total_expanded, deadline):
    node = path[-1]
    f    = g + haversine(G, node, goal)

    if f > threshold:
        return None, f

    # Hard limits inside recursion
    if time.perf_counter() > deadline or total_expanded[0] >= _MAX_EXPANDED:
        return None, math.inf

    total_expanded[0] += 1
    expansion_log.append(node)

    if node == goal:
        return path, threshold

    minimum = math.inf

    # Sort neighbours by f so most promising branch explored first
    neighbours = sorted(
        G.successors(node),
        key=lambda nb: g + min(d.get(weight, 1.0) for d in G[node][nb].values())
                         + haversine(G, nb, goal)
    )

    for nb in neighbours:
        if nb in visited:          # O(1) cycle check
            continue
        edge_cost = min(d.get(weight, 1.0) for d in G[node][nb].values())
        parent_map[nb] = node
        visited.add(nb)
        result, t = _dls(
            G, goal, weight,
            path + [nb], visited, g + edge_cost, threshold,
            expansion_log, parent_map, total_expanded, deadline,
        )
        visited.discard(nb)        # backtrack
        if result is not None:
            return result, t
        if t == math.inf:          # deadline/budget hit — propagate up
            return None, math.inf
        minimum = min(minimum, t)

    return None, minimum
