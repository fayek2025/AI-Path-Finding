import time
from heuristic import haversine
from models import SearchResult

_MAX_SECONDS  = 30
_MAX_EXPANDED = 500_000


def search(G, start: int, goal: int, weight: str = 'custom_weight') -> SearchResult:
    """
    IDS with f(n) = g(n) + h(n) threshold (IDA*-style).
    Cycle detection uses a set for O(1) lookup.
    Hard limits prevent hanging on large graphs.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    threshold      = haversine(G, start, goal)
    total_expanded = 0
    deadline       = time.perf_counter() + _MAX_SECONDS

    while True:
        if time.perf_counter() > deadline or total_expanded >= _MAX_EXPANDED:
            return SearchResult(path=[], nodes_expanded=total_expanded)

        result, expanded, next_threshold = _search(
            G, start, goal, weight, threshold, deadline
        )
        total_expanded += expanded

        if result is not None:
            return SearchResult(
                path=result['path'],
                nodes_expanded=total_expanded,
                expansion_log=result['expansion_log'],
                parent_map=result['parent_map'],
            )
        if next_threshold == float('inf'):
            return SearchResult(path=[], nodes_expanded=total_expanded)

        threshold = next_threshold


def _search(G, start, goal, weight, threshold, deadline):
    expansion_log  = []
    parent_map     = {}
    nodes_expanded = [0]

    def dfs(node, g, path_list, visited):
        if time.perf_counter() > deadline or nodes_expanded[0] >= _MAX_EXPANDED:
            return None, float('inf')

        f = g + haversine(G, node, goal)
        if f > threshold:
            return None, f

        nodes_expanded[0] += 1
        expansion_log.append(node)

        if node == goal:
            return path_list, 0

        min_next = float('inf')
        neighbours = sorted(
            G.successors(node),
            key=lambda nb: (
                g + min(d.get(weight, 1.0) for d in G[node][nb].values())
                + haversine(G, nb, goal)
            )
        )
        for nb in neighbours:
            if nb in visited:
                continue
            edge_cost = min(d.get(weight, 1.0) for d in G[node][nb].values())
            parent_map[nb] = node
            visited.add(nb)
            result, t = dfs(nb, g + edge_cost, path_list + [nb], visited)
            visited.discard(nb)
            if result is not None:
                return result, 0
            if t == float('inf'):
                return None, float('inf')
            min_next = min(min_next, t)

        return None, min_next

    path, next_t = dfs(start, 0.0, [start], {start})
    if path is not None:
        return {'path': path, 'expansion_log': expansion_log,
                'parent_map': parent_map}, nodes_expanded[0], 0
    return None, nodes_expanded[0], next_t
