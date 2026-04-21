import time
from heuristic import haversine
from models import SearchResult

_MAX_SECONDS = 20


def search(G, start: int, goal: int, weight: str = 'custom_weight') -> SearchResult:
    """
    Iterative Deepening A* (IDA*-style IDS).
    Uses f(n) = g(n) + h(n) as the threshold instead of raw depth.
    This makes IDS heuristic-guided: it only explores nodes whose
    estimated total cost is within the current threshold, pruning
    branches that can't possibly beat it.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    threshold = haversine(G, start, goal)
    total_expanded = 0
    deadline = time.time() + _MAX_SECONDS

    while True:
        if time.time() > deadline:
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
            # No path exists
            return SearchResult(path=[], nodes_expanded=total_expanded)

        threshold = next_threshold


def _search(G, start, goal, weight, threshold, deadline):
    expansion_log = []
    parent_map = {}
    nodes_expanded = [0]

    def dfs(node, g, path):
        if time.time() > deadline:
            return None, float('inf')

        f = g + haversine(G, node, goal)
        if f > threshold:
            return None, f  # prune — return f as candidate for next threshold

        nodes_expanded[0] += 1
        expansion_log.append(node)

        if node == goal:
            return path, 0

        min_next = float('inf')
        # Sort neighbors by f = g + edge + h so promising ones explored first
        neighbors = sorted(
            G.successors(node),
            key=lambda nb: (
                g + min(d.get(weight, 1.0) for d in G[node][nb].values())
                + haversine(G, nb, goal)
            )
        )
        for neighbor in neighbors:
            if neighbor in path:  # avoid cycles
                continue
            edge_cost = min(d.get(weight, 1.0) for d in G[node][neighbor].values())
            parent_map[neighbor] = node
            result, t = dfs(neighbor, g + edge_cost, path + [neighbor])
            if result is not None:
                return result, 0
            min_next = min(min_next, t)

        return None, min_next

    path, next_t = dfs(start, 0.0, [start])
    if path is not None:
        return {'path': path, 'expansion_log': expansion_log,
                'parent_map': parent_map}, nodes_expanded[0], 0
    return None, nodes_expanded[0], next_t
