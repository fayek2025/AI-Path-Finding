import time
from models import SearchResult

# IDS on a large real-world graph is impractical beyond ~30 hops.
# We cap depth and wall-clock time to keep it responsive.
_MAX_DEPTH   = 35
_MAX_SECONDS = 20


def search(G, start: int, goal: int, weight: str = None) -> SearchResult:
    """
    Iterative Deepening Search.
    Optimal for hop count (same as BFS) but uses O(depth) memory like DFS.
    Capped at depth=35 and 20 s wall-clock to stay practical on large graphs.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    total_expanded = 0
    deadline = time.time() + _MAX_SECONDS

    for depth in range(1, _MAX_DEPTH + 1):
        if time.time() > deadline:
            print(f"  [IDS] time limit reached at depth {depth}, returning no path.")
            return SearchResult(path=[], nodes_expanded=total_expanded)

        result, expanded = _depth_limited_search(G, start, goal, depth, deadline)
        total_expanded += expanded

        if result is not None:
            return SearchResult(
                path=result['path'],
                nodes_expanded=total_expanded,
                expansion_log=result['expansion_log'],
                parent_map=result['parent_map'],
            )

    return SearchResult(path=[], nodes_expanded=total_expanded)


def _depth_limited_search(G, start: int, goal: int, limit: int, deadline: float):
    expansion_log  = []
    parent_map     = {}
    nodes_expanded = [0]

    def dls(node, path, depth):
        if time.time() > deadline:
            return None
        nodes_expanded[0] += 1
        expansion_log.append(node)

        if node == goal:
            return path
        if depth == 0:
            return None

        for neighbor in G.successors(node):
            if neighbor not in path:          # avoid cycles within current path
                parent_map[neighbor] = node
                result = dls(neighbor, path + [neighbor], depth - 1)
                if result is not None:
                    return result
        return None

    path = dls(start, [start], limit)
    if path is not None:
        return {'path': path, 'expansion_log': expansion_log,
                'parent_map': parent_map}, nodes_expanded[0]
    return None, nodes_expanded[0]
