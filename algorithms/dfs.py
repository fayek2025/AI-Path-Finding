from models import SearchResult
from heuristic import haversine


def search(G, start: int, goal: int, weight: str = 'custom_weight') -> SearchResult:
    """DFS — heuristic-guided. Sorts neighbors by f = edge_cost + h(neighbor)."""
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    stack = [(start, [start], 0.0)]
    visited = set()
    parent_map = {}
    expansion_log = []
    nodes_expanded = 0

    while stack:
        node, path, g = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        nodes_expanded += 1
        expansion_log.append(node)

        if node == goal:
            return SearchResult(path=path, nodes_expanded=nodes_expanded,
                                expansion_log=expansion_log, parent_map=parent_map)

        neighbors = sorted(
            G.successors(node),
            key=lambda nb: (
                min(d.get(weight, 1.0) for d in G[node][nb].values())
                + haversine(G, nb, goal)
            ),
            reverse=True
        )
        for neighbor in neighbors:
            if neighbor not in visited:
                edge_cost = min(d.get(weight, 1.0) for d in G[node][neighbor].values())
                parent_map[neighbor] = node
                stack.append((neighbor, path + [neighbor], g + edge_cost))

    return SearchResult(path=[], nodes_expanded=nodes_expanded,
                        expansion_log=expansion_log, parent_map=parent_map)
