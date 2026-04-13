import heapq
from models import SearchResult
from heuristic import haversine


def search(G, start: int, goal: int, weight: str = 'custom_weight') -> SearchResult:
    """
    Greedy Best-First Search.
    Expands the node with the smallest h(n) = haversine distance to goal.
    Fast but NOT optimal — ignores actual path cost g(n).
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    counter = 0
    h_start = haversine(G, start, goal)
    heap = [(h_start, counter, start, [start])]
    visited = set()
    parent_map = {}
    expansion_log = []
    nodes_expanded = 0

    while heap:
        h, _, node, path = heapq.heappop(heap)

        if node in visited:
            continue
        visited.add(node)
        nodes_expanded += 1
        expansion_log.append(node)

        if node == goal:
            return SearchResult(
                path=path,
                nodes_expanded=nodes_expanded,
                expansion_log=expansion_log,
                parent_map=parent_map,
            )

        for neighbor in G.successors(node):
            if neighbor not in visited:
                parent_map[neighbor] = node
                h_n = haversine(G, neighbor, goal)
                counter += 1
                heapq.heappush(heap, (h_n, counter, neighbor, path + [neighbor]))

    return SearchResult(path=[], nodes_expanded=nodes_expanded,
                        expansion_log=expansion_log, parent_map=parent_map)
