import heapq
from models import SearchResult
from heuristic import haversine


def search(G, start: int, goal: int, weight: str = 'custom_weight') -> SearchResult:
    """
    A* Search.
    Expands nodes by f(n) = g(n) + h(n).
      g(n) = cumulative custom_weight from start to n
      h(n) = haversine distance from n to goal (admissible, consistent)
    Optimal and complete for non-negative weights with an admissible heuristic.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    counter = 0
    h_start = haversine(G, start, goal)
    heap = [(h_start, counter, 0.0, start, [start])]  # (f, tie, g, node, path)
    best_g = {}           # node -> best g(n) found so far
    parent_map = {}
    expansion_log = []
    nodes_expanded = 0

    while heap:
        f, _, g, node, path = heapq.heappop(heap)

        if node in best_g and best_g[node] <= g:
            continue
        best_g[node] = g
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
            edge_data = G[node][neighbor]
            edge_cost = min(d.get(weight, 1.0) for d in edge_data.values())
            new_g = g + edge_cost
            if neighbor not in best_g or best_g[neighbor] > new_g:
                parent_map[neighbor] = node
                h = haversine(G, neighbor, goal)
                counter += 1
                heapq.heappush(heap, (new_g + h, counter, new_g, neighbor, path + [neighbor]))

    return SearchResult(path=[], nodes_expanded=nodes_expanded,
                        expansion_log=expansion_log, parent_map=parent_map)
