import heapq
from models import SearchResult


def search(G, start: int, goal: int, weight: str = 'custom_weight') -> SearchResult:
    """
    Uniform Cost Search.
    Expands the node with the lowest cumulative path cost g(n).
    Optimal for any non-negative edge weights.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    # heap entry: (cost, tie_breaker, node, path)
    counter = 0
    heap = [(0.0, counter, start, [start])]
    visited = {}          # node -> best cost seen
    parent_map = {}
    expansion_log = []
    nodes_expanded = 0

    while heap:
        cost, _, node, path = heapq.heappop(heap)

        if node in visited and visited[node] <= cost:
            continue
        visited[node] = cost
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
            # MultiDiGraph: pick minimum weight among parallel edges
            edge_cost = min(d.get(weight, 1.0) for d in edge_data.values())
            new_cost = cost + edge_cost

            if neighbor not in visited or visited[neighbor] > new_cost:
                parent_map[neighbor] = node
                counter += 1
                heapq.heappush(heap, (new_cost, counter, neighbor, path + [neighbor]))

    return SearchResult(path=[], nodes_expanded=nodes_expanded,
                        expansion_log=expansion_log, parent_map=parent_map)
