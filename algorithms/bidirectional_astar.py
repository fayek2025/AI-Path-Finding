import heapq
import math
from models import SearchResult
from heuristic import haversine


def search(G, start: int, goal: int, weight: str = 'custom_weight') -> SearchResult:
    """
    Bidirectional A*.
    Runs two simultaneous A* frontiers — forward from start, backward from goal.
    Terminates when a node is settled by both frontiers, then reconstructs the path.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    # Reverse graph shares the same node/edge data as G (already weight-applied)
    G_rev = G.reverse(copy=False)

    counter = [1]

    def push(heap, f, g, node):
        counter[0] += 1
        heapq.heappush(heap, (f, counter[0], g, node))

    fwd_heap = [(haversine(G, start, goal), 0, 0.0, start)]
    bwd_heap = [(haversine(G, goal, start), 0, 0.0, goal)]

    fwd_g = {start: 0.0}
    bwd_g = {goal: 0.0}
    fwd_parent = {start: None}
    bwd_parent = {goal: None}
    fwd_settled = set()
    bwd_settled = set()

    expansion_log = []
    nodes_expanded = 0
    best_cost = math.inf
    meeting_node = None

    while fwd_heap or bwd_heap:
        fwd_min = fwd_heap[0][0] if fwd_heap else math.inf
        bwd_min = bwd_heap[0][0] if bwd_heap else math.inf
        if fwd_min + bwd_min >= best_cost:
            break

        if fwd_min <= bwd_min:
            _, _, g, node = heapq.heappop(fwd_heap)
            if node in fwd_settled:
                continue
            fwd_settled.add(node)
            nodes_expanded += 1
            expansion_log.append(node)

            if node in bwd_g:
                candidate = g + bwd_g[node]
                if candidate < best_cost:
                    best_cost = candidate
                    meeting_node = node

            for neighbor in G.successors(node):
                edge_data = G[node][neighbor]
                edge_cost = min(d.get(weight, 1.0) for d in edge_data.values())
                new_g = g + edge_cost
                if neighbor not in fwd_g or fwd_g[neighbor] > new_g:
                    fwd_g[neighbor] = new_g
                    fwd_parent[neighbor] = node
                    try:
                        h = haversine(G, neighbor, goal)
                    except (KeyError, Exception):
                        continue
                    push(fwd_heap, new_g + h, new_g, neighbor)
        else:
            _, _, g, node = heapq.heappop(bwd_heap)
            if node in bwd_settled:
                continue
            bwd_settled.add(node)
            nodes_expanded += 1
            expansion_log.append(node)

            if node in fwd_g:
                candidate = fwd_g[node] + g
                if candidate < best_cost:
                    best_cost = candidate
                    meeting_node = node

            for neighbor in G_rev.successors(node):
                edge_data = G_rev[node][neighbor]
                edge_cost = min(d.get(weight, 1.0) for d in edge_data.values())
                new_g = g + edge_cost
                if neighbor not in bwd_g or bwd_g[neighbor] > new_g:
                    bwd_g[neighbor] = new_g
                    bwd_parent[neighbor] = node
                    try:
                        h = haversine(G, neighbor, start)
                    except (KeyError, Exception):
                        continue
                    push(bwd_heap, new_g + h, new_g, neighbor)

    if meeting_node is None:
        return SearchResult(path=[], nodes_expanded=nodes_expanded,
                            expansion_log=expansion_log,
                            parent_map={**fwd_parent, **bwd_parent})

    fwd_path = _reconstruct(fwd_parent, start, meeting_node)
    bwd_path = _reconstruct(bwd_parent, goal, meeting_node)
    full_path = fwd_path + bwd_path[1:]

    return SearchResult(
        path=full_path,
        nodes_expanded=nodes_expanded,
        expansion_log=expansion_log,
        parent_map={**fwd_parent, **bwd_parent},
    )


def _reconstruct(parent_map: dict, start: int, end: int) -> list:
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent_map.get(node)
    path.reverse()
    # If we couldn't trace back to start, return just the end node
    if path and path[0] != start:
        return [end]
    return path
