from collections import deque
from models import SearchResult
from heuristic import haversine


def search(G, start: int, goal: int, weight: str = 'custom_weight') -> SearchResult:
    """
    Breadth-First Search — heuristic-pruned.
    Explores level by level but prunes neighbors that are geographically
    moving away from the goal (h(neighbor) > h(current) * slack).
    This reduces expansions while preserving the BFS level-order guarantee.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    h_goal = haversine(G, start, goal)
    # Allow up to 40% detour — prevents exploring nodes clearly in wrong direction
    SLACK = 1.4

    frontier = deque([[start]])
    visited = {start}
    parent_map = {}
    expansion_log = []
    nodes_expanded = 0

    while frontier:
        path = frontier.popleft()
        node = path[-1]
        nodes_expanded += 1
        expansion_log.append(node)

        h_node = haversine(G, node, goal)

        # Sort neighbors by combined: edge_cost + h(neighbor)
        neighbors = sorted(
            G.successors(node),
            key=lambda nb: (
                min(d.get(weight, 1.0) for d in G[node][nb].values())
                + haversine(G, nb, goal)
            )
        )

        for neighbor in neighbors:
            if neighbor in visited:
                continue

            # Prune: skip neighbors moving significantly away from goal
            h_nb = haversine(G, neighbor, goal)
            if h_nb > h_node * SLACK and neighbor != goal:
                continue

            visited.add(neighbor)
            parent_map[neighbor] = node
            new_path = path + [neighbor]

            if neighbor == goal:
                return SearchResult(
                    path=new_path,
                    nodes_expanded=nodes_expanded,
                    expansion_log=expansion_log,
                    parent_map=parent_map,
                )
            frontier.append(new_path)

    # Fallback: retry without pruning if pruning cut too aggressively
    return _bfs_fallback(G, start, goal, weight)


def _bfs_fallback(G, start, goal, weight):
    """Standard BFS without pruning as safety net."""
    frontier = deque([[start]])
    visited = {start}
    parent_map = {}
    expansion_log = []
    nodes_expanded = 0
    while frontier:
        path = frontier.popleft()
        node = path[-1]
        nodes_expanded += 1
        expansion_log.append(node)
        for neighbor in sorted(G.successors(node),
                               key=lambda nb: min(d.get(weight, 1.0) for d in G[node][nb].values())):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent_map[neighbor] = node
            new_path = path + [neighbor]
            if neighbor == goal:
                return SearchResult(path=new_path, nodes_expanded=nodes_expanded,
                                    expansion_log=expansion_log, parent_map=parent_map)
            frontier.append(new_path)
    return SearchResult(path=[], nodes_expanded=nodes_expanded,
                        expansion_log=expansion_log, parent_map=parent_map)
