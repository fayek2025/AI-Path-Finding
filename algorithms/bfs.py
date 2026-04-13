from collections import deque
from models import SearchResult


def search(G, start: int, goal: int, weight: str = None) -> SearchResult:
    """
    Breadth-First Search.
    Explores nodes level by level - optimal for hop count, ignores edge weights.
    Returns the path with the fewest edges from start to goal.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

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

        for neighbor in G.successors(node):
            if neighbor in visited:
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

    return SearchResult(path=[], nodes_expanded=nodes_expanded,
                        expansion_log=expansion_log, parent_map=parent_map)
