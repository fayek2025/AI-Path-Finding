from models import SearchResult


def search(G, start: int, goal: int, weight: str = None) -> SearchResult:
    """
    Depth-First Search (iterative).
    Uses a LIFO stack with a visited set to prevent cycles.
    Not optimal — may return a longer path than necessary.
    """
    if start == goal:
        return SearchResult(path=[start], nodes_expanded=0)

    # Stack holds (current_node, path_so_far)
    stack = [(start, [start])]
    visited = set()
    parent_map = {}
    expansion_log = []
    nodes_expanded = 0

    while stack:
        node, path = stack.pop()
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
                stack.append((neighbor, path + [neighbor]))

    return SearchResult(path=[], nodes_expanded=nodes_expanded,
                        expansion_log=expansion_log, parent_map=parent_map)
