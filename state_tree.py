import json
from models import SearchResult


def build_tree(result: SearchResult) -> dict:
    """
    Build a state-space tree from a SearchResult's expansion_log and parent_map.
    Returns a nested dict rooted at the start node.
    """
    if not result.expansion_log:
        return {}

    root = result.expansion_log[0]
    children_map: dict = {}

    for node in result.expansion_log:
        parent = result.parent_map.get(node)
        if parent is not None:
            children_map.setdefault(parent, []).append(node)

    solution_set = set(result.path)

    def build_node(node_id):
        return {
            'id': node_id,
            'in_solution': node_id in solution_set,
            'children': [build_node(c) for c in children_map.get(node_id, [])],
        }

    return build_node(root)


def serialize_tree(tree: dict) -> str:
    """Serialize the state-space tree to a JSON string."""
    return json.dumps(tree, indent=2)
