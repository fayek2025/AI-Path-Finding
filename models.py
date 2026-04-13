from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Result returned by every search algorithm."""
    path: list          # Ordered node IDs from start to goal (empty if no path)
    nodes_expanded: int # Number of nodes popped from the frontier
    expansion_log: list = field(default_factory=list)  # Expansion order (for state tree)
    parent_map: dict = field(default_factory=dict)     # node -> parent node


@dataclass
class EvaluationResult:
    """Stores f(n), g(n), h(n) for a single node during informed search."""
    node: int
    g: float  # Cumulative cost from start to this node
    h: float  # Heuristic estimate from this node to goal
    f: float  # g + h
