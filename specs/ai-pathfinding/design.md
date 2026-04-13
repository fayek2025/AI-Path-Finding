# Design Document: AI Pathfinding System

## Overview

This system implements a multi-metric AI pathfinding simulator on a real-world road network extracted from OpenStreetMap (OSM) for Dhaka, Bangladesh. It runs 8 classical AI search algorithms — 4 uninformed and 4 informed — on a weighted directed graph where each edge carries composite cost metrics (traffic, safety, potholes). Results are compared across standardized metrics and visualized through both a matplotlib-based static view and a Flask-based interactive web application.

This project falls squarely within the **Search and Problem Solving** area of classical AI — specifically **graph search in a weighted state space**. The problem is formulated as a single-source single-goal shortest path problem with a custom cost function, making it ideal for comparing uninformed vs. informed strategies.

---

## Architecture

```
ai-pathfinding/
├── map_loader.py          # OSMnx graph fetch, caching, metric assignment
├── graph_utils.py         # Custom node insertion, nearest-node lookup, weight computation
├── algorithms/
│   ├── bfs.py
│   ├── dfs.py
│   ├── ids.py
│   ├── ucs.py
│   ├── astar.py
│   ├── greedy.py
│   ├── idastar.py
│   └── bidirectional_astar.py
├── heuristic.py           # Haversine distance heuristic
├── metrics.py             # Path cost, hop count, nodes expanded recorder
├── state_tree.py          # State-space tree builder and serializer
├── comparison.py          # Aggregates results, produces JSON + bar chart
├── main.py                # Entry point: wires everything together, runs on local machine
└── tests/
    ├── test_metrics.py
    ├── test_graph_utils.py
    ├── test_algorithms.py
    ├── test_heuristic.py
    ├── test_comparison.py
    └── test_integration.py

> Note: Web app (FastAPI + Leaflet.js) is deferred to Phase 2. Phase 1 runs entirely on the local machine with matplotlib visualizations.
```

### Technology Stack

| Layer | Technology |
|---|---|
| Map data | OSMnx + OpenStreetMap |
| Graph engine | NetworkX (DiGraph) |
| Heuristic | Haversine formula (math only, no external lib) |
| Visualization (static) | Matplotlib |
| Web app (phase 2) | FastAPI + Leaflet.js |
| Property-based testing | Hypothesis |
| Data serialization | Python `json` stdlib |

---

## Components and Interfaces

### map_loader.py
```python
def load_graph(center: tuple, dist: int, seed: int) -> nx.MultiDiGraph
# Fetches OSM graph, assigns all edge metrics, returns annotated graph

def assign_metrics(G: nx.MultiDiGraph, seed: int) -> None
# Assigns traffic_factor, safety_factor, pothole_factor, custom_weight to all edges
```

### graph_utils.py
```python
def nearest_node(G, lat, lon) -> int
# Returns nearest node ID using projected graph lookup

def add_custom_node(G, node_id, lat, lon, k_neighbors=2) -> None
# Inserts a custom node and connects it to k nearest existing nodes

def verify_nodes(G, start, goal) -> None
# Raises ValueError if either node is not in G
```

### evaluation.py
```python
def compute_g(path: list[int], G, weight: str) -> float
# Cumulative cost from start to current node along path

def compute_h(G, node: int, goal: int) -> float
# Calls haversine heuristic

def compute_f(path: list[int], G, weight: str, goal: int) -> float
# Returns g(n) + h(n)
```

### heuristic.py
```python
def haversine(G, node_a: int, node_b: int) -> float
# Returns straight-line distance in meters between two nodes using their lat/lon
```

### algorithms/ (each module)
Each algorithm module exposes a single function:
```python
def search(G, start: int, goal: int, weight: str | None) -> SearchResult
```
Where `SearchResult` is a dataclass:
```python
@dataclass
class SearchResult:
    path: list[int]          # Ordered node IDs from start to goal
    nodes_expanded: int      # Count of nodes popped from frontier
    expansion_log: list[int] # Ordered list of expanded node IDs (for state tree)
```

### metrics.py
```python
def compute_path_cost(G, path: list[int], weight: str) -> float
def compute_hop_count(path: list[int]) -> int
def build_comparison_record(name: str, result: SearchResult, G, weight: str) -> dict
```

### state_tree.py
```python
def build_tree(expansion_log: list[int], parent_map: dict[int,int]) -> dict
# Returns nested dict representing the state-space tree

def serialize_tree(tree: dict) -> str
# Returns JSON string of the tree
```

### comparison.py
```python
def run_all(G, start, goal) -> list[dict]
# Runs all 8 algorithms, collects records, returns list

def export_json(records: list[dict], path: str) -> None
def plot_comparison(records: list[dict]) -> None
```

### web/app.py (Phase 2 — deferred)
FastAPI routes (to be implemented later):
- `GET /` — serves index.html with embedded Leaflet map
- `GET /api/graph` — returns GeoJSON of the road network
- `POST /api/run` — accepts `{algorithm, start, goal}`, returns path + metrics
- `GET /api/compare` — returns all algorithm comparison records
- `GET /api/tree/<algorithm>` — returns state-space tree JSON

---

## Data Models

### Edge attributes (stored in NetworkX edge data dict)
```python
{
  "length": float,           # meters, from OSM
  "traffic_factor": float,   # [1.0, 4.0]
  "safety_factor": float,    # [0.5, 1.0]
  "pothole_factor": float,   # [1.0, 3.0]
  "custom_weight": float     # length * traffic * pothole / safety
}
```

### SearchResult dataclass
```python
@dataclass
class SearchResult:
    path: list[int]
    nodes_expanded: int
    expansion_log: list[int]
    parent_map: dict[int, int]
```

### ComparisonRecord (dict / JSON)
```python
{
  "algorithm": str,
  "path_cost": float,
  "hop_count": int,
  "nodes_expanded": int,
  "path": list[int]
}
```

### StateSpaceTree (JSON)
```python
{
  "id": int,
  "children": [ { "id": int, "children": [...] } ]
}
```

### Custom Weight Formula
```
custom_weight = length * traffic_factor * pothole_factor / safety_factor
```
- Higher traffic → higher cost
- Higher potholes → higher cost
- Higher safety → lower cost (safer roads are preferred)

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

---

**Property 1: All edge metrics are within valid ranges**

*For any* graph produced by `assign_metrics`, every edge must have `traffic_factor` in [1.0, 4.0], `safety_factor` in [0.5, 1.0], and `pothole_factor` in [1.0, 3.0].

**Validates: Requirements 2.1, 2.2, 2.3**

---

**Property 2: Custom weight formula correctness**

*For any* edge in the graph, `custom_weight` must equal `length * traffic_factor * pothole_factor / safety_factor` within floating-point tolerance.

**Validates: Requirements 2.4**

---

**Property 3: Metric assignment is deterministic**

*For any* graph and fixed random seed, calling `assign_metrics` twice must produce identical `custom_weight` values on all edges.

**Validates: Requirements 2.5**

---

**Property 4: Custom node insertion**

*For any* valid (lat, lon) coordinate pair, after calling `add_custom_node`, the node must exist in the graph and have at least one outgoing edge with a valid `custom_weight > 0`.

**Validates: Requirements 3.2**

---

**Property 5: Path validity**

*For any* algorithm result with a non-empty path, every consecutive pair of nodes `(path[i], path[i+1])` must be connected by an edge in the graph, and no node ID must appear more than once in the path.

**Validates: Requirements 4.2, 5.2**

---

**Property 6: BFS and IDS return the shallowest path**

*For any* connected graph, start, and goal, BFS hop count and IDS hop count must both equal the minimum hop count returned by NetworkX's unweighted shortest path.

**Validates: Requirements 4.1, 4.3**

---

**Property 7: UCS returns minimum-cost path**

*For any* connected graph with positive `custom_weight` edges, UCS total path cost must equal the cost returned by NetworkX Dijkstra with `custom_weight`.

**Validates: Requirements 4.4**

---

**Property 8: Optimal informed algorithms agree with UCS**

*For any* connected graph, A* and IDA* must return a path whose total `custom_weight` equals the UCS optimal cost (since the haversine heuristic is admissible).

**Validates: Requirements 5.1, 5.3**

---

**Property 9: Heuristic admissibility and symmetry**

*For any* two nodes `a` and `b` in the graph: `haversine(a, b) >= 0`, `haversine(a, a) == 0`, and `haversine(a, b) == haversine(b, a)`. Additionally, `haversine(node, goal) <= actual_custom_weight_cost(node, goal)` for all nodes.

**Validates: Requirements 5.5**

---

**Property 10: Comparison record completeness**

*For any* algorithm execution, the resulting comparison record must contain non-null values for `algorithm`, `path_cost`, `hop_count`, and `nodes_expanded`.

**Validates: Requirements 6.1**

---

**Property 11: JSON serialization round-trip**

*For any* comparison record list or state-space tree, serializing to JSON and deserializing must produce a value equal to the original.

**Validates: Requirements 6.2, 7.3**

---

**Property 12: State-space tree parent coverage**

*For any* completed search, every node in `expansion_log` except the start node must have an entry in `parent_map`.

**Validates: Requirements 7.1**

---

## Error Handling

| Scenario | Handling |
|---|---|
| OSM fetch fails | Raise `ConnectionError` with message, exit cleanly |
| Start/goal node not in graph | Raise `ValueError("Node {id} not found in graph")` |
| No path between start and goal | Return `SearchResult(path=[], nodes_expanded=N, ...)` and log warning |
| Invalid custom node coordinates | Raise `ValueError("Coordinates out of graph bounding box")` |
| Division by zero in weight formula | `safety_factor` is clamped to minimum 0.1 before division |

---

## Testing Strategy

### Property-Based Testing (Hypothesis)

The primary testing tool is **Hypothesis** (Python). Each correctness property above maps to exactly one `@given`-decorated test. Tests are configured to run a minimum of 100 examples each.

Each property-based test is tagged with:
```
# Feature: ai-pathfinding, Property N: <property text>
```

Generators used:
- `st.integers()` filtered to valid node IDs sampled from a small synthetic graph
- `st.floats(min_value=..., max_value=...)` for metric values
- Small synthetic `nx.DiGraph` instances built via Hypothesis strategies (avoids slow OSM fetches in tests)

### Unit Testing (pytest)

Unit tests cover:
- Specific known examples (DU → Bashundhara path is non-empty)
- Edge cases: disconnected graph returns empty path, invalid node raises error
- Integration: `main.py` runs all 8 algorithms without exception on the cached graph

### Test File Layout

```
tests/
├── test_metrics.py          # Properties 1, 2, 3
├── test_graph_utils.py      # Properties 4, 12
├── test_algorithms.py       # Properties 5, 6, 7, 8
├── test_heuristic.py        # Property 9
├── test_comparison.py       # Properties 10, 11
└── test_integration.py      # Unit/integration examples
```

### Notes on Performance

- All property tests use small synthetic graphs (20–50 nodes) to keep test runtime under 30 seconds.
- The real OSM graph is only used in integration tests, which are marked `@pytest.mark.slow` and skipped by default.
- IDA* and Bidirectional A* are tested on graphs with guaranteed short paths to avoid exponential blowup.
- Web app (Phase 2) testing strategy will be defined when that phase begins.
