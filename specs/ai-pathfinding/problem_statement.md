# Problem Statement: Multi-Metric AI Pathfinding in Dhaka Road Network

## 1. Problem Domain

This problem falls within the **Search and Problem Solving** area of classical Artificial Intelligence. Specifically, it is a **single-source, single-goal graph search problem** in a **weighted, directed state space** derived from a real-world road network.

The system models route planning in Dhaka, Bangladesh as an AI search problem where the agent must find the best path from a start location to a goal location, optimizing across multiple real-world cost dimensions simultaneously.

---

## 2. Problem Formulation

### 2.1 State Space

- **State**: A node (road intersection) in the directed road graph G = (V, E), where V is the set of all intersections and E is the set of all directed road segments.
- **Initial State**: The node nearest to Dhaka University (DU) coordinates (23.7340°N, 90.3928°E).
- **Goal State**: The node nearest to Bashundhara Residential Area coordinates (23.8193°N, 90.4526°E).
- **State Transition**: Moving from node u to node v is valid if and only if a directed edge (u, v) exists in G.

### 2.2 Actions

At each state (node), the agent may move to any directly connected neighbor via an outgoing directed edge. The set of available actions at node u is:

```
Actions(u) = { v | (u, v) ∈ E }
```

### 2.3 Path Cost (Edge Weight)

Each edge (u, v) carries a composite cost called `custom_weight`, computed as:

```
custom_weight(u, v) = length(u,v) × traffic_factor(u,v) × pothole_factor(u,v)
                      ─────────────────────────────────────────────────────────
                                      safety_factor(u,v)
```

Where:
| Metric | Range | Meaning |
|---|---|---|
| `length` | > 0 meters | Physical road segment length from OSM |
| `traffic_factor` | [1.0, 4.0] | 1.0 = free flow, 4.0 = gridlock |
| `pothole_factor` | [1.0, 3.0] | 1.0 = smooth road, 3.0 = heavily damaged |
| `safety_factor` | [0.5, 1.0] | 1.0 = very safe, 0.5 = risky for women |

Higher `custom_weight` = less desirable road segment.

### 2.4 Goal Test

```
is_goal(node) = (node == goal_node)
```

### 2.5 Solution

A sequence of nodes [s, n₁, n₂, ..., g] where:
- s = start node
- g = goal node
- Each consecutive pair (nᵢ, nᵢ₊₁) is a valid directed edge in G
- Total path cost = Σ custom_weight(nᵢ, nᵢ₊₁)

---

## 3. Evaluation Function

The **evaluation function f(n)** determines the priority of a node n during search. It varies by algorithm:

### Uninformed Search
| Algorithm | f(n) |
|---|---|
| BFS | Depth level (hop count from start) |
| DFS | No explicit f(n); uses LIFO stack |
| IDS | Depth level, with increasing cutoff |
| UCS | g(n) = cumulative custom_weight from start to n |

### Informed Search
| Algorithm | f(n) |
|---|---|
| Greedy Best-First | h(n) |
| A* | g(n) + h(n) |
| IDA* | g(n) + h(n), with iterative threshold |
| Bidirectional A* | g(n) + h(n) from both directions |

Where:
- **g(n)** = actual cumulative cost from start to node n (sum of `custom_weight` along the path)
- **h(n)** = heuristic estimate of cost from n to goal (see Section 4)
- **f(n)** = total estimated cost of the cheapest solution through n

---

## 4. Heuristic Function

The **heuristic function h(n)** estimates the remaining cost from node n to the goal. We use the **Haversine distance** (straight-line geographic distance in meters) as our heuristic.

### Formula

```
h(n, goal) = 2R × arcsin( √( sin²(Δlat/2) + cos(lat_n) × cos(lat_goal) × sin²(Δlon/2) ) )
```

Where:
- R = 6,371,000 meters (Earth's radius)
- Δlat = lat_goal − lat_n (in radians)
- Δlon = lon_goal − lon_n (in radians)

### Admissibility

h(n) is **admissible** because:
- Roads cannot be shorter than the straight-line distance between two points
- Therefore h(n) ≤ actual_cost(n, goal) for all n
- This guarantees A* and IDA* return optimal solutions

### Consistency (Monotonicity)

h(n) is **consistent** because for any edge (n, m):
```
h(n) ≤ cost(n, m) + h(m)
```
This follows from the triangle inequality of geographic distances.

---

## 5. Algorithms to be Compared

### Uninformed (Blind) Search
1. **BFS** — Explores level by level; optimal for hop count, ignores weights
2. **DFS** — Explores deep first; not optimal, may find long paths
3. **IDS** — Combines DFS memory efficiency with BFS optimality for hop count
4. **UCS** — Expands cheapest node first; optimal for weighted cost

### Informed (Heuristic) Search
5. **Greedy Best-First** — Expands node closest to goal; fast but not optimal
6. **A\*** — Balances actual cost and heuristic; optimal and complete
7. **IDA\*** — Memory-efficient A* using iterative deepening on f-cost
8. **Bidirectional A\*** — Searches from both ends simultaneously; faster in practice

---

## 6. Comparison Metrics

Each algorithm will be evaluated on:

| Metric | Description |
|---|---|
| **Path Cost** | Total `custom_weight` of the returned path |
| **Hop Count** | Number of edges (road segments) in the path |
| **Nodes Expanded** | Total nodes popped from the frontier during search |
| **Optimality** | Whether the returned path is provably optimal |
| **Completeness** | Whether the algorithm always finds a path if one exists |

---

## 7. Simulated World Assumptions

- Edge metrics (traffic, safety, potholes) are randomly assigned with a fixed seed for reproducibility.
- The graph is treated as static (no real-time updates).
- All edge weights are positive (required for UCS, A*, IDA* correctness).
- The graph may not be fully connected; algorithms handle the no-path case gracefully.
