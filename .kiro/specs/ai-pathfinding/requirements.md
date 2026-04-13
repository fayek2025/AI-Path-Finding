# Requirements Document

## Introduction

This project implements a multi-metric AI pathfinding system on a real-world road network (Dhaka, Bangladesh) sourced from OpenStreetMap. The system simulates and compares 6–8 classical AI search algorithms — both uninformed and informed — across a weighted graph where edges carry custom metrics such as traffic congestion, road safety for women, and pothole density. The goal is to find the optimal route from a start location (Dhaka University) to a goal location (Bashundhara Residential Area) under different optimization criteria. The system includes a web-based visualization interface and a state-space tree view for academic presentation.

## Glossary

- **Graph (G)**: A directed road network graph loaded from OpenStreetMap via OSMnx, where nodes are intersections and edges are road segments.
- **Edge Metrics**: Numerical attributes assigned to each edge representing traffic factor, safety factor, and pothole factor.
- **Custom Weight**: A composite scalar cost assigned to each edge, computed from edge metrics, used by weighted search algorithms.
- **Uninformed Search**: Search algorithms that explore the graph without domain-specific heuristic knowledge (BFS, DFS, IDS, UCS).
- **Informed Search**: Search algorithms that use a heuristic function to guide exploration toward the goal (A*, Greedy Best-First, Bidirectional A*, IDA*).
- **Heuristic Function (h)**: An admissible estimate of the cost from a node to the goal, based on haversine geographic distance.
- **State Space Tree**: A tree representation of nodes expanded during a search, showing the exploration path from start to goal.
- **Node**: An OSM intersection point in the road graph, identified by a unique integer ID.
- **Custom Node**: A user-defined node added to the graph with manually specified coordinates and edge connections.
- **Pathfinding System**: The overall software system that loads the map, assigns metrics, runs search algorithms, and visualizes results.
- **Web App**: A browser-based interface for interactive visualization of the graph, paths, and algorithm comparisons.

## Requirements

### Requirement 1

**User Story:** As a student, I want to load a real road network from OpenStreetMap for a defined geographic area, so that I can run search algorithms on a realistic map.

#### Acceptance Criteria

1. WHEN the system initializes, THE Pathfinding System SHALL fetch the drivable road network graph for a bounding region covering Dhaka University and Bashundhara Residential Area using OSMnx.
2. WHEN the graph is loaded, THE Pathfinding System SHALL represent the road network as a directed graph where nodes are intersections and edges are road segments with a `length` attribute in meters.
3. IF the map data fetch fails due to a network error, THEN THE Pathfinding System SHALL raise a descriptive error and terminate gracefully without producing partial output.

---

### Requirement 2

**User Story:** As a student, I want to assign custom multi-metric weights to road edges, so that the search algorithms can optimize for traffic, safety, and road quality simultaneously.

#### Acceptance Criteria

1. WHEN the graph is loaded, THE Pathfinding System SHALL assign each edge a `traffic_factor` value in the range [1.0, 4.0] representing congestion level.
2. WHEN the graph is loaded, THE Pathfinding System SHALL assign each edge a `safety_factor` value in the range [0.5, 1.0] representing road safety for women.
3. WHEN the graph is loaded, THE Pathfinding System SHALL assign each edge a `pothole_factor` value in the range [1.0, 3.0] representing road surface quality.
4. WHEN all edge metrics are assigned, THE Pathfinding System SHALL compute a `custom_weight` for each edge using the formula: `length * traffic_factor * pothole_factor / safety_factor`.
5. WHEN the random seed is set to a fixed value before metric assignment, THE Pathfinding System SHALL produce identical metric values across separate runs.

---

### Requirement 3

**User Story:** As a student, I want to define start and goal nodes by real-world coordinates or by manual node ID, so that I can control the pathfinding problem instance.

#### Acceptance Criteria

1. WHEN geographic coordinates are provided, THE Pathfinding System SHALL resolve the nearest graph node to those coordinates using OSMnx nearest-node lookup.
2. WHEN a user provides a custom node ID and coordinates, THE Pathfinding System SHALL insert that node into the graph and connect it to its nearest existing neighbors with computed edge weights.
3. WHEN start and goal nodes are set, THE Pathfinding System SHALL verify both nodes exist in the graph and raise a descriptive error if either is missing.

---

### Requirement 4

**User Story:** As a student, I want to run uninformed search algorithms on the road graph, so that I can observe how they explore the state space without heuristic guidance.

#### Acceptance Criteria

1. WHEN BFS is executed, THE Pathfinding System SHALL return the path with the fewest edge hops from start to goal, treating all edges as equal.
2. WHEN DFS is executed, THE Pathfinding System SHALL return a valid path from start to goal found by depth-first traversal, without revisiting nodes.
3. WHEN IDS is executed, THE Pathfinding System SHALL return the shallowest path found by iteratively increasing depth limits from 1 until the goal is reached.
4. WHEN UCS is executed, THE Pathfinding System SHALL return the path with the minimum total `custom_weight` from start to goal.
5. WHEN any uninformed algorithm is executed and no path exists, THE Pathfinding System SHALL return an empty result and log a no-path message.

---

### Requirement 5

**User Story:** As a student, I want to run informed search algorithms on the road graph, so that I can observe how heuristics improve search efficiency.

#### Acceptance Criteria

1. WHEN A* is executed, THE Pathfinding System SHALL return the path minimizing `custom_weight + h(node, goal)` where `h` is the haversine distance heuristic.
2. WHEN Greedy Best-First Search is executed, THE Pathfinding System SHALL return the path found by always expanding the node with the smallest `h(node, goal)` value.
3. WHEN IDA* is executed, THE Pathfinding System SHALL return the optimal path using iterative deepening on the f-cost threshold `custom_weight + h`.
4. WHEN Bidirectional A* is executed, THE Pathfinding System SHALL search simultaneously from start and goal and return the path when the two frontiers meet.
5. WHEN the heuristic function is evaluated for any node, THE Pathfinding System SHALL compute the haversine distance in meters between that node's coordinates and the goal node's coordinates.
6. WHEN any informed algorithm is executed and no path exists, THE Pathfinding System SHALL return an empty result and log a no-path message.

---

### Requirement 6

**User Story:** As a student, I want to compare all algorithms on standardized metrics, so that I can present a meaningful analysis to my teachers.

#### Acceptance Criteria

1. WHEN all algorithms complete execution, THE Pathfinding System SHALL record for each algorithm: total path cost (sum of `custom_weight`), path length in hops, and number of nodes expanded during search.
2. WHEN comparison data is collected, THE Pathfinding System SHALL serialize the results to a JSON file with algorithm name, path cost, hop count, and nodes expanded.
3. WHEN comparison data is available, THE Pathfinding System SHALL render a bar chart comparing all algorithms across the three metrics.

---

### Requirement 7

**User Story:** As a student, I want to visualize the state-space tree for a selected algorithm, so that I can explain the search process to my teachers.

#### Acceptance Criteria

1. WHEN a search algorithm completes, THE Pathfinding System SHALL record the parent-child expansion relationships of all nodes visited during the search.
2. WHEN the state-space tree is requested, THE Pathfinding System SHALL render a tree diagram showing nodes expanded, with the solution path highlighted.
3. WHEN the state-space tree is serialized, THE Pathfinding System SHALL produce a JSON representation of the tree that can be consumed by the web app.

---

### Requirement 8

**User Story:** As a student, I want a web application to interactively visualize the map, paths, and algorithm comparisons, so that I can present my work effectively.

#### Acceptance Criteria

1. WHEN the web app loads, THE Pathfinding System SHALL display the road network map centered on the Dhaka University to Bashundhara corridor.
2. WHEN an algorithm is selected and run via the web interface, THE Pathfinding System SHALL highlight the resulting path on the map with a distinct color per algorithm.
3. WHEN multiple algorithms have been run, THE Pathfinding System SHALL display a side-by-side comparison panel showing path cost, hops, and nodes expanded for each algorithm.
4. WHEN the state-space tree view is requested, THE Pathfinding System SHALL render an interactive tree diagram in the browser.
5. WHEN a user clicks a node on the map, THE Pathfinding System SHALL display that node's ID, coordinates, and all edge metric values for connected roads.
