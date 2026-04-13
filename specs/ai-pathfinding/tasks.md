# Implementation Plan — AI Pathfinding (Phase 1)

- [x] 1. Set up project structure and core data types


  - Create the `algorithms/` directory and all module files as empty stubs
  - Define the `SearchResult` dataclass in a shared `models.py`
  - Define the `EvaluationResult` dataclass for storing f(n), g(n), h(n) per node
  - Add `hypothesis` and `pytest` to `pyproject.toml` dev dependencies
  - _Requirements: 1.1, 1.2_



- [ ] 2. Implement map loader and edge metric assignment
- [ ] 2.1 Implement `map_loader.py` with graph fetch and metric assignment
  - Fetch OSM graph using OSMnx for the DU–Bashundhara corridor
  - Assign `traffic_factor`, `safety_factor`, `pothole_factor`, and `custom_weight` to all edges
  - Cache the graph to disk to avoid repeated network calls
  - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 2.2 Write property test for metric range validity (Property 1)
  - **Property 1: All edge metrics are within valid ranges**
  - **Validates: Requirements 2.1, 2.2, 2.3**

- [ ]* 2.3 Write property test for custom weight formula (Property 2)
  - **Property 2: Custom weight formula correctness**
  - **Validates: Requirements 2.4**

- [ ]* 2.4 Write property test for deterministic metric assignment (Property 3)
  - **Property 3: Metric assignment is deterministic**


  - **Validates: Requirements 2.5**

- [ ] 3. Implement graph utilities
- [ ] 3.1 Implement `graph_utils.py` with nearest-node lookup, custom node insertion, and node verification
  - Use projected graph for nearest-node to avoid scikit-learn dependency on unprojected graphs
  - _Requirements: 3.1, 3.2, 3.3_



- [ ]* 3.2 Write property test for custom node insertion (Property 4)
  - **Property 4: Custom node insertion**
  - **Validates: Requirements 3.2**



- [ ] 4. Implement heuristic and evaluation functions
- [ ] 4.1 Implement `heuristic.py` with haversine distance heuristic h(n)
  - Pure math implementation using lat/lon from graph node attributes
  - h(n) = haversine distance in meters from node n to goal node
  - _Requirements: 5.5_

- [ ] 4.2 Implement `evaluation.py` with evaluation function f(n) = g(n) + h(n)
  - `g(n)`: cumulative `custom_weight` from start to node n along current path
  - `h(n)`: haversine heuristic from `heuristic.py`


  - `f(n)`: combined evaluation score used by informed algorithms
  - Expose `compute_f`, `compute_g`, `compute_h` as standalone functions
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 4.3 Write property test for heuristic admissibility and symmetry (Property 9)
  - **Property 9: Heuristic admissibility and symmetry**
  - **Validates: Requirements 5.5**

- [ ] 5. Implement uninformed search algorithms
- [ ] 5.1 Implement BFS in `algorithms/bfs.py`
  - Returns `SearchResult` with path, nodes_expanded, expansion_log, parent_map
  - _Requirements: 4.1, 4.5_

- [ ] 5.2 Implement DFS in `algorithms/dfs.py`
  - Iterative DFS with visited set to prevent cycles
  - _Requirements: 4.2, 4.5_

- [ ] 5.3 Implement IDS in `algorithms/ids.py`
  - Iteratively increases depth limit from 1 until goal is found
  - _Requirements: 4.3, 4.5_

- [ ] 5.4 Implement UCS in `algorithms/ucs.py`
  - Uses a priority queue ordered by cumulative `custom_weight`
  - _Requirements: 4.4, 4.5_

- [ ]* 5.5 Write property test for path validity across all algorithms (Property 5)
  - **Property 5: Path validity**
  - **Validates: Requirements 4.2, 5.2**

- [ ]* 5.6 Write property test for BFS and IDS returning shallowest path (Property 6)
  - **Property 6: BFS and IDS return the shallowest path**
  - **Validates: Requirements 4.1, 4.3**

- [ ]* 5.7 Write property test for UCS returning minimum-cost path (Property 7)
  - **Property 7: UCS returns minimum-cost path**
  - **Validates: Requirements 4.4**

- [ ] 6. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement informed search algorithms
- [ ] 7.1 Implement A* in `algorithms/astar.py`
  - Priority queue on f(n) = g(n) + h(n) using `evaluation.py`
  - _Requirements: 5.1, 5.6_

- [ ] 7.2 Implement Greedy Best-First Search in `algorithms/greedy.py`
  - Priority queue on h(n) only using `heuristic.py`
  - _Requirements: 5.2, 5.6_

- [ ] 7.3 Implement IDA* in `algorithms/idastar.py`
  - Recursive DFS with f-cost threshold using `evaluation.py`, increases threshold each iteration
  - _Requirements: 5.3, 5.6_

- [ ] 7.4 Implement Bidirectional A* in `algorithms/bidirectional_astar.py`
  - Two simultaneous A* frontiers using `evaluation.py`, merge when they meet
  - _Requirements: 5.4, 5.6_

- [ ]* 7.5 Write property test for optimal informed algorithms agreeing with UCS (Property 8)
  - **Property 8: Optimal informed algorithms agree with UCS**
  - **Validates: Requirements 5.1, 5.3**

- [ ] 8. Implement metrics collection and state-space tree
- [ ] 8.1 Implement `metrics.py` with path cost, hop count, and comparison record builder
  - _Requirements: 6.1_

- [ ] 8.2 Implement `state_tree.py` with tree builder and JSON serializer
  - _Requirements: 7.1, 7.3_

- [ ]* 8.3 Write property test for comparison record completeness (Property 10)
  - **Property 10: Comparison record completeness**
  - **Validates: Requirements 6.1**

- [ ]* 8.4 Write property test for JSON serialization round-trip (Property 11)
  - **Property 11: JSON serialization round-trip**
  - **Validates: Requirements 6.2, 7.3**

- [ ]* 8.5 Write property test for state-space tree parent coverage (Property 12)
  - **Property 12: State-space tree parent coverage**
  - **Validates: Requirements 7.1**

- [ ] 9. Implement comparison aggregator and matplotlib visualization
- [ ] 9.1 Implement `comparison.py` to run all 8 algorithms, collect records, export JSON, and plot bar chart
  - _Requirements: 6.1, 6.2, 6.3_

- [ ] 10. Wire everything together in `main.py`
- [ ] 10.1 Implement `main.py` as the single entry point
  - Load graph, set start/goal nodes, run all algorithms, print results table, show matplotlib plots
  - _Requirements: 1.1, 3.1, 4.1–4.5, 5.1–5.6, 6.1–6.3, 7.1–7.3_

- [ ] 11. Final Checkpoint — Ensure all tests pass, ask the user if questions arise.
