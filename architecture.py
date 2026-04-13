import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt

# 1. Configuration & Map Loading
place_name = "Dhaka, Bangladesh"
# We define a bounding box or a point-dist to cover DU to Bashundhara
# Coordinates for DU (approx) and Bashundhara (approx)
start_coords = (23.7340, 90.3928) # Dhaka University
goal_coords = (23.8193, 90.4526)  # Bashundhara R-A

print("Fetching map data... this might take a moment.")
# Fetch graph within a 8000m radius of the midpoint to ensure both areas are covered
G = ox.graph_from_point((23.7766, 90.4227), dist=8000, network_type='drive')

# 2. Assign Custom Metrics (Traffic & Safety)
# In AI formulation, weight = length * traffic * (1/safety)
random.seed(42) # For reproducibility
for u, v, k, data in G.edges(data=True, keys=True):
    length = data.get('length', 1)
    
    # Traffic: 1.0 (free flow) to 4.0 (gridlock)
    # Safety: 1.0 (very safe) to 0.5 (risky - increases cost)
    traffic_factor = random.uniform(1.0, 4.0)
    safety_factor = random.uniform(0.5, 1.0) 
    
    # Custom Weight Formulation
    # Higher weight = less desirable road
    data['custom_weight'] = (length * traffic_factor) / safety_factor

# 3. Setting Nodes Manually
# Finding the nearest network nodes to our real-world coordinates
start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
goal_node = ox.distance.nearest_nodes(G, goal_coords[1], goal_coords[0])

print(f"Start Node ID (DU): {start_node}")
print(f"Goal Node ID (Bashundhara): {goal_node}")

# 4. Implementing Search Algorithms

# A. BFS (Unweighted - ignores traffic/safety, only looks at number of steps)
bfs_path = nx.shortest_path(G, source=start_node, target=goal_node, weight=None)

# B. UCS (Weighted - considers distance, traffic, and safety)
ucs_path = nx.shortest_path(G, source=start_node, target=goal_node, weight='custom_weight')

# C. Depth Limited Search (DLS)
# We use a limit of 50 nodes deep for this example
try:
    dls_path = list(nx.dfs_preorder_nodes(G, source=start_node, depth_limit=50))
    # Note: nx.dfs_preorder_nodes returns a traversal, not necessarily the path to goal.
    # For a specific path to goal using DLS, you'd use a custom DFS function:
    def get_dls_path(graph, start, goal, limit):
        for path in nx.all_simple_paths(graph, source=start, target=goal, cutoff=limit):
            return path # Returns the first path found within limit
        return None
    path_dls = get_dls_path(G, start_node, goal_node, 100)
except nx.NetworkXNoPath:
    path_dls = None

# 5. Visualization
print("Visualizing Paths...")
# Red for UCS (The 'Smart' path), Blue for BFS (The 'Shortest Hops' path)
fig, ax = ox.plot_graph_routes(G, routes=[ucs_path, bfs_path], 
                               route_colors=['r', 'b'], 
                               route_linewidth=4, node_size=0)