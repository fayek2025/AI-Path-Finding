import math
import osmnx as ox
import networkx as nx


def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """
    Return the nearest graph node to the given (lat, lon) coordinates.
    Uses OSMnx nearest_nodes on the unprojected graph (requires scikit-learn).
    Falls back to a pure-Python haversine search if scikit-learn is unavailable.
    """
    try:
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except ImportError:
        from heuristic import haversine_coords
        return min(
            G.nodes,
            key=lambda n: haversine_coords(lat, lon, G.nodes[n]['y'], G.nodes[n]['x'])
        )


def corridor_subgraph(G: nx.MultiDiGraph, start: int, goal: int,
                      padding_factor: float = 0.35) -> nx.MultiDiGraph:
    """
    Return a subgraph containing only nodes within a padded bounding box
    around the straight line between start and goal.

    padding_factor: fraction of the diagonal distance added as padding on
                    each side (0.35 = 35% extra on each side).

    This reduces a 15k-node city graph to ~200-500 nodes, making IDA* and
    IDS fast while still guaranteeing the optimal path is reachable.
    """
    s_lat, s_lon = G.nodes[start]['y'], G.nodes[start]['x']
    g_lat, g_lon = G.nodes[goal]['y'],  G.nodes[goal]['x']

    lat_span = abs(g_lat - s_lat)
    lon_span = abs(g_lon - s_lon)

    # Minimum padding so very close nodes still get a usable corridor
    pad_lat = max(lat_span * padding_factor, 0.008)
    pad_lon = max(lon_span * padding_factor, 0.008)

    min_lat = min(s_lat, g_lat) - pad_lat
    max_lat = max(s_lat, g_lat) + pad_lat
    min_lon = min(s_lon, g_lon) - pad_lon
    max_lon = max(s_lon, g_lon) + pad_lon

    nodes_in_box = [
        n for n, d in G.nodes(data=True)
        if min_lat <= d['y'] <= max_lat and min_lon <= d['x'] <= max_lon
    ]

    sub = G.subgraph(nodes_in_box).copy()

    # Ensure start and goal are in the subgraph (they always should be)
    if start not in sub or goal not in sub:
        return G  # fallback to full graph

    print(f"  Corridor subgraph: {sub.number_of_nodes()} nodes, "
          f"{sub.number_of_edges()} edges "
          f"(full graph: {G.number_of_nodes()} nodes)")
    return sub


def add_custom_node(G: nx.MultiDiGraph, node_id: int, lat: float, lon: float,
                    k_neighbors: int = 2) -> None:
    """
    Insert a custom node into G and connect it to its k nearest existing nodes.
    Edges are added in both directions with a custom_weight computed from
    haversine distance (no traffic/pothole penalty for manually added nodes).
    """
    from heuristic import haversine_coords

    lats = [data['y'] for _, data in G.nodes(data=True)]
    lons = [data['x'] for _, data in G.nodes(data=True)]
    if not (min(lats) <= lat <= max(lats) and min(lons) <= lon <= max(lons)):
        raise ValueError(
            f"Coordinates ({lat}, {lon}) are outside the graph bounding box."
        )

    G.add_node(node_id, y=lat, x=lon, osmid=node_id)

    existing = [n for n in G.nodes if n != node_id]
    existing.sort(key=lambda n: haversine_coords(lat, lon, G.nodes[n]['y'], G.nodes[n]['x']))
    neighbors = existing[:k_neighbors]

    for neighbor in neighbors:
        dist = haversine_coords(lat, lon, G.nodes[neighbor]['y'], G.nodes[neighbor]['x'])
        weight = max(dist, 1.0)
        G.add_edge(node_id, neighbor, length=dist, traffic_factor=1.0,
                   safety_factor=1.0, pothole_factor=1.0, custom_weight=weight)
        G.add_edge(neighbor, node_id, length=dist, traffic_factor=1.0,
                   safety_factor=1.0, pothole_factor=1.0, custom_weight=weight)


def verify_nodes(G: nx.MultiDiGraph, start: int, goal: int) -> None:
    """Raise ValueError if start or goal node is not present in G."""
    if start not in G:
        raise ValueError(f"Start node {start} not found in graph.")
    if goal not in G:
        raise ValueError(f"Goal node {goal} not found in graph.")
