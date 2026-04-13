import math
import networkx as nx

_EARTH_RADIUS_M = 6_371_000  # meters


def haversine_coords(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the haversine (great-circle) distance in meters between two
    geographic coordinates.

    h(n) is admissible: straight-line distance <= any road path distance.
    h(n) is consistent: satisfies triangle inequality.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def haversine(G: nx.MultiDiGraph, node_a: int, node_b: int) -> float:
    """
    Compute haversine distance in meters between two nodes in graph G.
    Nodes must have 'y' (lat) and 'x' (lon) attributes.
    """
    a = G.nodes[node_a]
    b = G.nodes[node_b]
    return haversine_coords(a['y'], a['x'], b['y'], b['x'])
