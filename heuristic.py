import math
import networkx as nx

_EARTH_RADIUS_KM = 6_371.0  # kilometres


def haversine_coords(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine great-circle distance in METRES between two coordinates.
    Used by build_custom_graph for raw distance before /1000 conversion.
    """
    R_m = _EARTH_RADIUS_KM * 1000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * R_m * math.asin(math.sqrt(a))


def haversine(G, node_a: int, node_b: int) -> float:
    """
    Heuristic h(n): straight-line distance in KILOMETRES between two graph nodes.
    Matches the km-based custom_weight units so A* remains admissible.
    """
    a = G.nodes[node_a]
    b = G.nodes[node_b]
    return haversine_coords(a['y'], a['x'], b['y'], b['x']) / 1000.0
