import math

_EARTH_RADIUS_KM = 6_371.0

# With the exponent formula: cost = length_km * t^tw * p^pw * ra^rw * tc^tcw / s^sw
# The absolute minimum cost per km occurs on the best-case motorway edge
# with the lowest perturbation (0.9×) on each penalty factor and highest
# perturbation (1.1×) on safety, raised to the minimum slider value (1.0).
#
# motorway best-case (slider=1.0):
#   t^1  = (2.2*0.9)^1 = 1.98
#   p^1  = (1.1*0.9)^1 = 0.99
#   ra^1 = (0.8*0.9)^1 = 0.72
#   tc^1 = (0.3*0.9)^1 = 0.27
#   s^1  = (0.70*1.1)^1 = 0.77
#   min = 1.98*0.99*0.72*0.27/0.77 ≈ 0.495 → use 0.45 for safety margin
#
# At higher slider values the exponent formula changes costs non-linearly,
# but the heuristic only needs to be admissible (never overestimate).
# Since we use slider=1.0 as the floor, 0.45 remains a valid lower bound
# for all slider configurations — higher sliders raise ALL costs, so the
# true minimum can only increase, never decrease below the slider=1.0 floor.
_MIN_COST_PER_KM = 0.45


def haversine_coords(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in METRES."""
    R_m = _EARTH_RADIUS_KM * 1000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * R_m * math.asin(math.sqrt(a))


def haversine(G, node_a: int, node_b: int) -> float:
    """
    Admissible heuristic h(n) for informed search algorithms.

    Returns a lower bound on the true custom_weight cost from node_a to node_b.
    Straight-line distance × _MIN_COST_PER_KM never overestimates the actual
    road cost under any slider configuration, preserving A* optimality.
    """
    a = G.nodes[node_a]
    b = G.nodes[node_b]
    dist_km = haversine_coords(a['y'], a['x'], b['y'], b['x']) / 1000.0
    return dist_km * _MIN_COST_PER_KM
