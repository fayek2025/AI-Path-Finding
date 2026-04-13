import os
import pickle
import random
import osmnx as ox
import networkx as nx

# Bounding region: midpoint between DU and Bashundhara, 8km radius
_CENTER = (23.7766, 90.4227)
_DIST = 8000
_CACHE_FILE = "cache/graph.pkl"


def load_graph(center: tuple = _CENTER, dist: int = _DIST, seed: int = 42) -> nx.MultiDiGraph:
    """
    Load the OSM drivable road graph for the given area.
    Uses a local pickle cache to avoid repeated network calls.
    Assigns all custom edge metrics before returning.
    """
    G = _fetch_or_load_cached(center, dist)
    assign_metrics(G, seed)
    return G


def _fetch_or_load_cached(center: tuple, dist: int) -> nx.MultiDiGraph:
    """Return cached graph if available, otherwise fetch from OSM and cache it."""
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(_CACHE_FILE):
        print("Loading graph from cache...")
        with open(_CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Fetching map data from OpenStreetMap... this might take a moment.")
    try:
        G = ox.graph_from_point(center, dist=dist, network_type='drive')
    except Exception as e:
        raise ConnectionError(f"Failed to fetch OSM map data: {e}") from e

    with open(_CACHE_FILE, "wb") as f:
        pickle.dump(G, f)
    print("Graph cached to disk.")
    return G


def assign_metrics(G: nx.MultiDiGraph, seed: int = 42) -> None:
    """
    Assign custom edge metrics to every edge in G (in-place).

    Metrics:
      traffic_factor  : [1.0, 4.0]  — congestion level
      safety_factor   : [0.5, 1.0]  — road safety for women
      pothole_factor  : [1.0, 3.0]  — road surface quality
      custom_weight   : length * traffic * pothole / safety
    """
    rng = random.Random(seed)
    for u, v, k, data in G.edges(data=True, keys=True):
        length = data.get('length', 1.0)
        traffic = rng.uniform(1.0, 4.0)
        safety = max(rng.uniform(0.5, 1.0), 0.1)   # clamp to avoid div/0
        pothole = rng.uniform(1.0, 3.0)

        data['traffic_factor'] = traffic
        data['safety_factor'] = safety
        data['pothole_factor'] = pothole
        data['custom_weight'] = (length * traffic * pothole) / safety
