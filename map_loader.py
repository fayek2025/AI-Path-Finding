import os
import pickle
import random
import osmnx as ox
import networkx as nx

_CACHE_FILE = "cache/graph.pkl"

# Road types considered "highway" (accessible roads for this simulation)
# Ordered from highest to lowest class
HIGHWAY_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary',
    'tertiary', 'residential', 'unclassified', 'service'
}

# Road type → base metric values
# Higher class = faster/more traffic; residential = safer but slower
_ROAD_PROFILES = {
    'motorway':     {'traffic': 2.2, 'safety': 0.70, 'pothole': 1.1, 'speed_limit': 80, 'road_age': 0.8,  'turn_complexity': 0.3},
    'trunk':        {'traffic': 2.0, 'safety': 0.72, 'pothole': 1.15,'speed_limit': 60, 'road_age': 0.85, 'turn_complexity': 0.4},
    'primary':      {'traffic': 1.8, 'safety': 0.78, 'pothole': 1.2, 'speed_limit': 50, 'road_age': 0.9,  'turn_complexity': 0.5},
    'secondary':    {'traffic': 1.5, 'safety': 0.82, 'pothole': 1.3, 'speed_limit': 40, 'road_age': 1.0,  'turn_complexity': 0.6},
    'tertiary':     {'traffic': 1.3, 'safety': 0.85, 'pothole': 1.4, 'speed_limit': 30, 'road_age': 1.1,  'turn_complexity': 0.7},
    'residential':  {'traffic': 1.1, 'safety': 0.92, 'pothole': 1.5, 'speed_limit': 20, 'road_age': 1.3,  'turn_complexity': 0.9},
    'service':      {'traffic': 1.0, 'safety': 0.95, 'pothole': 1.6, 'speed_limit': 15, 'road_age': 1.5,  'turn_complexity': 1.0},
    'unclassified': {'traffic': 1.2, 'safety': 0.88, 'pothole': 1.45,'speed_limit': 25, 'road_age': 1.2,  'turn_complexity': 0.8},
}
_DEFAULT_PROFILE = {'traffic': 1.4, 'safety': 0.80, 'pothole': 1.35, 'speed_limit': 30, 'road_age': 1.1, 'turn_complexity': 0.7}

# Number of geographically diverse paths to seed the subgraph with.
K_PATHS = 4

# Road-type cost profiles used to build diverse candidate paths.
# Each profile biases edge weights so a different road class is preferred,
# ensuring the subgraph contains genuinely different route options.
_DIVERSITY_PROFILES = [
    # Profile 0 — raw length (shortest geographic path)
    {'motorway': 1.0, 'trunk': 1.0, 'primary': 1.0, 'secondary': 1.0,
     'tertiary': 1.0, 'residential': 1.0, 'service': 1.0, 'unclassified': 1.0},
    # Profile 1 — prefer high-speed roads (motorway/trunk/primary cheap)
    {'motorway': 0.2, 'trunk': 0.3, 'primary': 0.5, 'secondary': 0.8,
     'tertiary': 1.2, 'residential': 2.0, 'service': 2.5, 'unclassified': 1.5},
    # Profile 2 — prefer safe/quiet roads (residential/tertiary cheap)
    {'motorway': 3.0, 'trunk': 2.5, 'primary': 2.0, 'secondary': 1.5,
     'tertiary': 0.8, 'residential': 0.4, 'service': 0.6, 'unclassified': 0.9},
    # Profile 3 — balanced with slight secondary/tertiary preference
    {'motorway': 1.5, 'trunk': 1.3, 'primary': 1.0, 'secondary': 0.7,
     'tertiary': 0.6, 'residential': 1.0, 'service': 1.8, 'unclassified': 1.1},
]

def load_graph(center: tuple, dist: int = 8000, seed: int = 42) -> nx.MultiDiGraph:
    """
    Load OSM drivable road graph, assign road-type-aware metrics, return it.
    Uses a local pickle cache to avoid repeated network calls.
    """
    G = _fetch_or_load_cached(center, dist)
    assign_metrics(G, seed)
    return G


def _fetch_or_load_cached(center: tuple, dist: int) -> nx.MultiDiGraph:
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(_CACHE_FILE):
        print("  Loading OSM graph from cache...")
        with open(_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    print("  Fetching OSM road network... this may take a moment.")
    try:
        # network_type='drive' fetches all drivable roads from OSM
        # This includes motorway, trunk, primary, secondary, tertiary,
        # residential, service, unclassified — all real road types
        G = ox.graph_from_point(center, dist=dist, network_type='drive')
    except Exception as e:
        raise ConnectionError(f"Failed to fetch OSM map data: {e}") from e
    with open(_CACHE_FILE, "wb") as f:
        pickle.dump(G, f)
    print("  OSM graph cached to disk.")
    return G


def assign_metrics(G: nx.MultiDiGraph, seed: int = 42) -> None:
    """
    Assign road-type-aware metrics to every OSM edge using the 'highway' tag.

    Metrics:
      traffic_factor   : congestion level based on road class
      safety_factor    : safety score (residential = safer, motorway = less safe)
      pothole_factor   : road surface quality (motorway = smooth, service = rough)
      speed_limit      : effective speed limit in km/h (affects travel time)
      road_age_factor  : infrastructure age/quality (newer = lower cost)
      turn_complexity  : intersection complexity penalty
      custom_weight    : composite cost = length_km * traffic * pothole * road_age
                         * turn_complexity / safety
                         (higher = worse road to take)

    A ±10% random perturbation is added so parallel roads of the same type
    have slightly different costs, giving algorithms real choices.
    """
    rng = random.Random(seed)
    for u, v, k, data in G.edges(data=True, keys=True):
        length_km = data.get('length', 100.0) / 1000.0

        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0]
        base_type = highway.replace('_link', '').strip()
        profile = _ROAD_PROFILES.get(base_type, _DEFAULT_PROFILE)

        perturb = lambda v: max(v * rng.uniform(0.9, 1.1), 0.1)
        traffic    = round(perturb(profile['traffic']),         3)
        safety     = round(perturb(profile['safety']),          3)
        pothole    = round(perturb(profile['pothole']),         3)
        speed      = round(perturb(profile['speed_limit']),     1)
        road_age   = round(perturb(profile['road_age']),        3)
        turn_cmplx = round(perturb(profile['turn_complexity']), 3)

        data['highway_type']      = base_type
        data['traffic_factor']    = traffic
        data['safety_factor']     = safety
        data['pothole_factor']    = pothole
        data['speed_limit']       = speed
        data['road_age_factor']   = road_age
        data['turn_complexity']   = turn_cmplx
        data['custom_weight']     = round(
            (length_km * traffic * pothole * road_age * turn_cmplx) / safety, 4
        )


def _get_highway_type(osm_G: nx.MultiDiGraph, u: int, v: int, key: int) -> str:
    """Return the normalised highway type for an OSM edge."""
    hw = osm_G[u][v][key].get('highway', 'unclassified')
    if isinstance(hw, list):
        hw = hw[0]
    return hw.replace('_link', '').strip()


def build_search_graph(osm_G: nx.MultiDiGraph,
                       chosen_nodes: list,
                       seed: int = 42) -> nx.MultiDiGraph:
    """
    Extract a focused subgraph that contains genuinely diverse route options.

    Strategy:
      For each diversity profile (fast roads / safe roads / balanced / shortest),
      build a weighted DiGraph inside the bbox and find the shortest path under
      that profile's cost model. This guarantees the subgraph contains at least
      one motorway-biased path, one residential-biased path, etc.

      Algorithms then search this subgraph with the user's slider weights,
      so different slider settings genuinely select different paths.
    """
    start_id = chosen_nodes[0]['id']
    goal_id  = chosen_nodes[-1]['id']

    s_lat = osm_G.nodes[start_id]['y']
    s_lon = osm_G.nodes[start_id]['x']
    g_lat = osm_G.nodes[goal_id]['y']
    g_lon = osm_G.nodes[goal_id]['x']

    lat_pad = max(abs(g_lat - s_lat) * 0.25, 0.012)
    lon_pad = max(abs(g_lon - s_lon) * 0.25, 0.012)
    lat_min = min(s_lat, g_lat) - lat_pad
    lat_max = max(s_lat, g_lat) + lat_pad
    lon_min = min(s_lon, g_lon) - lon_pad
    lon_max = max(s_lon, g_lon) + lon_pad

    print(f"  Bbox: lat[{lat_min:.4f},{lat_max:.4f}] lon[{lon_min:.4f},{lon_max:.4f}]")

    # Pre-collect all bbox edges with their highway type and length
    bbox_edges = []
    for u, v, data in osm_G.edges(data=True):
        ud = osm_G.nodes[u]
        vd = osm_G.nodes[v]
        if not (lat_min <= ud['y'] <= lat_max and lon_min <= ud['x'] <= lon_max):
            continue
        if not (lat_min <= vd['y'] <= lat_max and lon_min <= vd['x'] <= lon_max):
            continue
        hw = data.get('highway', 'unclassified')
        if isinstance(hw, list):
            hw = hw[0]
        hw = hw.replace('_link', '').strip()
        if hw not in HIGHWAY_TYPES:
            continue
        length = data.get('length', 1.0)
        bbox_edges.append((u, v, hw, length))

    print(f"  Bbox edges: {len(bbox_edges)}")

    nodes_to_keep = set()
    edges_to_keep = {}

    # Find one path per diversity profile
    for profile_idx, profile in enumerate(_DIVERSITY_PROFILES):
        # Build a DiGraph weighted by this profile's road-type costs
        G_profile = nx.DiGraph()
        for u, v, hw, length in bbox_edges:
            hw_mult = profile.get(hw, 1.0)
            w = length * hw_mult
            if not G_profile.has_edge(u, v) or G_profile[u][v]['weight'] > w:
                G_profile.add_edge(u, v, weight=w)

        if start_id not in G_profile or goal_id not in G_profile:
            continue

        try:
            path = nx.shortest_path(G_profile, start_id, goal_id, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        print(f"  Profile {profile_idx}: {len(path)} nodes")
        nodes_to_keep.update(path)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not osm_G.has_edge(u, v):
                continue
            for edge_key, edge_data in osm_G[u][v].items():
                if _get_highway_type(osm_G, u, v, edge_key) not in HIGHWAY_TYPES:
                    continue
                ekey = (u, v, edge_key)
                if ekey not in edges_to_keep:
                    edges_to_keep[ekey] = dict(edge_data)

    # Also add K_PATHS geographically diverse paths (yen's algorithm on raw length)
    simple_G = nx.DiGraph()
    for u, v, hw, length in bbox_edges:
        if not simple_G.has_edge(u, v) or simple_G[u][v]['weight'] > length:
            simple_G.add_edge(u, v, weight=length)

    if start_id in simple_G and goal_id in simple_G:
        try:
            count = 0
            for path in nx.shortest_simple_paths(simple_G, start_id, goal_id, weight='weight'):
                nodes_to_keep.update(path)
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if not osm_G.has_edge(u, v):
                        continue
                    for edge_key, edge_data in osm_G[u][v].items():
                        if _get_highway_type(osm_G, u, v, edge_key) not in HIGHWAY_TYPES:
                            continue
                        ekey = (u, v, edge_key)
                        if ekey not in edges_to_keep:
                            edges_to_keep[ekey] = dict(edge_data)
                count += 1
                if count >= K_PATHS:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    # Build subgraph
    sub = nx.MultiDiGraph()
    for nid in nodes_to_keep:
        if nid not in osm_G.nodes:
            continue
        attrs = dict(osm_G.nodes[nid])
        for cn in chosen_nodes:
            if cn['id'] == nid:
                attrs['chosen'] = True
                attrs['label'] = cn['label']
                break
        sub.add_node(nid, **attrs)

    for (u, v, key), data in edges_to_keep.items():
        if u in sub and v in sub:
            sub.add_edge(u, v, key=key, **data)

    # Ensure START and GOAL are always present
    for cn in [chosen_nodes[0], chosen_nodes[-1]]:
        if cn['id'] not in sub and cn['id'] in osm_G.nodes:
            attrs = dict(osm_G.nodes[cn['id']])
            attrs['chosen'] = True
            attrs['label'] = cn['label']
            sub.add_node(cn['id'], **attrs)

    print(f"  Search graph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges\n")
    return sub
