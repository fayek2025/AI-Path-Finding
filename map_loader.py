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
    'motorway':     {'traffic': 2.2, 'safety': 0.70, 'pothole': 1.1},
    'trunk':        {'traffic': 2.0, 'safety': 0.72, 'pothole': 1.15},
    'primary':      {'traffic': 1.8, 'safety': 0.78, 'pothole': 1.2},
    'secondary':    {'traffic': 1.5, 'safety': 0.82, 'pothole': 1.3},
    'tertiary':     {'traffic': 1.3, 'safety': 0.85, 'pothole': 1.4},
    'residential':  {'traffic': 1.1, 'safety': 0.92, 'pothole': 1.5},
    'service':      {'traffic': 1.0, 'safety': 0.95, 'pothole': 1.6},
    'unclassified': {'traffic': 1.2, 'safety': 0.88, 'pothole': 1.45},
}
_DEFAULT_PROFILE = {'traffic': 1.4, 'safety': 0.80, 'pothole': 1.35}

# Number of alternative shortest paths to find between each node pair.
# More paths = richer graph = more route choices for algorithms.
K_PATHS = 3


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
      traffic_factor : congestion level based on road class
      safety_factor  : safety score (residential = safer, motorway = less safe)
      pothole_factor : road surface quality (motorway = smooth, service = rough)
      custom_weight  : length_km * traffic * pothole / safety

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
        traffic = round(perturb(profile['traffic']), 3)
        safety  = round(perturb(profile['safety']),  3)
        pothole = round(perturb(profile['pothole']), 3)

        data['highway_type']   = base_type
        data['traffic_factor'] = traffic
        data['safety_factor']  = safety
        data['pothole_factor'] = pothole
        data['custom_weight']  = round((length_km * traffic * pothole) / safety, 4)


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
    Extract a real OSM subgraph for the search algorithms.

    What this does:
    ──────────────
    1. For every pair of chosen nodes, find K shortest real road paths.
       (nx.shortest_simple_paths doesn't support MultiDiGraph, so we work
        on a collapsed DiGraph for path-finding, then pull real edges back.)
    2. Every OSM intersection along any of those paths becomes a node.
    3. ALL parallel edges between intersections are included (MultiDiGraph).
    4. Only roads in HIGHWAY_TYPES are included.
    5. Direct start→goal path edges get a congestion penalty.
    """
    start_id = chosen_nodes[0]['id']
    goal_id  = chosen_nodes[-1]['id']

    # ── Build a simple DiGraph for path-finding (collapse parallel edges) ────
    # Keep the minimum-weight edge between each node pair for routing.
    simple_G = nx.DiGraph()
    for u, v, data in osm_G.edges(data=True):
        hw = data.get('highway', 'unclassified')
        if isinstance(hw, list):
            hw = hw[0]
        hw = hw.replace('_link', '').strip()
        if hw not in HIGHWAY_TYPES:
            continue
        w = data.get('custom_weight', data.get('length', 1.0))
        if not simple_G.has_edge(u, v) or simple_G[u][v]['weight'] > w:
            simple_G.add_edge(u, v, weight=w, length=data.get('length', 1.0))

    nodes_to_keep = set()
    edges_to_keep = {}      # (u, v, key) → data dict
    direct_edges  = set()

    for a in chosen_nodes:
        for b in chosen_nodes:
            if a['id'] == b['id']:
                continue
            if a['id'] not in simple_G or b['id'] not in simple_G:
                continue

            is_direct = (a['id'] == start_id and b['id'] == goal_id)

            # Find K shortest simple paths on the collapsed DiGraph
            paths_found = []
            try:
                for p in nx.shortest_simple_paths(simple_G, a['id'], b['id'],
                                                   weight='weight'):
                    paths_found.append(p)
                    if len(paths_found) >= K_PATHS:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

            # Fallback to single shortest path
            if not paths_found:
                try:
                    paths_found = [nx.shortest_path(
                        simple_G, a['id'], b['id'], weight='weight')]
                except Exception:
                    pass

            for path in paths_found:
                nodes_to_keep.update(path)
                for k in range(len(path) - 1):
                    u, v = path[k], path[k + 1]
                    if not osm_G.has_edge(u, v):
                        continue
                    # Include ALL parallel edges from the original MultiDiGraph
                    for edge_key, edge_data in osm_G[u][v].items():
                        hw = _get_highway_type(osm_G, u, v, edge_key)
                        if hw not in HIGHWAY_TYPES:
                            continue
                        ekey = (u, v, edge_key)
                        if ekey not in edges_to_keep:
                            edges_to_keep[ekey] = dict(edge_data)
                        if is_direct:
                            direct_edges.add(ekey)

    # ── Build the subgraph ───────────────────────────────────────────────────
    sub = nx.MultiDiGraph()

    for nid in nodes_to_keep:
        if nid not in osm_G.nodes:
            continue
        attrs = dict(osm_G.nodes[nid])
        for cn in chosen_nodes:
            if cn['id'] == nid:
                attrs['chosen'] = True
                attrs['label']  = cn['label']
                break
        sub.add_node(nid, **attrs)

    for (u, v, key), data in edges_to_keep.items():
        if u not in sub or v not in sub:
            continue
        d = dict(data)
        if (u, v, key) in direct_edges:
            d['traffic_factor'] = round(d.get('traffic_factor', 1.4) * 1.8, 3)
            d['safety_factor']  = round(max(d.get('safety_factor', 0.8) * 0.7, 0.1), 3)
            d['pothole_factor'] = round(d.get('pothole_factor', 1.3) * 1.5, 3)
            lkm = d.get('length', 100) / 1000.0
            d['custom_weight']  = round(
                lkm * d['traffic_factor'] * d['pothole_factor']
                / d['safety_factor'], 4)
            d['congested'] = True
        sub.add_edge(u, v, key=key, **d)

    parallel = sum(
        1 for u, v in set((u, v) for u, v, _ in sub.edges(keys=True))
        if sub.number_of_edges(u, v) > 1
    )

    missing = [n for n in chosen_nodes if n['id'] not in sub]
    if missing:
        print(f"  Warning: {len(missing)} chosen node(s) not in subgraph.")

    print(f"\n  Search graph built:")
    print(f"    Nodes (intersections) : {sub.number_of_nodes()}")
    print(f"    Edges (road segments) : {sub.number_of_edges()}")
    print(f"    Parallel edge pairs   : {parallel}")
    print(f"    K paths per node pair : {K_PATHS}")
    print(f"    Road types included   : {', '.join(sorted(HIGHWAY_TYPES))}")
    print(f"    Direct start→goal     : congestion penalty applied\n")

    return sub
