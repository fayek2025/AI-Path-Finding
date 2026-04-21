"""
FastAPI backend for dynamic pathfinding dashboard.
Allows real-time weight adjustment and live node placement without re-running main.py.
"""
import json
import os
import pickle
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import networkx as nx

from comparison import run_all
from heuristic import haversine_coords
from map_loader import load_graph, build_search_graph
from visualization import generate_path_map, generate_complexity_graphs

# ── Global state ──────────────────────────────────────────────────────────────
OSM_G: Optional[nx.MultiDiGraph] = None   # full OSM graph for snapping
GRAPH: Optional[nx.MultiDiGraph] = None   # current search subgraph
NODES: list = []                           # chosen nodes [{id, lat, lon, label}]
START: Optional[int] = None
GOAL:  Optional[int] = None
GEOCACHE: dict = {}
_KDTREE = None        # pre-built KD-tree for fast nearest-node lookup
_KDTREE_IDS = None    # node IDs corresponding to KD-tree entries
_SEED = 42
_CENTER = (23.7766, 90.4227)
_GRAPH_CACHE: dict = {}   # frozenset(node_ids) → subgraph, avoids rebuild on same nodes


def _load_state():
    global GRAPH, NODES, START, GOAL, GEOCACHE, OSM_G, _KDTREE, _KDTREE_IDS, _GRAPH_CACHE

    print("  Loading OSM graph for node snapping...")
    OSM_G = load_graph(center=_CENTER, dist=15000, seed=_SEED)
    print(f"  OSM graph ready: {OSM_G.number_of_nodes()} nodes")

    # Build KD-tree once for O(log n) nearest-node lookup
    from scipy.spatial import KDTree
    import numpy as np
    node_ids = list(OSM_G.nodes)
    coords = np.array([[OSM_G.nodes[n]['y'], OSM_G.nodes[n]['x']] for n in node_ids])
    _KDTREE = KDTree(coords)
    _KDTREE_IDS = node_ids
    print(f"  KD-tree built for {len(node_ids)} nodes")

    # Clear subgraph cache so new diversity-aware build_search_graph is used
    _GRAPH_CACHE.clear()

    # Load last saved search state if available
    state_file = "cache/app_state.pkl"
    geocache_file = "cache/geocache.json"
    if os.path.exists(state_file):
        with open(state_file, 'rb') as f:
            data = pickle.load(f)
            NODES = data['nodes']
            START = data['start']
            GOAL  = data['goal']
        # Rebuild subgraph with current logic (not the stale cached version)
        if NODES and len(NODES) >= 2:
            print(f"  Rebuilding search graph for saved START→GOAL...")
            GRAPH = build_search_graph(OSM_G, NODES, seed=_SEED)
            _GRAPH_CACHE[(NODES[0]['id'], NODES[-1]['id'])] = GRAPH
            print(f"  Search graph ready: {GRAPH.number_of_nodes()} nodes")
    if os.path.exists(geocache_file):
        with open(geocache_file) as f:
            GEOCACHE = {int(k): v for k, v in json.load(f).items()}


@asynccontextmanager
async def lifespan(app):
    _load_state()
    yield


app = FastAPI(title="AI Pathfinding API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WeightParams(BaseModel):
    traffic_weight:     float = 1.0
    safety_weight:      float = 1.0
    pothole_weight:     float = 1.0
    road_age_weight:    float = 1.0   # higher = penalise older/worse roads more
    turn_weight:        float = 1.0   # higher = penalise complex intersections more


class SnapRequest(BaseModel):
    lat: float
    lon: float


class SetNodesRequest(BaseModel):
    node_ids: list[int]
    params: WeightParams = WeightParams()


# Road type sensitivity dicts removed — no longer needed after switching
# to the exponent-based weight formula in _apply_weights.


def _apply_weights(G: nx.MultiDiGraph, params: WeightParams) -> nx.MultiDiGraph:
    """
    Recompute custom_weight per edge so that each slider genuinely changes
    which roads are preferred.

    The key insight: when all sliders scale uniformly, the ratio between any
    two roads stays constant (slider^N cancels out). To break this, each
    slider raises only its own base factor to a power, making that dimension
    dominate non-linearly as the slider increases.

    Formula:
        custom_weight = length_km
                        * traffic^traffic_w
                        * pothole^pothole_w
                        * road_age^road_age_w
                        * turn^turn_w
                        / safety^safety_w

    At slider=1.0 this equals the base custom_weight from map_loader.
    At slider=3.0 the dominant factor overwhelms the others, genuinely
    shifting which road type is cheapest.
    """
    G = G.copy()
    for u, v, key, data in G.edges(keys=True, data=True):
        lkm = data.get('length', 100) / 1000.0
        t   = max(data.get('traffic_factor',  1.4), 1e-6)
        s   = max(data.get('safety_factor',   0.8), 1e-6)
        p   = max(data.get('pothole_factor',  1.3), 1e-6)
        ra  = max(data.get('road_age_factor', 1.1), 1e-6)
        tc  = max(data.get('turn_complexity', 0.7), 1e-6)

        G[u][v][key]['custom_weight'] = round(
            lkm
            * (t  ** params.traffic_weight)
            * (p  ** params.pothole_weight)
            * (ra ** params.road_age_weight)
            * (tc ** params.turn_weight)
            / (s  ** params.safety_weight),
            6
        )
    return G


def _reverse_geocode(lat: float, lon: float, node_id: int) -> str:
    """
    Return a human-readable place name for (lat, lon) using Nominatim.
    Falls back to cached value, then to coordinate string.
    Results are cached in GEOCACHE to avoid repeated API calls.
    """
    if node_id in GEOCACHE:
        return GEOCACHE[node_id]
    try:
        import urllib.request, urllib.parse
        params = urllib.parse.urlencode({'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 17, 'addressdetails': 1})
        url = f"https://nominatim.openstreetmap.org/reverse?{params}"
        req = urllib.request.Request(url, headers={'User-Agent': 'AI-Pathfinding-Dashboard/1.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
        addr = data.get('address', {})
        # Build a short readable name: road/suburb/neighbourhood
        name = (addr.get('road') or addr.get('pedestrian') or
                addr.get('neighbourhood') or addr.get('suburb') or
                addr.get('city_district') or data.get('display_name', '')[:60])
        GEOCACHE[node_id] = name
        # Persist cache
        os.makedirs('cache', exist_ok=True)
        with open('cache/geocache.json', 'w') as f:
            json.dump({str(k): v for k, v in GEOCACHE.items()}, f)
        return name
    except Exception:
        fallback = f"{lat:.4f}, {lon:.4f}"
        GEOCACHE[node_id] = fallback
        return fallback


@app.post("/api/snap-node")
def snap_node(req: SnapRequest):
    """Snap a lat/lon click to the nearest OSM road node using pre-built KD-tree — O(log n)."""
    if OSM_G is None or _KDTREE is None:
        raise HTTPException(400, "OSM graph not loaded")
    _, idx = _KDTREE.query([req.lat, req.lon])
    nid = _KDTREE_IDS[idx]
    data = OSM_G.nodes[nid]
    lat, lon = data['y'], data['x']
    name = _reverse_geocode(lat, lon, nid)
    return {"id": nid, "lat": lat, "lon": lon, "name": name}


@app.post("/api/set-nodes")
async def set_nodes(req: SetNodesRequest):
    """
    Rebuild the search graph from a new set of chosen node IDs,
    run all algorithms, and return results + graph data in one shot.
    """
    global GRAPH, NODES, START, GOAL

    if OSM_G is None:
        raise HTTPException(400, "OSM graph not loaded")
    if len(req.node_ids) < 2:
        raise HTTPException(400, "Need at least 2 nodes")

    # Build chosen_nodes list with labels and geocoded names
    chosen = []
    for i, nid in enumerate(req.node_ids):
        if nid not in OSM_G.nodes:
            raise HTTPException(400, f"Node {nid} not in OSM graph")
        d = OSM_G.nodes[nid]
        label = 'START' if i == 0 else ('GOAL' if i == len(req.node_ids) - 1 else f'N{i}')
        name = _reverse_geocode(d['y'], d['x'], nid)
        chosen.append({'id': nid, 'lat': d['y'], 'lon': d['x'], 'label': label, 'name': name})

    # Cache subgraph by START+GOAL only — intermediate nodes don't affect the subgraph
    cache_key = (req.node_ids[0], req.node_ids[-1])
    if cache_key not in _GRAPH_CACHE:
        GRAPH = await run_in_threadpool(build_search_graph, OSM_G, chosen, _SEED)
        _GRAPH_CACHE[cache_key] = GRAPH
    else:
        GRAPH = _GRAPH_CACHE[cache_key]

    NODES = chosen
    START = chosen[0]['id']
    GOAL  = chosen[-1]['id']

    # Intermediate nodes enrich the subgraph with more road options,
    # but are NOT forced waypoints — algorithms find the best path freely from START to GOAL.
    G = _apply_weights(GRAPH, req.params)
    records = await run_in_threadpool(run_all, G, START, GOAL, None)

    # Build graph data response
    pos = {nid: d for nid, d in GRAPH.nodes(data=True)}
    nodes_data = [
        {"id": nid, "lat": d.get('y'), "lon": d.get('x'),
         "label": d.get('label', ''), "name": GEOCACHE.get(nid, f"{d.get('y'):.4f},{d.get('x'):.4f}")}
        for nid, d in GRAPH.nodes(data=True)
    ]

    return {
        "records": records,
        "nodes": nodes_data,
        "start": START,
        "goal": GOAL,
        "chosen_nodes": NODES,
    }


@app.post("/api/run")
async def run_algorithms(params: WeightParams):
    """Re-run all algorithms with adjusted weights on the current graph."""
    if GRAPH is None:
        raise HTTPException(400, "No graph loaded. Place nodes first.")
    G = _apply_weights(GRAPH, params)
    records = await run_in_threadpool(run_all, G, START, GOAL, None)
    return {"records": records, "params": params.dict()}


@app.get("/api/graph-data")
def get_graph_data():
    """
    Return graph structure, nodes, edges, and geocoded names.
    Returns empty state if no graph is loaded yet (no nodes placed).
    """
    if GRAPH is None:
        return {
            "nodes": [], "edges": [],
            "start": None, "goal": None, "chosen_nodes": []
        }
    
    nodes_data = []
    for nid, data in GRAPH.nodes(data=True):
        nodes_data.append({
            "id": nid,
            "lat": data.get('y'),
            "lon": data.get('x'),
            "label": data.get('label', ''),
            "name": GEOCACHE.get(nid, f"{data.get('y'):.4f},{data.get('x'):.4f}")
        })
    
    edges_data = []
    for u, v, key, data in GRAPH.edges(keys=True, data=True):
        edges_data.append({
            "source": u,
            "target": v,
            "length": data.get('length'),
            "traffic": data.get('traffic_factor'),
            "safety": data.get('safety_factor'),
            "pothole": data.get('pothole_factor'),
            "weight": data.get('custom_weight')
        })
    
    return {
        "nodes": nodes_data,
        "edges": edges_data,
        "start": START,
        "goal": GOAL,
        "chosen_nodes": NODES
    }


class SaveSnapshotRequest(BaseModel):
    image: str        # base64 data-URL: "data:image/png;base64,..."
    label: str = ''   # e.g. 'all', 'A*', 'BFS' — used in filename


@app.post("/api/save-map-snapshot")
async def save_map_snapshot(req: SaveSnapshotRequest):
    """Receive a base64 PNG data-URL from the frontend and save it to disk."""
    import base64, re
    match = re.match(r'data:image/\w+;base64,(.*)', req.image, re.DOTALL)
    if not match:
        raise HTTPException(400, "Invalid image data")
    img_bytes = base64.b64decode(match.group(1))
    # Sanitise label for use in filename
    safe_label = re.sub(r'[^\w\-]', '_', req.label) if req.label else 'snapshot'
    out_path = f"output_map_{safe_label}.png"
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    return {"file": out_path, "message": f"Snapshot saved as {out_path}"}


@app.post("/api/generate-graphs")
async def generate_graphs(params: WeightParams):
    """
    Generate matplotlib visualisations for the current graph + algorithm results:
      - output_path_map.png      : all algorithm paths on the road network
      - output_complexity.png    : nodes expanded, time, memory, and complexity table
    Returns the filenames so the frontend can open/display them.
    """
    if GRAPH is None or START is None or GOAL is None:
        raise HTTPException(400, "No graph loaded. Place nodes first.")

    G = _apply_weights(GRAPH, params)
    records = await run_in_threadpool(run_all, G, START, GOAL, None)

    # Pass OSM_G as the full background map, GRAPH as the search subgraph,
    # and NODES so all chosen waypoints are marked on the map
    path_map_file = await run_in_threadpool(
        generate_path_map, OSM_G, GRAPH, G, records, START, GOAL, NODES
    )
    complexity_file = await run_in_threadpool(
        generate_complexity_graphs, G, START, GOAL, records
    )

    return {
        "path_map":   path_map_file,
        "complexity": complexity_file,
        "message":    "Graphs saved successfully.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
