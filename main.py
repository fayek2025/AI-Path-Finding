"""
main.py — AI Pathfinding System

Flow:
  1. Open ONE map in the browser.
  2. Click to place nodes: first click = START, last click = GOAL,
     everything in between = intermediate nodes.
  3. Python builds a small DiGraph from ONLY those nodes (~10-15),
     connecting every pair with haversine distance as the edge weight
     and random traffic/safety/pothole metrics.
  4. All 8 algorithms run on that tiny graph.
  5. Matplotlib shows the graph + all paths + comparison bar chart.
"""
import os
import json
import math
import random
import threading
import webbrowser
import http.server
import socketserver
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from heuristic import haversine_coords
from comparison import run_all, export_json, plot_comparison
from state_tree import build_tree, serialize_tree

COLORS = {
    'BFS':              '#1f77b4',
    'DFS':              '#9467bd',
    'IDS':              '#17becf',
    'UCS':              '#2ca02c',
    'A*':               '#d62728',
    'Greedy':           '#ff7f0e',
    'IDA*':             '#e377c2',
    'Bidirectional A*': '#bcbd22',
}

NAMED_LOCATIONS = {
    '1': ('Dhaka University (DU)',        23.7340, 90.3928),
    '2': ('Bashundhara Residential Area', 23.8193, 90.4526),
    '3': ('Gulshan 2',                    23.7925, 90.4148),
    '4': ('Banani',                       23.7937, 90.4066),
    '5': ('Mirpur 10',                    23.8073, 90.3664),
    '6': ('Motijheel',                    23.7279, 90.4176),
    '7': ('Uttara Sector 7',              23.8759, 90.3795),
    '8': ('Dhanmondi 27',                 23.7461, 90.3742),
}


# ── Step 1: Collect all nodes in one map session ──────────────────────────────

def collect_nodes_from_map(osm_G: nx.MultiDiGraph) -> list:
    """
    Open a single Leaflet map. User clicks to place nodes one by one.
    Each click is snapped to the nearest real OSM node ID.
    Returns list of dicts: [{id (OSM node ID), lat, lon, label}, ...]
    """
    state  = {'nodes': [], 'done': False}
    lock   = threading.Event()

    preset_js = json.dumps([
        {'name': v[0], 'lat': v[1], 'lon': v[2]}
        for v in NAMED_LOCATIONS.values()
    ])

    with socketserver.TCPServer(('', 0), None) as tmp:
        port = tmp.server_address[1]

    html = _build_collection_html(preset_js, port)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode())

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            body   = json.loads(self.rfile.read(length))
            action = body.get('action')

            if action == 'add_node':
                lat, lon = body['lat'], body['lon']
                # Snap to nearest real OSM node
                try:
                    osm_id = ox.distance.nearest_nodes(osm_G, X=lon, Y=lat)
                except Exception:
                    from heuristic import haversine_coords
                    osm_id = min(osm_G.nodes,
                                 key=lambda n: haversine_coords(
                                     lat, lon,
                                     osm_G.nodes[n]['y'],
                                     osm_G.nodes[n]['x']))
                # Use actual OSM coordinates of the snapped node
                snapped_lat = osm_G.nodes[osm_id]['y']
                snapped_lon = osm_G.nodes[osm_id]['x']
                # Avoid duplicate nodes
                if any(n['id'] == osm_id for n in state['nodes']):
                    resp = {'node_id': osm_id, 'total': len(state['nodes']),
                            'duplicate': True,
                            'node_lat': snapped_lat, 'node_lon': snapped_lon}
                else:
                    state['nodes'].append({
                        'id': osm_id,
                        'lat': snapped_lat,
                        'lon': snapped_lon,
                        'label': f'N{len(state["nodes"]) + 1}'
                    })
                    resp = {'node_id': osm_id, 'total': len(state['nodes']),
                            'node_lat': snapped_lat, 'node_lon': snapped_lon}

            elif action == 'remove_last':
                if state['nodes']:
                    state['nodes'].pop()
                resp = {'total': len(state['nodes'])}

            elif action == 'get_nodes':
                resp = {'nodes': state['nodes']}

            elif action == 'done':
                if len(state['nodes']) < 2:
                    resp = {'ok': False, 'error': 'Need at least 2 nodes (START + GOAL)'}
                else:
                    state['done'] = True
                    lock.set()
                    resp = {'ok': True}

            else:
                resp = {'error': 'unknown'}

            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(('localhost', port), Handler)

    def serve():
        while not state['done']:
            httpd.handle_request()
        httpd.server_close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()

    url = f'http://localhost:{port}'
    webbrowser.open(url)
    print(f"\n  Map opened at {url}")
    print("  → Click to place nodes. First = START, Last = GOAL.")
    print("  → Click 'Done' when finished (min 2 nodes).\n")

    lock.wait(timeout=600)

    nodes = state['nodes']
    if len(nodes) >= 2:
        nodes[0]['label']  = 'START'
        nodes[-1]['label'] = 'GOAL'
        for i, n in enumerate(nodes[1:-1], 1):
            n['label'] = f'N{i}'
    return nodes


def _build_collection_html(preset_js: str, port: int) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Place Nodes</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:Arial,sans-serif; }}
  #map {{ height:100vh; width:100%; }}
  #panel {{
    position:absolute; top:10px; left:50%; transform:translateX(-50%);
    z-index:1000; background:white; padding:14px 18px; border-radius:10px;
    box-shadow:0 3px 14px rgba(0,0,0,0.3); min-width:380px; text-align:center;
  }}
  h3 {{ margin:0 0 8px; color:#2c3e50; font-size:15px; }}
  #search-row {{ display:flex; gap:6px; margin-bottom:10px; }}
  #search-input {{ flex:1; padding:7px 10px; border:1px solid #ccc; border-radius:6px; font-size:13px; }}
  #search-btn {{ padding:7px 12px; background:#3498db; color:white; border:none; border-radius:6px; cursor:pointer; font-size:13px; }}
  #node-list {{ max-height:130px; overflow-y:auto; text-align:left; margin-bottom:10px;
                border:1px solid #eee; border-radius:6px; padding:4px 8px; font-size:12px; }}
  .node-item {{ padding:3px 0; border-bottom:1px solid #f5f5f5; }}
  .node-item:last-child {{ border-bottom:none; }}
  #status {{ font-size:13px; color:#555; margin-bottom:8px; }}
  .btn {{ padding:8px 14px; margin:0 3px; border:none; border-radius:6px;
          cursor:pointer; font-size:13px; font-weight:bold; }}
  #btn-undo {{ background:#ecf0f1; color:#555; }}
  #btn-done {{ background:#27ae60; color:white; }}
  #btn-done:disabled {{ background:#bdc3c7; cursor:not-allowed; }}
  #search-results {{ position:absolute; top:100%; left:0; right:0; background:white;
    border:1px solid #ddd; border-radius:0 0 6px 6px; max-height:160px;
    overflow-y:auto; z-index:2000; display:none; text-align:left; }}
  .sr-item {{ padding:7px 12px; cursor:pointer; font-size:12px; border-bottom:1px solid #f0f0f0; }}
  .sr-item:hover {{ background:#f0f7ff; }}
</style>
</head>
<body>
<div id="panel">
  <h3>🗺️ Place Your Nodes</h3>
  <div style="position:relative;">
    <div id="search-row">
      <input id="search-input" type="text" placeholder="Search place (e.g. Gulshan 1)..."
             onkeydown="if(event.key==='Enter') searchPlace()"/>
      <button id="search-btn" onclick="searchPlace()">🔍</button>
    </div>
    <div id="search-results"></div>
  </div>
  <div id="status">Click on the map to place your first node (START)</div>
  <div id="node-list"><i style="color:#aaa">No nodes placed yet</i></div>
  <button class="btn" id="btn-undo" onclick="undoLast()">↩ Undo Last</button>
  <button class="btn" id="btn-done" onclick="finishSelection()" disabled>✓ Done</button>
</div>
<div id="map"></div>
<script>
var map = L.map('map').setView([23.7766, 90.4227], 13);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution:'© OpenStreetMap contributors', maxZoom:19
}}).addTo(map);

var presets = {preset_js};
var markers = [];
var nodeColors = ['#27ae60','#3498db','#9b59b6','#e67e22','#1abc9c',
                  '#e74c3c','#f39c12','#2980b9','#8e44ad','#16a085',
                  '#d35400','#c0392b','#7f8c8d','#2ecc71','#e74c3c'];

presets.forEach(function(p) {{
  L.circleMarker([p.lat,p.lon],{{radius:6,color:'royalblue',fillColor:'royalblue',fillOpacity:0.6}})
    .addTo(map).bindTooltip(p.name);
}});

function getNodeColor(idx) {{
  if (idx === 0) return '#27ae60';
  return nodeColors[idx % nodeColors.length];
}}

function updateList(nodes) {{
  var list = document.getElementById('node-list');
  if (!nodes.length) {{
    list.innerHTML = '<i style="color:#aaa">No nodes placed yet</i>';
    document.getElementById('btn-done').disabled = true;
    document.getElementById('status').innerText = 'Click on the map to place your first node (START)';
    return;
  }}
  var html = '';
  nodes.forEach(function(n, i) {{
    var lbl = i===0 ? 'START' : (i===nodes.length-1 ? 'GOAL' : 'N'+i);
    var col = getNodeColor(i);
    html += '<div class="node-item"><span style="color:'+col+';font-weight:bold">'+lbl+'</span> — '+
            n.lat.toFixed(5)+', '+n.lon.toFixed(5)+'</div>';
  }});
  list.innerHTML = html;
  document.getElementById('btn-done').disabled = nodes.length < 2;
  var next = nodes.length === 0 ? 'START' : (nodes.length === 1 ? 'GOAL or more nodes' : 'another node or click Done');
  document.getElementById('status').innerText = nodes.length + ' node(s) placed. Click to add ' + next + '.';
}}

function refreshMarkers(nodes) {{
  markers.forEach(function(m) {{ map.removeLayer(m); }});
  markers = [];
  nodes.forEach(function(n, i) {{
    var lbl = i===0 ? 'START' : (i===nodes.length-1 && nodes.length>1 ? 'GOAL' : 'N'+i);
    var col = getNodeColor(i);
    var m = L.circleMarker([n.lat,n.lon],{{
      radius:12, color:col, fillColor:col, fillOpacity:0.9, weight:3
    }}).addTo(map).bindTooltip('<b>'+lbl+'</b><br>'+n.lat.toFixed(5)+', '+n.lon.toFixed(5),
      {{permanent:true, direction:'top'}});
    markers.push(m);
  }});
  // Draw lines between consecutive nodes
  for (var i=0; i<nodes.length-1; i++) {{
    var line = L.polyline([[nodes[i].lat,nodes[i].lon],[nodes[i+1].lat,nodes[i+1].lon]],
      {{color:'#95a5a6',weight:2,dashArray:'6'}}).addTo(map);
    markers.push(line);
  }}
}}

map.on('click', function(e) {{
  var lat = e.latlng.lat, lon = e.latlng.lng;
  document.getElementById('status').innerText = 'Snapping to nearest road node...';
  fetch('http://localhost:{port}', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{action:'add_node', lat:lat, lon:lon}})
  }})
  .then(r=>r.json())
  .then(function(d) {{
    if (d.duplicate) {{
      document.getElementById('status').innerText = '⚠ That node is already selected. Try a different location.';
      return;
    }}
    // Show snapped position, not raw click
    fetch('http://localhost:{port}', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{action:'get_nodes'}})
    }}).then(r=>r.json()).then(function(d2) {{
      updateList(d2.nodes);
      refreshMarkers(d2.nodes);
    }});
  }});
}});

function undoLast() {{
  fetch('http://localhost:{port}', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{action:'remove_last'}})
  }}).then(r=>r.json()).then(function() {{
    fetch('http://localhost:{port}', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{action:'get_nodes'}})
    }}).then(r=>r.json()).then(function(d) {{
      updateList(d.nodes);
      refreshMarkers(d.nodes);
    }});
  }});
}}

function finishSelection() {{
  fetch('http://localhost:{port}', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{action:'done'}})
  }}).then(r=>r.json()).then(function(d) {{
    if (d.ok) {{
      document.getElementById('status').innerText = '✓ Done! Close this tab and return to the terminal.';
      document.getElementById('btn-done').disabled = true;
    }} else {{
      document.getElementById('status').innerText = '⚠ ' + d.error;
    }}
  }});
}}

function searchPlace() {{
  var q = document.getElementById('search-input').value.trim();
  if (!q) return;
  if (q.toLowerCase().indexOf('dhaka')===-1 && q.toLowerCase().indexOf('bangladesh')===-1)
    q += ', Dhaka, Bangladesh';
  fetch('https://nominatim.openstreetmap.org/search?format=json&limit=5&q='+encodeURIComponent(q),
    {{headers:{{'Accept-Language':'en'}}}})
  .then(r=>r.json()).then(function(results) {{
    var box = document.getElementById('search-results');
    box.innerHTML = '';
    if (!results.length) {{
      box.innerHTML='<div class="sr-item" style="color:#999">No results</div>';
      box.style.display='block'; return;
    }}
    results.forEach(function(r) {{
      var div=document.createElement('div'); div.className='sr-item';
      div.innerText=r.display_name;
      div.onclick=function() {{
        map.setView([parseFloat(r.lat),parseFloat(r.lon)],16);
        box.style.display='none';
        document.getElementById('search-input').value=r.display_name.split(',')[0];
      }};
      box.appendChild(div);
    }});
    box.style.display='block';
  }});
}}
document.addEventListener('click',function(e) {{
  if (!e.target.closest('#panel'))
    document.getElementById('search-results').style.display='none';
}});
</script>
</body>
</html>"""


# ── Step 2: Build a small custom graph from the placed nodes ──────────────────

def build_custom_graph(nodes: list, seed: int = 42) -> nx.MultiDiGraph:
    """
    Complete directed graph — every node connects to every other node.

    Why complete graph is correct for this AI course demo:
    - BFS/DFS/IDS ignore weights → take the direct 1-hop start→goal edge
    - UCS/A* use weights → find cheapest path, which goes through intermediates
      because the direct start→goal edge is deliberately expensive
    - This produces genuinely different paths AND costs across algorithms

    The direct start→goal edge gets worst-case metrics (max traffic, max pothole,
    min safety) to model: "the direct road is congested/unsafe."
    All other edges get normal random metrics.
    """
    rng      = random.Random(seed)
    G        = nx.MultiDiGraph()
    start_id = nodes[0]['id']
    goal_id  = nodes[-1]['id']

    for n in nodes:
        G.add_node(n['id'], y=n['lat'], x=n['lon'], label=n['label'])

    for a in nodes:
        for b in nodes:
            if a['id'] == b['id']:
                continue
            dist_km = haversine_coords(a['lat'], a['lon'],
                                       b['lat'], b['lon']) / 1000.0
            # Direct start→goal: deliberately expensive so weighted algorithms
            # prefer going through intermediate nodes
            if a['id'] == start_id and b['id'] == goal_id:
                traffic = 2.4
                safety  = 0.62
                pothole = 1.95
            else:
                traffic = rng.uniform(1.0, 2.0)
                safety  = rng.uniform(0.7, 1.0)
                pothole = rng.uniform(1.0, 1.6)

            weight = (dist_km * traffic * pothole) / safety
            G.add_edge(a['id'], b['id'],
                       length=round(dist_km, 4),
                       traffic_factor=round(traffic, 3),
                       safety_factor=round(safety, 3),
                       pothole_factor=round(pothole, 3),
                       custom_weight=round(weight, 4))

    print(f"  Complete graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges  "
          f"(direct start→goal edge is high-cost, seed={seed})")
    return G


# ── Node name lookup ──────────────────────────────────────────────────────────

def _save_graph_state(G: nx.MultiDiGraph, nodes: list, start: int, goal: int, geocache: dict):
    """Save graph + geocache to disk so the React dashboard / API can load it."""
    import pickle
    os.makedirs('cache', exist_ok=True)
    with open('cache/app_state.pkl', 'wb') as f:
        pickle.dump({'graph': G, 'nodes': nodes, 'start': start, 'goal': goal}, f)
    with open('cache/geocache.json', 'w') as f:
        json.dump({str(k): v for k, v in geocache.items()}, f, indent=2)
    print("  Graph state saved to cache/")


def _get_node_name(nid: int, lat: float, lon: float) -> str:
    """
    Reverse-geocode a node to get its street/place name via Nominatim.
    Returns a short name (road name or neighbourhood).
    Falls back to coordinates if geocoding fails.
    """
    import urllib.request, urllib.parse
    try:
        url = (
            "https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json&zoom=17&addressdetails=1"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'ai-pathfinding/1.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
        addr = data.get('address', {})
        # Try progressively broader name fields
        name = (addr.get('road') or addr.get('pedestrian') or
                addr.get('neighbourhood') or addr.get('suburb') or
                addr.get('quarter') or data.get('display_name', '').split(',')[0])
        return name.strip()[:30]   # cap length for readability
    except Exception:
        return f"{lat:.4f},{lon:.4f}"


def _geocode_all_nodes(G: nx.MultiDiGraph, chosen_nodes: list) -> dict:
    """
    Reverse-geocode ONLY the chosen nodes (START/GOAL/intermediates).
    All other graph nodes get coordinate-based fallback names.
    Returns {node_id: display_name}.
    """
    import time
    chosen_ids = {n['id']: n['label'] for n in chosen_nodes}
    names = {}

    print(f"  Geocoding {len(chosen_nodes)} chosen nodes...", end='', flush=True)
    for i, node in enumerate(chosen_nodes):
        nid = node['id']
        name = _get_node_name(nid, node['lat'], node['lon'])
        names[nid] = f"{chosen_ids[nid]}: {name}"
        print('.', end='', flush=True)
        time.sleep(0.2)

    # Fallback: coord string for every other node in the graph
    for nid, data in G.nodes(data=True):
        if nid not in names:
            names[nid] = f"{data.get('y', 0):.4f},{data.get('x', 0):.4f}"

    print(' done.')
    return names


# ── Step 3: Visualize graph + paths with matplotlib ──────────────────────────

# ── Dashboard ─────────────────────────────────────────────────────────────────

def build_dashboard(G: nx.MultiDiGraph, records: list, nodes: list,
                    start: int, goal: int, node_names: dict) -> None:
    """
    Build a full-screen dashboard served via localhost so OSM tiles load correctly.
    Layout:
      - Full-screen Leaflet map (left ~60%) — Google Maps-style zoom/pan
      - Slide-out right panel with comparison table + bar charts + node info
      - Toggle button to show/hide the panel
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    pos = {nid: (d['x'], d['y']) for nid, d in G.nodes(data=True)
           if 'x' in d and 'y' in d}
    chosen_map = {n['id']: n for n in nodes}
    valid      = [r for r in records if r['path'] and len(r['path']) > 1]
    optimal_set = {'UCS', 'A*', 'IDA*', 'Bidirectional A*'}
    best_cost   = min((r['path_cost'] for r in records if r.get('path_cost')), default=None)

    center_lat = sum(n['lat'] for n in nodes) / len(nodes)
    center_lon = sum(n['lon'] for n in nodes) / len(nodes)

    # ── Leaflet GeoJSON data ──────────────────────────────────────────────────
    # Road network edges
    road_features = []
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        road_features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[pos[u][0], pos[u][1]],
                                         [pos[v][0], pos[v][1]]]},
            "properties": {
                "highway": data.get('highway_type', ''),
                "congested": data.get('congested', False),
            }
        })

    # Algorithm paths
    path_features = {}
    for record in valid:
        path  = record['path']
        color = COLORS.get(record['algorithm'], '#999')
        coords = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u not in pos or v not in pos:
                continue
            coords.append([[pos[u][0], pos[u][1]], [pos[v][0], pos[v][1]]])
        path_features[record['algorithm']] = {
            'color': color,
            'coords': coords,
            'cost': record['path_cost'],
            'hops': record['hop_count'],
            'expanded': record['nodes_expanded'],
        }

    # Intersection nodes
    inter_nodes = []
    for nid, (lon, lat) in pos.items():
        if nid in chosen_map:
            continue
        inter_nodes.append({
            'lat': lat, 'lon': lon,
            'name': node_names.get(nid, f'{lat:.5f},{lon:.5f}')
        })

    # Chosen nodes
    chosen_nodes_data = []
    for n in nodes:
        nid = n['id']
        if nid not in pos:
            continue
        lon, lat = pos[nid]
        color = '#27ae60' if nid == start else '#e74c3c' if nid == goal else '#3498db'
        chosen_nodes_data.append({
            'lat': lat, 'lon': lon,
            'label': n['label'],
            'street': node_names.get(nid, ''),
            'color': color,
            'isStart': nid == start,
            'isGoal': nid == goal,
        })

    # ── Bar charts (Plotly, inline) ───────────────────────────────────────────
    def bar_html(metric_key, title, color):
        names  = [r['algorithm'] for r in records if r.get(metric_key) is not None]
        values = [r[metric_key]  for r in records if r.get(metric_key) is not None]
        f = go.Figure(go.Bar(
            x=names, y=values, marker_color=color,
            text=[f'{v:.2f}' if isinstance(v, float) else str(v) for v in values],
            textposition='outside',
        ))
        f.update_layout(
            title=dict(text=title, font=dict(size=11)),
            margin=dict(l=30, r=10, t=36, b=55), height=220,
            paper_bgcolor='white', plot_bgcolor='#f8f9fa',
            xaxis=dict(tickangle=-30, tickfont=dict(size=8)),
            yaxis=dict(gridcolor='#e0e0e0'),
        )
        return pio.to_html(f, include_plotlyjs=False, full_html=False)

    chart_cost     = bar_html('path_cost',     'Path Cost',      '#3498db')
    chart_hops     = bar_html('hop_count',      'Hop Count',      '#e67e22')
    chart_expanded = bar_html('nodes_expanded', 'Nodes Expanded', '#27ae60')

    # ── Table rows ────────────────────────────────────────────────────────────
    table_rows = ''
    for r in records:
        found = '✓' if r.get('path_cost') else '✗'
        cost  = f"{r['path_cost']:.2f}" if r.get('path_cost') else '—'
        opt   = '✓' if r['algorithm'] in optimal_set else ''
        is_best = r.get('path_cost') == best_cost and r['algorithm'] in optimal_set
        row_cls = 'optimal' if is_best else ('nopath' if not r.get('path_cost') else '')
        dot = f'<span class="dot" style="background:{COLORS.get(r["algorithm"],"#999")}"></span>'
        table_rows += (
            f'<tr class="{row_cls}">'
            f'<td>{dot}{r["algorithm"]}</td>'
            f'<td>{cost}</td><td>{r["hop_count"]}</td>'
            f'<td>{r["nodes_expanded"]}</td>'
            f'<td>{len(set(r.get("path",[])))}</td>'
            f'<td style="color:#27ae60;font-weight:700">{opt}</td>'
            f'<td class="{"ok" if found=="✓" else "fail"}">{found}</td>'
            f'</tr>\n'
        )

    node_rows = ''
    for n in nodes:
        street = node_names.get(n['id'], '—')
        c = '#27ae60' if n['label']=='START' else '#e74c3c' if n['label']=='GOAL' else '#3498db'
        node_rows += (
            f'<tr><td><b style="color:{c}">{n["label"]}</b></td>'
            f'<td>{street}</td><td>{n["lat"]:.5f}</td>'
            f'<td>{n["lon"]:.5f}</td></tr>\n'
        )

    # ── Serialise data for JS ─────────────────────────────────────────────────
    road_js    = json.dumps(road_features)
    paths_js   = json.dumps(path_features)
    inter_js   = json.dumps(inter_nodes)
    chosen_js  = json.dumps(chosen_nodes_data)
    colors_js  = json.dumps(COLORS)

    # ── HTML ──────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>AI Pathfinding Dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
html,body {{ height:100%; font-family:'Segoe UI',Arial,sans-serif; overflow:hidden; }}

/* ── Header ── */
#header {{
  position:fixed; top:0; left:0; right:0; height:44px; z-index:2000;
  background:linear-gradient(135deg,#1a252f,#2980b9);
  color:white; display:flex; align-items:center; padding:0 16px;
  gap:16px; box-shadow:0 2px 8px rgba(0,0,0,0.3);
}}
#header h1 {{ font-size:15px; font-weight:700; white-space:nowrap; }}
#header .meta {{ font-size:11px; opacity:0.8; }}
#toggle-btn {{
  margin-left:auto; padding:6px 14px; background:rgba(255,255,255,0.15);
  border:1px solid rgba(255,255,255,0.3); color:white; border-radius:5px;
  cursor:pointer; font-size:12px; white-space:nowrap;
  transition:background 0.2s;
}}
#toggle-btn:hover {{ background:rgba(255,255,255,0.25); }}

/* ── Map ── */
#map {{
  position:fixed; top:44px; left:0; right:0; bottom:0; z-index:1;
  transition:right 0.3s ease;
}}
#map.panel-open {{ right:420px; }}

/* ── Layer control panel (path toggles) ── */
#layer-panel {{
  position:fixed; top:54px; left:10px; z-index:1500;
  background:rgba(255,255,255,0.95); border-radius:8px;
  padding:10px 14px; box-shadow:0 2px 10px rgba(0,0,0,0.2);
  font-size:12px; min-width:160px;
}}
#layer-panel h4 {{ font-size:11px; color:#555; margin-bottom:8px;
                   text-transform:uppercase; letter-spacing:0.5px; }}
.layer-row {{ display:flex; align-items:center; gap:8px;
              margin-bottom:5px; cursor:pointer; }}
.layer-row:hover {{ opacity:0.75; }}
.layer-swatch {{ width:22px; height:4px; border-radius:2px; flex-shrink:0; }}
.layer-label {{ font-size:11px; color:#333; }}

/* ── Right panel ── */
#panel {{
  position:fixed; top:44px; right:-420px; width:420px; bottom:0;
  z-index:1500; background:#f4f6f8; overflow-y:auto;
  box-shadow:-3px 0 12px rgba(0,0,0,0.15);
  transition:right 0.3s ease;
  border-left:1px solid #dde;
}}
#panel.open {{ right:0; }}
#panel::-webkit-scrollbar {{ width:5px; }}
#panel::-webkit-scrollbar-thumb {{ background:#ccc; border-radius:3px; }}

.section {{
  background:white; margin:10px; border-radius:8px; padding:12px;
  box-shadow:0 1px 4px rgba(0,0,0,0.07);
}}
.section h2 {{
  font-size:11px; font-weight:700; color:#2c3e50; margin-bottom:10px;
  text-transform:uppercase; letter-spacing:0.6px;
  border-bottom:2px solid #3498db; padding-bottom:5px;
}}
table {{ width:100%; border-collapse:collapse; font-size:11px; }}
th {{ background:#2c3e50; color:white; padding:6px 8px;
      text-align:left; font-size:10px; font-weight:600; }}
td {{ padding:5px 8px; border-bottom:1px solid #f0f0f0; }}
tr:hover td {{ background:#f0f7ff; }}
tr.optimal td {{ background:#eafaf1; font-weight:600; }}
tr.nopath td {{ color:#bbb; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%;
        margin-right:5px; vertical-align:middle; }}
.ok   {{ color:#27ae60; font-weight:700; }}
.fail {{ color:#e74c3c; font-weight:700; }}
</style>
</head>
<body>

<div id="header">
  <h1>🗺️ AI Pathfinding Dashboard</h1>
  <div class="meta">
    {len(nodes)} nodes · {G.number_of_nodes()} intersections ·
    {G.number_of_edges()} road segments ·
    {sum(1 for r in records if r.get('path_cost'))} paths found
  </div>
  <button id="toggle-btn" onclick="togglePanel()">☰ Analysis Panel</button>
</div>

<div id="map"></div>

<!-- Layer toggles -->
<div id="layer-panel">
  <h4>Toggle Paths</h4>
  <div id="layer-rows"></div>
</div>

<!-- Right panel -->
<div id="panel">

  <div class="section">
    <h2>Algorithm Comparison</h2>
    <table>
      <thead><tr>
        <th>Algorithm</th><th>Cost</th><th>Hops</th>
        <th>Expanded</th><th>Nodes</th><th>Opt</th><th>Found</th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    <div style="font-size:9px;color:#999;margin-top:6px;line-height:1.7">
      Cost = Σ(km×traffic×pothole/safety) · Hops = road segments ·
      Expanded = frontier pops · <span style="color:#27ae60">■</span> = optimal
    </div>
  </div>

  <div class="section">
    <h2>Path Cost</h2>{chart_cost}
  </div>
  <div class="section">
    <h2>Hop Count</h2>{chart_hops}
  </div>
  <div class="section">
    <h2>Nodes Expanded</h2>{chart_expanded}
  </div>

  <div class="section">
    <h2>Chosen Nodes</h2>
    <table>
      <thead><tr><th>Role</th><th>Street</th><th>Lat</th><th>Lon</th></tr></thead>
      <tbody>{node_rows}</tbody>
    </table>
  </div>

</div>

<script>
// ── Data from Python ──────────────────────────────────────────────────────────
var ROAD_FEATURES  = {road_js};
var PATH_FEATURES  = {paths_js};
var INTER_NODES    = {inter_js};
var CHOSEN_NODES   = {chosen_js};
var COLORS         = {colors_js};

// ── Map init ──────────────────────────────────────────────────────────────────
var map = L.map('map', {{
  center: [{center_lat}, {center_lon}],
  zoom: 15,
  zoomControl: true,
}});

L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  maxZoom: 19,
}}).addTo(map);

// ── Road network ──────────────────────────────────────────────────────────────
ROAD_FEATURES.forEach(function(f) {{
  var c = f.properties.congested ? '#e74c3c' : '#888888';
  var w = f.properties.congested ? 2.5 : 1.2;
  L.polyline(
    f.geometry.coordinates.map(function(c) {{ return [c[1], c[0]]; }}),
    {{color:c, weight:w, opacity:0.6}}
  ).addTo(map);
}});

// ── Algorithm paths ───────────────────────────────────────────────────────────
var pathLayers = {{}};
var layerVisible = {{}};

Object.keys(PATH_FEATURES).forEach(function(algo) {{
  var pf = PATH_FEATURES[algo];
  var group = L.layerGroup();
  pf.coords.forEach(function(seg) {{
    L.polyline(
      seg.map(function(c) {{ return [c[1], c[0]]; }}),
      {{color: pf.color, weight: 5, opacity: 0.85,
        smoothFactor: 1}}
    ).bindTooltip(
      '<b>' + algo + '</b><br>Cost: ' + (pf.cost ? pf.cost.toFixed(2) : '—') +
      '<br>Hops: ' + pf.hops + '<br>Expanded: ' + pf.expanded,
      {{sticky: true}}
    ).addTo(group);
  }});
  group.addTo(map);
  pathLayers[algo] = group;
  layerVisible[algo] = true;
}});

// ── Layer toggle UI ───────────────────────────────────────────────────────────
var rowsDiv = document.getElementById('layer-rows');
Object.keys(PATH_FEATURES).forEach(function(algo) {{
  var pf = PATH_FEATURES[algo];
  var row = document.createElement('div');
  row.className = 'layer-row';
  row.id = 'row-' + algo;
  row.innerHTML =
    '<div class="layer-swatch" style="background:' + pf.color + '"></div>' +
    '<span class="layer-label">' + algo + '</span>';
  row.onclick = function() {{
    if (layerVisible[algo]) {{
      map.removeLayer(pathLayers[algo]);
      row.style.opacity = '0.35';
    }} else {{
      map.addLayer(pathLayers[algo]);
      row.style.opacity = '1';
    }}
    layerVisible[algo] = !layerVisible[algo];
  }};
  rowsDiv.appendChild(row);
}});

// ── Intersection nodes ────────────────────────────────────────────────────────
INTER_NODES.forEach(function(n) {{
  L.circleMarker([n.lat, n.lon], {{
    radius: 3, color: '#777', fillColor: '#777',
    fillOpacity: 0.7, weight: 1,
  }}).bindTooltip(n.name + '<br>' + n.lat.toFixed(5) + ', ' + n.lon.toFixed(5))
    .addTo(map);
}});

// ── Chosen nodes ──────────────────────────────────────────────────────────────
CHOSEN_NODES.forEach(function(n) {{
  var icon = L.divIcon({{
    className: '',
    html: '<div style="' +
      'background:' + n.color + ';' +
      'width:' + (n.isStart || n.isGoal ? '18px' : '14px') + ';' +
      'height:' + (n.isStart || n.isGoal ? '18px' : '14px') + ';' +
      'border-radius:50%;border:3px solid white;' +
      'box-shadow:0 0 6px rgba(0,0,0,0.4);' +
      '"></div>',
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  }});
  L.marker([n.lat, n.lon], {{icon: icon}})
    .bindPopup(
      '<b style="color:' + n.color + '">' + n.label + '</b><br>' +
      (n.street ? n.street + '<br>' : '') +
      n.lat.toFixed(5) + ', ' + n.lon.toFixed(5),
      {{maxWidth: 220}}
    )
    .bindTooltip('<b>' + n.label + '</b>' + (n.street ? '<br>' + n.street : ''),
                 {{permanent: false}})
    .addTo(map);
}});

// ── Panel toggle ──────────────────────────────────────────────────────────────
var panelOpen = false;
function togglePanel() {{
  panelOpen = !panelOpen;
  document.getElementById('panel').classList.toggle('open', panelOpen);
  document.getElementById('map').classList.toggle('panel-open', panelOpen);
  setTimeout(function() {{ map.invalidateSize(); }}, 320);
  document.getElementById('toggle-btn').textContent =
    panelOpen ? '✕ Close Panel' : '☰ Analysis Panel';
}}
</script>
</body>
</html>"""

    # Serve via localhost so Leaflet CDN + OSM tiles load correctly
    import tempfile, http.server, socketserver

    tmpdir = tempfile.mkdtemp()
    html_path = os.path.join(tmpdir, 'dashboard.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    # Also save a copy locally
    with open('dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html)

    with socketserver.TCPServer(('', 0), None) as s:
        port = s.server_address[1]

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=tmpdir, **kwargs)
        def log_message(self, *a): pass

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(('', port), Handler)

    def serve():
        threading.Timer(120, httpd.shutdown).start()
        httpd.serve_forever()
        httpd.server_close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()

    url = f'http://localhost:{port}/dashboard.html'
    webbrowser.open(url)
    print(f"\n  Dashboard opened at {url}")
    print("  → Full-screen map with Google Maps-style zoom/pan")
    print("  → Click '☰ Analysis Panel' to open comparison charts")
    print("  → Toggle individual algorithm paths via the left panel")
    print("  → Click any node marker for street name + coordinates")
    t.join(timeout=125)





# ── Table ─────────────────────────────────────────────────────────────────────

def print_table(records: list) -> None:
    """
    Print comparison table with a legend explaining each column.

    Hops     = number of edges in the path (path length - 1).
               hops=1 means a direct start→goal edge was used.
               hops=2 means start→intermediate→goal, etc.

    Expanded = total node expansions (pops from frontier).
               For IDA* this counts across ALL threshold iterations,
               so it can exceed the graph size — that's the trade-off
               for IDA*'s low memory usage vs A*'s single-pass expansion.

    Path Cost = sum of custom_weight along the chosen path
                (dist_km × traffic × pothole / safety).
    """
    header = (f"{'Algorithm':<20} {'Path Cost':>12} {'Hops':>6} "
              f"{'Expanded':>10} {'Unique Nodes':>14} {'Optimal':>8} {'Found':>6}")
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    # Algorithms known to be cost-optimal
    optimal_algos = {'UCS', 'A*', 'IDA*', 'Bidirectional A*'}
    # Find best cost for reference
    costs = [r['path_cost'] for r in records if r['path_cost'] is not None]
    best_cost = min(costs) if costs else None

    for r in records:
        found    = "Yes" if r['path_cost'] is not None else "No"
        cost     = f"{r['path_cost']:.2f}" if r['path_cost'] is not None else "—"
        optimal  = "✓" if r['algorithm'] in optimal_algos and r['path_cost'] == best_cost else "—"
        unique   = len(set(r.get('path', [])))
        print(f"{r['algorithm']:<20} {cost:>12} {r['hop_count']:>6} "
              f"{r['nodes_expanded']:>10} {unique:>14} {optimal:>8} {found:>6}")

    print(sep)
    print("\n  Legend:")
    print("  Hops         = edges in path (1 = direct start→goal edge)")
    print("  Expanded     = total node pops from frontier")
    print("                 (IDA* re-expands nodes each iteration — expected)")
    print("  Unique Nodes = distinct nodes in the solution path")
    print("  Optimal      = ✓ if algorithm guarantees minimum cost path")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("   AI Pathfinding System — OSM Road Network")
    print("=" * 52)

    import time
    from map_loader import load_graph, build_search_graph

    # Step 1: Load OSM graph for Dhaka area upfront
    # so node snapping works correctly during map selection
    print("\nStep 1: Loading OSM road network for Dhaka...")
    _CENTER = (23.7766, 90.4227)
    seed = int(time.time()) % 100000
    osm_G = load_graph(center=_CENTER, dist=15000, seed=seed)

    # Step 2: Place nodes on map (snaps to real OSM node IDs)
    print("\nStep 2: Place your nodes on the map.")
    print("  • First click  = START")
    print("  • Last click   = GOAL")
    print("  • Middle clicks = intermediate nodes")
    print("  • Recommended: 5-15 nodes\n")

    nodes = collect_nodes_from_map(osm_G)
    if len(nodes) < 2:
        print("Not enough nodes selected. Exiting.")
        return

    print(f"\n  Nodes collected: {len(nodes)}")
    for n in nodes:
        print(f"    [{n['label']:>6}] id={n['id']}  ({n['lat']:.5f}, {n['lon']:.5f})")

    start = nodes[0]['id']
    goal  = nodes[-1]['id']

    # Step 3: Build real OSM search graph between chosen nodes
    print(f"\nStep 3: Building real OSM search graph...  (seed={seed})")
    G = build_search_graph(osm_G, nodes, seed=seed)

    # Step 4: Run all algorithms
    print("\nStep 4: Running algorithms...")
    records = run_all(G, start, goal)

    # Print table
    print_table(records)

    # Export JSON
    export_json(records)

    # Save A* state-space tree
    astar_rec = next((r for r in records if r['algorithm'] == 'A*'), None)
    if astar_rec and astar_rec['path']:
        from models import SearchResult
        dummy = SearchResult(path=astar_rec['path'],
                             nodes_expanded=astar_rec['nodes_expanded'])
        with open('astar_tree.json', 'w') as f:
            f.write(serialize_tree(build_tree(dummy)))
        print("A* state-space tree saved to astar_tree.json")

    # Comparison bar chart
    plot_comparison(records)

    # Step 5: Geocode chosen nodes only + save state for React dashboard
    print("\nStep 5: Geocoding nodes and saving state...")
    node_names = _geocode_all_nodes(G, nodes)
    _save_graph_state(G, nodes, start, goal, node_names)
    print("  → Run 'python api.py' then 'cd dashboard && npm run dev' for the React dashboard")

    # Build legacy HTML dashboard (optional)
    build_dashboard(G, records, nodes, start, goal, node_names)


if __name__ == '__main__':
    main()
