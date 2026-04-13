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

def collect_nodes_from_map() -> list:
    """
    Open a single Leaflet map. User clicks to place nodes one by one.
    Each click snaps to the nearest OSM road node (via Nominatim reverse geocode).
    Returns list of dicts: [{id, lat, lon, label}, ...]
    First node = START, last node = GOAL, rest = intermediate.
    Minimum 2 nodes required.
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
                idx = len(state['nodes'])
                node_id = idx + 1
                state['nodes'].append({
                    'id': node_id, 'lat': lat, 'lon': lon,
                    'label': body.get('label', f'N{node_id}')
                })
                resp = {'node_id': node_id, 'total': len(state['nodes'])}

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
  fetch('http://localhost:{port}', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{action:'add_node', lat:lat, lon:lon}})
  }})
  .then(r=>r.json())
  .then(function(d) {{
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
    Build a fully-connected directed graph from the user-placed nodes.
    Uses MultiDiGraph so edge access G[u][v].values() works the same
    as in all algorithm implementations.
    """
    rng = random.Random(seed)
    G = nx.MultiDiGraph()

    for n in nodes:
        G.add_node(n['id'], y=n['lat'], x=n['lon'], label=n['label'])

    for a in nodes:
        for b in nodes:
            if a['id'] == b['id']:
                continue
            dist    = haversine_coords(a['lat'], a['lon'], b['lat'], b['lon'])
            traffic = rng.uniform(1.0, 4.0)
            safety  = max(rng.uniform(0.5, 1.0), 0.1)
            pothole = rng.uniform(1.0, 3.0)
            weight  = (dist * traffic * pothole) / safety
            G.add_edge(a['id'], b['id'],
                       length=dist,
                       traffic_factor=traffic,
                       safety_factor=safety,
                       pothole_factor=pothole,
                       custom_weight=weight)

    print(f"  Custom graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── Step 3: Visualize graph + paths with matplotlib ──────────────────────────

def visualize_matplotlib(G: nx.MultiDiGraph, records: list, nodes: list,
                         start: int, goal: int) -> None:
    """
    Draw the custom graph with all algorithm paths overlaid using matplotlib.
    Nodes are positioned by their real lat/lon coordinates.
    """
    pos = {n['id']: (n['lon'], n['lat']) for n in nodes}
    labels = {n['id']: n['label'] for n in nodes}

    valid = [r for r in records if r['path'] and len(r['path']) > 1]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    # Draw all edges (light grey)
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#cccccc',
                                   lw=1.0, connectionstyle='arc3,rad=0.05'))

    # Draw each algorithm path
    for record in valid:
        path  = record['path']
        color = COLORS.get(record['algorithm'], 'gray')
        for i in range(len(path) - 1):
            x0, y0 = pos[path[i]]
            x1, y1 = pos[path[i + 1]]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color=color,
                                       lw=2.5, alpha=0.75,
                                       connectionstyle='arc3,rad=0.08'))

    # Draw nodes
    for n in nodes:
        nid = n['id']
        x, y = pos[nid]
        if nid == start:
            c, ec, s = '#27ae60', '#1a7a44', 220
        elif nid == goal:
            c, ec, s = '#e74c3c', '#a93226', 220
        else:
            c, ec, s = '#3498db', '#1a5276', 160
        ax.scatter(x, y, s=s, c=c, edgecolors=ec, linewidths=2, zorder=5)
        ax.annotate(labels[nid], (x, y),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # Legend
    legend_handles = [
        Line2D([0], [0], color=COLORS.get(r['algorithm'], 'gray'), lw=2.5,
               label=f"{r['algorithm']}  (cost={r['path_cost']:.0f}, "
                     f"hops={r['hop_count']}, exp={r['nodes_expanded']})")
        for r in valid
    ]
    legend_handles += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
               markersize=10, label='START'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=10, label='GOAL'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8,
              framealpha=0.95, facecolor='white')

    ax.set_title('AI Pathfinding — Custom Node Graph\n'
                 f'{len(nodes)} nodes | All algorithm paths overlaid',
                 fontsize=13, pad=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    plt.savefig('paths_map.png', dpi=150, bbox_inches='tight')
    print("Graph visualization saved to paths_map.png")
    plt.show()


# ── Table ─────────────────────────────────────────────────────────────────────

def print_table(records: list) -> None:
    header = f"{'Algorithm':<20} {'Path Cost':>14} {'Hops':>6} {'Expanded':>10} {'Found':>6}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in records:
        found = "Yes" if r['path_cost'] is not None else "No"
        cost  = f"{r['path_cost']:.2f}" if r['path_cost'] is not None else "—"
        print(f"{r['algorithm']:<20} {cost:>14} {r['hop_count']:>6} "
              f"{r['nodes_expanded']:>10} {found:>6}")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("   AI Pathfinding System — Custom Node Graph")
    print("=" * 52)
    print("\nStep 1: Place your nodes on the map.")
    print("  • First click  = START")
    print("  • Last click   = GOAL")
    print("  • Middle clicks = intermediate nodes")
    print("  • Recommended: 5-15 nodes\n")

    # Collect nodes from map
    nodes = collect_nodes_from_map()
    if len(nodes) < 2:
        print("Not enough nodes selected. Exiting.")
        return

    print(f"\n  Nodes collected: {len(nodes)}")
    for n in nodes:
        print(f"    [{n['label']:>6}] id={n['id']}  ({n['lat']:.5f}, {n['lon']:.5f})")

    start = nodes[0]['id']
    goal  = nodes[-1]['id']

    # Build custom graph
    print("\nStep 2: Building custom graph...")
    G = build_custom_graph(nodes)

    # Run all algorithms
    print("\nStep 3: Running algorithms...")
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

    # Graph + paths visualization
    print("\nStep 4: Visualizing...")
    visualize_matplotlib(G, records, nodes, start, goal)


if __name__ == '__main__':
    main()
