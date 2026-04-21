import { useEffect, useState, useCallback, useRef } from 'react'
import './index.css'
import MapView from './components/MapView'
import type { MapViewHandle } from './components/MapView'
import WeightControls from './components/WeightControls'
import ResultsTable from './components/ResultsTable'
import Charts from './components/Charts'
import { fetchGraphData, runAlgorithms, snapNode, setNodes, generateGraphs, saveMapSnapshot } from './api'
import type { AlgorithmRecord, WeightParams } from './types'

const COLORS: Record<string, string> = {
  'BFS': '#1f77b4', 'DFS': '#9467bd', 'IDS': '#17becf',
  'UCS': '#2ca02c', 'A*': '#d62728', 'Greedy': '#ff7f0e',
}
const ALL_ALGOS = Object.keys(COLORS)

interface ChosenNode { id: number; lat: number; lon: number; label: string; name?: string }
interface GraphNode  { id: number; lat: number; lon: number; label: string; name: string }

export default function App() {
  const [allNodes, setAllNodes]       = useState<GraphNode[]>([])
  const [chosenNodes, setChosenNodes] = useState<ChosenNode[]>([])
  const [records, setRecords]         = useState<AlgorithmRecord[]>([])
  const [params, setParams] = useState<WeightParams>({ traffic_weight: 1, safety_weight: 1, road_age_weight: 1, turn_weight: 1 })
  const [loading, setLoading]         = useState(false)
  const [snapping, setSnapping]       = useState(false)
  const [graphGenLoading, setGraphGenLoading] = useState(false)
  const [graphGenMsg, setGraphGenMsg] = useState<string | null>(null)
  const [error, setError]             = useState<string | null>(null)
  const [panelOpen, setPanelOpen]     = useState(true)
  const [editMode, setEditMode]       = useState(false)
  const [visibleAlgos, setVisibleAlgos] = useState<Set<string>>(new Set(ALL_ALGOS))
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mapRef      = useRef<MapViewHandle>(null)

  // Initial load
  useEffect(() => {
    fetchGraphData()
      .then(data => {
        setAllNodes(data.nodes)
        setChosenNodes(data.chosen_nodes.map((n, i) => ({
          ...n,
          label: i === 0 ? 'START' : i === data.chosen_nodes.length - 1 ? 'GOAL' : `N${i}`
        })))
        // Only run algorithms if there's an existing graph (saved state)
        if (data.chosen_nodes.length >= 2) {
          return runAlgorithms(params).then(res => setRecords(res.records))
        }
      })
      .catch(e => setError(e.message))
  }, [])

  // Weight slider change — debounced re-run
  const handleParamChange = useCallback((newParams: WeightParams) => {
    setParams(newParams)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const res = await runAlgorithms(newParams)
        setRecords(res.records)
      } catch (e: any) { setError(e.message) }
      finally { setLoading(false) }
    }, 400)
  }, [])

  // Guard against concurrent clicks
  const clickInFlightRef = useRef(false)
  const pendingRunRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Map click — snap to nearest node, add to list
  const handleMapClick = useCallback(async (lat: number, lon: number) => {
    if (clickInFlightRef.current) return   // drop concurrent clicks
    clickInFlightRef.current = true
    setSnapping(true)
    try {
      const snapped = await snapNode(lat, lon)
      // Avoid duplicates
      if (chosenNodes.some(n => n.id === snapped.id)) return

      const newChosen = [...chosenNodes, { ...snapped, label: '', name: snapped.name ?? '' }]
      // Relabel
      newChosen.forEach((n, i) => {
        n.label = i === 0 ? 'START' : i === newChosen.length - 1 ? 'GOAL' : `N${i}`
      })
      setChosenNodes(newChosen)

      // Debounce the heavy set-nodes call so rapid placements only trigger once
      if (newChosen.length >= 2) {
        if (pendingRunRef.current) clearTimeout(pendingRunRef.current)
        pendingRunRef.current = setTimeout(async () => {
          setLoading(true)
          try {
            const res = await setNodes(newChosen.map(n => n.id), params)
            setAllNodes(res.nodes)
            setRecords(res.records)
            setChosenNodes(res.chosen_nodes.map((n, i) => ({
              ...n,
              label: i === 0 ? 'START' : i === res.chosen_nodes.length - 1 ? 'GOAL' : `N${i}`
            })))
          } catch (e: any) { setError(e.message) }
          finally { setLoading(false) }
        }, 300)
      }
    } catch (e: any) { setError(e.message) }
    finally { setSnapping(false); clickInFlightRef.current = false }
  }, [chosenNodes, params])

  const removeLastNode = useCallback(async () => {
    if (chosenNodes.length === 0) return
    const newChosen = chosenNodes.slice(0, -1)
    newChosen.forEach((n, i) => {
      n.label = i === 0 ? 'START' : i === newChosen.length - 1 ? 'GOAL' : `N${i}`
    })
    setChosenNodes(newChosen)
    if (newChosen.length >= 2) {
      setLoading(true)
      try {
        const res = await setNodes(newChosen.map(n => n.id), params)
        setAllNodes(res.nodes)
        setRecords(res.records)
      } catch (e: any) { setError(e.message) }
      finally { setLoading(false) }
    } else {
      setRecords([])
    }
  }, [chosenNodes, params])

  const clearNodes = () => { setChosenNodes([]); setRecords([]) }

  const toggleAlgo = (algo: string) => {
    setVisibleAlgos(prev => {
      const next = new Set(prev)
      next.has(algo) ? next.delete(algo) : next.add(algo)
      return next
    })
  }

  const handleGenerateGraphs = useCallback(async () => {
    setGraphGenLoading(true)
    setGraphGenMsg(null)
    try {
      const res = await generateGraphs(params)
      const expCount = (res as any).expansion_maps?.length ?? 0
      setGraphGenMsg(`✅ Saved: path map, complexity + ${expCount} expansion maps`)
    } catch (e: any) {
      setGraphGenMsg(`❌ ${e.message}`)
    } finally {
      setGraphGenLoading(false)
    }
  }, [params])

  const handleMapSnapshot = useCallback(async () => {
    if (!mapRef.current) return
    setGraphGenLoading(true)
    setGraphGenMsg(null)
    try {
      // 1. Full combined snapshot
      const dataUrl = await mapRef.current.snapshot()
      await saveMapSnapshot(dataUrl, 'all')

      // 2. One snapshot per algorithm (only those with a found path)
      const algosWithPaths = records.filter(r => r.path_cost !== null).map(r => r.algorithm)
      for (const algo of algosWithPaths) {
        const algoUrl = await mapRef.current.snapshotAlgo(algo)
        await saveMapSnapshot(algoUrl, algo)
      }

      setGraphGenMsg(`✅ Saved ${1 + algosWithPaths.length} snapshots (all + per-algorithm)`)
    } catch (e: any) {
      setGraphGenMsg(`❌ ${e.message}`)
    } finally {
      setGraphGenLoading(false)
    }
  }, [records])

  if (error) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#0f1923', color: '#e74c3c', flexDirection: 'column', gap: 12 }}>
      <div style={{ fontSize: 18 }}>API Error</div>
      <div style={{ fontSize: 13, color: '#888' }}>{error}</div>
      <button onClick={() => setError(null)} style={{ padding: '6px 14px', background: '#2980b9', color: 'white', border: 'none', borderRadius: 5, cursor: 'pointer' }}>Dismiss</button>
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#0f1923', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ height: 44, background: 'linear-gradient(135deg,#1a252f,#2980b9)', display: 'flex', alignItems: 'center', padding: '0 16px', gap: 12, flexShrink: 0 }}>
        <span style={{ color: 'white', fontWeight: 700, fontSize: 15 }}>🗺️ AI Pathfinding Dashboard</span>
        <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: 11 }}>
          {chosenNodes.length} nodes · {records.filter(r => r.path_cost).length} paths found
        </span>

        {/* Edit mode toggle */}
        <button onClick={() => setEditMode(e => !e)} style={{
          padding: '5px 12px', borderRadius: 5, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
          background: editMode ? '#27ae60' : 'rgba(255,255,255,0.15)',
          color: 'white', transition: 'background 0.2s'
        }}>
          {editMode ? '✏️ Placing nodes...' : '✏️ Edit Nodes'}
        </button>

        {editMode && chosenNodes.length > 0 && (
          <>
            <button onClick={removeLastNode} style={{ padding: '5px 10px', background: '#e67e22', color: 'white', border: 'none', borderRadius: 5, cursor: 'pointer', fontSize: 12 }}>
              ↩ Undo
            </button>
            <button onClick={clearNodes} style={{ padding: '5px 10px', background: '#e74c3c', color: 'white', border: 'none', borderRadius: 5, cursor: 'pointer', fontSize: 12 }}>
              🗑 Clear
            </button>
          </>
        )}

        {(loading || snapping) && (
          <span style={{ color: '#3498db', fontSize: 11 }}>
            {snapping ? 'Snapping...' : 'Running algorithms...'}
          </span>
        )}

        {/* Generate Graphs button — visible once we have results */}
        {records.length > 0 && (
          <button
            onClick={handleGenerateGraphs}
            disabled={graphGenLoading}
            title="Generate path map + complexity graphs as PNG files"
            style={{
              padding: '5px 12px', borderRadius: 5, border: 'none', cursor: graphGenLoading ? 'wait' : 'pointer',
              fontSize: 12, fontWeight: 600,
              background: graphGenLoading ? '#555' : 'linear-gradient(135deg,#8e44ad,#2980b9)',
              color: 'white', opacity: graphGenLoading ? 0.7 : 1, transition: 'opacity 0.2s',
            }}
          >
            {graphGenLoading ? '⏳ Generating...' : '📊 Generate Graphs'}
          </button>
        )}

        {/* Map snapshot button */}
        {records.length > 0 && (
          <button
            onClick={handleMapSnapshot}
            disabled={graphGenLoading}
            title="Save a screenshot of the current map view"
            style={{
              padding: '5px 12px', borderRadius: 5, border: 'none', cursor: graphGenLoading ? 'wait' : 'pointer',
              fontSize: 12, fontWeight: 600,
              background: graphGenLoading ? '#555' : 'linear-gradient(135deg,#16a085,#27ae60)',
              color: 'white', opacity: graphGenLoading ? 0.7 : 1, transition: 'opacity 0.2s',
            }}
          >
            {graphGenLoading ? '⏳ ...' : '📷 Snapshot Map'}
          </button>
        )}
        {graphGenMsg && (
          <span style={{ fontSize: 10, color: graphGenMsg.startsWith('✅') ? '#2ecc71' : '#e74c3c', maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {graphGenMsg}
          </span>
        )}

        <button onClick={() => setPanelOpen(p => !p)} style={{ marginLeft: 'auto', padding: '5px 12px', background: 'rgba(255,255,255,0.15)', border: '1px solid rgba(255,255,255,0.3)', color: 'white', borderRadius: 5, cursor: 'pointer', fontSize: 12 }}>
          {panelOpen ? '✕ Close Panel' : '☰ Analysis Panel'}
        </button>
      </div>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Map */}
        <div style={{ flex: 1, position: 'relative' }}>
          <MapView
            ref={mapRef}
            chosenNodes={chosenNodes}
            allNodes={allNodes}
            records={records}
            visibleAlgos={visibleAlgos}
            editMode={editMode}
            onMapClick={handleMapClick}
          />

          {/* Edit mode hint */}
          {editMode && (
            <div style={{ position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)', zIndex: 1000, background: 'rgba(39,174,96,0.9)', color: 'white', padding: '8px 16px', borderRadius: 20, fontSize: 12, pointerEvents: 'none' }}>
              Click map to place nodes · First = START · Last = GOAL
            </div>
          )}

          {/* Layer toggles */}
          <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1000, background: 'rgba(15,25,35,0.92)', borderRadius: 8, padding: '10px 14px', minWidth: 160 }}>
            <div style={{ color: '#888', fontSize: 10, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Paths</div>
            {ALL_ALGOS.map(algo => (
              <div key={algo} onClick={() => toggleAlgo(algo)}
                style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5, cursor: 'pointer', opacity: visibleAlgos.has(algo) ? 1 : 0.35 }}>
                <div style={{ width: 22, height: 4, borderRadius: 2, background: COLORS[algo] }} />
                <span style={{ fontSize: 11, color: '#ccc' }}>{algo}</span>
              </div>
            ))}
          </div>

          {/* Node list */}
          {chosenNodes.length > 0 && (
            <div style={{ position: 'absolute', top: 10, right: panelOpen ? 0 : 10, zIndex: 1000, background: 'rgba(15,25,35,0.92)', borderRadius: 8, padding: '10px 14px', maxWidth: 200 }}>
              <div style={{ color: '#888', fontSize: 10, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>Chosen Nodes</div>
              {chosenNodes.map((n, i) => {
                const color = i === 0 ? '#27ae60' : i === chosenNodes.length - 1 ? '#e74c3c' : '#3498db'
                const displayName = n.name && !n.name.match(/^\d+\.\d+/) ? n.name : `${n.lat.toFixed(4)}, ${n.lon.toFixed(4)}`
                return (
                  <div key={n.id} style={{ fontSize: 11, color: '#ccc', marginBottom: 5 }}>
                    <span style={{ color, fontWeight: 700 }}>{n.label}</span>
                    <span style={{ color: '#aaa', marginLeft: 6, display: 'block', fontSize: 10, lineHeight: 1.3 }}>{displayName}</span>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Side panel */}
        {panelOpen && (
          <div style={{ width: 380, background: '#131e2b', overflowY: 'auto', borderLeft: '1px solid #1e2d3d', display: 'flex', flexDirection: 'column', gap: 10, padding: 10 }}>
            <WeightControls params={params} onChange={handleParamChange} loading={loading} />
            {records.length > 0 && (
              <>
                <div style={{ background: '#1a2535', borderRadius: 8, padding: 12 }}>
                  <div style={{ color: '#aaa', fontSize: 11, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>Results</div>
                  <ResultsTable records={records} />
                </div>
                <div style={{ background: '#1a2535', borderRadius: 8, padding: 12 }}>
                  <div style={{ color: '#aaa', fontSize: 11, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>Charts</div>
                  <Charts records={records} />
                </div>
              </>
            )}
            {records.length === 0 && chosenNodes.length < 2 && (
              <div style={{ color: '#555', fontSize: 12, textAlign: 'center', padding: 20 }}>
                Click "Edit Nodes" then place at least 2 nodes on the map to run algorithms.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
