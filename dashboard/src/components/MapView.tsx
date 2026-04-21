import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import type { AlgorithmRecord } from '../types'

const COLORS: Record<string, string> = {
  'BFS': '#1f77b4', 'DFS': '#9467bd', 'IDS': '#17becf',
  'UCS': '#2ca02c', 'A*': '#d62728', 'Greedy': '#ff7f0e',
}

interface ChosenNode { id: number; lat: number; lon: number; label: string; name?: string }
interface GraphNode  { id: number; lat: number; lon: number; label: string; name: string }

interface Props {
  chosenNodes: ChosenNode[]
  allNodes: GraphNode[]
  records: AlgorithmRecord[]
  visibleAlgos: Set<string>
  editMode: boolean
  onMapClick: (lat: number, lon: number) => void
}

export interface MapViewHandle {
  /** Capture the current map view and return a base64 PNG data-URL. */
  snapshot: () => Promise<string>
  /** Show only one algorithm, snapshot, then restore previous visibility. */
  snapshotAlgo: (algo: string) => Promise<string>
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload  = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

const MapView = forwardRef<MapViewHandle, Props>(function MapView(
  { chosenNodes, allNodes, records, visibleAlgos, editMode, onMapClick },
  ref
) {
  const mapRef           = useRef<L.Map | null>(null)
  const screenshoterRef  = useRef<any>(null)
  const pathLayersRef    = useRef<Record<string, L.LayerGroup>>({})
  const nodeLayersRef    = useRef<L.LayerGroup | null>(null)

  // Expose snapshot() and snapshotAlgo() to parent via ref
  useImperativeHandle(ref, () => ({
    snapshot: async () => {
      const screenshoter = screenshoterRef.current
      if (!screenshoter) throw new Error('Map not ready')
      const blob = await screenshoter.takeScreen('blob') as Blob
      return blobToDataUrl(blob)
    },

    snapshotAlgo: async (algo: string) => {
      const map = mapRef.current
      const screenshoter = screenshoterRef.current
      if (!map || !screenshoter) throw new Error('Map not ready')

      // Hide all paths except the target algorithm
      const prev: Record<string, boolean> = {}
      Object.entries(pathLayersRef.current).forEach(([name, group]) => {
        prev[name] = map.hasLayer(group)
        if (name !== algo) { if (map.hasLayer(group)) map.removeLayer(group) }
        else               { if (!map.hasLayer(group)) group.addTo(map) }
      })

      // Small delay so Leaflet repaints before capture
      await new Promise(r => setTimeout(r, 120))
      const blob = await screenshoter.takeScreen('blob') as Blob
      const dataUrl = await blobToDataUrl(blob)

      // Restore previous visibility
      Object.entries(pathLayersRef.current).forEach(([name, group]) => {
        if (prev[name]) { if (!map.hasLayer(group)) group.addTo(map) }
        else            { if (map.hasLayer(group))  map.removeLayer(group) }
      })

      return dataUrl
    },
  }))

  // Init map once
  useEffect(() => {
    if (mapRef.current) return
    const center: [number, number] = chosenNodes.length
      ? [chosenNodes[0].lat, chosenNodes[0].lon]
      : [23.7766, 90.4227]

    const map = L.map('leaflet-map').setView(center, 14)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors', maxZoom: 19,
    }).addTo(map)

    // Attach screenshoter after dynamic import resolves
    import('leaflet-simple-map-screenshoter').then(m => {
      // The package's type definition is a namespace, not a class — cast to any
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const Cls = (m.default ?? m) as any
      screenshoterRef.current = new Cls({ hidden: true }).addTo(map)
    })

    mapRef.current = map
    nodeLayersRef.current = L.layerGroup().addTo(map)
  }, [])

  // Map click handler
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    map.off('click')
    if (editMode) {
      map.on('click', (e: L.LeafletMouseEvent) => onMapClick(e.latlng.lat, e.latlng.lng))
      map.getContainer().style.cursor = 'crosshair'
    } else {
      map.getContainer().style.cursor = ''
    }
  }, [editMode, onMapClick])

  // Redraw chosen node markers
  useEffect(() => {
    const group = nodeLayersRef.current
    if (!group) return
    group.clearLayers()
    chosenNodes.forEach((n, i) => {
      const color = i === 0 ? '#27ae60' : i === chosenNodes.length - 1 ? '#e74c3c' : '#3498db'
      const displayName = n.name && !n.name.match(/^\d+\.\d+/) ? n.name : `${n.lat.toFixed(4)}, ${n.lon.toFixed(4)}`
      const tooltipHtml = `<b style="color:${color}">${n.label}</b><br/><span style="font-size:11px">${displayName}</span>`
      L.circleMarker([n.lat, n.lon], { radius: 11, color, fillColor: color, fillOpacity: 0.9, weight: 3 })
        .addTo(group)
        .bindTooltip(tooltipHtml, { permanent: true, direction: 'top', className: 'node-tooltip' })
    })
  }, [chosenNodes])

  // Redraw paths
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const nodeById = Object.fromEntries(allNodes.map(n => [n.id, n]))
    Object.values(pathLayersRef.current).forEach(lg => map.removeLayer(lg))
    pathLayersRef.current = {}

    records.forEach(r => {
      if (!r.path || r.path.length < 2) return
      const color = COLORS[r.algorithm] ?? '#999'
      const group = L.layerGroup()
      for (let i = 0; i < r.path.length - 1; i++) {
        const a = nodeById[r.path[i]]
        const b = nodeById[r.path[i + 1]]
        if (!a || !b) continue
        L.polyline([[a.lat, a.lon], [b.lat, b.lon]], { color, weight: 8, opacity: 0.8 })
          .addTo(group).bindTooltip(r.algorithm)
      }
      pathLayersRef.current[r.algorithm] = group
      if (visibleAlgos.has(r.algorithm)) group.addTo(map)
    })
  }, [records, allNodes])

  // Toggle visibility
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    Object.entries(pathLayersRef.current).forEach(([algo, group]) => {
      if (visibleAlgos.has(algo)) { if (!map.hasLayer(group)) group.addTo(map) }
      else { if (map.hasLayer(group)) map.removeLayer(group) }
    })
  }, [visibleAlgos])

  return <div id="leaflet-map" style={{ width: '100%', height: '100%' }} />
})

export default MapView
