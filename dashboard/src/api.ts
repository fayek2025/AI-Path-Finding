import type { GraphData, WeightParams, AlgorithmRecord } from './types'

const BASE = 'http://localhost:8000'

export async function fetchGraphData(): Promise<GraphData> {
  const res = await fetch(`${BASE}/api/graph-data`)
  if (!res.ok) throw new Error('Failed to load graph data')
  return res.json()
}

export async function runAlgorithms(
  params: WeightParams
): Promise<{ records: AlgorithmRecord[] }> {
  const res = await fetch(`${BASE}/api/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error('Failed to run algorithms')
  return res.json()
}

export async function snapNode(lat: number, lon: number): Promise<{ id: number; lat: number; lon: number; name: string }> {
  const res = await fetch(`${BASE}/api/snap-node`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lat, lon }),
  })
  if (!res.ok) throw new Error('Failed to snap node')
  return res.json()
}

export async function saveMapSnapshot(
  dataUrl: string,
  label?: string
): Promise<{ file: string; message: string }> {
  const res = await fetch(`${BASE}/api/save-map-snapshot`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl, label: label ?? '' }),
  })
  if (!res.ok) throw new Error('Failed to save map snapshot')
  return res.json()
}

export async function generateGraphs(
  params: WeightParams
): Promise<{ path_map: string; complexity: string; message: string }> {
  const res = await fetch(`${BASE}/api/generate-graphs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error('Failed to generate graphs')
  return res.json()
}

export async function setNodes(
  nodeIds: number[],
  params: WeightParams
): Promise<{ records: AlgorithmRecord[]; nodes: any[]; start: number; goal: number; chosen_nodes: any[] }> {
  const res = await fetch(`${BASE}/api/set-nodes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ node_ids: nodeIds, params }),
  })
  if (!res.ok) throw new Error('Failed to set nodes')
  return res.json()
}
