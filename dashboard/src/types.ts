export interface AlgorithmRecord {
  algorithm: string
  path_cost: number | null
  hop_count: number
  nodes_expanded: number
  path: number[]
  is_optimal: boolean
}

export interface NodeData {
  id: number
  lat: number
  lon: number
  label: string
  name: string
}

export interface EdgeData {
  source: number
  target: number
  length: number
  traffic: number
  safety: number
  pothole: number
  weight: number
}

export interface GraphData {
  nodes: NodeData[]
  edges: EdgeData[]
  start: number
  goal: number
  chosen_nodes: NodeData[]
}

export interface WeightParams {
  traffic_weight:  number   // Traffic Intensity
  safety_weight:   number   // Safety Index (benefit — divides cost)
  road_age_weight: number   // Road Quality (older = higher cost)
  turn_weight:     number   // Turn Complexity
  // pothole_weight removed — excluded from weight matrix
}
