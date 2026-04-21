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
  traffic_weight:  number
  safety_weight:   number
  pothole_weight:  number
  road_age_weight: number
  turn_weight:     number
}
