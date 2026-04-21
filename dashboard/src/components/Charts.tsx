import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import type { AlgorithmRecord } from '../types'

const COLORS: Record<string, string> = {
  'BFS': '#1f77b4', 'DFS': '#9467bd', 'IDS': '#17becf',
  'UCS': '#2ca02c', 'A*': '#d62728', 'Greedy': '#ff7f0e',
  'IDA*': '#e377c2', 'Bidirectional A*': '#bcbd22',
}

interface ChartProps { records: AlgorithmRecord[] }

function MiniBar({ data, dataKey, label }: { data: any[]; dataKey: string; label: string }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ color: '#aaa', fontSize: 10, marginBottom: 4, textTransform: 'uppercase' }}>{label}</div>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 20 }}>
          <XAxis dataKey="algorithm" tick={{ fontSize: 8, fill: '#888' }} angle={-30} textAnchor="end" />
          <YAxis tick={{ fontSize: 8, fill: '#888' }} />
          <Tooltip
            contentStyle={{ background: '#1e2a38', border: 'none', fontSize: 11 }}
            labelStyle={{ color: '#fff' }}
          />
          <Bar dataKey={dataKey} radius={[3, 3, 0, 0]}>
            {data.map(d => <Cell key={d.algorithm} fill={COLORS[d.algorithm] ?? '#999'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Charts({ records }: ChartProps) {
  const valid = records.filter(r => r.path_cost != null)
  return (
    <>
      <MiniBar data={valid} dataKey="path_cost" label="Path Cost" />
      <MiniBar data={records} dataKey="hop_count" label="Hop Count" />
      <MiniBar data={records} dataKey="nodes_expanded" label="Nodes Expanded" />
    </>
  )
}
