import type { AlgorithmRecord } from '../types'

const COLORS: Record<string, string> = {
  'BFS': '#1f77b4', 'DFS': '#9467bd', 'IDS': '#17becf',
  'UCS': '#2ca02c', 'A*': '#d62728', 'Greedy': '#ff7f0e',
  'IDA*': '#e377c2', 'Bidirectional A*': '#bcbd22',
}

interface Props { records: AlgorithmRecord[] }

export default function ResultsTable({ records }: Props) {

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
      <thead>
        <tr style={{ background: '#2c3e50', color: 'white' }}>
          {['Algorithm', 'Cost', 'Hops', 'Expanded', 'Optimal', 'Found'].map(h => (
            <th key={h} style={{ padding: '6px 8px', textAlign: 'left', fontSize: 10 }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {records.map(r => (
          <tr key={r.algorithm}
            style={{ background: r.is_optimal ? '#1a3a2a' : 'transparent', borderBottom: '1px solid #2a3a4a' }}>
            <td style={{ padding: '5px 8px', color: '#ddd' }}>
              <span style={{
                display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                background: COLORS[r.algorithm] ?? '#999', marginRight: 6
              }} />
              {r.algorithm}
            </td>
            <td style={{ padding: '5px 8px', color: r.path_cost ? '#fff' : '#555' }}>
              {r.path_cost ? r.path_cost.toFixed(2) : '—'}
            </td>
            <td style={{ padding: '5px 8px', color: '#ccc' }}>{r.hop_count}</td>
            <td style={{ padding: '5px 8px', color: '#ccc' }}>{r.nodes_expanded}</td>
            <td style={{ padding: '5px 8px', color: '#27ae60' }}>
              {r.is_optimal ? '✓' : ''}
            </td>
            <td style={{ padding: '5px 8px', color: r.path_cost ? '#27ae60' : '#e74c3c', fontWeight: 700 }}>
              {r.path_cost ? '✓' : '✗'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
