import type { WeightParams } from '../types'

interface Props {
  params: WeightParams
  onChange: (p: WeightParams) => void
  loading: boolean
}

const sliders: { key: keyof WeightParams; label: string; color: string }[] = [
  { key: 'traffic_weight',  label: 'Traffic Intensity',  color: '#e74c3c' },
  { key: 'safety_weight',   label: 'Safety Factor',      color: '#27ae60' },
  { key: 'pothole_weight',  label: 'Pothole Factor',     color: '#e67e22' },
  { key: 'road_age_weight', label: 'Road Age / Quality', color: '#9b59b6' },
  { key: 'turn_weight',     label: 'Turn Complexity',    color: '#1abc9c' },
]

export default function WeightControls({ params, onChange, loading }: Props) {
  return (
    <div style={{ padding: '12px 16px', background: '#1e2a38', borderRadius: 8 }}>
      <div style={{ color: '#aaa', fontSize: 11, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
        Weight Parameters
      </div>
      {sliders.map(({ key, label, color }) => (
        <div key={key} style={{ marginBottom: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ color: '#ccc', fontSize: 12 }}>{label}</span>
            <span style={{ color, fontSize: 12, fontWeight: 700 }}>{params[key].toFixed(1)}×</span>
          </div>
          <input
            type="range" min={0.1} max={3} step={0.1}
            value={params[key]}
            disabled={loading}
            onChange={e => onChange({ ...params, [key]: parseFloat(e.target.value) })}
            style={{ width: '100%', accentColor: color }}
          />
        </div>
      ))}
      {loading && (
        <div style={{ color: '#3498db', fontSize: 11, textAlign: 'center', marginTop: 4 }}>
          Running algorithms...
        </div>
      )}
    </div>
  )
}
