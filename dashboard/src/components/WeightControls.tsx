import type { WeightParams } from '../types'

interface Props {
  params: WeightParams
  onChange: (p: WeightParams) => void
  loading: boolean
}

// Pothole factor excluded — weight matrix uses:
// Traffic Intensity × Road Quality × Turn Complexity / Safety Index
const sliders: { key: keyof WeightParams; label: string; color: string; hint: string }[] = [
  { key: 'traffic_weight',  label: 'Traffic Intensity',  color: '#e74c3c', hint: 'Higher = avoid congested roads' },
  { key: 'safety_weight',   label: 'Safety Index',       color: '#27ae60', hint: 'Higher = strongly prefer safer roads' },
  { key: 'road_age_weight', label: 'Road Quality',       color: '#9b59b6', hint: 'Higher = avoid older/degraded roads' },
  { key: 'turn_weight',     label: 'Turn Complexity',    color: '#1abc9c', hint: 'Higher = avoid complex intersections' },
]

export default function WeightControls({ params, onChange, loading }: Props) {
  return (
    <div style={{ padding: '12px 16px', background: '#1e2a38', borderRadius: 8 }}>
      <div style={{ color: '#aaa', fontSize: 11, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
        Weight Parameters
      </div>
      <div style={{ color: '#555', fontSize: 10, marginBottom: 12 }}>
        Formula: length × traffic × road_quality × turn / safety
      </div>
      {sliders.map(({ key, label, color, hint }) => (
        <div key={key} style={{ marginBottom: 14 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
            <span style={{ color: '#ccc', fontSize: 12 }}>{label}</span>
            <span style={{ color, fontSize: 12, fontWeight: 700 }}>{params[key].toFixed(1)}×</span>
          </div>
          <div style={{ color: '#555', fontSize: 10, marginBottom: 4 }}>{hint}</div>
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
