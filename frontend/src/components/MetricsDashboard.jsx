import { useState, useEffect } from 'react'
import api from '../api'

// Backend returns: { "Policy Gate Accuracy (PGA)": { value, threshold, pass, detail }, ... }
function MetricTile({ label, metric }) {
  const pass = metric.pass  // true | false | null
  const tileColor  = pass === true ? '#f0fdf4' : pass === false ? '#fef2f2' : '#f8fafc'
  const borderColor = pass === true ? '#86efac' : pass === false ? '#fca5a5' : '#e2e8f0'
  const valueColor  = pass === true ? '#15803d' : pass === false ? '#dc2626' : '#475569'

  // Extract 2-5 char abbreviation from the parenthetical e.g. "Policy Gate Accuracy (PGA)" → "PGA"
  const abbrMatch = label.match(/\(([^)]+)\)$/)
  const abbr = abbrMatch ? abbrMatch[1] : label.slice(0, 3).toUpperCase()
  const shortLabel = abbrMatch ? label.replace(/\s*\([^)]+\)$/, '') : label

  return (
    <div style={{ ...s.tile, background: tileColor, borderColor }}>
      <div style={s.abbr}>{abbr}</div>
      <div style={{ ...s.value, color: valueColor }}>{metric.value}</div>
      <div style={s.tileLabel}>{shortLabel}</div>
      {metric.threshold && (
        <div style={s.threshold}>Threshold: {metric.threshold}</div>
      )}
      {metric.detail && (
        <div style={s.detail}>{metric.detail}</div>
      )}
      <div style={s.badge}>
        {pass === true  ? <span style={s.pass}>✅ PASS</span>
         : pass === false ? <span style={s.fail}>❌ FAIL</span>
         : <span style={s.na}>⏳ PENDING</span>}
      </div>
    </div>
  )
}

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastRefresh, setLastRefresh] = useState(null)

  const fetchMetrics = () => {
    setLoading(true)
    api.get('/admin/metrics')
      .then(r => { setMetrics(r.data); setLastRefresh(new Date()) })
      .catch(e => setError(e.response?.data?.detail || 'Failed to load metrics'))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchMetrics() }, [])

  return (
    <div style={s.wrap}>
      <div style={s.header}>
        <div>
          <h2 style={s.title}>📊 Governance Metrics</h2>
          {lastRefresh && (
            <div style={s.lastRefresh}>
              Last updated: {lastRefresh.toLocaleTimeString()}
            </div>
          )}
        </div>
        <button style={s.refreshBtn} onClick={fetchMetrics} disabled={loading}>
          {loading ? '⏳ Loading…' : '↻ Refresh Metrics'}
        </button>
      </div>

      {error && <div style={s.error}>{error}</div>}

      {loading && !metrics && (
        <div style={s.loading}>Computing metrics from database…</div>
      )}

      {metrics && (
        <div style={s.grid}>
          {Object.entries(metrics)
            .filter(([key]) => key !== '_meta')
            .map(([key, metric]) => (
              <MetricTile key={key} label={key} metric={metric} />
            ))}
        </div>
      )}
    </div>
  )
}

const s = {
  wrap: { background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb',
          padding: 24, marginBottom: 24 },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
            marginBottom: 20 },
  title: { margin: '0 0 4px', fontSize: 18, color: '#1e293b' },
  lastRefresh: { fontSize: 12, color: '#94a3b8' },
  refreshBtn: { padding: '8px 16px', background: '#2563eb', color: '#fff', border: 'none',
                borderRadius: 8, cursor: 'pointer', fontSize: 13, fontWeight: 600,
                opacity: 1, transition: 'opacity 0.2s' },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 16 },
  tile: { border: '2px solid', borderRadius: 10, padding: 16, textAlign: 'center' },
  abbr: { fontSize: 11, fontWeight: 800, color: '#94a3b8', letterSpacing: 1.5,
          textTransform: 'uppercase', marginBottom: 8 },
  value: { fontSize: 26, fontWeight: 800, marginBottom: 4 },
  tileLabel: { fontSize: 11, color: '#64748b', marginBottom: 6, lineHeight: 1.4 },
  threshold: { fontSize: 10, color: '#94a3b8', marginBottom: 4 },
  detail: { fontSize: 10, color: '#94a3b8', marginBottom: 6, fontStyle: 'italic', lineHeight: 1.4 },
  badge: { marginTop: 4 },
  pass: { background: '#dcfce7', color: '#15803d', padding: '2px 10px', borderRadius: 20,
          fontSize: 11, fontWeight: 700 },
  fail: { background: '#fee2e2', color: '#dc2626', padding: '2px 10px', borderRadius: 20,
          fontSize: 11, fontWeight: 700 },
  na: { background: '#f1f5f9', color: '#64748b', padding: '2px 10px', borderRadius: 20,
        fontSize: 11, fontWeight: 700 },
  error: { color: '#ef4444', fontSize: 13, marginBottom: 16 },
  loading: { color: '#64748b', fontSize: 13, padding: '24px 0', textAlign: 'center' },
}
