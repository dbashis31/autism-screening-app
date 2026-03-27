import { useState, useEffect } from 'react'
import api from '../api'

export default function AbstentionHistory({ childId }) {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!childId) return
    setLoading(true)
    setError(null)
    // Find sessions for this child and load abstentions
    api.get('/clinician/sessions')
      .then(res => {
        // Filter sessions for this child that have abstention info
        const childSessions = res.data.filter(s => s.child_id === childId)
        // Collect abstentions from the abstention_history embedded in each session
        // or directly from pipeline_result
        const abslist = []
        childSessions.forEach(s => {
          if (s.pipeline_result?.abstaining) {
            abslist.push({
              session_id: s.id,
              reason: s.pipeline_result?.abstention_reason || 'Abstained',
              created_at: s.created_at,
              pipeline_status: s.pipeline_status,
            })
          }
        })
        setHistory(abslist)
      })
      .catch(e => setError(e.response?.data?.detail || 'Failed to load history'))
      .finally(() => setLoading(false))
  }, [childId])

  if (!childId) {
    return (
      <div style={s.wrap}>
        <h3 style={s.title}>📋 Abstention History</h3>
        <div style={s.empty}>Select a child from the escalation queue to view their history.</div>
      </div>
    )
  }

  return (
    <div style={s.wrap}>
      <h3 style={s.title}>📋 Abstention History — <span style={{ color: '#2563eb' }}>{childId}</span></h3>

      {loading && <div style={s.loading}>Loading…</div>}
      {error   && <div style={s.error}>{error}</div>}

      {!loading && !error && history.length === 0 && (
        <div style={s.empty}>No abstention records found for this child.</div>
      )}

      {!loading && history.length > 0 && (
        <div style={s.timeline}>
          {history.map((item, i) => (
            <div key={item.session_id} style={s.timelineItem}>
              <div style={s.dot} />
              {i < history.length - 1 && <div style={s.line} />}
              <div style={s.content}>
                <div style={s.date}>{new Date(item.created_at).toLocaleString()}</div>
                <div style={s.sessionRef}>
                  Session: <code style={s.code}>{item.session_id?.slice(0, 8)}…</code>
                </div>
                <div style={s.reason}>{item.reason}</div>
                <div style={{ ...s.statusChip,
                  background: item.pipeline_status === 'complete' ? '#f0fdf4' : '#fffbeb',
                  color: item.pipeline_status === 'complete' ? '#15803d' : '#b45309',
                }}>
                  {item.pipeline_status}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const s = {
  wrap: { background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb',
          padding: 20, marginTop: 16 },
  title: { margin: '0 0 16px', fontSize: 15, color: '#1e293b' },
  timeline: { paddingLeft: 8 },
  timelineItem: { display: 'flex', gap: 12, position: 'relative', marginBottom: 20 },
  dot: { width: 12, height: 12, borderRadius: '50%', background: '#f59e0b',
         border: '2px solid #fcd34d', flexShrink: 0, marginTop: 3 },
  line: { position: 'absolute', left: 5, top: 18, bottom: -20,
          width: 2, background: '#e5e7eb' },
  content: { flex: 1, paddingBottom: 4 },
  date: { fontSize: 12, color: '#94a3b8', marginBottom: 2 },
  sessionRef: { fontSize: 12, color: '#64748b', marginBottom: 4 },
  code: { background: '#f1f5f9', padding: '1px 4px', borderRadius: 3 },
  reason: { fontSize: 13, color: '#374151', marginBottom: 6, fontStyle: 'italic' },
  statusChip: { display: 'inline-block', padding: '2px 8px', borderRadius: 12, fontSize: 11,
                fontWeight: 600 },
  empty: { color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: '20px 0' },
  loading: { color: '#64748b', fontSize: 13, padding: '12px 0' },
  error: { color: '#ef4444', fontSize: 13 },
}
