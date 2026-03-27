import { useState, useEffect } from 'react'
import api from '../api'

export default function EscalationQueue({ onSelectChild }) {
  const [queue, setQueue] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    api.get('/clinician/queue')
      .then(r => setQueue(r.data))
      .catch(e => setError(e.response?.data?.detail || 'Failed to load queue'))
      .finally(() => setLoading(false))
  }, [])

  const handleSelect = (childId) => {
    setSelected(childId)
    onSelectChild?.(childId)
  }

  if (loading) return <div style={s.loading}>Loading escalation queue…</div>
  if (error)   return <div style={s.error}>{error}</div>

  return (
    <div style={s.wrap}>
      <div style={s.header}>
        <h3 style={s.title}>🚨 Escalation Queue</h3>
        <span style={s.count}>{queue.length} child{queue.length !== 1 ? 'ren' : ''}</span>
      </div>
      <p style={s.sub}>Children with 2 or more consecutive abstentions requiring clinician review.</p>

      {queue.length === 0 ? (
        <div style={s.empty}>
          <div style={s.emptyIcon}>✅</div>
          <div style={s.emptyText}>No escalations — all screenings proceeding normally.</div>
        </div>
      ) : (
        <div style={s.list}>
          {queue.map(item => (
            <div key={item.child_id}
                 style={{ ...s.item, ...(selected === item.child_id ? s.itemSelected : {}) }}
                 onClick={() => handleSelect(item.child_id)}>
              <div style={s.itemLeft}>
                <div style={s.childId}>{item.child_id}</div>
                <div style={s.itemMeta}>
                  {item.abstention_count} abstentions · Last: {new Date(item.last_abstention).toLocaleDateString()}
                </div>
                <div style={s.reasonText}>{item.last_reason}</div>
              </div>
              <div style={s.pill}>
                <span style={s.count2}>{item.abstention_count}</span>
                <span style={s.pillLabel}>abstentions</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const s = {
  wrap: { background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb', overflow: 'hidden' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '16px 20px', background: '#fef2f2', borderBottom: '1px solid #fecaca' },
  title: { margin: 0, fontSize: 15, color: '#991b1b' },
  count: { background: '#ef4444', color: '#fff', borderRadius: 20, padding: '2px 10px',
           fontSize: 12, fontWeight: 700 },
  sub: { margin: 0, padding: '10px 20px', fontSize: 12, color: '#64748b',
         borderBottom: '1px solid #f1f5f9' },
  list: { maxHeight: 320, overflowY: 'auto' },
  item: { display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '14px 20px', borderBottom: '1px solid #f1f5f9',
          cursor: 'pointer', transition: 'background 0.15s' },
  itemSelected: { background: '#fef2f2' },
  itemLeft: { flex: 1 },
  childId: { fontWeight: 700, fontSize: 14, color: '#1e293b' },
  itemMeta: { fontSize: 12, color: '#64748b', marginTop: 2 },
  reasonText: { fontSize: 12, color: '#ef4444', marginTop: 4, fontStyle: 'italic' },
  pill: { display: 'flex', flexDirection: 'column', alignItems: 'center',
          background: '#fef2f2', border: '1px solid #fecaca',
          borderRadius: 8, padding: '6px 12px', minWidth: 60 },
  count2: { fontSize: 20, fontWeight: 800, color: '#ef4444' },
  pillLabel: { fontSize: 10, color: '#f87171', marginTop: 1 },
  empty: { padding: '32px 20px', textAlign: 'center' },
  emptyIcon: { fontSize: 32, marginBottom: 8 },
  emptyText: { color: '#22c55e', fontWeight: 600, fontSize: 13 },
  loading: { padding: 20, color: '#64748b', fontSize: 13 },
  error: { padding: 20, color: '#ef4444', fontSize: 13 },
}
