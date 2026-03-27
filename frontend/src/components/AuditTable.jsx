import React, { useState, useEffect } from 'react'
import api from '../api'

const PAGE_SIZE = 50

const DECISION_COLORS = {
  ALLOW:   { bg: '#f0fdf4', color: '#15803d', border: '#86efac' },
  BLOCK:   { bg: '#fef2f2', color: '#dc2626', border: '#fca5a5' },
  ABSTAIN: { bg: '#fffbeb', color: '#b45309', border: '#fcd34d' },
  REPORT:  { bg: '#eff6ff', color: '#1d4ed8', border: '#93c5fd' },
  APPROVE: { bg: '#f0fdf4', color: '#15803d', border: '#86efac' },
  REJECT:  { bg: '#fef2f2', color: '#dc2626', border: '#fca5a5' },
  WARN:    { bg: '#fffbeb', color: '#b45309', border: '#fcd34d' },
  ESCALATE:{ bg: '#fdf4ff', color: '#7c3aed', border: '#c4b5fd' },
}

function DecisionBadge({ decision }) {
  const c = DECISION_COLORS[decision] || { bg: '#f8fafc', color: '#64748b', border: '#e2e8f0' }
  return (
    <span style={{ background: c.bg, color: c.color, border: `1px solid ${c.border}`,
                   padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 700,
                   whiteSpace: 'nowrap' }}>
      {decision}
    </span>
  )
}

export default function AuditTable() {
  const [entries, setEntries] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [page, setPage] = useState(0)
  const [filterAgent, setFilterAgent] = useState('')
  const [filterDecision, setFilterDecision] = useState('')
  const [filterSession, setFilterSession] = useState('')
  const [expanded, setExpanded] = useState(null)

  const fetchLog = (pg = 0) => {
    setLoading(true)
    const params = new URLSearchParams({
      limit: PAGE_SIZE,
      skip: pg * PAGE_SIZE,
      ...(filterAgent    ? { agent: filterAgent }       : {}),
      ...(filterDecision ? { decision: filterDecision } : {}),
      ...(filterSession  ? { session_id: filterSession }: {}),
    })
    api.get(`/admin/audit-log?${params}`)
      .then(r => { setEntries(r.data.entries); setTotal(r.data.total) })
      .catch(e => setError(e.response?.data?.detail || 'Failed to load audit log'))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchLog(page) }, [page])

  const applyFilters = () => { setPage(0); fetchLog(0) }
  const clearFilters = () => {
    setFilterAgent(''); setFilterDecision(''); setFilterSession('')
    setPage(0)
    setLoading(true)
    api.get(`/admin/audit-log?limit=${PAGE_SIZE}&offset=0`)
      .then(r => { setEntries(r.data.entries); setTotal(r.data.total) })
      .catch(e => setError(e.response?.data?.detail || 'Failed to load audit log'))
      .finally(() => setLoading(false))
  }

  const exportCSV = () => {
    const header = ['timestamp', 'agent', 'session_id', 'decision', 'reason']
    const rows = entries.map(e => [
      e.timestamp, e.agent, e.session_id, e.decision, `"${(e.reason||'').replace(/"/g,'""')}"`
    ])
    const csv = [header, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `audit_log_${Date.now()}.csv`; a.click()
    URL.revokeObjectURL(url)
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div style={s.wrap}>
      <div style={s.header}>
        <div>
          <h2 style={s.title}>📋 Audit Log</h2>
          <div style={s.totalText}>{total.toLocaleString()} total entries</div>
        </div>
        <div style={s.actions}>
          <button style={s.exportBtn} onClick={exportCSV}>⬇ Export CSV</button>
        </div>
      </div>

      {/* Filters */}
      <div style={s.filters}>
        <input style={s.filterInput} placeholder="Filter by agent…"
               value={filterAgent} onChange={e => setFilterAgent(e.target.value)} />
        <input style={s.filterInput} placeholder="Filter by decision…"
               value={filterDecision} onChange={e => setFilterDecision(e.target.value)} />
        <input style={s.filterInput} placeholder="Filter by session ID…"
               value={filterSession} onChange={e => setFilterSession(e.target.value)} />
        <button style={s.applyBtn} onClick={applyFilters}>Apply</button>
        <button style={s.clearBtn} onClick={clearFilters}>Clear</button>
      </div>

      {error && <div style={s.error}>{error}</div>}
      {loading && <div style={s.loading}>Loading audit log…</div>}

      {!loading && entries.length === 0 && (
        <div style={s.loading}>No entries match your filters.</div>
      )}

      {!loading && entries.length > 0 && (
        <div style={s.tableWrap}>
          <table style={s.table}>
            <thead>
              <tr>
                {['Timestamp', 'Agent', 'Session', 'Decision', 'Reason', ''].map(h => (
                  <th key={h} style={s.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {entries.map(entry => (
                <React.Fragment key={entry.id}>
                  <tr style={s.tr}
                      onClick={() => setExpanded(expanded === entry.id ? null : entry.id)}>
                    <td style={s.td}>
                      <span style={{ fontSize: 11, color: '#64748b' }}>
                        {new Date(entry.timestamp).toLocaleString()}
                      </span>
                    </td>
                    <td style={s.td}>
                      <code style={s.code}>{entry.agent}</code>
                    </td>
                    <td style={s.td}>
                      <code style={{ ...s.code, color: '#6b7280' }}>
                        {entry.session_id?.slice(0, 8)}…
                      </code>
                    </td>
                    <td style={s.td}>
                      <DecisionBadge decision={entry.decision} />
                    </td>
                    <td style={{ ...s.td, maxWidth: 260 }}>
                      <span style={{ fontSize: 12, color: '#374151' }}>{entry.reason}</span>
                    </td>
                    <td style={s.td}>
                      {entry.details && Object.keys(entry.details).length > 0 && (
                        <button style={s.detailsBtn}>
                          {expanded === entry.id ? '▲' : '▼'}
                        </button>
                      )}
                    </td>
                  </tr>
                  {expanded === entry.id && entry.details && (
                    <tr>
                      <td colSpan={6} style={s.detailCell}>
                        <pre style={s.pre}>{JSON.stringify(entry.details, null, 2)}</pre>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={s.pagination}>
          <button style={s.pageBtn} disabled={page === 0}
                  onClick={() => setPage(p => p - 1)}>
            ← Prev
          </button>
          <span style={s.pageInfo}>Page {page + 1} of {totalPages}</span>
          <button style={s.pageBtn} disabled={page >= totalPages - 1}
                  onClick={() => setPage(p => p + 1)}>
            Next →
          </button>
        </div>
      )}
    </div>
  )
}

const s = {
  wrap: { background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb', overflow: 'hidden' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
            padding: '20px 24px', borderBottom: '1px solid #f1f5f9' },
  title: { margin: '0 0 4px', fontSize: 18, color: '#1e293b' },
  totalText: { fontSize: 12, color: '#94a3b8' },
  actions: { display: 'flex', gap: 8 },
  exportBtn: { padding: '8px 14px', background: '#f1f5f9', color: '#334155',
               border: '1px solid #e2e8f0', borderRadius: 8, cursor: 'pointer',
               fontSize: 13, fontWeight: 600 },
  filters: { display: 'flex', gap: 8, padding: '14px 24px', background: '#f8fafc',
             borderBottom: '1px solid #e5e7eb', flexWrap: 'wrap' },
  filterInput: { padding: '7px 12px', border: '1px solid #d1d5db', borderRadius: 7,
                 fontSize: 12, minWidth: 150 },
  applyBtn: { padding: '7px 14px', background: '#2563eb', color: '#fff', border: 'none',
              borderRadius: 7, cursor: 'pointer', fontSize: 12, fontWeight: 600 },
  clearBtn: { padding: '7px 12px', background: '#f1f5f9', color: '#64748b',
              border: '1px solid #e2e8f0', borderRadius: 7, cursor: 'pointer', fontSize: 12 },
  tableWrap: { overflowX: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { padding: '10px 16px', fontSize: 11, fontWeight: 700, color: '#94a3b8',
        textTransform: 'uppercase', letterSpacing: 0.6, textAlign: 'left',
        background: '#f8fafc', borderBottom: '1px solid #e5e7eb' },
  tr: { cursor: 'pointer', transition: 'background 0.1s',
        ':hover': { background: '#f8fafc' } },
  td: { padding: '10px 16px', borderBottom: '1px solid #f1f5f9', verticalAlign: 'middle' },
  code: { background: '#f1f5f9', padding: '2px 6px', borderRadius: 4,
          fontSize: 11, color: '#334155' },
  detailsBtn: { background: 'none', border: 'none', cursor: 'pointer',
                fontSize: 12, color: '#6b7280', fontWeight: 700 },
  detailCell: { background: '#f8fafc', padding: '0 16px 12px 48px' },
  pre: { margin: 0, fontSize: 11, color: '#334155', lineHeight: 1.5,
         maxHeight: 200, overflow: 'auto' },
  pagination: { display: 'flex', justifyContent: 'center', alignItems: 'center',
                gap: 16, padding: '16px 24px', borderTop: '1px solid #f1f5f9' },
  pageBtn: { padding: '7px 16px', background: '#f1f5f9', color: '#334155',
             border: '1px solid #e2e8f0', borderRadius: 7, cursor: 'pointer', fontSize: 13 },
  pageInfo: { fontSize: 13, color: '#64748b' },
  error: { color: '#ef4444', fontSize: 13, padding: '12px 24px' },
  loading: { color: '#64748b', fontSize: 13, padding: '32px 24px', textAlign: 'center' },
}
