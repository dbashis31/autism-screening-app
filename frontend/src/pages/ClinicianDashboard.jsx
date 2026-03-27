import { useState, useEffect } from 'react'
import api from '../api'
import EscalationQueue from '../components/EscalationQueue'
import AbstentionHistory from '../components/AbstentionHistory'
import ClinicianReportCard from '../components/ClinicianReportCard'

export default function ClinicianDashboard() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedChild, setSelectedChild] = useState(null)
  const [search, setSearch] = useState('')
  const [filterStatus, setFilterStatus] = useState('all')

  const fetchSessions = () => {
    setLoading(true)
    api.get('/clinician/sessions')
      .then(r => setSessions(r.data))
      .catch(e => setError(e.response?.data?.detail || 'Failed to load sessions'))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchSessions() }, [])

  const filtered = sessions.filter(s => {
    const matchSearch = !search || s.child_id?.includes(search) || s.id?.includes(search)
    const matchStatus = filterStatus === 'all' || s.pipeline_status === filterStatus
    return matchSearch && matchStatus
  })

  return (
    <div style={s.page}>
      <div style={s.pageHeader}>
        <h1 style={s.pageTitle}>🩺 Clinician Dashboard</h1>
        <p style={s.pageSub}>Review screening results, abstentions, and escalated cases</p>
        <button style={s.refreshBtn} onClick={fetchSessions}>↻ Refresh</button>
      </div>

      <div style={s.layout}>
        {/* Left panel — escalation + abstention history */}
        <div style={s.leftPanel}>
          <EscalationQueue onSelectChild={setSelectedChild} />
          <AbstentionHistory childId={selectedChild} />
        </div>

        {/* Right panel — session reports */}
        <div style={s.rightPanel}>
          <div style={s.reportHeader}>
            <h2 style={s.reportTitle}>All Sessions</h2>
            <div style={s.controls}>
              <input
                style={s.search}
                placeholder="Search child ID or session…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
              <select style={s.filter} value={filterStatus}
                      onChange={e => setFilterStatus(e.target.value)}>
                <option value="all">All statuses</option>
                <option value="complete">Complete</option>
                <option value="blocked">Blocked</option>
                <option value="abstained">Abstained</option>
                <option value="pending">Pending</option>
              </select>
            </div>
          </div>

          {loading && <div style={s.msg}>Loading sessions…</div>}
          {error   && <div style={{ ...s.msg, color: '#ef4444' }}>{error}</div>}
          {!loading && !error && filtered.length === 0 && (
            <div style={s.msg}>
              {sessions.length === 0
                ? 'No sessions yet. Run a caregiver screening to see results here.'
                : 'No sessions match your filters.'}
            </div>
          )}

          {filtered.map(session => (
            <ClinicianReportCard key={session.id} session={session} />
          ))}
        </div>
      </div>
    </div>
  )
}

const s = {
  page: { minHeight: '100vh', background: '#f8fafc' },
  pageHeader: { background: 'linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%)',
                color: '#fff', padding: '28px 32px', display: 'flex',
                alignItems: 'center', gap: 16, flexWrap: 'wrap' },
  pageTitle: { margin: 0, fontSize: 24, fontWeight: 700 },
  pageSub: { margin: 0, fontSize: 14, color: '#94a3b8', flex: 1 },
  refreshBtn: { padding: '8px 16px', background: '#334155', color: '#f1f5f9',
                border: '1px solid #475569', borderRadius: 8, cursor: 'pointer',
                fontSize: 13, fontWeight: 600 },
  layout: { display: 'grid', gridTemplateColumns: '340px 1fr', gap: 24,
            padding: '24px 32px', maxWidth: 1400, margin: '0 auto' },
  leftPanel: { minWidth: 0 },
  rightPanel: { minWidth: 0 },
  reportHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  marginBottom: 16, flexWrap: 'wrap', gap: 12 },
  reportTitle: { margin: 0, fontSize: 18, color: '#1e293b' },
  controls: { display: 'flex', gap: 8 },
  search: { padding: '7px 12px', border: '1px solid #d1d5db', borderRadius: 7,
            fontSize: 13, width: 200 },
  filter: { padding: '7px 10px', border: '1px solid #d1d5db', borderRadius: 7,
            fontSize: 13, background: '#fff' },
  msg: { color: '#64748b', fontSize: 14, padding: '32px 0', textAlign: 'center' },
}
