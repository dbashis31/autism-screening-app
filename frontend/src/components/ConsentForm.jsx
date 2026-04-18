import { useState } from 'react'
import api from '../api'

const TODAY = new Date().toISOString().split('T')[0]
const DEFAULT_EXPIRY = `${new Date().getFullYear() + 2}-12-31`

export default function ConsentForm({ onDone }) {
  const [childId, setChildId] = useState('')
  const [expiry, setExpiry] = useState(DEFAULT_EXPIRY)
  const [ops, setOps] = useState(['inference'])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const toggleOp = (op) =>
    setOps(prev => prev.includes(op) ? prev.filter(o => o !== op) : [...prev, op])

  const loadDemo = async (sid) => {
    const res = await api.get('/dev/scenarios')
    return res.data[sid]
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!childId.trim()) { setError('Child ID is required'); return }
    setLoading(true); setError(null)
    try {
      const role = localStorage.getItem('role') || 'caregiver'
      const sessRes = await api.post('/sessions', { child_id: childId, role })
      const sessionId = sessRes.data.session_id
      await api.post(`/sessions/${sessionId}/consent`, {
        permitted_ops: ops, expiry_date: expiry
      })
      onDone({ sessionId, childId })
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create session')
    } finally { setLoading(false) }
  }

  return (
    <div style={s.card}>
      <h2 style={s.h2}>Step 1 — Consent & Registration</h2>
      <p style={s.sub}>Enter your child's ID and confirm consent before the screening begins.</p>
      <form onSubmit={handleSubmit}>
        <label style={s.label}>Child ID
          <input style={s.input} value={childId} onChange={e => setChildId(e.target.value)}
                 placeholder="e.g. SYN-001 or your assigned ID" required />
        </label>

        <label style={s.label}>Consent expiry date
          <input style={s.input} type="date" value={expiry} min={TODAY}
                 onChange={e => setExpiry(e.target.value)} required />
        </label>

        <fieldset style={s.fieldset}>
          <legend style={s.legend}>Permitted operations</legend>
          {['inference', 'longitudinal_tracking'].map(op => (
            <label key={op} style={s.check}>
              <input type="checkbox" checked={ops.includes(op)}
                     onChange={() => toggleOp(op)} /> {op}
            </label>
          ))}
        </fieldset>

        {error && <p style={s.err}>{error}</p>}
        <button style={s.btn} type="submit" disabled={loading}>
          {loading ? 'Creating session…' : 'Confirm Consent & Continue →'}
        </button>
      </form>
    </div>
  )
}

const s = {
  card: { background:'#fff', borderRadius:12, padding:32, boxShadow:'0 1px 4px #0001', maxWidth:540 },
  h2: { margin:'0 0 8px', color:'#1e293b' },
  sub: { color:'#64748b', marginBottom:24, fontSize:14 },
  label: { display:'flex', flexDirection:'column', gap:6, marginBottom:16, fontWeight:600, fontSize:14, color:'#374151' },
  input: { padding:'8px 12px', borderRadius:6, border:'1px solid #d1d5db', fontSize:14 },
  fieldset: { border:'1px solid #e5e7eb', borderRadius:8, padding:'12px 16px', marginBottom:16 },
  legend: { fontWeight:600, fontSize:13, color:'#374151', padding:'0 4px' },
  check: { display:'flex', alignItems:'center', gap:8, marginBottom:8, fontWeight:400, fontSize:14 },
  err: { color:'#ef4444', fontSize:13, marginBottom:12 },
  btn: { width:'100%', padding:'10px 0', background:'#2563eb', color:'#fff', border:'none',
         borderRadius:8, fontWeight:600, fontSize:15, cursor:'pointer' },
}
