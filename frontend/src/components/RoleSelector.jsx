import { useNavigate } from 'react-router-dom'
import { useRole } from '../context/RoleContext'

const ROLES = [
  { value: 'caregiver', label: '👨‍👩‍👧 Caregiver', path: '/caregiver' },
  { value: 'clinician', label: '🩺 Clinician', path: '/clinician' },
  { value: 'admin',     label: '🔧 Admin',     path: '/admin' },
]

export default function RoleSelector() {
  const { role, setRole, sessionId } = useRole()
  const navigate = useNavigate()

  const handleChange = (e) => {
    const chosen = ROLES.find(r => r.value === e.target.value)
    if (chosen) { setRole(chosen.value); navigate(chosen.path) }
  }

  return (
    <nav style={styles.nav}>
      <span style={styles.title}>🧩 ASD Screening MVP</span>
      <div style={styles.right}>
        {sessionId && <span style={styles.sid}>Session: {sessionId.slice(0,8)}…</span>}
        <select value={role} onChange={handleChange} style={styles.select}>
          {ROLES.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
        </select>
      </div>
    </nav>
  )
}

const styles = {
  nav: { display:'flex', justifyContent:'space-between', alignItems:'center',
         background:'#1e293b', color:'#f1f5f9', padding:'12px 24px',
         position:'sticky', top:0, zIndex:100 },
  title: { fontWeight:700, fontSize:18 },
  right: { display:'flex', alignItems:'center', gap:16 },
  sid: { fontSize:12, color:'#94a3b8' },
  select: { padding:'6px 12px', borderRadius:6, border:'none',
            background:'#334155', color:'#f1f5f9', cursor:'pointer', fontSize:14 },
}
