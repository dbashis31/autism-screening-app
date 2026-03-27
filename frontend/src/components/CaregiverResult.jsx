const VOCAB = [
  "Your child's developmental screen is complete. A clinician will review and follow up with you.",
  "Additional information is needed before results can be shared. A clinician will contact you.",
  "Your child's screening session has been logged. No further action is needed at this time.",
]

const VARIANTS = {
  [VOCAB[0]]: { bg:'#f0fdf4', border:'#86efac', icon:'✅', title:'Screening Complete' },
  [VOCAB[1]]: { bg:'#fffbeb', border:'#fcd34d', icon:'ℹ️', title:'Follow-up Needed' },
  [VOCAB[2]]: { bg:'#f8fafc', border:'#cbd5e1', icon:'📋', title:'Session Logged' },
}

export default function CaregiverResult({ result, onReset }) {
  const msg = result?.caregiver_report
  const variant = VARIANTS[msg] || { bg:'#fef2f2', border:'#fca5a5', icon:'⚠️', title:'No result' }

  return (
    <div style={s.card}>
      <h2 style={s.h2}>Step 3 — Your Result</h2>

      <div style={{ ...s.resultBox, background: variant.bg, borderColor: variant.border }}>
        <div style={s.icon}>{variant.icon}</div>
        <div style={s.title}>{variant.title}</div>
        <p style={s.msg}>{msg || 'No caregiver message was generated.'}</p>
      </div>

      <div style={s.meta}>
        <span>Session: <code>{result?.session_id?.slice(0,8)}…</code></span>
        <span>Status: <b>{result?.pipeline_status}</b></span>
      </div>

      <button style={s.btn} onClick={onReset}>Start New Session</button>
    </div>
  )
}

const s = {
  card: { background:'#fff', borderRadius:12, padding:32, boxShadow:'0 1px 4px #0001', maxWidth:540 },
  h2: { margin:'0 0 20px', color:'#1e293b' },
  resultBox: { border:'2px solid', borderRadius:10, padding:24, textAlign:'center', marginBottom:20 },
  icon: { fontSize:40, marginBottom:8 },
  title: { fontWeight:700, fontSize:18, color:'#1e293b', marginBottom:8 },
  msg: { fontSize:15, color:'#374151', lineHeight:1.6, margin:0 },
  meta: { display:'flex', gap:24, fontSize:12, color:'#94a3b8', marginBottom:20 },
  btn: { padding:'10px 24px', background:'#e2e8f0', color:'#334155', border:'none',
         borderRadius:8, fontWeight:600, cursor:'pointer', fontSize:14 },
}
