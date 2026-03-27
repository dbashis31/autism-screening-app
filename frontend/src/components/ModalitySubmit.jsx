import { useState } from 'react'
import api from '../api'

export default function ModalitySubmit({ sessionId, childId, onDone }) {
  const [snr, setSnr] = useState(20)
  const [useAudio, setUseAudio] = useState(true)
  const [useVideo, setUseVideo] = useState(true)
  const [useText, setUseText] = useState(true)
  const [useQuestionnaire, setUseQuestionnaire] = useState(true)
  const [ageMonths, setAgeMonths] = useState('')
  const [conflict, setConflict] = useState(false)
  const [forceAbstain, setForceAbstain] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [scenario, setScenario] = useState('custom')
  const [scenarios, setScenarios] = useState(null)

  const loadScenarios = async () => {
    if (scenarios) return
    const res = await api.get('/dev/scenarios')
    setScenarios(res.data)
  }

  const applyScenario = (id) => {
    if (!scenarios || !scenarios[id]) return
    const sc = scenarios[id]
    setScenario(id)
    setSnr(sc.audio_snr_db ?? 20)
    setUseAudio(true); setUseVideo(true); setUseText(true); setUseQuestionnaire(true)
    setAgeMonths(sc.child_age_months ?? '')
    setConflict(sc.cross_modal_conflict ?? false)
    setForceAbstain(sc.force_abstain ?? false)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true); setError(null)
    const modalities = [
      ...(useAudio ? ['audio'] : []),
      ...(useVideo ? ['video'] : []),
      ...(useText ? ['text'] : []),
      ...(useQuestionnaire ? ['questionnaire'] : []),
    ]
    try {
      const payload = {
        modalities,
        audio_snr_db: useAudio ? snr : null,
        child_age_months: ageMonths ? parseInt(ageMonths) : null,
        cross_modal_conflict: conflict,
        force_abstain: forceAbstain,
      }
      const res = await api.post(`/sessions/${sessionId}/submit`, payload)
      onDone(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Submission failed')
    } finally { setLoading(false) }
  }

  const snrColor = snr < 15 ? '#ef4444' : snr < 20 ? '#f59e0b' : '#22c55e'

  return (
    <div style={s.card}>
      <h2 style={s.h2}>Step 2 — Submit Screening Data</h2>
      <p style={s.sub}>Session: <code>{sessionId.slice(0,8)}…</code> &nbsp;|&nbsp; Child: <b>{childId}</b></p>

      <div style={s.demoBox}>
        <span style={s.demoLabel}>🧪 Load research scenario:</span>
        <select style={s.sel} value={scenario}
                onChange={e => { applyScenario(e.target.value) }}
                onFocus={loadScenarios}>
          <option value="custom">Custom input</option>
          {scenarios && Object.entries(scenarios).map(([id, sc]) => (
            <option key={id} value={id}>{sc.label}</option>
          ))}
        </select>
      </div>

      <form onSubmit={handleSubmit}>
        <fieldset style={s.fieldset}>
          <legend style={s.legend}>Modalities</legend>
          {[['audio','🎙 Audio',useAudio,setUseAudio],['video','📹 Video',useVideo,setUseVideo],
            ['text','📝 Text notes',useText,setUseText],['questionnaire','📋 Questionnaire',useQuestionnaire,setUseQuestionnaire]]
            .map(([key,label,val,set]) => (
            <label key={key} style={s.check}>
              <input type="checkbox" checked={val} onChange={e => set(e.target.checked)} /> {label}
            </label>
          ))}
        </fieldset>

        {useAudio && (
          <label style={s.label}>
            Audio quality (SNR) — <span style={{color:snrColor, fontWeight:700}}>{snr} dB
              {snr < 15 ? ' ⚠ below threshold, audio will be excluded' : ''}</span>
            <input type="range" min={0} max={30} value={snr}
                   onChange={e => setSnr(Number(e.target.value))} style={{marginTop:6}} />
          </label>
        )}

        <label style={s.label}>Child age (months) — optional
          <input style={s.input} type="number" min={1} max={120} value={ageMonths}
                 onChange={e => setAgeMonths(e.target.value)} placeholder="Leave blank if unknown" />
        </label>

        <details style={s.details}>
          <summary style={s.summary}>⚙ Advanced / test flags</summary>
          <label style={{...s.check, marginTop:10}}>
            <input type="checkbox" checked={conflict} onChange={e => setConflict(e.target.checked)} />
            Simulate cross-modal conflict (triggers abstention)
          </label>
          <label style={s.check}>
            <input type="checkbox" checked={forceAbstain} onChange={e => setForceAbstain(e.target.checked)} />
            Force abstention (insufficient data)
          </label>
        </details>

        {error && <p style={s.err}>{error}</p>}
        <button style={s.btn} type="submit" disabled={loading}>
          {loading ? 'Running governance pipeline…' : 'Submit for Screening →'}
        </button>
      </form>
    </div>
  )
}

const s = {
  card: { background:'#fff', borderRadius:12, padding:32, boxShadow:'0 1px 4px #0001', maxWidth:540 },
  h2: { margin:'0 0 8px', color:'#1e293b' },
  sub: { color:'#64748b', marginBottom:16, fontSize:13 },
  demoBox: { display:'flex', alignItems:'center', gap:12, background:'#f1f5f9', padding:'10px 14px',
             borderRadius:8, marginBottom:20 },
  demoLabel: { fontSize:13, fontWeight:600, color:'#475569', whiteSpace:'nowrap' },
  sel: { flex:1, padding:'6px 10px', borderRadius:6, border:'1px solid #d1d5db', fontSize:13 },
  label: { display:'flex', flexDirection:'column', gap:4, marginBottom:16, fontWeight:600, fontSize:14, color:'#374151' },
  input: { padding:'8px 12px', borderRadius:6, border:'1px solid #d1d5db', fontSize:14 },
  fieldset: { border:'1px solid #e5e7eb', borderRadius:8, padding:'12px 16px', marginBottom:16 },
  legend: { fontWeight:600, fontSize:13, color:'#374151', padding:'0 4px' },
  check: { display:'flex', alignItems:'center', gap:8, marginBottom:8, fontWeight:400, fontSize:14 },
  details: { marginBottom:16 },
  summary: { fontSize:13, color:'#6b7280', cursor:'pointer', fontWeight:600 },
  err: { color:'#ef4444', fontSize:13, marginBottom:12 },
  btn: { width:'100%', padding:'10px 0', background:'#2563eb', color:'#fff', border:'none',
         borderRadius:8, fontWeight:600, fontSize:15, cursor:'pointer' },
}
