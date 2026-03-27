import { useState } from 'react'
import { useRole } from '../context/RoleContext'
import ConsentForm from '../components/ConsentForm'
import ModalitySubmit from '../components/ModalitySubmit'
import CaregiverResult from '../components/CaregiverResult'

const STEPS = ['Consent & Registration', 'Submit Screening Data', 'Your Result']

export default function CaregiverApp() {
  const { setSessionId: setGlobalSessionId, setChildId: setGlobalChildId } = useRole()
  const [step, setStep] = useState(1)
  const [sessionId, setSessionId] = useState(null)
  const [childId, setChildId] = useState(null)
  const [result, setResult] = useState(null)

  const handleConsentDone = ({ sessionId: sid, childId: cid }) => {
    setSessionId(sid)
    setChildId(cid)
    setGlobalSessionId(sid)
    setGlobalChildId(cid)
    setStep(2)
  }

  const handleSubmitDone = (res) => {
    setResult(res)
    setStep(3)
  }

  const handleReset = () => {
    setStep(1)
    setSessionId(null)
    setChildId(null)
    setResult(null)
    setGlobalSessionId(null)
    setGlobalChildId('')
  }

  return (
    <div style={s.page}>
      <div style={s.header}>
        <h1 style={s.title}>Developmental Screening Portal</h1>
        <p style={s.subtitle}>Private · Secure · Governed by clinical ethics protocol</p>
      </div>

      {/* Step indicator */}
      <div style={s.stepper}>
        {STEPS.map((label, i) => {
          const num = i + 1
          const active = step === num
          const done = step > num
          return (
            <div key={num} style={s.stepItem}>
              <div style={{
                ...s.stepCircle,
                background: done ? '#22c55e' : active ? '#2563eb' : '#e2e8f0',
                color: done || active ? '#fff' : '#94a3b8',
              }}>
                {done ? '✓' : num}
              </div>
              <span style={{
                ...s.stepLabel,
                color: active ? '#1e293b' : done ? '#22c55e' : '#94a3b8',
                fontWeight: active ? 700 : 400,
              }}>
                {label}
              </span>
              {i < STEPS.length - 1 && (
                <div style={{
                  ...s.connector,
                  background: done ? '#22c55e' : '#e2e8f0',
                }} />
              )}
            </div>
          )
        })}
      </div>

      {/* Active step */}
      <div style={s.content}>
        {step === 1 && <ConsentForm onDone={handleConsentDone} />}
        {step === 2 && sessionId && (
          <ModalitySubmit
            sessionId={sessionId}
            childId={childId}
            onDone={handleSubmitDone}
          />
        )}
        {step === 3 && result && (
          <CaregiverResult result={result} onReset={handleReset} />
        )}
      </div>

      <footer style={s.footer}>
        <p>This tool supports clinical decision-making. Results must be reviewed by a qualified clinician.</p>
        <p style={{ marginTop: 4 }}>Data is processed under your consent agreement and protected by our governance protocol.</p>
      </footer>
    </div>
  )
}

const s = {
  page: { minHeight: '100vh', background: '#f8fafc', paddingBottom: 40 },
  header: { background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)', color: '#fff',
            padding: '32px 24px', textAlign: 'center' },
  title: { margin: '0 0 8px', fontSize: 26, fontWeight: 700 },
  subtitle: { margin: 0, fontSize: 14, color: '#94a3b8' },
  stepper: { display: 'flex', alignItems: 'center', justifyContent: 'center',
             padding: '28px 24px', background: '#fff', borderBottom: '1px solid #e5e7eb',
             flexWrap: 'wrap', gap: 0 },
  stepItem: { display: 'flex', alignItems: 'center', gap: 8 },
  stepCircle: { width: 32, height: 32, borderRadius: '50%', display: 'flex',
                alignItems: 'center', justifyContent: 'center',
                fontWeight: 700, fontSize: 14, flexShrink: 0 },
  stepLabel: { fontSize: 13, whiteSpace: 'nowrap' },
  connector: { width: 48, height: 2, margin: '0 8px', flexShrink: 0 },
  content: { maxWidth: 600, margin: '32px auto', padding: '0 16px' },
  footer: { maxWidth: 600, margin: '24px auto', padding: '16px', textAlign: 'center',
            fontSize: 12, color: '#94a3b8', borderTop: '1px solid #e5e7eb' },
}
