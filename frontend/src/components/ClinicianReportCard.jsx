import { useState } from 'react'

const MODALITY_COLORS = {
  audio: '#3b82f6',
  video: '#8b5cf6',
  text: '#10b981',
  questionnaire: '#f59e0b',
}

function ConfidenceBar({ label, value }) {
  const pct = Math.round(value * 100)
  const color = value >= 0.85 ? '#22c55e' : value >= 0.70 ? '#f59e0b' : '#ef4444'
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12,
                    color: '#475569', marginBottom: 3 }}>
        <span>{label}</span>
        <span style={{ fontWeight: 700, color }}>{pct}%</span>
      </div>
      <div style={{ height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color,
                      borderRadius: 4, transition: 'width 0.4s ease' }} />
      </div>
    </div>
  )
}

export default function ClinicianReportCard({ session }) {
  const [showRaw, setShowRaw] = useState(false)
  const r = session.pipeline_result || {}
  const conf = session.confidence_scores || {}
  const modalities = session.modalities || []
  const warnings = r.applicability_warnings || []

  const statusColor = {
    complete: '#22c55e',
    blocked: '#ef4444',
    abstained: '#f59e0b',
    pending: '#94a3b8',
  }[session.pipeline_status] || '#94a3b8'

  return (
    <div style={s.card}>
      {/* Header */}
      <div style={s.cardHeader}>
        <div>
          <div style={s.childId}>Child: <strong>{session.child_id}</strong></div>
          <div style={s.sessionId}>Session: <code>{session.id?.slice(0, 8)}…</code></div>
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
            {new Date(session.created_at).toLocaleString()}
          </div>
        </div>
        <div style={{ ...s.statusBadge, background: statusColor + '22', color: statusColor,
                      border: `1px solid ${statusColor}` }}>
          {session.pipeline_status?.toUpperCase()}
        </div>
      </div>

      {/* Modality badges */}
      <div style={s.section}>
        <div style={s.sectionLabel}>Modalities Used</div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {modalities.map(m => (
            <span key={m} style={{ ...s.badge,
              background: (MODALITY_COLORS[m] || '#64748b') + '22',
              color: MODALITY_COLORS[m] || '#64748b',
              border: `1px solid ${MODALITY_COLORS[m] || '#64748b'}` }}>
              {m}
            </span>
          ))}
          {modalities.length === 0 && <span style={{ color: '#94a3b8', fontSize: 12 }}>None recorded</span>}
        </div>
      </div>

      {/* Confidence scores */}
      {Object.keys(conf).length > 0 && (
        <div style={s.section}>
          <div style={s.sectionLabel}>Confidence Scores</div>
          {Object.entries(conf).map(([k, v]) => (
            <ConfidenceBar key={k} label={k} value={v} />
          ))}
          {r.uncertainty_bounds && (
            <div style={s.bounds}>
              Uncertainty bounds: [{(r.uncertainty_bounds[0] * 100).toFixed(1)}%,{' '}
              {(r.uncertainty_bounds[1] * 100).toFixed(1)}%]
            </div>
          )}
        </div>
      )}

      {/* Applicability warnings */}
      {warnings.length > 0 && (
        <div style={s.section}>
          <div style={s.sectionLabel}>Applicability Warnings</div>
          {warnings.map((w, i) => (
            <div key={i} style={s.warning}>⚠ {w}</div>
          ))}
        </div>
      )}

      {/* Abstention reason */}
      {r.abstention_reason && (
        <div style={{ ...s.section, background: '#fffbeb', borderRadius: 8, padding: 12,
                      border: '1px solid #fcd34d' }}>
          <div style={{ ...s.sectionLabel, color: '#b45309' }}>Abstention Reason</div>
          <p style={{ margin: 0, fontSize: 13, color: '#92400e' }}>{r.abstention_reason}</p>
        </div>
      )}

      {/* Clinician report text */}
      {r.clinician_report && (
        <div style={s.section}>
          <div style={s.sectionLabel}>Clinician Report</div>
          <p style={{ margin: 0, fontSize: 13, color: '#374151', lineHeight: 1.6 }}>
            {r.clinician_report}
          </p>
        </div>
      )}

      {/* Model info */}
      {r.model_used && (
        <div style={{ ...s.section, display: 'flex', gap: 20, fontSize: 12, color: '#64748b' }}>
          <span>Model: <code>{r.model_used}</code></span>
          {r.report_type && <span>Report type: <code>{r.report_type}</code></span>}
        </div>
      )}

      {/* Raw JSON toggle */}
      <div style={{ marginTop: 8 }}>
        <button style={s.rawBtn} onClick={() => setShowRaw(v => !v)}>
          {showRaw ? '▲ Hide raw JSON' : '▼ Show raw pipeline output'}
        </button>
        {showRaw && (
          <pre style={s.pre}>{JSON.stringify(r, null, 2)}</pre>
        )}
      </div>
    </div>
  )
}

const s = {
  card: { background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20,
          boxShadow: '0 1px 3px #0001', marginBottom: 16 },
  cardHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
                marginBottom: 16 },
  childId: { fontSize: 15, color: '#1e293b' },
  sessionId: { fontSize: 12, color: '#64748b', marginTop: 2 },
  statusBadge: { padding: '4px 10px', borderRadius: 20, fontSize: 11, fontWeight: 700,
                 whiteSpace: 'nowrap' },
  section: { marginBottom: 16 },
  sectionLabel: { fontSize: 11, fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase',
                  letterSpacing: 0.8, marginBottom: 8 },
  badge: { padding: '3px 10px', borderRadius: 20, fontSize: 12, fontWeight: 600 },
  bounds: { fontSize: 12, color: '#64748b', marginTop: 6, fontStyle: 'italic' },
  warning: { background: '#fffbeb', border: '1px solid #fcd34d', borderRadius: 6,
             padding: '6px 10px', fontSize: 12, color: '#92400e', marginBottom: 6 },
  rawBtn: { fontSize: 12, color: '#6b7280', background: 'none', border: 'none',
            cursor: 'pointer', padding: 0, fontWeight: 600 },
  pre: { background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 8, padding: 12,
         fontSize: 11, overflow: 'auto', maxHeight: 300, marginTop: 8, color: '#334155' },
}
