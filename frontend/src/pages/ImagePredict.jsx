import { useState, useRef } from 'react'

export default function ImagePredict() {
  const [file,     setFile]     = useState(null)
  const [preview,  setPreview]  = useState(null)
  const [result,   setResult]   = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef()

  const loadFile = (f) => {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const onInputChange = (e) => loadFile(e.target.files[0])
  const onDrop        = (e) => { e.preventDefault(); setDragOver(false); loadFile(e.dataTransfer.files[0]) }
  const onDragOver    = (e) => { e.preventDefault(); setDragOver(true) }
  const onDragLeave   = ()  => setDragOver(false)

  const predict = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch('/predict/image', { method: 'POST', body: fd })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Prediction failed')
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null) }

  const isAutistic = result?.prediction === 'autistic'

  return (
    <div style={s.page}>
      <div style={s.inner}>

        {/* Header */}
        <div style={s.headerWrap}>
          <div style={s.badge}>🧠 CNN-BiLSTM Multimodal · AUC 1.00 · Visual Pathway</div>
          <h1 style={s.h1}>Autism Detection — Image Analysis</h1>
          <p style={s.sub}>
            Upload a child's facial image · Model returns prediction + Grad-CAM attention map
          </p>
        </div>

        <div style={s.grid}>
          {/* Upload panel */}
          <div style={s.card}>
            <h2 style={s.h2}>📤 Upload Image</h2>

            <div
              style={{
                ...s.dropZone,
                ...(dragOver ? s.dropZoneActive : {}),
                height: preview ? 224 : 176,
              }}
              onClick={() => inputRef.current.click()}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
            >
              {preview ? (
                <img src={preview} alt="preview" style={s.previewImg} />
              ) : (
                <>
                  <span style={{ fontSize: 36, marginBottom: 8 }}>🖼️</span>
                  <p style={s.dropText}>Click or drag & drop image here</p>
                  <p style={s.dropSub}>JPG, PNG, WEBP</p>
                </>
              )}
              <input ref={inputRef} type="file" accept="image/*"
                     style={{ display: 'none' }} onChange={onInputChange} />
            </div>

            {file && (
              <p style={s.fileName}>
                📎 {file.name} ({(file.size / 1024).toFixed(1)} KB)
              </p>
            )}

            <div style={s.btnRow}>
              <button
                onClick={predict}
                disabled={!file || loading}
                style={{ ...s.predictBtn, ...((!file || loading) ? s.predictBtnDisabled : {}) }}
              >
                {loading ? '⟳ Analysing…' : '🔍 Run Prediction'}
              </button>
              {(file || result) && (
                <button onClick={reset} style={s.resetBtn}>Reset</button>
              )}
            </div>

            {error && (
              <div style={s.errorBox}>⚠️ {error}</div>
            )}
          </div>

          {/* Result panel */}
          <div style={{
            ...s.card,
            background: result
              ? isAutistic ? '#fff5f5' : '#f0fdf4'
              : '#fff',
            borderColor: result
              ? isAutistic ? '#fca5a5' : '#86efac'
              : '#e5e7eb',
          }}>
            <h2 style={s.h2}>📊 Prediction Result</h2>

            {!result && !loading && (
              <div style={s.emptyResult}>
                <span style={{ fontSize: 48, marginBottom: 12 }}>🧪</span>
                <p style={{ color: '#94a3b8', fontSize: 14 }}>
                  Upload an image and click "Run Prediction"
                </p>
              </div>
            )}

            {loading && (
              <div style={s.emptyResult}>
                <div style={s.spinner} />
                <p style={{ color: '#64748b', fontSize: 14, marginTop: 12 }}>
                  Running CNN-BiLSTM + Grad-CAM…
                </p>
              </div>
            )}

            {result && (
              <>
                <div style={s.verdict}>
                  <span style={{ fontSize: 40 }}>{isAutistic ? '🔴' : '🟢'}</span>
                  <div>
                    <div style={{
                      ...s.labelBadge,
                      background: isAutistic ? '#ef4444' : '#22c55e',
                    }}>
                      {result.label}
                    </div>
                    <p style={s.confText}>
                      {(result.confidence * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                </div>

                {[
                  { label: 'Autistic',     prob: result.autistic_prob,     color: '#f87171' },
                  { label: 'Non-Autistic', prob: result.non_autistic_prob, color: '#4ade80' },
                ].map(({ label, prob, color }) => (
                  <div key={label} style={{ marginBottom: 10 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between',
                                  fontSize: 12, color: '#475569', marginBottom: 4 }}>
                      <span>{label}</span>
                      <span style={{ fontWeight: 700 }}>{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: 10, background: '#e2e8f0', borderRadius: 5, overflow: 'hidden' }}>
                      <div style={{ width: `${prob * 100}%`, height: '100%',
                                    background: color, borderRadius: 5,
                                    transition: 'width 0.6s ease' }} />
                    </div>
                  </div>
                ))}

                <p style={s.disclaimer}>
                  ⚠️ <strong>Research use only.</strong> This tool is not a clinical diagnostic
                  device. Always consult a qualified clinician for ASD assessment.
                </p>
              </>
            )}
          </div>
        </div>

        {/* Grad-CAM panel */}
        {result && (
          <div style={{ ...s.card, marginTop: 24 }}>
            <h2 style={s.h2}>🔥 Grad-CAM Attention Map</h2>
            <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 16 }}>
              Warmer colours (red/yellow) show regions the model focused on most when making
              the prediction.
            </p>
            <div style={s.camGrid}>
              <div style={{ textAlign: 'center' }}>
                <p style={s.camLabel}>Original Image</p>
                <img src={preview} alt="original" style={s.camImg} />
              </div>
              <div style={{ textAlign: 'center' }}>
                <p style={s.camLabel}>Grad-CAM Overlay</p>
                <img src={`data:image/png;base64,${result.gradcam_image}`}
                     alt="grad-cam" style={s.camImg} />
              </div>
            </div>

            <div style={s.metaGrid}>
              {[
                ['Model',    'CNN-BiLSTM (ResNet-18 Visual)'],
                ['Device',   result.device],
                ['Val AUC',  '1.00'],
                ['Heads',    '5 (A/V/Q/T/G)'],
                ['Training', 'Synthetic'],
                ['Classes',  'Autistic / Non-Autistic'],
              ].map(([k, v]) => (
                <div key={k} style={s.metaTile}>
                  <p style={s.metaKey}>{k}</p>
                  <p style={s.metaVal}>{v}</p>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

const s = {
  page: { minHeight: '100vh', background: 'linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%)',
          padding: 24 },
  inner: { maxWidth: 960, margin: '0 auto' },
  headerWrap: { textAlign: 'center', marginBottom: 32 },
  badge: { display: 'inline-block', background: '#2563eb', color: '#fff',
           padding: '6px 16px', borderRadius: 20, fontSize: 13, fontWeight: 600,
           marginBottom: 12 },
  h1: { margin: '0 0 8px', fontSize: 28, fontWeight: 800, color: '#1e293b' },
  sub: { color: '#64748b', fontSize: 14, margin: 0 },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 },
  card: { background: '#fff', borderRadius: 16, border: '1px solid #e5e7eb',
          padding: 24, boxShadow: '0 1px 4px #0001' },
  h2: { margin: '0 0 16px', fontSize: 16, fontWeight: 700, color: '#374151' },
  dropZone: { border: '2px dashed #cbd5e1', borderRadius: 12, cursor: 'pointer',
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', overflow: 'hidden', marginBottom: 12,
              transition: 'border-color 0.2s, background 0.2s' },
  dropZoneActive: { borderColor: '#3b82f6', background: '#eff6ff' },
  previewImg: { width: '100%', height: '100%', objectFit: 'contain' },
  dropText: { color: '#64748b', fontSize: 14, fontWeight: 600, margin: '0 0 4px' },
  dropSub: { color: '#94a3b8', fontSize: 12, margin: 0 },
  fileName: { fontSize: 12, color: '#64748b', marginBottom: 12,
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  btnRow: { display: 'flex', gap: 12 },
  predictBtn: { flex: 1, padding: '10px 0', background: '#2563eb', color: '#fff',
                border: 'none', borderRadius: 12, fontWeight: 700, fontSize: 14,
                cursor: 'pointer' },
  predictBtnDisabled: { background: '#cbd5e1', cursor: 'not-allowed' },
  resetBtn: { padding: '10px 16px', background: '#fff', color: '#64748b',
              border: '1px solid #e5e7eb', borderRadius: 12, cursor: 'pointer',
              fontSize: 14 },
  errorBox: { marginTop: 12, background: '#fef2f2', border: '1px solid #fca5a5',
              borderRadius: 10, padding: '10px 14px', color: '#dc2626', fontSize: 13 },
  emptyResult: { display: 'flex', flexDirection: 'column', alignItems: 'center',
                 justifyContent: 'center', minHeight: 200 },
  spinner: { width: 48, height: 48, border: '4px solid #dbeafe',
             borderTopColor: '#2563eb', borderRadius: '50%',
             animation: 'spin 0.8s linear infinite' },
  verdict: { display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 },
  labelBadge: { display: 'inline-block', color: '#fff', fontSize: 11, fontWeight: 800,
                padding: '3px 12px', borderRadius: 20, textTransform: 'uppercase',
                letterSpacing: 1, marginBottom: 4 },
  confText: { fontSize: 22, fontWeight: 800, color: '#1e293b', margin: 0 },
  disclaimer: { marginTop: 12, background: 'rgba(255,255,255,0.6)',
                border: '1px solid rgba(255,255,255,0.8)', borderRadius: 10,
                padding: '10px 14px', fontSize: 12, color: '#475569', lineHeight: 1.5 },
  camGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 },
  camLabel: { fontSize: 12, fontWeight: 600, color: '#64748b', marginBottom: 8 },
  camImg: { maxHeight: 256, maxWidth: '100%', objectFit: 'contain',
            borderRadius: 12, border: '1px solid #e5e7eb' },
  metaGrid: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 },
  metaTile: { background: '#f8fafc', borderRadius: 10, padding: '10px 14px' },
  metaKey: { fontSize: 11, color: '#94a3b8', fontWeight: 600, margin: '0 0 2px' },
  metaVal: { fontSize: 13, color: '#334155', fontWeight: 700, margin: 0,
             overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
}
