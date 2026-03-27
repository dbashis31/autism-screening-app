import MetricsDashboard from '../components/MetricsDashboard'
import AuditTable from '../components/AuditTable'

export default function AdminAuditLog() {
  return (
    <div style={s.page}>
      <div style={s.pageHeader}>
        <h1 style={s.pageTitle}>🔧 Admin — Governance Oversight</h1>
        <p style={s.pageSub}>
          Real-time governance metrics and full audit trail for every pipeline decision
        </p>
        <div style={s.badge}>MLHC 2026 · Research Mode</div>
      </div>

      <div style={s.content}>
        {/* 7-metric dashboard */}
        <MetricsDashboard />

        {/* Audit log table */}
        <AuditTable />
      </div>

      <footer style={s.footer}>
        <p>All pipeline decisions are immutably logged. Access is restricted to admin role.</p>
        <p style={{ marginTop: 4 }}>
          Metrics are computed live from the database on every page load or manual refresh.
        </p>
      </footer>
    </div>
  )
}

const s = {
  page: { minHeight: '100vh', background: '#f8fafc' },
  pageHeader: { background: 'linear-gradient(135deg, #0f172a 0%, #312e81 100%)',
                color: '#fff', padding: '28px 32px', display: 'flex',
                alignItems: 'center', gap: 16, flexWrap: 'wrap' },
  pageTitle: { margin: 0, fontSize: 24, fontWeight: 700 },
  pageSub: { margin: 0, fontSize: 14, color: '#a5b4fc', flex: 1 },
  badge: { background: '#4f46e5', color: '#c7d2fe', padding: '4px 14px',
           borderRadius: 20, fontSize: 12, fontWeight: 700,
           border: '1px solid #6366f1' },
  content: { maxWidth: 1400, margin: '24px auto', padding: '0 32px' },
  footer: { maxWidth: 1400, margin: '0 auto', padding: '16px 32px 32px',
            fontSize: 12, color: '#94a3b8', borderTop: '1px solid #e5e7eb' },
}
