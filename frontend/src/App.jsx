import { useEffect } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { useRole } from './context/RoleContext'
import RoleSelector from './components/RoleSelector'
import CaregiverApp from './pages/CaregiverApp'
import ClinicianDashboard from './pages/ClinicianDashboard'
import AdminAuditLog from './pages/AdminAuditLog'
import ImagePredict from './pages/ImagePredict'

// Auto-sync role from URL so direct navigation / refresh works correctly.
// e.g. opening http://localhost:5173/clinician sets role = 'clinician'
function RoleSync() {
  const { setRole } = useRole()
  const { pathname } = useLocation()
  useEffect(() => {
    if (pathname.startsWith('/clinician')) setRole('clinician')
    else if (pathname.startsWith('/admin'))    setRole('admin')
    else                                       setRole('caregiver')
  }, [pathname])
  return null
}

export default function App() {
  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', fontFamily: 'system-ui, sans-serif' }}>
      <RoleSync />
      <RoleSelector />
      <Routes>
        <Route path="/" element={<Navigate to="/caregiver" replace />} />
        <Route path="/caregiver" element={<CaregiverApp />} />
        <Route path="/clinician" element={<ClinicianDashboard />} />
        <Route path="/admin" element={<AdminAuditLog />} />
        <Route path="/predict" element={<ImagePredict />} />
      </Routes>
    </div>
  )
}
