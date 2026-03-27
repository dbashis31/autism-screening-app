import { createContext, useContext, useState } from 'react'

const RoleContext = createContext(null)

export function RoleProvider({ children }) {
  const [role, setRoleState] = useState(() => localStorage.getItem('role') || 'caregiver')
  const [sessionId, setSessionId] = useState(null)
  const [childId, setChildId] = useState('')

  const setRole = (r) => {
    localStorage.setItem('role', r)
    setRoleState(r)
  }

  return (
    <RoleContext.Provider value={{ role, setRole, sessionId, setSessionId, childId, setChildId }}>
      {children}
    </RoleContext.Provider>
  )
}

export const useRole = () => useContext(RoleContext)
