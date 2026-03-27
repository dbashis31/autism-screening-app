import axios from 'axios'

const api = axios.create({ baseURL: '' })

// Attach the current role from sessionStorage before each request
api.interceptors.request.use((config) => {
  const role = sessionStorage.getItem('role') || 'caregiver'
  config.headers['X-Role'] = role
  return config
})

export default api
