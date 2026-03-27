import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // /sessions — always an API call
      '/sessions': { target: 'http://localhost:8000', changeOrigin: true },

      // /clinician — API sub-paths proxy to backend; bare route served by React Router
      '/clinician': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass(req) {
          if (req.url === '/clinician' || req.url === '/clinician/') {
            return '/index.html'
          }
        },
      },

      // /admin — same pattern
      '/admin': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass(req) {
          if (req.url === '/admin' || req.url === '/admin/') {
            return '/index.html'
          }
        },
      },

      '/dev':     { target: 'http://localhost:8000', changeOrigin: true },
      '/predict': { target: 'http://localhost:8000', changeOrigin: true },
      '/ml':      { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
