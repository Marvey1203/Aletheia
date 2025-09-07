// FILE: vite.config.js (Simplified and Corrected)
import { defineConfig } from 'vite'

export default defineConfig({
  // prevent vite from obscuring rust errors
  clearScreen: false,
  // Tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
  },
})