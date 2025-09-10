// FILE: vite.config.js (Corrected Hierarchy)
import { defineConfig } from 'vite'

export default defineConfig({
  // prevent vite from obscuring rust errors
  clearScreen: false,
  // Tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    // --- THIS IS THE CORRECTED PLACEMENT ---
    // The 'watch' object must be a key inside the 'server' object.
    watch: {
      ignored: [
        "**/venv/**",
        "**/memory_galaxy/**",
        "**/src-tauri/target/**"
      ],
    },
  },
})