import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/token': { target: 'http://localhost:7860', changeOrigin: false },
      '/healthz': { target: 'http://localhost:7860', changeOrigin: false },
    },
  },
});
