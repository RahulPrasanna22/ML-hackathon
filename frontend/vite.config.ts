/// <reference types="vitest/config" />
import { defineConfig, loadEnv, type ProxyOptions } from 'vite';
import react from '@vitejs/plugin-react';

// PATHSNAP_PROXY_TARGET (optional, NOT exposed to the browser):
// when set, requests to /api/* on the Laptop 2 dev/preview server are forwarded
// to Laptop 1. Use together with VITE_API_BASE_URL=/api as a CORS-free fallback.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.PATHSNAP_PROXY_TARGET?.trim();
  const proxy: Record<string, ProxyOptions> | undefined = target
    ? {
        '/api': {
          target,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      }
    : undefined;

  return {
    plugins: [react()],
    server: { port: 5173, strictPort: true, proxy },
    preview: { port: 4173, strictPort: true, proxy },
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: ['./src/test/setup.ts'],
      css: false,
      include: ['src/**/*.test.{ts,tsx}'],
    },
  };
});
