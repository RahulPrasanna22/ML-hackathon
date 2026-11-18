// Separate vitest config so the contract test lives outside Laptop 2's frontend/ sources.
// Plain object (no `vite` import) because this folder has no node_modules of its own.
// Run from frontend/ (which provides vitest):
//   npx vitest run --config ../integration/vitest.config.mts
import { fileURLToPath } from 'node:url';

export default {
  root: fileURLToPath(new URL('..', import.meta.url)),
  test: {
    globals: true,
    environment: 'node',
    include: ['integration/**/*.test.ts'],
  },
};
