/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_API_TIMEOUT_MS?: string;
  readonly VITE_EVALUATION_TIMEOUT_MS?: string;
  readonly VITE_POLL_INTERVAL_MS?: string;
  readonly VITE_DEV_FALLBACK_TRIPS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
