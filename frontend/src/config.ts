/**
 * Runtime configuration for the Laptop 2 frontend.
 *
 * Backend URL resolution order (first match wins):
 *   1. `?api=<url>` query parameter (also remembered for this browser)
 *   2. URL saved from the Connection panel (localStorage, this browser only)
 *   3. VITE_API_BASE_URL from the environment (.env.local / build env)
 *   4. none -> the UI asks the evaluator to configure Laptop 1's address
 *
 * There is deliberately no hardcoded localhost default.
 */

const STORAGE_KEY = 'pathsnap.apiBaseUrl';

export type BaseUrlSource = 'query' | 'saved' | 'env' | 'none';

export interface ResolvedBaseUrl {
  url: string | null;
  source: BaseUrlSource;
}

function positiveNumber(raw: string | undefined, fallback: number): number {
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

export function getTimeouts() {
  const env = import.meta.env;
  return {
    apiTimeoutMs: positiveNumber(env.VITE_API_TIMEOUT_MS, 10_000),
    evaluationTimeoutMs: positiveNumber(env.VITE_EVALUATION_TIMEOUT_MS, 180_000),
    pollIntervalMs: positiveNumber(env.VITE_POLL_INTERVAL_MS, 1_500),
  };
}

/**
 * Development-only fallback trip IDs. Always empty in production builds.
 * Read lazily so tests can stub the environment.
 */
export function getDevFallbackTrips(): string[] {
  if (!import.meta.env.DEV) return [];
  const raw = import.meta.env.VITE_DEV_FALLBACK_TRIPS ?? '';
  return [...new Set(raw.split(',').map((s) => s.trim()).filter(Boolean))];
}

/**
 * Normalises user/env input into a base URL without a trailing slash.
 * Accepts "192.168.1.20:8000" (http:// is assumed), full http(s) URLs and
 * same-origin relative paths such as "/api" (used with the Vite proxy).
 * Returns null when the input is not usable.
 */
export function normalizeBaseUrl(raw: string | null | undefined): string | null {
  const s = (raw ?? '').trim();
  if (!s) return null;
  if (s.startsWith('/')) return s.replace(/\/+$/, '');
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(s) ? s : `http://${s}`;
  try {
    const u = new URL(withScheme);
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return null;
    if (!u.hostname) return null;
    return `${u.origin}${u.pathname}`.replace(/\/+$/, '');
  } catch {
    return null;
  }
}

function readSaved(): string | null {
  try {
    return normalizeBaseUrl(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return null;
  }
}

export function saveBaseUrl(url: string | null): void {
  try {
    if (url === null) window.localStorage.removeItem(STORAGE_KEY);
    else window.localStorage.setItem(STORAGE_KEY, url);
  } catch {
    /* storage unavailable (private mode etc.) - the in-memory value still works */
  }
}

export function getEnvBaseUrl(): string | null {
  return normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL);
}

export function resolveInitialBaseUrl(search: string = window.location.search): ResolvedBaseUrl {
  const fromQuery = normalizeBaseUrl(new URLSearchParams(search).get('api'));
  if (fromQuery) {
    saveBaseUrl(fromQuery);
    return { url: fromQuery, source: 'query' };
  }
  const saved = readSaved();
  if (saved) return { url: saved, source: 'saved' };
  const env = getEnvBaseUrl();
  if (env) return { url: env, source: 'env' };
  return { url: null, source: 'none' };
}

/** Human-readable label for a base URL, e.g. "192.168.1.20:8000". */
export function describeBaseUrl(url: string | null): string {
  if (url === null) return 'Not configured';
  if (url === '' || url.startsWith('/')) return `${url || '/'} (via proxy)`;
  try {
    const u = new URL(url);
    return `${u.host}${u.pathname === '/' ? '' : u.pathname}`;
  } catch {
    return url;
  }
}

/** Options for constructing a PathSnapClient from the environment. */
export function clientOptions() {
  const t = getTimeouts();
  return { timeoutMs: t.apiTimeoutMs, evaluationTimeoutMs: t.evaluationTimeoutMs, pollIntervalMs: t.pollIntervalMs };
}
