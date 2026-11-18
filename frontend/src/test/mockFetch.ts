/** Test-only fetch router. */
import { vi } from 'vitest';

export type Handler = (req: { url: string; method: string; body: unknown }) => Response | Promise<Response>;

export const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

export interface FetchMock {
  calls: Array<{ url: string; method: string; body: unknown }>;
}

/** Routes "METHOD /path" keys to handlers; unmatched requests reject like a network error. */
export function mockFetch(routes: Record<string, Handler>): FetchMock {
  const state: FetchMock = { calls: [] };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? 'GET').toUpperCase();
      const body = typeof init?.body === 'string' ? JSON.parse(init.body) : undefined;
      state.calls.push({ url, method, body });
      const path = new URL(url, 'http://same-origin.invalid').pathname;
      const handler = routes[`${method} ${path}`];
      if (!handler) throw new TypeError('Failed to fetch');
      if (init?.signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      return handler({ url, method, body });
    }),
  );
  return state;
}

/** A promise you resolve manually, to hold a request "in flight". */
export function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}
