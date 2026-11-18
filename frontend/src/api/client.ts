/**
 * PathSnap API client - the only module that talks to Laptop 1.
 * UI components never call fetch directly.
 */
import { ApiError, ValidationError } from './errors';
import type { EvaluationResult, HealthStatus, TripList } from './types';
import { parseEvaluation, parseHealth, parseTripList } from './validate';

export interface ClientOptions {
  timeoutMs: number;
  evaluationTimeoutMs: number;
  pollIntervalMs: number;
}

interface RequestOptions {
  method?: 'GET' | 'POST';
  body?: unknown;
  timeoutMs: number;
  signal?: AbortSignal;
}

export interface RunOptions {
  signal?: AbortSignal;
  /** Called with interim results while the backend reports queued/running. */
  onProgress?: (result: EvaluationResult) => void;
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const id = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(id);
        reject(new ApiError('aborted', 'The request was cancelled.'));
      },
      { once: true },
    );
  });

function extractDetail(body: unknown): string | null {
  if (typeof body === 'string') return body.slice(0, 300) || null;
  if (body && typeof body === 'object') {
    const detail = (body as Record<string, unknown>).detail ?? (body as Record<string, unknown>).error;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      // FastAPI validation errors: [{loc, msg, type}]
      return detail
        .map((d) => (d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : String(d)))
        .join('; ');
    }
  }
  return null;
}

export class PathSnapClient {
  readonly baseUrl: string;
  private readonly opts: ClientOptions;

  constructor(baseUrl: string, opts: ClientOptions) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.opts = opts;
  }

  /** Absolute or same-origin URL for an API path. */
  url(path: string): string {
    return `${this.baseUrl}${path.startsWith('/') ? path : `/${path}`}`;
  }

  /** Resolve a URL returned by the backend (may be relative to the API base). */
  resolve(pathOrUrl: string): string {
    if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
    return this.url(pathOrUrl);
  }

  private async request<T>(path: string, req: RequestOptions, parse: (raw: unknown) => T): Promise<T> {
    const url = this.url(path);
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, req.timeoutMs);
    const onExternalAbort = () => controller.abort();
    req.signal?.addEventListener('abort', onExternalAbort, { once: true });

    let res: Response;
    try {
      res = await fetch(url, {
        method: req.method ?? 'GET',
        headers: req.body !== undefined ? { 'Content-Type': 'application/json', Accept: 'application/json' } : { Accept: 'application/json' },
        body: req.body !== undefined ? JSON.stringify(req.body) : undefined,
        signal: controller.signal,
      });
    } catch (err) {
      if (timedOut) {
        throw new ApiError('timeout', `No response from Laptop 1 within ${Math.round(req.timeoutMs / 1000)} s (${url}).`, { url });
      }
      if (req.signal?.aborted) throw new ApiError('aborted', 'The request was cancelled.', { url });
      const origin = typeof window !== 'undefined' ? window.location.origin : 'this page';
      throw new ApiError(
        'network',
        `Could not reach ${url}. Check that the API is running on Laptop 1, both laptops are on the same network, ` +
          `the firewall allows the port, and the backend's CORS settings allow ${origin}.`,
        { url, detail: err instanceof Error ? err.message : null },
      );
    } finally {
      clearTimeout(timer);
      req.signal?.removeEventListener('abort', onExternalAbort);
    }

    const text = await res.text().catch(() => '');
    let body: unknown = undefined;
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = text;
      }
    }

    if (!res.ok) {
      const detail = extractDetail(body);
      throw new ApiError('http', detail ? `HTTP ${res.status}: ${detail}` : `HTTP ${res.status} from ${url}`, {
        status: res.status,
        detail,
        url,
      });
    }
    if (typeof body !== 'object' || body === null) {
      throw new ApiError('validation', `Expected JSON from ${url} but received ${text ? 'non-JSON text' : 'an empty body'}.`, { url });
    }
    try {
      return parse(body);
    } catch (err) {
      if (err instanceof ValidationError) {
        throw new ApiError('validation', `Response from ${path} did not match the API contract (${err.message}).`, {
          url,
          detail: err.message,
        });
      }
      throw err;
    }
  }

  health(signal?: AbortSignal): Promise<HealthStatus> {
    return this.request('/health', { timeoutMs: this.opts.timeoutMs, signal }, parseHealth);
  }

  listTrips(signal?: AbortSignal): Promise<TripList> {
    return this.request('/evaluation/trips', { timeoutMs: this.opts.timeoutMs, signal }, parseTripList);
  }

  getEvaluation(evaluationId: string, signal?: AbortSignal, timeoutMs = this.opts.timeoutMs): Promise<EvaluationResult> {
    return this.request(
      `/evaluation/${encodeURIComponent(evaluationId)}`,
      { timeoutMs, signal },
      parseEvaluation,
    );
  }

  /**
   * POST /evaluation/run. Supports both synchronous responses (completed /
   * failed immediately) and asynchronous ones (queued/running + polling of
   * GET /evaluation/{id}) within the overall evaluation timeout.
   */
  async runEvaluation(tripId: string, opts: RunOptions = {}): Promise<EvaluationResult> {
    const deadline = Date.now() + this.opts.evaluationTimeoutMs;
    let result = await this.request(
      '/evaluation/run',
      { method: 'POST', body: { trip_id: tripId }, timeoutMs: this.opts.evaluationTimeoutMs, signal: opts.signal },
      parseEvaluation,
    );

    while (result.status === 'queued' || result.status === 'running') {
      opts.onProgress?.(result);
      if (!result.evaluationId) {
        throw new ApiError('validation', 'Laptop 1 reported the evaluation as in progress but did not return an evaluation_id to poll.');
      }
      const remaining = deadline - Date.now();
      if (remaining <= 0) {
        throw new ApiError('timeout', `Evaluation of trip ${tripId} did not finish within ${Math.round(this.opts.evaluationTimeoutMs / 1000)} s.`);
      }
      await sleep(Math.min(this.opts.pollIntervalMs, remaining), opts.signal);
      result = await this.getEvaluation(result.evaluationId, opts.signal, Math.min(this.opts.timeoutMs, Math.max(remaining, 1000)));
    }

    if (result.tripId !== tripId) {
      throw new ApiError(
        'validation',
        `Laptop 1 returned results for trip ${result.tripId} but trip ${tripId} was requested.`,
      );
    }
    return result;
  }
}
