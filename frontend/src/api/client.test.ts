import { describe, expect, it, vi } from 'vitest';
import { completeEvaluation, healthOk, TEST_API, tripList } from '../test/fixtures';
import { json, mockFetch } from '../test/mockFetch';
import { PathSnapClient } from './client';
import { ApiError } from './errors';

const opts = { timeoutMs: 1000, evaluationTimeoutMs: 2000, pollIntervalMs: 5 };
const client = () => new PathSnapClient(`${TEST_API}/`, opts);

async function expectApiError(p: Promise<unknown>, kind: ApiError['kind']) {
  const err = await p.then(
    () => null,
    (e: unknown) => e,
  );
  expect(err).toBeInstanceOf(ApiError);
  expect((err as ApiError).kind).toBe(kind);
  return err as ApiError;
}

describe('PathSnapClient', () => {
  it('strips the trailing slash and calls GET /health', async () => {
    const f = mockFetch({ 'GET /health': () => json(healthOk) });
    const h = await client().health();
    expect(f.calls[0].url).toBe(`${TEST_API}/health`);
    expect(h).toMatchObject({ status: 'ok', modelLoaded: true, modelName: 'test-model', hmmEnabled: true });
  });

  it('retrieves trips from GET /evaluation/trips', async () => {
    mockFetch({ 'GET /evaluation/trips': () => json(tripList) });
    const list = await client().listTrips();
    expect(list.partition).toBe('evaluator_test');
    expect(list.trips.map((t) => t.tripId)).toEqual(['T-001', 'T-002', 'T-003']);
    expect(list.trips[2].numObservations).toBeNull();
  });

  it('sends only the trip id in POST /evaluation/run', async () => {
    const f = mockFetch({ 'POST /evaluation/run': () => json(completeEvaluation('T-001')) });
    const r = await client().runEvaluation('T-001');
    expect(f.calls[0]).toMatchObject({ method: 'POST', body: { trip_id: 'T-001' } });
    expect(r.status).toBe('completed');
    expect(r.predictedRoute?.arcIds).toEqual([1, 2]);
  });

  it('polls GET /evaluation/{id} when the backend answers queued/running', async () => {
    let polls = 0;
    const progress = vi.fn();
    mockFetch({
      'POST /evaluation/run': () => json({ evaluation_id: 'e1', trip_id: 'T-001', status: 'queued' }, 202),
      'GET /evaluation/e1': () => {
        polls += 1;
        return json(polls < 2 ? { evaluation_id: 'e1', trip_id: 'T-001', status: 'running' } : { ...completeEvaluation('T-001'), evaluation_id: 'e1' });
      },
    });
    const r = await client().runEvaluation('T-001', { onProgress: progress });
    expect(r.status).toBe('completed');
    expect(polls).toBe(2);
    expect(progress).toHaveBeenCalledTimes(2);
  });

  it('maps HTTP 404 with a FastAPI detail to an http error', async () => {
    mockFetch({ 'POST /evaluation/run': () => json({ detail: 'Trip T-999 not found' }, 404) });
    const err = await expectApiError(client().runEvaluation('T-999'), 'http');
    expect(err.status).toBe(404);
    expect(err.message).toContain('Trip T-999 not found');
  });

  it('flattens FastAPI 422 validation details', async () => {
    mockFetch({ 'POST /evaluation/run': () => json({ detail: [{ loc: ['body', 'trip_id'], msg: 'field required' }] }, 422) });
    const err = await expectApiError(client().runEvaluation('x'), 'http');
    expect(err.detail).toBe('field required');
  });

  it('reports an unreachable backend (offline or CORS) as a network error', async () => {
    mockFetch({});
    const err = await expectApiError(client().health(), 'network');
    expect(err.message).toContain('CORS');
  });

  it('times out a request that never answers', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise((_, reject) => init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))),
      ),
    );
    const c = new PathSnapClient(TEST_API, { ...opts, timeoutMs: 20 });
    await expectApiError(c.health(), 'timeout');
  });

  it('rejects a response that breaks the contract', async () => {
    mockFetch({ 'POST /evaluation/run': () => json({ status: 'completed' }) });
    const err = await expectApiError(client().runEvaluation('T-001'), 'validation');
    expect(err.message).toContain('trip_id');
  });

  it('rejects non-JSON bodies', async () => {
    mockFetch({ 'GET /health': () => new Response('<html>hello</html>', { status: 200 }) });
    await expectApiError(client().health(), 'validation');
  });

  it('rejects results for a different trip than requested', async () => {
    mockFetch({ 'POST /evaluation/run': () => json(completeEvaluation('T-002')) });
    await expectApiError(client().runEvaluation('T-001'), 'validation');
  });

  it('returns failed evaluations as results so the UI can show the backend error', async () => {
    mockFetch({ 'POST /evaluation/run': () => json({ evaluation_id: 'e', trip_id: 'T-001', status: 'failed', error: 'No candidates' }) });
    const r = await client().runEvaluation('T-001');
    expect(r.status).toBe('failed');
    expect(r.error).toBe('No candidates');
  });

  it('resolves backend-relative URLs against the API base', () => {
    const c = client();
    expect(c.resolve('/evaluation/e1/map')).toBe(`${TEST_API}/evaluation/e1/map`);
    expect(c.resolve('http://other/x')).toBe('http://other/x');
    expect(new PathSnapClient('/api', opts).url('/health')).toBe('/api/health');
  });
});
