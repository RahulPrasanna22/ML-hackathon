import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { completeEvaluation, healthOk, minimalEvaluation, TEST_API, tripList } from './test/fixtures';
import { deferred, json, mockFetch, type Handler } from './test/mockFetch';

vi.mock('react-leaflet', async () => (await import('./test/leafletMock')).leafletMock);

const baseRoutes = (): Record<string, Handler> => ({
  'GET /health': () => json(healthOk),
  'GET /evaluation/trips': () => json(tripList),
});

beforeEach(() => {
  vi.stubEnv('VITE_API_BASE_URL', TEST_API);
  vi.stubEnv('VITE_DEV_FALLBACK_TRIPS', '');
});

async function selectAndRun(user: ReturnType<typeof userEvent.setup>, tripId: string) {
  await user.click(await screen.findByTestId(`trip-${tripId}`));
  await user.click(screen.getByRole('button', { name: `Run evaluation on ${tripId}` }));
}

describe('dashboard', () => {
  it('renders branding and asks for the Laptop 1 address when none is configured', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '');
    const f = mockFetch({});
    render(<App />);
    expect(screen.getByText('PathSnap')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /Distributed Map Matching & Route Reconstruction/ })).toBeInTheDocument();
    expect(screen.getByTestId('connection-status')).toHaveTextContent('Not configured');
    expect(screen.getByLabelText('Laptop 1 API address')).toBeInTheDocument();
    expect(screen.getByText('Connect to Laptop 1 first')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Select a trip to run' })).toBeDisabled();
    expect(f.calls).toHaveLength(0);
  });

  it('connects using an address entered in the connection panel', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '');
    const user = userEvent.setup();
    const f = mockFetch(baseRoutes());
    render(<App />);
    await user.type(screen.getByLabelText('Laptop 1 API address'), 'test-laptop1.invalid:8000');
    await user.click(screen.getByRole('button', { name: 'Save and connect' }));
    expect(await screen.findByTestId('trip-T-001')).toBeInTheDocument();
    expect(f.calls.some((c) => c.url === `${TEST_API}/evaluation/trips`)).toBe(true);
    expect(window.localStorage.getItem('pathsnap.apiBaseUrl')).toBe(TEST_API);
  });

  it('shows connection status and the trips retrieved from the backend', async () => {
    mockFetch(baseRoutes());
    render(<App />);
    await waitFor(() => expect(screen.getByTestId('connection-status')).toHaveTextContent('Connected'));
    expect(await screen.findByTestId('trip-T-001')).toBeInTheDocument();
    expect(screen.getByTestId('trip-T-002')).toHaveTextContent('3 points');
    expect(screen.getByText(/3 trips from Laptop 1 \(evaluator_test\)/)).toBeInTheDocument();
    expect(screen.getByText('test-model')).toBeInTheDocument();
  });

  it('shows a loading state while trips are being fetched', async () => {
    const pending = deferred<Response>();
    mockFetch({ 'GET /health': () => json(healthOk), 'GET /evaluation/trips': () => pending.promise });
    render(<App />);
    expect(await screen.findByLabelText('Loading trips')).toHaveAttribute('aria-busy', 'true');
    await act(async () => pending.resolve(json(tripList)));
    expect(await screen.findByTestId('trip-T-001')).toBeInTheDocument();
  });

  it('reports an offline backend and recovers on retry', async () => {
    const user = userEvent.setup();
    let online = false;
    mockFetch({
      'GET /health': () => (online ? json(healthOk) : Promise.reject(new TypeError('Failed to fetch'))),
      'GET /evaluation/trips': () => (online ? json(tripList) : Promise.reject(new TypeError('Failed to fetch'))),
    });
    render(<App />);
    expect(await screen.findByText(/Could not load trips: Laptop 1 is unreachable/)).toBeInTheDocument();
    expect(await screen.findByText('Laptop 1 API is not responding')).toBeInTheDocument();
    expect(screen.queryByTestId('trip-T-001')).not.toBeInTheDocument();

    online = true;
    const tripsPanel = screen.getByRole('region', { name: 'Evaluation trips' });
    await user.click(within(tripsPanel).getByRole('button', { name: 'Retry' }));
    expect(await screen.findByTestId('trip-T-001')).toBeInTheDocument();
  });

  it('labels development fallback trips and never presents them as backend data', async () => {
    vi.stubEnv('VITE_DEV_FALLBACK_TRIPS', 'DEV-1,DEV-2');
    mockFetch({ 'GET /health': () => json(healthOk), 'GET /evaluation/trips': () => json({ detail: 'boom' }, 500) });
    render(<App />);
    expect(await screen.findByText('Showing development fallback trips')).toBeInTheDocument();
    expect(screen.getByText('Dev fallback, not backend data')).toBeInTheDocument();
    expect(screen.getByTestId('trip-DEV-1')).toBeInTheDocument();
    expect(screen.getByText('dev fallback')).toBeInTheDocument();
  });

  it('rejects an invalid trip-list response with a clear message', async () => {
    mockFetch({ 'GET /health': () => json(healthOk), 'GET /evaluation/trips': () => json({ nope: true }) });
    render(<App />);
    expect(await screen.findByText(/Could not load trips: Unexpected response from Laptop 1/)).toBeInTheDocument();
  });
});

describe('evaluation flow', () => {
  it('selects a trip, runs it, blocks duplicate submissions and renders the results', async () => {
    const user = userEvent.setup();
    const pending = deferred<Response>();
    const f = mockFetch({ ...baseRoutes(), 'POST /evaluation/run': () => pending.promise });
    render(<App />);

    expect(await screen.findByRole('button', { name: 'Select a trip to run' })).toBeDisabled();
    await selectAndRun(user, 'T-001');

    // Running state
    const runBtn = screen.getByRole('button', { name: 'Running evaluation…' });
    expect(runBtn).toBeDisabled();
    expect(screen.getByTestId('run-status')).toHaveTextContent('Evaluating trip T-001');
    expect(screen.getByText('Waiting for the reconstructed route…')).toBeInTheDocument();
    expect(screen.getByTestId('trip-T-002')).toBeDisabled();
    await user.click(runBtn);
    expect(f.calls.filter((c) => c.method === 'POST')).toHaveLength(1);
    expect(f.calls.find((c) => c.method === 'POST')?.body).toEqual({ trip_id: 'T-001' });

    await act(async () => pending.resolve(json(completeEvaluation('T-001'))));

    const results = await screen.findByTestId('results');
    const tile = (label: string) => within(results).getByText(label).parentElement!;
    expect(tile('GPS observations')).toHaveTextContent('4');
    expect(tile('Predicted road segments')).toHaveTextContent('2');
    expect(tile('Top-1 accuracy')).toHaveTextContent('50.0%');
    expect(tile('Macro F1-score')).toHaveTextContent('25.0%');
    expect(tile('Route continuity')).toHaveTextContent('75.0%');
    expect(tile('Log loss')).toHaveTextContent('0.1235');
    expect(tile('Inference time')).toHaveTextContent('42 ms');
    expect(tile('HMM decoding')).toHaveTextContent('Applied');
    // Additional metrics from the API are listed, including unknown ones.
    expect(within(results).getByText('Candidate miss rate').nextSibling).toHaveTextContent('10.0%');
    expect(within(results).getByText('Custom score').nextSibling).toHaveTextContent('7');

    expect(screen.getByTestId('run-status')).toHaveTextContent('Trip T-001 evaluated');
    expect(screen.getByRole('button', { name: 'Run evaluation on T-001' })).toBeEnabled();
  });

  it('renders map layers from the API response with a legend and layer controls', async () => {
    const user = userEvent.setup();
    mockFetch({ ...baseRoutes(), 'POST /evaluation/run': () => json(completeEvaluation('T-001')) });
    render(<App />);
    await selectAndRun(user, 'T-001');

    const map = await screen.findByTestId('route-map');
    expect(within(map).getByTestId('overlay-GPS observations')).toHaveAttribute('data-checked', 'true');
    expect(within(map).getByTestId('overlay-Predicted route')).toBeInTheDocument();
    expect(within(map).getByTestId('overlay-Road network')).toBeInTheDocument();
    expect(within(map).queryByTestId('overlay-Reference route')).not.toBeInTheDocument();
    expect(within(map).getByTestId('base-Dark')).toBeInTheDocument();
    expect(within(map).getByTestId('base-OpenStreetMap')).toBeInTheDocument();

    expect(within(within(map).getByTestId('overlay-GPS observations')).getAllByTestId('circle-marker')).toHaveLength(4);
    expect(within(within(map).getByTestId('overlay-Start / end markers')).getAllByTestId('circle-marker')).toHaveLength(2);
    expect(within(within(map).getByTestId('overlay-Predicted route')).getAllByTestId('polyline')).toHaveLength(2);
    expect(JSON.parse(within(map).getByTestId('map-container').dataset.bounds!)).toEqual([
      [20, 10],
      [20.003, 10.003],
    ]);

    const legend = within(map).getByTestId('map-legend');
    expect(legend).toHaveTextContent('Raw GPS observations (4)');
    expect(legend).toHaveTextContent('Predicted route');
    expect(legend).toHaveTextContent('Road network');
    expect(legend).not.toHaveTextContent('Reference route');
  });

  it('shows the reference route layer (off by default) only when the backend sends one', async () => {
    const user = userEvent.setup();
    const payload = { ...completeEvaluation('T-001'), reference_route: { segments: [{ arc_id: 9, coordinates: [[10, 20], [10.003, 20.003]] }] } };
    mockFetch({ ...baseRoutes(), 'POST /evaluation/run': () => json(payload) });
    render(<App />);
    await selectAndRun(user, 'T-001');
    const overlay = await screen.findByTestId('overlay-Reference route');
    expect(overlay).toHaveAttribute('data-checked', 'false');
    expect(screen.getByTestId('map-legend')).toHaveTextContent('Reference route (from backend)');
  });

  it('shows N/A for metrics the backend did not return and skips charts without data', async () => {
    const user = userEvent.setup();
    mockFetch({ ...baseRoutes(), 'POST /evaluation/run': () => json(minimalEvaluation('T-002')) });
    render(<App />);
    await selectAndRun(user, 'T-002');

    const results = await screen.findByTestId('results');
    const tile = (label: string) => within(results).getByText(label).parentElement!;
    for (const label of ['Top-1 accuracy', 'Macro F1-score', 'Route continuity', 'Log loss', 'Inference time', 'HMM decoding']) {
      expect(tile(label)).toHaveTextContent('N/A');
    }
    expect(tile('GPS observations')).toHaveTextContent('2');
    expect(screen.queryByTestId('charts')).not.toBeInTheDocument();
  });

  it('renders charts only for metrics that were returned', async () => {
    const user = userEvent.setup();
    mockFetch({ ...baseRoutes(), 'POST /evaluation/run': () => json(completeEvaluation('T-001')) });
    render(<App />);
    await selectAndRun(user, 'T-001');
    const charts = await screen.findByTestId('charts');
    expect(within(charts).getByText('Route quality, trip T-001')).toBeInTheDocument();
    expect(within(charts).getByText('Effect of HMM/Viterbi decoding')).toBeInTheDocument();
    expect(within(charts).getByText('Server timing breakdown')).toBeInTheDocument();
    // Needs at least two evaluated trips.
    expect(within(charts).queryByText(/across trips evaluated this session/)).not.toBeInTheDocument();
  });

  it('shows a backend failure and does not display results as a success', async () => {
    const user = userEvent.setup();
    mockFetch({ ...baseRoutes(), 'POST /evaluation/run': () => json({ detail: 'Model crashed' }, 500) });
    render(<App />);
    await selectAndRun(user, 'T-001');
    expect(await screen.findByText(/Evaluation of trip T-001 failed: Laptop 1 reported a server error/)).toBeInTheDocument();
    expect(within(screen.getByRole('alert')).getByText('HTTP 500: Model crashed')).toBeInTheDocument();
    expect(screen.queryByTestId('results')).not.toBeInTheDocument();
    expect(screen.getByTestId('map-empty')).toBeInTheDocument();
    expect(within(screen.getByTestId('history')).getByText('HTTP 500: Model crashed')).toBeInTheDocument();
  });

  it('treats status "failed" in a 200 response as a failure', async () => {
    const user = userEvent.setup();
    mockFetch({
      ...baseRoutes(),
      'POST /evaluation/run': () => json({ evaluation_id: 'e', trip_id: 'T-001', status: 'failed', error: 'No candidates within radius' }),
    });
    render(<App />);
    await selectAndRun(user, 'T-001');
    expect(await screen.findByText(/failed: Laptop 1 reported a failure/)).toBeInTheDocument();
    expect(within(screen.getByRole('alert')).getByText('No candidates within radius')).toBeInTheDocument();
    expect(screen.queryByTestId('results')).not.toBeInTheDocument();
  });

  it('reports a missing trip (404)', async () => {
    const user = userEvent.setup();
    mockFetch({ ...baseRoutes(), 'POST /evaluation/run': () => json({ detail: 'Unknown trip T-003' }, 404) });
    render(<App />);
    await selectAndRun(user, 'T-003');
    expect(await screen.findByText(/failed: Not found on Laptop 1/)).toBeInTheDocument();
  });

  it('warns about an empty predicted route and missing coordinates', async () => {
    const user = userEvent.setup();
    const payload = { evaluation_id: 'e', trip_id: 'T-001', status: 'completed', observations: [{ lon: 10, lat: 20 }, { lon: null, lat: 20 }], predicted_route: { arc_ids: [], segments: [] } };
    mockFetch({ ...baseRoutes(), 'POST /evaluation/run': () => json(payload) });
    render(<App />);
    await selectAndRun(user, 'T-001');
    expect(await screen.findByText('Predicted route is empty')).toBeInTheDocument();
    expect(screen.getByText(/1 GPS observation\(s\) had invalid coordinates/)).toBeInTheDocument();
  });

  it('explains when a response has no map geometry at all', async () => {
    const user = userEvent.setup();
    mockFetch({ ...baseRoutes(), 'POST /evaluation/run': () => json({ evaluation_id: 'e', trip_id: 'T-001', status: 'completed' }) });
    render(<App />);
    await selectAndRun(user, 'T-001');
    expect(await screen.findByText('No map geometry in this response')).toBeInTheDocument();
    expect(screen.queryByTestId('route-map')).not.toBeInTheDocument();
  });
});

describe('evaluation history', () => {
  it('records each run and restores an earlier result from history', async () => {
    const user = userEvent.setup();
    mockFetch({ ...baseRoutes(), 'POST /evaluation/run': ({ body }) => json(completeEvaluation((body as { trip_id: string }).trip_id)) });
    render(<App />);

    await selectAndRun(user, 'T-001');
    await screen.findByText(/Results for trip/);
    await selectAndRun(user, 'T-002');
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Results for trip T-002' })).toBeInTheDocument());

    const history = screen.getByTestId('history');
    const items = within(history).getAllByRole('listitem');
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent('T-002');
    expect(items[1]).toHaveTextContent('T-001');
    expect(items[1]).toHaveTextContent('Top-1 50.0%');

    // Session chart appears with two evaluated trips.
    expect(screen.getByText('Top-1 accuracy across trips evaluated this session')).toBeInTheDocument();

    await user.click(within(items[1]).getByRole('button'));
    expect(screen.getByRole('heading', { name: 'Results for trip T-001' })).toBeInTheDocument();

    // Persisted summary survives a re-mount (same tab session).
    const stored = JSON.parse(window.sessionStorage.getItem('pathsnap.history.v1')!);
    expect(stored).toHaveLength(2);
    expect(JSON.stringify(stored)).not.toContain('coordinates');

    await user.click(within(history).getByRole('button', { name: 'Clear' }));
    expect(within(screen.getByTestId('history')).queryAllByRole('listitem')).toHaveLength(0);
    expect(window.sessionStorage.getItem('pathsnap.history.v1')).toBeNull();
  });

  it('reloads a past evaluation from GET /evaluation/{id} after a page reload', async () => {
    const user = userEvent.setup();
    window.sessionStorage.setItem(
      'pathsnap.history.v1',
      JSON.stringify([{ id: 'x', evaluationId: 'eval-T-003', tripId: 'T-003', status: 'completed', timestamp: new Date().toISOString(), metrics: { top1_accuracy: 0.5 }, inferenceMs: null, roundTripMs: null, error: null }]),
    );
    const f = mockFetch({ ...baseRoutes(), 'GET /evaluation/eval-T-003': () => json(completeEvaluation('T-003')) });
    render(<App />);
    await user.click(within(screen.getByTestId('history')).getByTitle('Show this result'));
    expect(await screen.findByRole('heading', { name: 'Results for trip T-003' })).toBeInTheDocument();
    expect(f.calls.some((c) => c.url.endsWith('/evaluation/eval-T-003'))).toBe(true);
  });
});
