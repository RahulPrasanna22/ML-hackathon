import { useCallback, useMemo, useRef, useState } from 'react';
import { PathSnapClient } from './api/client';
import { errorMessage } from './api/errors';
import type { EvaluationResult } from './api/types';
import { Charts } from './components/Charts';
import { ConnectionPanel } from './components/ConnectionPanel';
import { Header } from './components/Header';
import { HistoryPanel } from './components/HistoryPanel';
import { MapPanel } from './components/MapPanel';
import { Notice } from './components/Notice';
import { OverviewBar } from './components/OverviewBar';
import { ResultsPanel } from './components/ResultsPanel';
import { RunStatus } from './components/RunStatus';
import { TripSelector } from './components/TripSelector';
import { clientOptions, getEnvBaseUrl, resolveInitialBaseUrl, saveBaseUrl, type ResolvedBaseUrl } from './config';
import { useBackendStatus } from './hooks/useBackendStatus';
import { useEvaluation } from './hooks/useEvaluation';
import { useTrips } from './hooks/useTrips';
import { clearStoredHistory, entryFromError, entryFromResult, loadHistory, MAX_HISTORY, saveHistory, type HistoryEntry } from './lib/history';

export default function App() {
  const [base, setBase] = useState<ResolvedBaseUrl>(() => resolveInitialBaseUrl());
  const [settingsOpen, setSettingsOpen] = useState(() => base.url === null);
  const [selectedTrip, setSelectedTrip] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());
  const [historyError, setHistoryError] = useState<string | null>(null);
  /** Full results for this page session only (in memory, never persisted). */
  const resultCache = useRef(new Map<string, EvaluationResult>());

  const client = useMemo(() => (base.url === null ? null : new PathSnapClient(base.url, clientOptions())), [base.url]);
  const { status, check } = useBackendStatus(client);
  const trips = useTrips(client);

  const pushHistory = useCallback((entry: HistoryEntry) => {
    setHistory((prev) => {
      const next = [entry, ...prev].slice(0, MAX_HISTORY);
      saveHistory(next);
      return next;
    });
  }, []);

  const evaluation = useEvaluation(client, {
    onFinished: ({ tripId, result, error, roundTripMs }) => {
      if (result?.evaluationId) resultCache.current.set(result.evaluationId, result);
      pushHistory(result && result.status !== 'queued' && result.status !== 'running' ? entryFromResult(result, roundTripMs) : entryFromError(tripId, errorMessage(error), roundTripMs));
    },
  });
  const running = evaluation.phase === 'running';

  function applyBaseUrl(url: string | null) {
    saveBaseUrl(url);
    const envUrl = getEnvBaseUrl();
    setBase(url === null ? { url: envUrl, source: envUrl ? 'env' : 'none' } : { url, source: 'saved' });
    setSelectedTrip(null);
    if (url !== null || envUrl) setSettingsOpen(false);
  }

  async function viewHistory(entry: HistoryEntry) {
    if (!entry.evaluationId) return;
    setHistoryError(null);
    const cached = resultCache.current.get(entry.evaluationId);
    if (cached) {
      evaluation.show(cached);
      setSelectedTrip(cached.tripId);
      return;
    }
    if (!client) return;
    try {
      const result = await client.getEvaluation(entry.evaluationId);
      resultCache.current.set(entry.evaluationId, result);
      evaluation.show(result);
      setSelectedTrip(result.tripId);
    } catch (err) {
      setHistoryError(`Could not reload evaluation ${entry.evaluationId} for trip ${entry.tripId}: ${errorMessage(err)}`);
    }
  }

  function clearHistory() {
    clearStoredHistory();
    setHistory([]);
  }

  const latestSuccess = history.find((h) => h.status === 'completed') ?? null;
  const shown = evaluation.result && evaluation.result.status === 'completed' ? evaluation.result : null;

  return (
    <div className="app">
      <Header baseUrl={base.url} status={status} settingsOpen={settingsOpen} onToggleSettings={() => setSettingsOpen((o) => !o)} />
      {settingsOpen && (
        <ConnectionPanel key={base.url ?? 'none'} baseUrl={base.url} source={base.source} status={status} onApply={applyBaseUrl} onRetry={() => void check()} />
      )}

      {status.state === 'offline' && (
        <div className="banner">
          <Notice
            tone="error"
            title="Laptop 1 API is not responding"
            action={
              <button type="button" className="btn btn--ghost btn--sm" onClick={() => void check()}>
                Retry
              </button>
            }
          >
            {status.error}
          </Notice>
        </div>
      )}

      <OverviewBar
        status={status}
        tripCount={trips.trips.length > 0 || trips.status === 'ready' ? trips.trips.length : null}
        tripsFromBackend={trips.source === 'backend'}
        selectedTrip={selectedTrip}
        phase={evaluation.phase}
        runningTrip={running ? evaluation.tripId : null}
        latestSuccess={latestSuccess}
      />

      <div className="layout">
        <aside className="rail">
          <TripSelector
            trips={trips}
            configured={client !== null}
            selected={selectedTrip}
            onSelect={setSelectedTrip}
            onReload={() => void trips.reload()}
            onRun={() => selectedTrip && void evaluation.run(selectedTrip)}
            running={running}
            history={history}
          />
          <RunStatus evaluation={evaluation} onCancel={evaluation.cancel} onRetry={() => evaluation.tripId && void evaluation.run(evaluation.tripId)} />
          {historyError && (
            <Notice tone="error" title="History item unavailable">
              {historyError}
            </Notice>
          )}
          <HistoryPanel
            entries={history}
            activeEvaluationId={shown?.evaluationId ?? null}
            onView={(e) => void viewHistory(e)}
            onClear={clearHistory}
            disabled={running}
          />
        </aside>

        <main className="main">
          <MapPanel result={shown} running={running} resolveUrl={(p) => (client ? client.resolve(p) : p)} />
          {shown && <ResultsPanel result={shown} roundTripMs={evaluation.roundTripMs} />}
          <Charts result={shown} history={history} />
        </main>
      </div>

      <footer className="footer">
        PathSnap evaluator UI (Laptop 2). Every route, metric and map layer shown here comes from the Laptop 1 API response; nothing is computed or stored locally beyond session summaries.
      </footer>
    </div>
  );
}
