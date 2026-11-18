import { errorMessage, errorTitle } from '../api/errors';
import type { TripSummary } from '../api/types';
import type { TripsState } from '../hooks/useTrips';
import type { HistoryEntry } from '../lib/history';
import { Notice } from './Notice';

interface TripSelectorProps {
  trips: TripsState;
  configured: boolean;
  selected: string | null;
  onSelect: (tripId: string) => void;
  onReload: () => void;
  onRun: () => void;
  running: boolean;
  history: HistoryEntry[];
}

function lastOutcome(history: HistoryEntry[], tripId: string): HistoryEntry['status'] | null {
  return history.find((h) => h.tripId === tripId)?.status ?? null;
}

function tripMeta(t: TripSummary): string | null {
  const parts: string[] = [];
  if (t.numObservations !== null) parts.push(`${t.numObservations.toLocaleString('en-US')} points`);
  if (t.durationS !== null) parts.push(`${Math.round(t.durationS / 60)} min`);
  return parts.length ? parts.join(', ') : null;
}

export function TripSelector({ trips, configured, selected, onSelect, onReload, onRun, running, history }: TripSelectorProps) {
  const isFallback = trips.source === 'dev-fallback';
  const canRun = configured && selected !== null && !running;

  return (
    <section className="panel trips" aria-labelledby="trips-heading">
      <div className="panel__head">
        <h2 id="trips-heading" className="panel__title">
          Evaluation trips
        </h2>
        {configured && (
          <button type="button" className="linkbtn" onClick={onReload} disabled={trips.status === 'loading'}>
            {trips.status === 'loading' ? 'Loading…' : 'Refresh'}
          </button>
        )}
      </div>

      {!configured && (
        <Notice tone="info" title="Connect to Laptop 1 first">
          Open <strong>Connection</strong> and enter the Laptop 1 API address to load the evaluation trips.
        </Notice>
      )}

      {configured && trips.status === 'loading' && trips.trips.length === 0 && (
        <ul className="triplist" aria-busy="true" aria-label="Loading trips">
          {Array.from({ length: 6 }, (_, i) => (
            <li key={i} className="trip trip--skeleton" />
          ))}
        </ul>
      )}

      {configured && trips.status === 'error' && (
        <Notice
          tone={isFallback ? 'warning' : 'error'}
          title={isFallback ? 'Showing development fallback trips' : `Could not load trips: ${errorTitle(trips.error)}`}
          action={
            <button type="button" className="btn btn--ghost btn--sm" onClick={onReload}>
              Retry
            </button>
          }
        >
          {isFallback
            ? 'These IDs come from VITE_DEV_FALLBACK_TRIPS, not from Laptop 1. Runs are still sent to the backend.'
            : errorMessage(trips.error)}
        </Notice>
      )}

      {configured && trips.status === 'ready' && trips.trips.length === 0 && (
        <Notice tone="warning" title="Laptop 1 returned no evaluation trips">
          Check that the evaluation partition is loaded on the backend, then refresh.
        </Notice>
      )}

      {trips.trips.length > 0 && (
        <>
          <p className="trips__caption">
            {isFallback ? (
              <span className="tag tag--warn">Dev fallback, not backend data</span>
            ) : (
              <>
                {trips.trips.length} trip{trips.trips.length === 1 ? '' : 's'} from Laptop 1
                {trips.partition ? ` (${trips.partition})` : ''}
              </>
            )}
          </p>
          <ul className="triplist" role="radiogroup" aria-label="Evaluation trips">
            {trips.trips.map((t) => {
              const outcome = lastOutcome(history, t.tripId);
              const meta = tripMeta(t);
              const isSel = selected === t.tripId;
              return (
                <li key={t.tripId}>
                  <button
                    type="button"
                    role="radio"
                    aria-checked={isSel}
                    className={`trip${isSel ? ' trip--selected' : ''}`}
                    onClick={() => onSelect(t.tripId)}
                    disabled={running}
                    data-testid={`trip-${t.tripId}`}
                  >
                    <span className="trip__id num">{t.label ?? t.tripId}</span>
                    {meta && <span className="trip__meta">{meta}</span>}
                    {outcome && (
                      <span className={`trip__done trip__done--${outcome}`} title={outcome === 'completed' ? 'Evaluated this session' : 'Last run failed'}>
                        {outcome === 'completed' ? 'Evaluated' : 'Failed'}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </>
      )}

      <button type="button" className="btn btn--primary btn--run" onClick={onRun} disabled={!canRun} aria-busy={running}>
        {running ? 'Running evaluation…' : selected ? `Run evaluation on ${selected}` : 'Select a trip to run'}
      </button>
    </section>
  );
}
