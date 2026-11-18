import type { HistoryEntry } from '../lib/history';
import { formatMetric, lookupMetric, PRIMARY_METRICS } from '../lib/metrics';

interface HistoryPanelProps {
  entries: HistoryEntry[];
  activeEvaluationId: string | null;
  onView: (entry: HistoryEntry) => void;
  onClear: () => void;
  disabled: boolean;
}

const [TOP1, F1, CONT] = PRIMARY_METRICS;

export function HistoryPanel({ entries, activeEvaluationId, onView, onClear, disabled }: HistoryPanelProps) {
  return (
    <section className="panel history" aria-labelledby="history-heading" data-testid="history">
      <div className="panel__head">
        <h2 id="history-heading" className="panel__title">
          Session history
        </h2>
        {entries.length > 0 && (
          <button type="button" className="linkbtn" onClick={onClear} disabled={disabled}>
            Clear
          </button>
        )}
      </div>
      {entries.length === 0 ? (
        <p className="muted small">Runs from this browser tab are listed here. Only summary metrics are kept, never routes or reference labels.</p>
      ) : (
        <ol className="historylist">
          {entries.map((e) => {
            const active = e.evaluationId !== null && e.evaluationId === activeEvaluationId;
            return (
              <li key={e.id}>
                <button
                  type="button"
                  className={`hist hist--${e.status}${active ? ' hist--active' : ''}`}
                  onClick={() => onView(e)}
                  disabled={disabled || e.status !== 'completed' || e.evaluationId === null}
                  title={e.status === 'failed' ? e.error ?? 'Failed' : e.evaluationId ? 'Show this result' : 'No evaluation id to reload'}
                >
                  <span className="hist__row">
                    <span className="hist__trip num">{e.tripId}</span>
                    <span className="hist__time">{new Date(e.timestamp).toLocaleTimeString()}</span>
                  </span>
                  {e.status === 'completed' ? (
                    <span className="hist__metrics num">
                      Top-1 {formatMetric(lookupMetric(e.metrics, TOP1), 'fraction')} · F1 {formatMetric(lookupMetric(e.metrics, F1), 'fraction')} · Cont.{' '}
                      {formatMetric(lookupMetric(e.metrics, CONT), 'fraction')}
                    </span>
                  ) : (
                    <span className="hist__error">{e.error ?? 'Failed'}</span>
                  )}
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
