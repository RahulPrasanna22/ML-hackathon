import type { BackendStatus } from '../hooks/useBackendStatus';
import type { EvaluationPhase } from '../hooks/useEvaluation';
import type { HistoryEntry } from '../lib/history';
import { formatMetric, PRIMARY_METRICS, lookupMetric } from '../lib/metrics';

interface OverviewBarProps {
  status: BackendStatus;
  tripCount: number | null;
  tripsFromBackend: boolean;
  selectedTrip: string | null;
  phase: EvaluationPhase;
  runningTrip: string | null;
  latestSuccess: HistoryEntry | null;
}

const PHASE_TEXT: Record<EvaluationPhase, string> = {
  idle: 'Ready',
  running: 'Running',
  success: 'Completed',
  failed: 'Failed',
};

export function OverviewBar({ status, tripCount, tripsFromBackend, selectedTrip, phase, runningTrip, latestSuccess }: OverviewBarProps) {
  const health = status.health;
  const top1 = latestSuccess ? lookupMetric(latestSuccess.metrics, PRIMARY_METRICS[0]) : undefined;
  return (
    <dl className="overview" aria-label="Session overview">
      <div className="overview__cell">
        <dt>Model on Laptop 1</dt>
        <dd>
          {status.state === 'online'
            ? health?.modelName ?? (health?.modelLoaded === true ? 'Loaded' : health?.modelLoaded === false ? 'Not loaded' : 'Not reported')
            : '—'}
        </dd>
      </div>
      <div className="overview__cell">
        <dt>Evaluation trips</dt>
        <dd>
          {tripCount === null ? '—' : tripCount}
          {tripCount !== null && !tripsFromBackend && <span className="tag tag--warn">dev fallback</span>}
        </dd>
      </div>
      <div className="overview__cell">
        <dt>Selected trip</dt>
        <dd className="num">{selectedTrip ?? 'None'}</dd>
      </div>
      <div className="overview__cell">
        <dt>Execution</dt>
        <dd className={`phase phase--${phase}`}>
          {PHASE_TEXT[phase]}
          {phase === 'running' && runningTrip && <span className="num"> · {runningTrip}</span>}
        </dd>
      </div>
      <div className="overview__cell overview__cell--wide">
        <dt>Latest successful evaluation</dt>
        <dd>
          {latestSuccess ? (
            <>
              <span className="num">{latestSuccess.tripId}</span>
              <span className="muted"> · Top-1 {formatMetric(top1, 'fraction')}</span>
              <span className="muted"> · {new Date(latestSuccess.timestamp).toLocaleTimeString()}</span>
            </>
          ) : (
            'None yet'
          )}
        </dd>
      </div>
    </dl>
  );
}
