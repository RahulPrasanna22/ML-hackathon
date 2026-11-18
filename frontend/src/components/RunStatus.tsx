import { useEffect, useState } from 'react';
import { errorTitle } from '../api/errors';
import type { EvaluationState } from '../hooks/useEvaluation';
import { formatMetric } from '../lib/metrics';
import { Notice } from './Notice';

interface RunStatusProps {
  evaluation: EvaluationState & { errorText: string | null };
  onCancel: () => void;
  onRetry: () => void;
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, []);
  return <span className="num">{((now - since) / 1000).toFixed(1)} s</span>;
}

export function RunStatus({ evaluation, onCancel, onRetry }: RunStatusProps) {
  const { phase, tripId } = evaluation;
  if (phase === 'idle') return null;

  if (phase === 'running') {
    return (
      <div className="runstatus runstatus--running" role="status" aria-live="polite" data-testid="run-status">
        <div className="runstatus__pulse" aria-hidden="true" />
        <div className="runstatus__body">
          <p className="runstatus__title">
            Evaluating trip <span className="num">{tripId}</span>
          </p>
          <p className="runstatus__text">
            Waiting for Laptop 1 to return the reconstructed route
            {evaluation.backendStatus ? ` (backend status: ${evaluation.backendStatus})` : ''}.{' '}
            {evaluation.startedAt && <Elapsed since={evaluation.startedAt} />}
          </p>
        </div>
        <button type="button" className="btn btn--ghost btn--sm" onClick={onCancel}>
          Cancel
        </button>
      </div>
    );
  }

  if (phase === 'failed') {
    return (
      <Notice
        tone="error"
        title={`Evaluation of trip ${tripId ?? ''} failed: ${errorTitle(evaluation.error)}`}
        action={
          tripId ? (
            <button type="button" className="btn btn--ghost btn--sm" onClick={onRetry}>
              Run again
            </button>
          ) : undefined
        }
      >
        {evaluation.errorText}
      </Notice>
    );
  }

  return (
    <p className="runstatus runstatus--done" role="status" data-testid="run-status">
      Trip <span className="num">{tripId}</span> evaluated
      {evaluation.roundTripMs !== null && <> in {formatMetric(evaluation.roundTripMs, 'ms')} (browser round trip)</>}.
    </p>
  );
}
