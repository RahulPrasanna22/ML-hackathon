import type { EvaluationResult } from '../api/types';
import {
  extraMetrics,
  formatMetric,
  headlineInferenceMs,
  humanizeKey,
  inferKind,
  lookupMetric,
  NA,
  PRIMARY_METRICS,
  SECONDARY_METRICS,
  TIMING_METRICS,
} from '../lib/metrics';
import { Notice } from './Notice';

interface ResultsPanelProps {
  result: EvaluationResult;
  roundTripMs: number | null;
}

function Tile({ label, value, hint, accent }: { label: string; value: string; hint?: string; accent?: 'route' | 'gps' }) {
  return (
    <div className={`tile${value === NA ? ' tile--na' : ''}${accent ? ` tile--${accent}` : ''}`}>
      <dt className="tile__label">{label}</dt>
      <dd className="tile__value num">{value}</dd>
      {hint && <dd className="tile__hint">{hint}</dd>}
    </div>
  );
}

function hmmText(result: EvaluationResult): { value: string; hint?: string } {
  const h = result.hmm;
  if (!h) return { value: NA, hint: 'Not reported' };
  const value = h.applied === true ? 'Applied' : h.applied === false ? 'Not applied' : h.status ?? NA;
  const hint = [h.applied !== null ? h.status : null, h.message].filter(Boolean).join(' · ') || undefined;
  return { value, hint };
}

export function ResultsPanel({ result, roundTripMs }: ResultsPanelProps) {
  const numObs = result.observations.length > 0 ? result.observations.length : result.numObservations;
  const route = result.predictedRoute;
  const numSegments = route ? (route.arcIds.length > 0 ? route.arcIds.length : route.segments.length) : null;
  const inference = headlineInferenceMs(result.timing);
  const hmm = hmmText(result);

  const secondary = SECONDARY_METRICS.map((d) => [d, lookupMetric(result.metrics, d)] as const).filter(([, v]) => v !== undefined);
  const extras = extraMetrics(result.metrics);
  const timingRows = [
    ...TIMING_METRICS.map((d) => [d.label, lookupMetric(result.timing, d)] as const).filter(([, v]) => v !== undefined),
    ...extraMetrics(result.timing, TIMING_METRICS).map(([k, v]) => [humanizeKey(k), v] as const),
  ];

  return (
    <section className="panel results" aria-labelledby="results-heading" data-testid="results">
      <div className="panel__head">
        <h2 id="results-heading" className="panel__title">
          Results for trip <span className="num">{result.tripId}</span>
        </h2>
        <p className="results__meta">
          {result.model?.name && <span>Model {result.model.name}{result.model.version ? ` v${result.model.version}` : ''}</span>}
          {result.evaluationId && <span>Evaluation {result.evaluationId}</span>}
          {result.createdAt && <span>{new Date(result.createdAt).toLocaleString()}</span>}
        </p>
      </div>

      {result.warnings.length > 0 && (
        <Notice tone="warning" title="Laptop 1 returned warnings">
          <ul className="plainlist">
            {result.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Notice>
      )}

      <dl className="tiles">
        <Tile label="GPS observations" value={formatMetric(numObs, 'count')} accent="gps" />
        <Tile label="Predicted road segments" value={formatMetric(numSegments, 'count')} accent="route" />
        {PRIMARY_METRICS.map((d) => (
          <Tile key={d.key} label={d.label} value={formatMetric(lookupMetric(result.metrics, d), d.kind)} />
        ))}
        <Tile
          label="Inference time"
          value={formatMetric(inference, 'ms')}
          hint={roundTripMs !== null ? `${formatMetric(roundTripMs, 'ms')} round trip from this browser` : undefined}
        />
        <Tile label="HMM decoding" value={hmm.value} hint={hmm.hint} />
      </dl>

      {(secondary.length > 0 || extras.length > 0 || timingRows.length > 0) && (
        <div className="results__more">
          {(secondary.length > 0 || extras.length > 0) && (
            <table className="kv">
              <caption>Additional metrics from Laptop 1</caption>
              <tbody>
                {secondary.map(([d, v]) => (
                  <tr key={d.key}>
                    <th scope="row">{d.label}</th>
                    <td className="num">{formatMetric(v, d.kind)}</td>
                  </tr>
                ))}
                {extras.map(([k, v]) => (
                  <tr key={k}>
                    <th scope="row">{humanizeKey(k)}</th>
                    <td className="num">{formatMetric(v, inferKind(k))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {timingRows.length > 0 && (
            <table className="kv">
              <caption>Server timing</caption>
              <tbody>
                {timingRows.map(([label, v]) => (
                  <tr key={label}>
                    <th scope="row">{label}</th>
                    <td className="num">{formatMetric(v, 'ms')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </section>
  );
}
