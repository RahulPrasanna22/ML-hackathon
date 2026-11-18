/**
 * Model comparison: the recorded TRAIN cross-validation results of the candidate
 * classifiers, served by Laptop 1 (GET /models/comparison). These are NOT
 * evaluator-test results. Values missing from the recorded results are shown as
 * "not recorded"; nothing is computed or estimated here.
 */
import { useMemo, useState } from 'react';
import { errorMessage } from '../api/errors';
import type { ModelComparison as ModelComparisonData, ModelCvRow } from '../api/types';
import type { ModelComparisonState } from '../hooks/useModelComparison';

type SortKey = 'name' | 'cvTop1Accuracy' | 'cvMacroF1' | 'cvLogLoss' | 'fitTimeSAllTrain' | 'inferenceMsPerObservation' | 'selected';

interface Column {
  key: SortKey;
  label: string;
  title: string;
  /** true = larger is better (default sort direction when first clicked is "best first") */
  higherIsBetter: boolean;
  numeric: boolean;
}

const COLUMNS: Column[] = [
  { key: 'name', label: 'Model', title: 'Classifier and algorithm family', higherIsBetter: false, numeric: false },
  { key: 'cvTop1Accuracy', label: 'CV top-1 accuracy', title: 'Mean over 5 trip-grouped TRAIN folds, ± fold standard deviation', higherIsBetter: true, numeric: true },
  { key: 'cvMacroF1', label: 'CV macro-F1', title: 'Candidate-level macro-F1 (threshold 0.5), mean over 5 TRAIN folds', higherIsBetter: true, numeric: true },
  { key: 'cvLogLoss', label: 'CV log loss', title: 'Lower = better-calibrated probabilities (used as HMM emissions)', higherIsBetter: false, numeric: true },
  { key: 'fitTimeSAllTrain', label: 'Training time', title: 'Fit on all TRAIN rows (indicative; CPU was shared with other jobs)', higherIsBetter: false, numeric: true },
  { key: 'inferenceMsPerObservation', label: 'Inference / GPS fix', title: 'Classifier scoring time per GPS fix', higherIsBetter: false, numeric: true },
  { key: 'selected', label: 'Selection', title: 'Final classifier deployed on Laptop 1', higherIsBetter: true, numeric: false },
];

const NR = 'not recorded';
const pct = (v: number | null, digits = 2) => (v === null ? NR : `${(v * 100).toFixed(digits)}%`);
const dec = (v: number | null, digits = 4) => (v === null ? NR : v.toFixed(digits));
const secs = (v: number | null) => (v === null ? NR : v < 10 ? `${v.toFixed(1)} s` : `${Math.round(v)} s`);
const ms = (v: number | null) => (v === null ? NR : `${v < 0.1 ? v.toFixed(4) : v.toFixed(3)} ms`);

function sortValue(m: ModelCvRow, key: SortKey): number | string | null {
  if (key === 'name') return m.name.toLowerCase();
  if (key === 'selected') return m.selected ? 1 : 0;
  return m[key];
}

export function sortModels(models: ModelCvRow[], key: SortKey, dir: 'asc' | 'desc'): ModelCvRow[] {
  return [...models].sort((a, b) => {
    const va = sortValue(a, key);
    const vb = sortValue(b, key);
    if (va === null && vb === null) return 0;
    if (va === null) return 1; // missing values always last
    if (vb === null) return -1;
    const cmp = va < vb ? -1 : va > vb ? 1 : 0;
    return dir === 'asc' ? cmp : -cmp;
  });
}

/** Dot plot of CV top-1 accuracy with ±1 fold-std whiskers on an explicitly labelled zoomed axis. */
function AccuracyDotPlot({ models }: { models: ModelCvRow[] }) {
  const rows = models.filter((m) => m.cvTop1Accuracy !== null).sort((a, b) => b.cvTop1Accuracy! - a.cvTop1Accuracy!);
  if (rows.length === 0) return null;
  const lo = Math.floor((Math.min(...rows.map((m) => m.cvTop1Accuracy! - (m.cvTop1AccuracyStd ?? 0))) - 0.005) * 100) / 100;
  const hi = Math.ceil((Math.max(...rows.map((m) => m.cvTop1Accuracy! + (m.cvTop1AccuracyStd ?? 0))) + 0.005) * 100) / 100;
  const rowH = 28;
  const labelW = 170;
  const valueW = 72;
  const width = 640;
  const plotW = width - labelW - valueW;
  const axisH = 26;
  const x = (v: number) => labelW + ((v - lo) / (hi - lo)) * plotW;
  const step = Math.round((hi - lo) * 100) > 8 ? 2 : 1; // percentage points between grid lines
  const ticks: number[] = [];
  for (let t = Math.ceil(lo * 100); t <= Math.floor(hi * 100); t += step) ticks.push(t / 100);
  return (
    <svg
      className="chart modelplot"
      viewBox={`0 0 ${width} ${rows.length * rowH + axisH}`}
      role="img"
      aria-label={`Cross-validated top-1 accuracy per model, axis from ${Math.round(lo * 100)}% to ${Math.round(hi * 100)}%`}
      preserveAspectRatio="xMinYMin meet"
    >
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={0} y2={rows.length * rowH} className="modelplot__grid" />
          <text x={x(t)} y={rows.length * rowH + 16} textAnchor="middle" className="chart__value chart__value--muted">
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      {rows.map((m, i) => {
        const y = i * rowH + rowH / 2;
        const v = m.cvTop1Accuracy!;
        const s = m.cvTop1AccuracyStd ?? 0;
        return (
          <g key={m.key} className={m.selected ? 'modelplot__row modelplot__row--selected' : 'modelplot__row'}>
            <text x={0} y={y} dominantBaseline="middle" className="chart__label">
              {m.name}
              {m.selected ? ' ★' : ''}
            </text>
            <line x1={x(Math.max(lo, v - s))} x2={x(Math.min(hi, v + s))} y1={y} y2={y} className="modelplot__whisker" />
            <circle cx={x(v)} cy={y} r={m.selected ? 6 : 4.5} className="modelplot__dot" />
            <text x={width} y={y} dominantBaseline="middle" textAnchor="end" className="chart__value">
              {pct(v)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function ModelComparisonPanel({ state }: { state: ModelComparisonState }) {
  const [sort, setSort] = useState<{ key: SortKey; dir: 'asc' | 'desc' }>({ key: 'cvTop1Accuracy', dir: 'desc' });
  const data: ModelComparisonData | null = state.data;
  const rows = useMemo(() => (data ? sortModels(data.models, sort.key, sort.dir) : []), [data, sort]);

  function onSort(col: Column) {
    setSort((s) =>
      s.key === col.key ? { key: col.key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key: col.key, dir: col.higherIsBetter || col.key === 'name' ? (col.key === 'name' ? 'asc' : 'desc') : 'asc' },
    );
  }

  return (
    <section className="panel models" aria-labelledby="models-heading" data-testid="model-comparison">
      <div className="panel__head">
        <h2 className="panel__title" id="models-heading">
          Model comparison
        </h2>
        <span className="tag models__scope">TRAIN cross-validation · not evaluator-test results</span>
      </div>

      {state.status === 'idle' && <p className="muted">Connect to Laptop 1 to load the recorded model comparison.</p>}
      {state.status === 'loading' && <p className="muted">Loading the recorded comparison from Laptop 1…</p>}
      {state.status === 'error' && (
        <p className="muted" data-testid="model-comparison-error">
          Model comparison unavailable: {errorMessage(state.error)}
        </p>
      )}

      {data && (
        <>
          <p className="models__caption">
            {data.models.length} candidate classifiers evaluated with <strong>{data.evaluation}</strong>. Selection rule:{' '}
            {data.selectionMetric ?? NR}. These numbers come from model development on the TRAIN partition; official evaluator-test
            scores are produced per trip by <em>Run evaluation</em>.
          </p>
          <div className="models__tablewrap">
            <table className="models__table">
              <caption className="visually-hidden">TRAIN cross-validation comparison of {data.models.length} classifiers (sortable)</caption>
              <thead>
                <tr>
                  {COLUMNS.map((c) => (
                    <th
                      key={c.key}
                      scope="col"
                      className={c.numeric ? 'num' : undefined}
                      aria-sort={sort.key === c.key ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}
                    >
                      <button type="button" className="models__sort" onClick={() => onSort(c)} title={c.title}>
                        {c.label}
                        <span className="models__arrow" aria-hidden="true">
                          {sort.key === c.key ? (sort.dir === 'asc' ? '▲' : '▼') : '↕'}
                        </span>
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((m) => (
                  <tr key={m.key} className={m.selected ? 'models__row--selected' : undefined} data-testid={`model-row-${m.key}`}>
                    <th scope="row">
                      <span className="models__name">{m.name}</span>
                      {m.family && <span className="models__family">{m.family}</span>}
                    </th>
                    <td className="num">
                      {pct(m.cvTop1Accuracy)}
                      {m.cvTop1AccuracyStd !== null && <span className="models__std"> ± {pct(m.cvTop1AccuracyStd)}</span>}
                    </td>
                    <td className="num">{dec(m.cvMacroF1)}</td>
                    <td className="num">{dec(m.cvLogLoss)}</td>
                    <td className="num">{secs(m.fitTimeSAllTrain)}</td>
                    <td className="num">{ms(m.inferenceMsPerObservation)}</td>
                    <td>{m.selected ? <span className="tag tag--selected">Selected · final model</span> : <span className="muted">Not selected</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="models__foot">
            {data.baseline && (
              <p className="models__note">
                Reference baseline (no learning), {data.baseline.name}: CV top-1 accuracy {pct(data.baseline.cvTop1Accuracy)}, macro-F1{' '}
                {dec(data.baseline.cvMacroF1)}.
              </p>
            )}
            {data.tunedFinal && (
              <p className="models__note" data-testid="model-tuned-final">
                Deployed variant, {data.tunedFinal.name}: CV top-1 {pct(data.tunedFinal.cvTop1Accuracy)}
                {data.tunedFinal.cvTop1AccuracyStd !== null && ` ± ${pct(data.tunedFinal.cvTop1AccuracyStd)}`}, macro-F1{' '}
                {dec(data.tunedFinal.cvMacroF1)}, log loss {dec(data.tunedFinal.cvLogLoss)}. Bayesian optimisation changed top-1 by less
                than one fold standard deviation.
              </p>
            )}
          </div>

          <figure className="chartcard models__chart">
            <figcaption className="chartcard__title">CV top-1 accuracy per model (TRAIN, mean ± 1 fold std)</figcaption>
            <AccuracyDotPlot models={data.models} />
            <p className="chartcard__note">
              The axis is zoomed to the range of the results so that differences are visible. Overlapping whiskers mean the
              difference is within fold-to-fold variation. ★ = selected final classifier.
            </p>
          </figure>

          {data.notes.length > 0 && (
            <ul className="models__notes">
              {data.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
