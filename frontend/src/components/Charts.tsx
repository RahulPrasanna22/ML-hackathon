/**
 * Small dependency-free SVG charts. Each chart renders only metrics that are
 * present in actual API responses; a chart with no data is not rendered.
 */
import type { ReactNode } from 'react';
import type { EvaluationResult } from '../api/types';
import type { HistoryEntry } from '../lib/history';
import { formatMetric, lookupMetric, PRIMARY_METRICS, SECONDARY_METRICS, TIMING_METRICS, type MetricDef } from '../lib/metrics';

const FRACTION_DEFS: MetricDef[] = [...PRIMARY_METRICS, ...SECONDARY_METRICS].filter((d) => d.kind === 'fraction');
const LOG_LOSS = PRIMARY_METRICS.find((d) => d.key === 'log_loss')!;

interface Bar {
  label: string;
  value: number;
  display: string;
}

/** Horizontal bars. `max` fixes the scale (1 for fractions); otherwise scaled to the largest value. */
function BarChart({ bars, max, tone, ariaLabel }: { bars: Bar[]; max?: number; tone: 'route' | 'gps' | 'neutral'; ariaLabel: string }) {
  const scale = max ?? Math.max(...bars.map((b) => b.value), Number.EPSILON);
  const rowH = 30;
  const labelW = 150;
  const valueW = 70;
  const width = 520;
  const plotW = width - labelW - valueW;
  return (
    <svg className={`chart chart--${tone}`} viewBox={`0 0 ${width} ${bars.length * rowH}`} role="img" aria-label={ariaLabel} preserveAspectRatio="xMinYMin meet">
      {bars.map((b, i) => {
        const w = Math.max(0, Math.min(1, b.value / scale)) * plotW;
        const y = i * rowH;
        return (
          <g key={b.label} transform={`translate(0 ${y})`}>
            <text x={0} y={rowH / 2} dominantBaseline="middle" className="chart__label">
              {b.label}
            </text>
            <rect x={labelW} y={8} width={plotW} height={rowH - 16} rx={3} className="chart__track" />
            <rect x={labelW} y={8} width={w} height={rowH - 16} rx={3} className="chart__bar" />
            <text x={width} y={rowH / 2} dominantBaseline="middle" textAnchor="end" className="chart__value">
              {b.display}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Grouped comparison: classifier-only vs classifier + HMM, per metric. */
function ComparisonChart({ rows }: { rows: Array<{ label: string; before: number; after: number; def: MetricDef }> }) {
  const rowH = 46;
  const labelW = 150;
  const valueW = 70;
  const width = 520;
  const plotW = width - labelW - valueW;
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${rows.length * rowH}`} role="img" aria-label="Classifier-only versus classifier plus HMM" preserveAspectRatio="xMinYMin meet">
      {rows.map((r, i) => {
        const y = i * rowH;
        const fb = Math.max(0, Math.min(1, r.before));
        const fa = Math.max(0, Math.min(1, r.after));
        return (
          <g key={r.label} transform={`translate(0 ${y})`}>
            <text x={0} y={rowH / 2} dominantBaseline="middle" className="chart__label">
              {r.label}
            </text>
            <rect x={labelW} y={8} width={fb * plotW} height={12} rx={2} className="chart__bar chart__bar--muted" />
            <rect x={labelW} y={24} width={fa * plotW} height={12} rx={2} className="chart__bar" />
            <text x={width} y={14} dominantBaseline="middle" textAnchor="end" className="chart__value chart__value--muted">
              {formatMetric(r.before, r.def.kind)}
            </text>
            <text x={width} y={30} dominantBaseline="middle" textAnchor="end" className="chart__value">
              {formatMetric(r.after, r.def.kind)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ChartCard({ title, children, note }: { title: string; children: ReactNode; note?: ReactNode }) {
  return (
    <figure className="chartcard">
      <figcaption className="chartcard__title">{title}</figcaption>
      {children}
      {note && <p className="chartcard__note">{note}</p>}
    </figure>
  );
}

const present = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v);

export function Charts({ result, history }: { result: EvaluationResult | null; history: HistoryEntry[] }) {
  const cards: ReactNode[] = [];

  if (result) {
    const fractionBars = FRACTION_DEFS.flatMap((d) => {
      const v = lookupMetric(result.metrics, d);
      return present(v) && v >= 0 && v <= 1 ? [{ label: d.label, value: v, display: formatMetric(v, d.kind) }] : [];
    });
    if (fractionBars.length) {
      cards.push(
        <ChartCard key="quality" title={`Route quality, trip ${result.tripId}`}>
          <BarChart bars={fractionBars} max={1} tone="route" ariaLabel="Route quality metrics" />
        </ChartCard>,
      );
    }

    const before = result.classifierOnlyMetrics;
    if (before) {
      const rows = FRACTION_DEFS.flatMap((d) => {
        const b = lookupMetric(before, d);
        const a = lookupMetric(result.metrics, d);
        return present(b) && present(a) ? [{ label: d.label, before: b, after: a, def: d }] : [];
      });
      const lb = lookupMetric(before, LOG_LOSS);
      const la = lookupMetric(result.metrics, LOG_LOSS);
      if (rows.length) {
        cards.push(
          <ChartCard
            key="hmm"
            title="Effect of HMM/Viterbi decoding"
            note={
              <>
                <span className="key key--muted" /> Classifier only <span className="key" /> Classifier + HMM
                {present(lb) && present(la) && <> · Log loss {formatMetric(lb, 'loss')} → {formatMetric(la, 'loss')}</>}
              </>
            }
          >
            <ComparisonChart rows={rows} />
          </ChartCard>,
        );
      }
    }

    const timingBars = TIMING_METRICS.flatMap((d) => {
      const v = lookupMetric(result.timing, d);
      return present(v) ? [{ label: d.label, value: v, display: formatMetric(v, 'ms') }] : [];
    });
    if (timingBars.length) {
      cards.push(
        <ChartCard key="timing" title="Server timing breakdown">
          <BarChart bars={timingBars} tone="neutral" ariaLabel="Server timing in milliseconds" />
        </ChartCard>,
      );
    }
  }

  // Session comparison: latest completed run per trip, Top-1 accuracy.
  const latestByTrip = new Map<string, HistoryEntry>();
  for (const h of history) {
    if (h.status === 'completed' && !latestByTrip.has(h.tripId)) latestByTrip.set(h.tripId, h);
  }
  const top1Def = PRIMARY_METRICS[0];
  const sessionBars = [...latestByTrip.values()].flatMap((h) => {
    const v = lookupMetric(h.metrics, top1Def);
    return present(v) && v >= 0 && v <= 1 ? [{ label: `Trip ${h.tripId}`, value: v, display: formatMetric(v, 'fraction') }] : [];
  });
  if (sessionBars.length >= 2) {
    sessionBars.sort((a, b) => a.label.localeCompare(b.label));
    cards.push(
      <ChartCard key="session" title="Top-1 accuracy across trips evaluated this session">
        <BarChart bars={sessionBars} max={1} tone="gps" ariaLabel="Top-1 accuracy per evaluated trip" />
      </ChartCard>,
    );
  }

  if (!cards.length) return null;
  return (
    <section className="panel charts" aria-labelledby="charts-heading" data-testid="charts">
      <div className="panel__head">
        <h2 id="charts-heading" className="panel__title">
          Evaluation charts
        </h2>
      </div>
      <div className="charts__grid">{cards}</div>
    </section>
  );
}
