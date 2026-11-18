/**
 * Metric catalogue and formatting. Only values actually present in an API
 * response are ever displayed; anything missing is rendered as "N/A".
 */
import type { MetricMap } from '../api/types';

export type MetricKind = 'fraction' | 'loss' | 'ms' | 'count' | 'number';

export interface MetricDef {
  key: string;
  label: string;
  kind: MetricKind;
  /** Alternative key names accepted from the backend. */
  aliases?: string[];
  /** Higher is better (used to annotate comparisons). */
  higherIsBetter?: boolean;
}

export const PRIMARY_METRICS: MetricDef[] = [
  { key: 'top1_accuracy', label: 'Top-1 accuracy', kind: 'fraction', aliases: ['observation_top1_accuracy', 'top_1_accuracy', 'top1'], higherIsBetter: true },
  { key: 'macro_f1', label: 'Macro F1-score', kind: 'fraction', aliases: ['macro_f1_score', 'f1_macro'], higherIsBetter: true },
  { key: 'route_continuity', label: 'Route continuity', kind: 'fraction', aliases: ['continuity'], higherIsBetter: true },
  { key: 'log_loss', label: 'Log loss', kind: 'loss', aliases: ['logloss'], higherIsBetter: false },
];

export const SECONDARY_METRICS: MetricDef[] = [
  { key: 'accuracy', label: 'Accuracy', kind: 'fraction', higherIsBetter: true },
  { key: 'precision', label: 'Macro precision', kind: 'fraction', aliases: ['macro_precision'], higherIsBetter: true },
  { key: 'recall', label: 'Macro recall', kind: 'fraction', aliases: ['macro_recall'], higherIsBetter: true },
  { key: 'candidate_miss_rate', label: 'Candidate miss rate', kind: 'fraction', higherIsBetter: false },
  { key: 'sequence_accuracy', label: 'Sequence accuracy', kind: 'fraction', aliases: ['sequence_level_accuracy'], higherIsBetter: true },
];

export const TIMING_METRICS: MetricDef[] = [
  { key: 'inference_ms', label: 'Inference', kind: 'ms' },
  { key: 'candidate_generation_ms', label: 'Candidate generation', kind: 'ms' },
  { key: 'classifier_ms', label: 'Classifier', kind: 'ms' },
  { key: 'hmm_ms', label: 'HMM / Viterbi', kind: 'ms', aliases: ['viterbi_ms', 'decoding_ms'] },
  { key: 'total_ms', label: 'Total server time', kind: 'ms' },
];

const ALL_DEFS = [...PRIMARY_METRICS, ...SECONDARY_METRICS];

/** Finds a metric value by key or alias. undefined = not returned at all. */
export function lookupMetric(metrics: MetricMap | null | undefined, def: MetricDef): number | null | undefined {
  if (!metrics) return undefined;
  for (const k of [def.key, ...(def.aliases ?? [])]) {
    if (k in metrics) return metrics[k];
  }
  return undefined;
}

function claimedKeys(defs: MetricDef[]): Set<string> {
  return new Set(defs.flatMap((d) => [d.key, ...(d.aliases ?? [])]));
}

/** Metrics returned by the backend that are not in the known catalogue. */
export function extraMetrics(metrics: MetricMap, defs: MetricDef[] = ALL_DEFS): Array<[string, number | null]> {
  const claimed = claimedKeys(defs);
  return Object.entries(metrics).filter(([k]) => !claimed.has(k));
}

export function humanizeKey(key: string): string {
  const s = key.replace(/_ms$/, '').replace(/_/g, ' ').trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function inferKind(key: string): MetricKind {
  if (key.endsWith('_ms')) return 'ms';
  if (key.startsWith('num_') || key.endsWith('_count')) return 'count';
  return 'number';
}

export const NA = 'N/A';

export function formatMetric(value: number | null | undefined, kind: MetricKind): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NA;
  switch (kind) {
    case 'fraction':
      // Contract: fractions are in [0, 1]. Anything else is shown verbatim, not rescaled.
      return value >= 0 && value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
    case 'loss':
      return value.toFixed(4);
    case 'ms':
      return value < 1000 ? `${value.toFixed(value < 10 ? 1 : 0)} ms` : `${(value / 1000).toFixed(2)} s`;
    case 'count':
      return Math.round(value).toLocaleString('en-US');
    case 'number':
      return Number.isInteger(value) ? value.toLocaleString('en-US') : value.toFixed(4);
  }
}

/** Inference time to headline: explicit inference_ms, else total server time. */
export function headlineInferenceMs(timing: MetricMap): number | null {
  const inf = lookupMetric(timing, TIMING_METRICS[0]);
  if (inf !== undefined && inf !== null) return inf;
  const total = lookupMetric(timing, TIMING_METRICS[4]);
  return total ?? null;
}
