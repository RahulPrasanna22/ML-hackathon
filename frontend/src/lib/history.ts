/**
 * Session-level evaluation history.
 *
 * Persisted to sessionStorage (cleared when the tab closes). Only summary data
 * is stored: trip id, status, timestamp and the aggregate metrics the backend
 * returned. No geometry, per-observation data or reference/ground-truth
 * information is ever written to browser storage.
 */
import type { EvaluationResult, MetricMap } from '../api/types';
import { headlineInferenceMs } from './metrics';

export interface HistoryEntry {
  id: string;
  evaluationId: string | null;
  tripId: string;
  status: 'completed' | 'failed';
  timestamp: string;
  metrics: MetricMap;
  inferenceMs: number | null;
  roundTripMs: number | null;
  error: string | null;
}

const STORAGE_KEY = 'pathsnap.history.v1';
export const MAX_HISTORY = 50;

let counter = 0;
function localId(): string {
  counter += 1;
  return `${Date.now().toString(36)}-${counter}`;
}

export function entryFromResult(result: EvaluationResult, roundTripMs: number | null): HistoryEntry {
  return {
    id: localId(),
    evaluationId: result.evaluationId,
    tripId: result.tripId,
    status: result.status === 'completed' ? 'completed' : 'failed',
    timestamp: new Date().toISOString(),
    metrics: { ...result.metrics },
    inferenceMs: headlineInferenceMs(result.timing),
    roundTripMs,
    error: result.status === 'completed' ? null : result.error ?? 'Evaluation failed',
  };
}

export function entryFromError(tripId: string, error: string, roundTripMs: number | null): HistoryEntry {
  return {
    id: localId(),
    evaluationId: null,
    tripId,
    status: 'failed',
    timestamp: new Date().toISOString(),
    metrics: {},
    inferenceMs: null,
    roundTripMs,
    error,
  };
}

function isEntry(v: unknown): v is HistoryEntry {
  if (!v || typeof v !== 'object') return false;
  const e = v as Record<string, unknown>;
  return (
    typeof e.id === 'string' &&
    typeof e.tripId === 'string' &&
    (e.status === 'completed' || e.status === 'failed') &&
    typeof e.timestamp === 'string' &&
    typeof e.metrics === 'object' &&
    e.metrics !== null
  );
}

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isEntry).slice(0, MAX_HISTORY) : [];
  } catch {
    return [];
  }
}

export function saveHistory(entries: HistoryEntry[]): void {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
  } catch {
    /* storage unavailable - history still works in memory */
  }
}

export function clearStoredHistory(): void {
  try {
    window.sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
