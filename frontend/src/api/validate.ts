/**
 * Runtime validation of Laptop 1 responses.
 *
 * Structural problems (wrong top-level types, missing trip_id, ...) throw a
 * ValidationError so the UI never renders a half-understood response as a
 * success. Optional fields are tolerated when absent and become null/empty so
 * the UI shows "N/A". Individual bad coordinates are NOT fixed or invented
 * here: they are kept as NaN and filtered (and counted) by lib/geo.ts.
 */
import { ValidationError } from './errors';
import type {
  EvaluationResult,
  EvaluationStatus,
  HealthStatus,
  HmmInfo,
  LonLat,
  MetricMap,
  Observation,
  RouteGeometry,
  RouteSegment,
  TripList,
  TripSummary,
} from './types';

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v);
const isFiniteNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

function optString(o: Obj, key: string, path: string): string | null {
  const v = o[key];
  if (v === undefined || v === null) return null;
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return String(v);
  throw new ValidationError(`${path}.${key}`, 'expected a string');
}

function optNumber(o: Obj, key: string, path: string): number | null {
  const v = o[key];
  if (v === undefined || v === null) return null;
  if (isFiniteNum(v)) return v;
  throw new ValidationError(`${path}.${key}`, 'expected a finite number');
}

function optBool(o: Obj, key: string, path: string): boolean | null {
  const v = o[key];
  if (v === undefined || v === null) return null;
  if (typeof v === 'boolean') return v;
  throw new ValidationError(`${path}.${key}`, 'expected a boolean');
}

function optArray(o: Obj, key: string, path: string): unknown[] | null {
  const v = o[key];
  if (v === undefined || v === null) return null;
  if (Array.isArray(v)) return v;
  throw new ValidationError(`${path}.${key}`, 'expected an array');
}

function optObject(o: Obj, key: string, path: string): Obj | null {
  const v = o[key];
  if (v === undefined || v === null) return null;
  if (isObj(v)) return v;
  throw new ValidationError(`${path}.${key}`, 'expected an object');
}

function toId(v: unknown, path: string): string {
  if (typeof v === 'string' && v.trim()) return v.trim();
  if (isFiniteNum(v)) return String(v);
  throw new ValidationError(path, 'expected a non-empty string or number id');
}

/* ------------------------------------------------------------------ health */

export function parseHealth(raw: unknown): HealthStatus {
  if (!isObj(raw)) throw new ValidationError('health', 'expected a JSON object');
  return {
    status: optString(raw, 'status', 'health') ?? 'unknown',
    modelLoaded: optBool(raw, 'model_loaded', 'health'),
    modelName: optString(raw, 'model_name', 'health'),
    hmmEnabled: optBool(raw, 'hmm_enabled', 'health'),
    version: optString(raw, 'version', 'health'),
    numEvaluationTrips: optNumber(raw, 'num_evaluation_trips', 'health'),
  };
}

/* ------------------------------------------------------------------- trips */

function parseTrip(raw: unknown, path: string): TripSummary {
  if (typeof raw === 'string' || typeof raw === 'number') {
    return { tripId: toId(raw, path), numObservations: null, durationS: null, label: null };
  }
  if (!isObj(raw)) throw new ValidationError(path, 'expected a trip id or trip object');
  return {
    tripId: toId(raw.trip_id, `${path}.trip_id`),
    numObservations: optNumber(raw, 'num_observations', path),
    durationS: optNumber(raw, 'duration_s', path),
    label: optString(raw, 'label', path),
  };
}

export function parseTripList(raw: unknown): TripList {
  let items: unknown[];
  let partition: string | null = null;
  if (Array.isArray(raw)) {
    items = raw;
  } else if (isObj(raw)) {
    const arr = optArray(raw, 'trips', 'trips');
    if (arr === null) throw new ValidationError('trips', 'missing "trips" array');
    items = arr;
    partition = optString(raw, 'partition', 'trips');
  } else {
    throw new ValidationError('trips', 'expected an object with a "trips" array');
  }

  const seen = new Set<string>();
  const trips: TripSummary[] = [];
  items.forEach((item, i) => {
    const trip = parseTrip(item, `trips[${i}]`);
    if (!seen.has(trip.tripId)) {
      seen.add(trip.tripId);
      trips.push(trip);
    }
  });
  return { partition, trips };
}

/* -------------------------------------------------------------- evaluation */

const STATUS_ALIASES: Record<string, EvaluationStatus> = {
  queued: 'queued',
  pending: 'queued',
  running: 'running',
  processing: 'running',
  in_progress: 'running',
  completed: 'completed',
  complete: 'completed',
  success: 'completed',
  succeeded: 'completed',
  done: 'completed',
  ok: 'completed',
  failed: 'failed',
  failure: 'failed',
  error: 'failed',
};

function parseStatus(o: Obj): EvaluationStatus {
  const raw = o.status;
  if (raw === undefined || raw === null) {
    // No explicit status: only a response with an error is a failure.
    return o.error ? 'failed' : 'completed';
  }
  if (typeof raw !== 'string') throw new ValidationError('evaluation.status', 'expected a string');
  const status = STATUS_ALIASES[raw.trim().toLowerCase()];
  if (!status) throw new ValidationError('evaluation.status', `unknown status "${raw}"`);
  return status;
}

function coordOrNaN(v: unknown): number {
  return isFiniteNum(v) ? v : Number.NaN;
}

function parseLonLat(raw: unknown, path: string): LonLat {
  if (!Array.isArray(raw) || raw.length < 2) {
    throw new ValidationError(path, 'expected a [longitude, latitude] pair');
  }
  return [coordOrNaN(raw[0]), coordOrNaN(raw[1])];
}

function parseObservations(arr: unknown[] | null): Observation[] {
  if (!arr) return [];
  return arr.map((item, i): Observation => {
    const path = `evaluation.observations[${i}]`;
    if (Array.isArray(item)) {
      // [lon, lat, t?]
      return {
        index: i,
        lon: coordOrNaN(item[0]),
        lat: coordOrNaN(item[1]),
        t: isFiniteNum(item[2]) ? item[2] : null,
        matchedArcId: null,
      };
    }
    if (!isObj(item)) throw new ValidationError(path, 'expected an object or [lon, lat, t] array');
    const index = item.index;
    const arc = item.matched_arc_id;
    return {
      index: isFiniteNum(index) ? index : i,
      lon: coordOrNaN(item.lon),
      lat: coordOrNaN(item.lat),
      t: isFiniteNum(item.t) ? item.t : null,
      matchedArcId: isFiniteNum(arc) ? arc : null,
    };
  });
}

function parseSegment(raw: unknown, path: string): RouteSegment {
  if (!isObj(raw)) throw new ValidationError(path, 'expected a segment object');
  const coords = optArray(raw, 'coordinates', path);
  if (coords === null) throw new ValidationError(`${path}.coordinates`, 'missing coordinates');
  const arc = raw.arc_id;
  return {
    arcId: isFiniteNum(arc) ? arc : null,
    coordinates: coords.map((c, i) => parseLonLat(c, `${path}.coordinates[${i}]`)),
  };
}

export function parseRoute(raw: Obj | null, path: string): RouteGeometry | null {
  if (raw === null) return null;
  const arcIdsRaw = optArray(raw, 'arc_ids', path) ?? [];
  const arcIds = arcIdsRaw.map((v, i) => {
    if (!isFiniteNum(v)) throw new ValidationError(`${path}.arc_ids[${i}]`, 'expected a number');
    return v;
  });
  const segments = (optArray(raw, 'segments', path) ?? []).map((s, i) =>
    parseSegment(s, `${path}.segments[${i}]`),
  );
  // A single continuous polyline is also accepted.
  const line = optArray(raw, 'coordinates', path);
  if (line && line.length > 0) {
    segments.push({
      arcId: null,
      coordinates: line.map((c, i) => parseLonLat(c, `${path}.coordinates[${i}]`)),
    });
  }
  return { arcIds, segments };
}

function parseMetricMap(raw: Obj | null): MetricMap {
  const out: MetricMap = {};
  if (!raw) return out;
  for (const [k, v] of Object.entries(raw)) {
    if (isFiniteNum(v)) out[k] = v;
    else if (v === null || typeof v === 'string') out[k] = null; // "N/A", null, etc.
    // nested objects / booleans are ignored: not displayable as a single metric
  }
  return out;
}

function parseHmm(raw: Obj | null): HmmInfo | null {
  if (!raw) return null;
  return {
    applied: optBool(raw, 'applied', 'evaluation.hmm'),
    status: optString(raw, 'status', 'evaluation.hmm'),
    message: optString(raw, 'message', 'evaluation.hmm'),
  };
}

export function parseEvaluation(raw: unknown): EvaluationResult {
  if (!isObj(raw)) throw new ValidationError('evaluation', 'expected a JSON object');
  const p = 'evaluation';
  const evalId = raw.evaluation_id;
  const model = optObject(raw, 'model', p);
  const warnings = (optArray(raw, 'warnings', p) ?? []).filter(
    (w): w is string => typeof w === 'string',
  );

  return {
    evaluationId: evalId === undefined || evalId === null ? null : toId(evalId, `${p}.evaluation_id`),
    tripId: toId(raw.trip_id, `${p}.trip_id`),
    status: parseStatus(raw),
    createdAt: optString(raw, 'created_at', p),
    error: optString(raw, 'error', p),
    numObservations: optNumber(raw, 'num_observations', p),
    model: model
      ? { name: optString(model, 'name', `${p}.model`), version: optString(model, 'version', `${p}.model`) }
      : null,
    observations: parseObservations(optArray(raw, 'observations', p)),
    predictedRoute: parseRoute(optObject(raw, 'predicted_route', p), `${p}.predicted_route`),
    roadNetwork: parseRoute(optObject(raw, 'road_network', p), `${p}.road_network`),
    referenceRoute: parseRoute(optObject(raw, 'reference_route', p), `${p}.reference_route`),
    metrics: parseMetricMap(optObject(raw, 'metrics', p)),
    classifierOnlyMetrics: raw.classifier_only_metrics
      ? parseMetricMap(optObject(raw, 'classifier_only_metrics', p))
      : null,
    timing: parseMetricMap(optObject(raw, 'timing', p)),
    hmm: parseHmm(optObject(raw, 'hmm', p)),
    foliumMapUrl: optString(raw, 'folium_map_url', p),
    foliumMapHtml: optString(raw, 'folium_map_html', p),
    warnings,
  };
}
