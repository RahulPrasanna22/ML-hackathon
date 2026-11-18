/**
 * Frontend domain types. These are the *validated* shapes the UI consumes.
 * The wire format (snake_case JSON) is documented in docs/API_CONTRACT.md and
 * converted by ./validate.ts.
 *
 * Coordinates are kept in the dataset's native [longitude, latitude] order and
 * only converted to Leaflet's [lat, lng] order at render time (lib/geo.ts).
 */

export type LonLat = [number, number];

export interface HealthStatus {
  status: string;
  modelLoaded: boolean | null;
  modelName: string | null;
  hmmEnabled: boolean | null;
  version: string | null;
  numEvaluationTrips: number | null;
}

export interface TripSummary {
  tripId: string;
  numObservations: number | null;
  durationS: number | null;
  label: string | null;
}

export interface TripList {
  partition: string | null;
  trips: TripSummary[];
}

export interface Observation {
  index: number;
  lon: number;
  lat: number;
  t: number | null;
  /** Arc predicted for this observation by the backend (never ground truth). */
  matchedArcId: number | null;
}

export interface RouteSegment {
  arcId: number | null;
  coordinates: LonLat[];
}

export interface RouteGeometry {
  arcIds: number[];
  segments: RouteSegment[];
}

/** Metric name -> value. `null` means the backend explicitly returned no value. */
export type MetricMap = Record<string, number | null>;

export interface HmmInfo {
  applied: boolean | null;
  status: string | null;
  message: string | null;
}

export type EvaluationStatus = 'queued' | 'running' | 'completed' | 'failed';

export interface EvaluationResult {
  evaluationId: string | null;
  tripId: string;
  status: EvaluationStatus;
  createdAt: string | null;
  error: string | null;
  numObservations: number | null;
  model: { name: string | null; version: string | null } | null;
  observations: Observation[];
  predictedRoute: RouteGeometry | null;
  roadNetwork: RouteGeometry | null;
  /** Only present if the backend chooses to send an authorised reference route. */
  referenceRoute: RouteGeometry | null;
  metrics: MetricMap;
  classifierOnlyMetrics: MetricMap | null;
  timing: MetricMap;
  hmm: HmmInfo | null;
  foliumMapUrl: string | null;
  foliumMapHtml: string | null;
  warnings: string[];
}
