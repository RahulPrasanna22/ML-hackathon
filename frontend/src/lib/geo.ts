/**
 * Converts validated API geometry into Leaflet-ready layers.
 * Invalid coordinates are dropped and counted - never repaired or invented.
 */
import type { EvaluationResult, LonLat, RouteGeometry } from '../api/types';

/** Leaflet order: [latitude, longitude]. */
export type LatLng = [number, number];
export type Bounds = [LatLng, LatLng];

export function isValidLonLat(lon: number, lat: number): boolean {
  return (
    Number.isFinite(lon) &&
    Number.isFinite(lat) &&
    lon >= -180 &&
    lon <= 180 &&
    lat >= -90 &&
    lat <= 90 &&
    !(lon === 0 && lat === 0) // null-island placeholder, not a real fix
  );
}

export const toLatLng = ([lon, lat]: LonLat): LatLng => [lat, lon];

export interface MapObservation {
  index: number;
  position: LatLng;
  t: number | null;
  matchedArcId: number | null;
}

export interface PreparedLine {
  arcId: number | null;
  positions: LatLng[];
}

export interface PreparedLayers {
  observations: MapObservation[];
  predicted: PreparedLine[];
  road: PreparedLine[];
  reference: PreparedLine[];
  bounds: Bounds | null;
  issues: {
    invalidObservations: number;
    invalidRoutePoints: number;
    droppedRouteSegments: number;
  };
}

function prepareLines(route: RouteGeometry | null, issues: PreparedLayers['issues'], countIssues: boolean): PreparedLine[] {
  if (!route) return [];
  const lines: PreparedLine[] = [];
  for (const seg of route.segments) {
    const positions: LatLng[] = [];
    for (const c of seg.coordinates) {
      if (isValidLonLat(c[0], c[1])) positions.push(toLatLng(c));
      else if (countIssues) issues.invalidRoutePoints += 1;
    }
    if (positions.length >= 2) lines.push({ arcId: seg.arcId, positions });
    else if (countIssues) issues.droppedRouteSegments += 1;
  }
  return lines;
}

export function computeBounds(groups: LatLng[][]): Bounds | null {
  let minLat = Infinity;
  let minLng = Infinity;
  let maxLat = -Infinity;
  let maxLng = -Infinity;
  for (const group of groups) {
    for (const [lat, lng] of group) {
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
    }
  }
  if (!Number.isFinite(minLat)) return null;
  return [
    [minLat, minLng],
    [maxLat, maxLng],
  ];
}

export function prepareMapLayers(result: EvaluationResult): PreparedLayers {
  const issues = { invalidObservations: 0, invalidRoutePoints: 0, droppedRouteSegments: 0 };

  const observations: MapObservation[] = [];
  for (const o of result.observations) {
    if (isValidLonLat(o.lon, o.lat)) {
      observations.push({ index: o.index, position: [o.lat, o.lon], t: o.t, matchedArcId: o.matchedArcId });
    } else {
      issues.invalidObservations += 1;
    }
  }

  const predicted = prepareLines(result.predictedRoute, issues, true);
  const reference = prepareLines(result.referenceRoute, issues, true);
  // Road-network context is background only; don't alarm the evaluator over it.
  const road = prepareLines(result.roadNetwork, issues, false);

  // Frame the trip itself; fall back to the road network only if nothing else exists.
  const focus = [observations.map((o) => o.position), ...predicted.map((l) => l.positions), ...reference.map((l) => l.positions)];
  let bounds = computeBounds(focus);
  if (!bounds) bounds = computeBounds(road.map((l) => l.positions));

  return { observations, predicted, road, reference, bounds, issues };
}
