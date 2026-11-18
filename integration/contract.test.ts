/**
 * Contract check: responses captured from the REAL Laptop 1 FastAPI server
 * (integration/fixtures/*.json, written by integration/capture_fixtures.py) are parsed
 * with Laptop 2's own, unmodified validator (frontend/src/api/validate.ts).
 *
 * Run from frontend/:  npx vitest run --config ../integration/vitest.config.mts
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseEvaluation, parseHealth, parseTripList } from '../frontend/src/api/validate';

const here = dirname(fileURLToPath(import.meta.url));
const load = (name: string): unknown => JSON.parse(readFileSync(join(here, 'fixtures', name), 'utf-8'));

describe('real Laptop 1 responses satisfy the Laptop 2 contract', () => {
  it('GET /health', () => {
    const h = parseHealth(load('health.json'));
    expect(h.status).toBe('ok');
    expect(h.modelLoaded).toBe(true);
    expect(h.hmmEnabled).toBe(true);
    expect(h.modelName).toMatch(/HMM/);
    expect(h.numEvaluationTrips).toBe(10);
  });

  it('GET /evaluation/trips keeps zero-padded ids and has no route data', () => {
    const raw = load('trips.json') as Record<string, unknown>;
    const t = parseTripList(raw);
    expect(t.trips).toHaveLength(10);
    for (const trip of t.trips) {
      expect(trip.tripId).toMatch(/^\d{8}$/);
      expect(trip.numObservations).toBeGreaterThan(0);
    }
    expect(JSON.stringify(raw)).not.toMatch(/route|arc/i);
  });

  for (const tid of ['00000052', '00000007']) {
    it(`POST /evaluation/run ${tid}`, () => {
      const raw = load(`run_${tid}.json`) as Record<string, unknown>;
      const r = parseEvaluation(raw);
      expect(r.tripId).toBe(tid);
      expect(r.status).toBe('completed');
      expect(r.evaluationId).toBeTruthy();
      // observations: [lon, lat] in range, predicted arc present
      expect(r.observations.length).toBe(r.numObservations);
      for (const o of r.observations) {
        expect(Math.abs(o.lon)).toBeLessThanOrEqual(180);
        expect(Math.abs(o.lat)).toBeLessThanOrEqual(90);
      }
      expect(r.observations.filter((o) => o.matchedArcId === null)).toHaveLength(0);
      // predicted route geometry: one 2-point segment per arc, finite coordinates
      expect(r.predictedRoute).not.toBeNull();
      expect(r.predictedRoute!.segments.length).toBe(r.predictedRoute!.arcIds.length);
      for (const s of r.predictedRoute!.segments) {
        expect(s.coordinates).toHaveLength(2);
        for (const [lon, lat] of s.coordinates) {
          expect(Number.isFinite(lon) && Number.isFinite(lat)).toBe(true);
        }
      }
      expect(r.roadNetwork!.segments.length).toBeGreaterThan(0);
      // the first predicted segment is near the GPS track (sanity check of [lon, lat] order)
      const [lon0, lat0] = r.predictedRoute!.segments[0].coordinates[0];
      const d = Math.min(...r.observations.map((o) => Math.hypot(o.lon - lon0, o.lat - lat0)));
      expect(d).toBeLessThan(0.01);
      // metrics: known keys present, fractions in [0, 1]
      for (const k of ['top1_accuracy', 'macro_f1', 'route_continuity', 'candidate_miss_rate', 'precision', 'recall', 'sequence_accuracy']) {
        expect(r.metrics[k]).toBeGreaterThanOrEqual(0);
        expect(r.metrics[k]).toBeLessThanOrEqual(1);
      }
      expect(r.metrics.log_loss).toBeGreaterThan(0);
      expect(r.classifierOnlyMetrics!.top1_accuracy).toBeGreaterThan(0);
      expect(r.timing.inference_ms).toBeGreaterThan(0);
      expect(r.timing.hmm_ms).toBeGreaterThan(0);
      expect(r.hmm!.applied).toBe(true);
      // ground-truth isolation
      expect(r.referenceRoute).toBeNull();
      expect(JSON.stringify(raw)).not.toMatch(/reference_route|label_arc/);
    });
  }

  it('GET /evaluation/{id} returns the same result', () => {
    const run = parseEvaluation(load('run_00000052.json'));
    const got = parseEvaluation(load('get_00000052.json'));
    expect(got).toEqual(run);
  });

  it('errors use FastAPI {detail}', () => {
    expect((load('err_404.json') as { detail: string }).detail).toMatch(/Unknown trip_id/);
    expect(Array.isArray((load('err_422.json') as { detail: unknown[] }).detail)).toBe(true);
  });
});
