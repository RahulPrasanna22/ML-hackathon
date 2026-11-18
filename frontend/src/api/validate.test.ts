import { describe, expect, it } from 'vitest';
import { completeEvaluation, minimalEvaluation } from '../test/fixtures';
import { ValidationError } from './errors';
import { parseEvaluation, parseHealth, parseTripList } from './validate';

describe('parseTripList', () => {
  it('accepts plain id arrays, numbers and de-duplicates', () => {
    expect(parseTripList(['a', 'b', 'a', 7]).trips.map((t) => t.tripId)).toEqual(['a', 'b', '7']);
  });

  it('rejects objects without a trips array and trips without ids', () => {
    expect(() => parseTripList({ items: [] })).toThrow(ValidationError);
    expect(() => parseTripList({ trips: [{ num_observations: 3 }] })).toThrow(/trip_id/);
  });
});

describe('parseHealth', () => {
  it('tolerates a bare {status}', () => {
    expect(parseHealth({ status: 'ok' })).toMatchObject({ status: 'ok', modelLoaded: null, modelName: null });
  });
  it('rejects wrong types', () => {
    expect(() => parseHealth({ status: 'ok', model_loaded: 'yes' })).toThrow(ValidationError);
  });
});

describe('parseEvaluation', () => {
  it('parses a complete response', () => {
    const r = parseEvaluation(completeEvaluation());
    expect(r.observations).toHaveLength(4);
    expect(r.observations[2].matchedArcId).toBe(2);
    expect(r.predictedRoute?.segments).toHaveLength(2);
    expect(r.roadNetwork?.segments).toHaveLength(3);
    expect(r.metrics.top1_accuracy).toBe(0.5);
    expect(r.classifierOnlyMetrics?.top1_accuracy).toBe(0.4);
    expect(r.hmm).toEqual({ applied: true, status: 'ok', message: null });
    expect(r.referenceRoute).toBeNull();
  });

  it('leaves missing optional sections empty instead of inventing values', () => {
    const r = parseEvaluation(minimalEvaluation());
    expect(r.metrics).toEqual({});
    expect(r.timing).toEqual({});
    expect(r.hmm).toBeNull();
    expect(r.classifierOnlyMetrics).toBeNull();
    expect(r.observations[0].t).toBeNull();
  });

  it('normalises status aliases and rejects unknown statuses', () => {
    expect(parseEvaluation({ trip_id: 'a', status: 'SUCCESS' }).status).toBe('completed');
    expect(parseEvaluation({ trip_id: 'a', status: 'error', error: 'x' }).status).toBe('failed');
    expect(parseEvaluation({ trip_id: 'a', error: 'boom' }).status).toBe('failed');
    expect(() => parseEvaluation({ trip_id: 'a', status: 'weird' })).toThrow(/unknown status/);
  });

  it('keeps invalid coordinates as NaN for the map layer to drop', () => {
    const r = parseEvaluation({ trip_id: 'a', observations: [{ lon: 'x', lat: 1 }, [10, null, 3]] });
    expect(Number.isNaN(r.observations[0].lon)).toBe(true);
    expect(Number.isNaN(r.observations[1].lat)).toBe(true);
    expect(r.observations[1].t).toBe(3);
  });

  it('turns string/null metric values into null and ignores nested objects', () => {
    const r = parseEvaluation({ trip_id: 'a', metrics: { top1_accuracy: 'N/A', macro_f1: null, nested: { a: 1 }, ok: 0.3 } });
    expect(r.metrics).toEqual({ top1_accuracy: null, macro_f1: null, ok: 0.3 });
  });

  it('accepts a single-polyline route and rejects malformed segments', () => {
    const r = parseEvaluation({ trip_id: 'a', predicted_route: { coordinates: [[1, 2], [3, 4]] } });
    expect(r.predictedRoute?.segments[0].coordinates).toEqual([[1, 2], [3, 4]]);
    expect(() => parseEvaluation({ trip_id: 'a', predicted_route: { segments: [{ arc_id: 1 }] } })).toThrow(/coordinates/);
    expect(() => parseEvaluation({ trip_id: 'a', predicted_route: { arc_ids: ['x'] } })).toThrow(ValidationError);
    expect(() => parseEvaluation({ trip_id: 'a', observations: 'nope' })).toThrow(ValidationError);
  });
});
