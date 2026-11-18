import { describe, expect, it, vi } from 'vitest';
import { parseEvaluation } from '../api/validate';
import { getDevFallbackTrips, normalizeBaseUrl, resolveInitialBaseUrl } from '../config';
import { completeEvaluation } from '../test/fixtures';
import { prepareMapLayers } from './geo';
import { entryFromResult, loadHistory, saveHistory } from './history';
import { extraMetrics, formatMetric, headlineInferenceMs, lookupMetric, NA, PRIMARY_METRICS } from './metrics';

describe('geo.prepareMapLayers', () => {
  it('converts [lon, lat] to Leaflet [lat, lng] and frames the trip', () => {
    const layers = prepareMapLayers(parseEvaluation(completeEvaluation()));
    expect(layers.observations[0].position).toEqual([20.0, 10.0]);
    expect(layers.predicted).toHaveLength(2);
    expect(layers.road).toHaveLength(3);
    // Bounds come from GPS + predicted route, not the wider road network.
    expect(layers.bounds).toEqual([
      [20.0, 10.0],
      [20.003, 10.003],
    ]);
  });

  it('drops and counts invalid coordinates without repairing them', () => {
    const layers = prepareMapLayers(
      parseEvaluation({
        trip_id: 'a',
        observations: [{ lon: 10, lat: 20 }, { lon: 0, lat: 0 }, { lon: 200, lat: 20 }, { lon: 'x', lat: 1 }],
        predicted_route: { segments: [{ coordinates: [[10, 20], [null, 1]] }, { coordinates: [[10, 20], [10.1, 20.1]] }] },
      }),
    );
    expect(layers.observations).toHaveLength(1);
    expect(layers.issues).toEqual({ invalidObservations: 3, invalidRoutePoints: 1, droppedRouteSegments: 1 });
    expect(layers.predicted).toHaveLength(1);
  });

  it('returns null bounds when there is no valid geometry', () => {
    expect(prepareMapLayers(parseEvaluation({ trip_id: 'a' })).bounds).toBeNull();
  });
});

describe('metrics', () => {
  it('formats values and shows N/A for missing ones', () => {
    expect(formatMetric(0.9123, 'fraction')).toBe('91.2%');
    expect(formatMetric(1.5, 'fraction')).toBe('1.5000'); // out of contract range: shown verbatim
    expect(formatMetric(0.123456, 'loss')).toBe('0.1235');
    expect(formatMetric(42, 'ms')).toBe('42 ms');
    expect(formatMetric(2500, 'ms')).toBe('2.50 s');
    expect(formatMetric(2503, 'count')).toBe('2,503');
    expect(formatMetric(undefined, 'fraction')).toBe(NA);
    expect(formatMetric(null, 'ms')).toBe(NA);
    expect(formatMetric(Number.NaN, 'loss')).toBe(NA);
  });

  it('looks up aliases and distinguishes "not returned" from null', () => {
    const top1 = PRIMARY_METRICS[0];
    expect(lookupMetric({ observation_top1_accuracy: 0.7 }, top1)).toBe(0.7);
    expect(lookupMetric({ top1_accuracy: null }, top1)).toBeNull();
    expect(lookupMetric({}, top1)).toBeUndefined();
  });

  it('lists unknown metrics and picks the headline inference time', () => {
    expect(extraMetrics({ top1_accuracy: 1, custom: 2 })).toEqual([['custom', 2]]);
    expect(headlineInferenceMs({ total_ms: 90 })).toBe(90);
    expect(headlineInferenceMs({ inference_ms: 40, total_ms: 90 })).toBe(40);
    expect(headlineInferenceMs({})).toBeNull();
  });
});

describe('history', () => {
  it('stores only summary data - no geometry, observations or reference routes', () => {
    const result = parseEvaluation({ ...completeEvaluation(), reference_route: { segments: [{ coordinates: [[1, 2], [3, 4]] }] } });
    const entry = entryFromResult(result, 100);
    saveHistory([entry]);
    const raw = window.sessionStorage.getItem('pathsnap.history.v1')!;
    expect(raw).not.toContain('coordinates');
    expect(raw).not.toContain('observations');
    expect(raw).not.toContain('reference');
    expect(raw).not.toContain('matched_arc');
    expect(loadHistory()[0]).toMatchObject({ tripId: 'T-001', status: 'completed', inferenceMs: 42, roundTripMs: 100 });
  });

  it('ignores corrupted storage', () => {
    window.sessionStorage.setItem('pathsnap.history.v1', '{not json');
    expect(loadHistory()).toEqual([]);
    window.sessionStorage.setItem('pathsnap.history.v1', JSON.stringify([{ bogus: true }]));
    expect(loadHistory()).toEqual([]);
  });
});

describe('config', () => {
  it('normalises backend addresses', () => {
    expect(normalizeBaseUrl('192.168.1.20:8000')).toBe('http://192.168.1.20:8000');
    expect(normalizeBaseUrl(' http://10.0.0.5:8000/ ')).toBe('http://10.0.0.5:8000');
    expect(normalizeBaseUrl('https://host/api/')).toBe('https://host/api');
    expect(normalizeBaseUrl('/api/')).toBe('/api');
    expect(normalizeBaseUrl('ftp://x')).toBeNull();
    expect(normalizeBaseUrl('')).toBeNull();
  });

  it('resolves the base URL from query > saved > env > none', () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://env-host:8000');
    expect(resolveInitialBaseUrl('')).toEqual({ url: 'http://env-host:8000', source: 'env' });
    window.localStorage.setItem('pathsnap.apiBaseUrl', 'http://saved:8000');
    expect(resolveInitialBaseUrl('')).toEqual({ url: 'http://saved:8000', source: 'saved' });
    expect(resolveInitialBaseUrl('?api=10.1.1.1:9000')).toEqual({ url: 'http://10.1.1.1:9000', source: 'query' });
    window.localStorage.clear();
    vi.stubEnv('VITE_API_BASE_URL', '');
    expect(resolveInitialBaseUrl('')).toEqual({ url: null, source: 'none' });
  });

  it('only offers dev fallback trips in development', () => {
    vi.stubEnv('VITE_DEV_FALLBACK_TRIPS', ' a, b ,a,');
    expect(getDevFallbackTrips()).toEqual(['a', 'b']);
    vi.stubEnv('DEV', false);
    expect(getDevFallbackTrips()).toEqual([]);
  });
});
