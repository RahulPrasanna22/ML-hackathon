/**
 * ===================== AUTOMATED-TEST FIXTURES ONLY =====================
 * Synthetic payloads that follow docs/API_CONTRACT.md. They are imported only
 * by *.test.ts(x) files, are never bundled into the app, and do not represent
 * real PathSnap trips, routes or model results. Coordinates are a tiny
 * made-up grid and every metric value is an arbitrary placeholder.
 * ========================================================================
 */

export const TEST_API = 'http://test-laptop1.invalid:8000';

export const healthOk = {
  status: 'ok',
  model_loaded: true,
  model_name: 'test-model',
  hmm_enabled: true,
  version: '0.0.0-test',
};

export const tripList = {
  partition: 'evaluator_test',
  trips: [
    { trip_id: 'T-001', num_observations: 4 },
    { trip_id: 'T-002', num_observations: 3 },
    { trip_id: 'T-003' },
  ],
};

/** Complete evaluation response with every optional section present. */
export function completeEvaluation(tripId = 'T-001') {
  return {
    evaluation_id: `eval-${tripId}`,
    trip_id: tripId,
    status: 'completed',
    created_at: '2026-01-01T10:00:00Z',
    model: { name: 'test-model', version: '1' },
    observations: [
      { index: 0, lon: 10.0, lat: 20.0, t: 1, matched_arc_id: 1 },
      { index: 1, lon: 10.001, lat: 20.001, t: 2, matched_arc_id: 1 },
      { index: 2, lon: 10.002, lat: 20.002, t: 3, matched_arc_id: 2 },
      { index: 3, lon: 10.003, lat: 20.003, t: 4, matched_arc_id: 2 },
    ],
    predicted_route: {
      arc_ids: [1, 2],
      segments: [
        { arc_id: 1, coordinates: [[10.0, 20.0], [10.0015, 20.0015]] },
        { arc_id: 2, coordinates: [[10.0015, 20.0015], [10.003, 20.003]] },
      ],
    },
    road_network: {
      segments: [
        { arc_id: 1, coordinates: [[10.0, 20.0], [10.0015, 20.0015]] },
        { arc_id: 2, coordinates: [[10.0015, 20.0015], [10.003, 20.003]] },
        { arc_id: 3, coordinates: [[10.0015, 20.0015], [10.0015, 20.004]] },
      ],
    },
    metrics: {
      top1_accuracy: 0.5,
      macro_f1: 0.25,
      route_continuity: 0.75,
      log_loss: 0.123456,
      candidate_miss_rate: 0.1,
      custom_score: 7,
    },
    classifier_only_metrics: { top1_accuracy: 0.4, macro_f1: 0.2, route_continuity: 0.5 },
    timing: { inference_ms: 42, hmm_ms: 12, total_ms: 60 },
    hmm: { applied: true, status: 'ok', message: null },
    warnings: [],
  };
}

/** Minimal valid response: no metrics, no timing, no HMM block. */
export function minimalEvaluation(tripId = 'T-002') {
  return {
    evaluation_id: `eval-min-${tripId}`,
    trip_id: tripId,
    status: 'completed',
    observations: [
      { index: 0, lon: 10.0, lat: 20.0 },
      { index: 1, lon: 10.001, lat: 20.001 },
    ],
    predicted_route: { arc_ids: [5], segments: [{ arc_id: 5, coordinates: [[10.0, 20.0], [10.001, 20.001]] }] },
  };
}
