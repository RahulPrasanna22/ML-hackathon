import { useCallback, useEffect, useRef, useState } from 'react';
import type { PathSnapClient } from '../api/client';
import { ApiError, errorMessage, EvaluationFailedError } from '../api/errors';
import type { EvaluationResult } from '../api/types';

export type EvaluationPhase = 'idle' | 'running' | 'success' | 'failed';

export interface EvaluationState {
  phase: EvaluationPhase;
  tripId: string | null;
  result: EvaluationResult | null;
  error: unknown;
  startedAt: number | null;
  /** Browser-measured round trip for the run (request sent -> final response). */
  roundTripMs: number | null;
  backendStatus: string | null;
}

export interface EvaluationCallbacks {
  onFinished?: (outcome: { tripId: string; result: EvaluationResult | null; error: unknown; roundTripMs: number }) => void;
}

const INITIAL: EvaluationState = {
  phase: 'idle',
  tripId: null,
  result: null,
  error: null,
  startedAt: null,
  roundTripMs: null,
  backendStatus: null,
};

export function useEvaluation(client: PathSnapClient | null, callbacks: EvaluationCallbacks = {}) {
  const [state, setState] = useState<EvaluationState>(INITIAL);
  const running = useRef(false);
  const controllerRef = useRef<AbortController | null>(null);
  const cbRef = useRef(callbacks);
  cbRef.current = callbacks;

  const run = useCallback(
    async (tripId: string) => {
      // Duplicate-submission guard: synchronous ref check, independent of render timing.
      if (running.current) return;
      if (!client) {
        setState({ ...INITIAL, phase: 'failed', tripId, error: new ApiError('config', 'Set the Laptop 1 API address before running an evaluation.') });
        return;
      }
      running.current = true;
      const controller = new AbortController();
      controllerRef.current = controller;
      const started = performance.now();
      setState({ ...INITIAL, phase: 'running', tripId, startedAt: Date.now(), backendStatus: null });

      let result: EvaluationResult | null = null;
      let error: unknown = null;
      try {
        result = await client.runEvaluation(tripId, {
          signal: controller.signal,
          onProgress: (r) => setState((s) => ({ ...s, backendStatus: r.status })),
        });
      } catch (err) {
        error = err;
      }
      const roundTripMs = Math.round(performance.now() - started);
      running.current = false;
      controllerRef.current = null;

      if (error instanceof ApiError && error.kind === 'aborted') {
        setState({ ...INITIAL, tripId });
        return;
      }
      if (result && result.status === 'completed') {
        setState({ ...INITIAL, phase: 'success', tripId, result, roundTripMs });
      } else {
        const failure = error ?? new EvaluationFailedError(result?.error ?? `Laptop 1 reported that the evaluation of trip ${tripId} failed.`);
        setState({ ...INITIAL, phase: 'failed', tripId, result, error: failure, roundTripMs });
        error = failure;
      }
      cbRef.current.onFinished?.({ tripId, result, error, roundTripMs });
    },
    [client],
  );

  const cancel = useCallback(() => controllerRef.current?.abort(), []);

  /** Show a previously obtained result (e.g. from history). */
  const show = useCallback((result: EvaluationResult) => {
    if (running.current) return;
    setState({ ...INITIAL, phase: result.status === 'completed' ? 'success' : 'failed', tripId: result.tripId, result, error: result.status === 'completed' ? null : new EvaluationFailedError(result.error ?? 'Evaluation failed') });
  }, []);

  useEffect(() => () => controllerRef.current?.abort(), []);

  return { ...state, run, cancel, show, errorText: state.error ? errorMessage(state.error) : null };
}
