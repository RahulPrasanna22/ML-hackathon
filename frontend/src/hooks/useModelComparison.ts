import { useCallback, useEffect, useRef, useState } from 'react';
import type { PathSnapClient } from '../api/client';
import { ApiError } from '../api/errors';
import type { ModelComparison } from '../api/types';

export interface ModelComparisonState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  data: ModelComparison | null;
  error: unknown;
}

const INITIAL: ModelComparisonState = { status: 'idle', data: null, error: null };

/** Loads the recorded TRAIN cross-validation comparison from Laptop 1 (GET /models/comparison). */
export function useModelComparison(client: PathSnapClient | null) {
  const [state, setState] = useState<ModelComparisonState>(INITIAL);
  const inFlight = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    inFlight.current?.abort();
    if (!client) {
      setState(INITIAL);
      return;
    }
    const controller = new AbortController();
    inFlight.current = controller;
    setState((s) => ({ ...s, status: 'loading', error: null }));
    try {
      const data = await client.getModelComparison(controller.signal);
      if (controller.signal.aborted) return;
      setState({ status: 'ready', data, error: null });
    } catch (err) {
      if (controller.signal.aborted || (err instanceof ApiError && err.kind === 'aborted')) return;
      setState({ status: 'error', data: null, error: err });
    }
  }, [client]);

  useEffect(() => {
    void load();
    return () => inFlight.current?.abort();
  }, [load]);

  return { ...state, reload: load };
}
