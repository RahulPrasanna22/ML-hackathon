import { useCallback, useEffect, useRef, useState } from 'react';
import type { PathSnapClient } from '../api/client';
import { ApiError } from '../api/errors';
import type { TripSummary } from '../api/types';
import { getDevFallbackTrips } from '../config';

export type TripsSource = 'backend' | 'dev-fallback';

export interface TripsState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  trips: TripSummary[];
  partition: string | null;
  source: TripsSource;
  error: unknown;
}

const INITIAL: TripsState = { status: 'idle', trips: [], partition: null, source: 'backend', error: null };

export function useTrips(client: PathSnapClient | null) {
  const [state, setState] = useState<TripsState>(INITIAL);
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
      const list = await client.listTrips(controller.signal);
      if (controller.signal.aborted) return;
      setState({ status: 'ready', trips: list.trips, partition: list.partition, source: 'backend', error: null });
    } catch (err) {
      if (controller.signal.aborted || (err instanceof ApiError && err.kind === 'aborted')) return;
      const fallback = getDevFallbackTrips();
      if (fallback.length > 0) {
        // Development only: keep the error visible and label the list as fallback.
        setState({
          status: 'error',
          trips: fallback.map((tripId) => ({ tripId, numObservations: null, durationS: null, label: null })),
          partition: null,
          source: 'dev-fallback',
          error: err,
        });
      } else {
        setState({ status: 'error', trips: [], partition: null, source: 'backend', error: err });
      }
    }
  }, [client]);

  useEffect(() => {
    void load();
    return () => inFlight.current?.abort();
  }, [load]);

  return { ...state, reload: load };
}
