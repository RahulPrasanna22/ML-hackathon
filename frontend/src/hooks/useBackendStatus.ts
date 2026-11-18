import { useCallback, useEffect, useRef, useState } from 'react';
import type { PathSnapClient } from '../api/client';
import { errorMessage } from '../api/errors';
import type { HealthStatus } from '../api/types';

export type ConnectionState = 'unconfigured' | 'checking' | 'online' | 'offline';

export interface BackendStatus {
  state: ConnectionState;
  health: HealthStatus | null;
  error: string | null;
  latencyMs: number | null;
  lastChecked: Date | null;
}

const POLL_MS = 15_000;

export function useBackendStatus(client: PathSnapClient | null) {
  const [status, setStatus] = useState<BackendStatus>({
    state: client ? 'checking' : 'unconfigured',
    health: null,
    error: null,
    latencyMs: null,
    lastChecked: null,
  });
  const inFlight = useRef<AbortController | null>(null);

  const check = useCallback(async () => {
    if (!client) {
      setStatus({ state: 'unconfigured', health: null, error: null, latencyMs: null, lastChecked: null });
      return;
    }
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setStatus((s) => ({ ...s, state: s.state === 'online' ? 'online' : 'checking' }));
    const started = performance.now();
    try {
      const health = await client.health(controller.signal);
      if (controller.signal.aborted) return;
      setStatus({ state: 'online', health, error: null, latencyMs: Math.round(performance.now() - started), lastChecked: new Date() });
    } catch (err) {
      if (controller.signal.aborted) return;
      setStatus({ state: 'offline', health: null, error: errorMessage(err), latencyMs: null, lastChecked: new Date() });
    }
  }, [client]);

  useEffect(() => {
    void check();
    if (!client) return;
    const id = setInterval(() => void check(), POLL_MS);
    return () => {
      clearInterval(id);
      inFlight.current?.abort();
    };
  }, [check, client]);

  return { status, check };
}
