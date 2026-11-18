import { useState, type FormEvent } from 'react';
import { PathSnapClient } from '../api/client';
import { errorMessage } from '../api/errors';
import { clientOptions, getEnvBaseUrl, normalizeBaseUrl, type BaseUrlSource } from '../config';
import type { BackendStatus } from '../hooks/useBackendStatus';

interface ConnectionPanelProps {
  baseUrl: string | null;
  source: BaseUrlSource;
  status: BackendStatus;
  onApply: (url: string | null) => void;
  onRetry: () => void;
}

type TestState = { kind: 'idle' } | { kind: 'testing' } | { kind: 'ok'; text: string } | { kind: 'fail'; text: string };

const SOURCE_TEXT: Record<BaseUrlSource, string> = {
  query: 'from the ?api= link parameter',
  saved: 'saved in this browser',
  env: 'from VITE_API_BASE_URL',
  none: 'not set',
};

export function ConnectionPanel({ baseUrl, source, status, onApply, onRetry }: ConnectionPanelProps) {
  const [draft, setDraft] = useState(baseUrl ?? '');
  const [test, setTest] = useState<TestState>({ kind: 'idle' });
  const envUrl = getEnvBaseUrl();
  const normalized = normalizeBaseUrl(draft);
  const origin = window.location.origin;

  async function runTest() {
    if (normalized === null) {
      setTest({ kind: 'fail', text: 'Enter an address such as http://192.168.1.20:8000' });
      return;
    }
    setTest({ kind: 'testing' });
    const started = performance.now();
    try {
      const health = await new PathSnapClient(normalized, clientOptions()).health();
      const ms = Math.round(performance.now() - started);
      setTest({ kind: 'ok', text: `Reached ${normalized} in ${ms} ms · status "${health.status}"${health.modelName ? ` · model ${health.modelName}` : ''}` });
    } catch (err) {
      setTest({ kind: 'fail', text: errorMessage(err) });
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    if (normalized === null) {
      setTest({ kind: 'fail', text: 'Enter an address such as http://192.168.1.20:8000' });
      return;
    }
    onApply(normalized);
    setTest({ kind: 'idle' });
  }

  return (
    <section id="connection-panel" className="connpanel" aria-label="Laptop 1 connection settings">
      <form className="connpanel__form" onSubmit={submit}>
        <label htmlFor="api-url" className="field-label">
          Laptop 1 API address
        </label>
        <div className="connpanel__row">
          <input
            id="api-url"
            className="input"
            type="text"
            inputMode="url"
            autoComplete="off"
            spellCheck={false}
            placeholder="http://192.168.1.20:8000"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <button type="button" className="btn btn--ghost" onClick={runTest} disabled={test.kind === 'testing'}>
            {test.kind === 'testing' ? 'Testing…' : 'Test connection'}
          </button>
          <button type="submit" className="btn btn--primary">
            Save and connect
          </button>
        </div>
        {test.kind === 'ok' && <p className="connpanel__result connpanel__result--ok">{test.text}</p>}
        {test.kind === 'fail' && <p className="connpanel__result connpanel__result--fail">{test.text}</p>}
        <p className="connpanel__meta">
          Current: <strong>{baseUrl ?? 'none'}</strong> ({SOURCE_TEXT[source]}).{' '}
          {status.state === 'offline' && (
            <button type="button" className="linkbtn" onClick={onRetry}>
              Retry now
            </button>
          )}
          {envUrl && envUrl !== baseUrl && (
            <button type="button" className="linkbtn" onClick={() => { setDraft(envUrl); onApply(null); }}>
              Reset to {envUrl}
            </button>
          )}
        </p>
      </form>
      <div className="connpanel__help">
        <p className="connpanel__help-title">Connecting the two laptops</p>
        <ul>
          <li>Put both laptops on the same Wi-Fi or hotspot.</li>
          <li>
            On Laptop 1 run <code>ipconfig</code> and use the <em>IPv4 Address</em> of the active adapter, e.g.{' '}
            <code>http://192.168.1.20:8000</code>.
          </li>
          <li>
            Start the API on Laptop 1 with <code>--host 0.0.0.0</code> and allow the port through Windows Firewall.
          </li>
          <li>
            The backend's CORS settings must allow this page's origin: <code>{origin}</code>.
          </li>
        </ul>
      </div>
    </section>
  );
}
