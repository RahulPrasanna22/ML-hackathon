import { describeBaseUrl } from '../config';
import type { BackendStatus } from '../hooks/useBackendStatus';

const STATE_LABEL: Record<BackendStatus['state'], string> = {
  unconfigured: 'Not configured',
  checking: 'Connecting…',
  online: 'Connected',
  offline: 'Offline',
};

interface HeaderProps {
  baseUrl: string | null;
  status: BackendStatus;
  settingsOpen: boolean;
  onToggleSettings: () => void;
}

export function BrandMark() {
  return (
    <svg className="brand__mark" viewBox="0 0 40 40" aria-hidden="true">
      <path d="M5 31 L15 19 L23 23 L35 9" fill="none" stroke="var(--c-predicted)" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="6.5" cy="25" r="2.2" fill="var(--c-gps)" />
      <circle cx="13" cy="14.5" r="2.2" fill="var(--c-gps)" />
      <circle cx="25" cy="28" r="2.2" fill="var(--c-gps)" />
      <circle cx="31" cy="14" r="2.2" fill="var(--c-gps)" />
    </svg>
  );
}

export function Header({ baseUrl, status, settingsOpen, onToggleSettings }: HeaderProps) {
  return (
    <header className="topbar">
      <div className="brand">
        <BrandMark />
        <div>
          <p className="brand__name">PathSnap</p>
          <h1 className="brand__title">Distributed Map Matching &amp; Route Reconstruction</h1>
        </div>
      </div>
      <div className="topbar__right">
        <div className={`conn conn--${status.state}`} aria-live="polite" data-testid="connection-status">
          <span className="conn__dot" aria-hidden="true" />
          <span className="conn__label">
            <span className="conn__state">Laptop 1 API · {STATE_LABEL[status.state]}</span>
            <span className="conn__host">
              {describeBaseUrl(baseUrl)}
              {status.state === 'online' && status.latencyMs !== null && ` · ${status.latencyMs} ms`}
            </span>
          </span>
        </div>
        <button type="button" className="btn btn--ghost" aria-expanded={settingsOpen} aria-controls="connection-panel" onClick={onToggleSettings}>
          {settingsOpen ? 'Close connection' : 'Connection'}
        </button>
      </div>
    </header>
  );
}
