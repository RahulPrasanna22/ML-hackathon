import { useState } from 'react';
import type { EvaluationResult } from '../api/types';
import { RouteMap } from './RouteMap';

interface MapPanelProps {
  result: EvaluationResult | null;
  running: boolean;
  /** Resolves a backend-relative URL (e.g. folium_map_url) against the API base. */
  resolveUrl: (pathOrUrl: string) => string;
}

type Tab = 'interactive' | 'folium';

export function MapPanel({ result, running, resolveUrl }: MapPanelProps) {
  const [tab, setTab] = useState<Tab>('interactive');
  const hasFolium = !!result && (!!result.foliumMapUrl || !!result.foliumMapHtml);
  const active: Tab = hasFolium ? tab : 'interactive';

  return (
    <section className="panel mappanel" aria-labelledby="map-heading">
      <div className="panel__head">
        <h2 id="map-heading" className="panel__title">
          {result ? (
            <>
              Route reconstruction, trip <span className="num">{result.tripId}</span>
            </>
          ) : (
            'Route reconstruction'
          )}
        </h2>
        {hasFolium && (
          <div className="tabs" role="tablist" aria-label="Map view">
            <button type="button" role="tab" aria-selected={active === 'interactive'} className="tab" onClick={() => setTab('interactive')}>
              Interactive map
            </button>
            <button type="button" role="tab" aria-selected={active === 'folium'} className="tab" onClick={() => setTab('folium')}>
              Folium map from Laptop 1
            </button>
          </div>
        )}
      </div>

      {!result && (
        <div className={`mapempty${running ? ' mapempty--running' : ''}`} data-testid="map-empty">
          <p className="mapempty__title">{running ? 'Waiting for the reconstructed route…' : 'No route yet'}</p>
          <p className="mapempty__text">
            {running
              ? 'The map appears as soon as Laptop 1 responds.'
              : 'Select an evaluation trip and run it. The raw GPS trace and the route reconstructed by Laptop 1 will be drawn here.'}
          </p>
        </div>
      )}

      {result && active === 'interactive' && <RouteMap key={result.evaluationId ?? result.tripId} result={result} />}

      {result && active === 'folium' && (
        <div className="routemap__frame">
          {result.foliumMapHtml ? (
            // srcdoc without allow-same-origin: the map runs in an opaque origin.
            <iframe title={`Folium map for trip ${result.tripId}`} className="folium" sandbox="allow-scripts" srcDoc={result.foliumMapHtml} />
          ) : (
            <iframe
              title={`Folium map for trip ${result.tripId}`}
              className="folium"
              sandbox="allow-scripts allow-same-origin"
              src={resolveUrl(result.foliumMapUrl!)}
            />
          )}
        </div>
      )}
    </section>
  );
}
