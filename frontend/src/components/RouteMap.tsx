/**
 * Interactive route map (Leaflet - the same engine Folium generates).
 * Every layer is drawn from the API response; nothing is synthesised here.
 */
import 'leaflet/dist/leaflet.css';
import { useEffect, useMemo } from 'react';
import { CircleMarker, LayersControl, LayerGroup, MapContainer, Polyline, TileLayer, Tooltip, useMap } from 'react-leaflet';
import type { EvaluationResult } from '../api/types';
import { prepareMapLayers, type Bounds, type PreparedLayers } from '../lib/geo';
import { Notice } from './Notice';

const OSM_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

const COLORS = {
  gps: '#f2b544',
  predicted: '#3fd0e0',
  road: '#6f7f9c',
  reference: '#e46fb5',
  start: '#4cc38a',
  end: '#f2675a',
};

/** Above this many points, markers are thinned for rendering only (the track line keeps every point). */
const MAX_MARKERS = 4000;

function FitBounds({ bounds }: { bounds: Bounds | null }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) map.fitBounds(bounds, { padding: [28, 28], maxZoom: 18 });
  }, [map, bounds]);
  return null;
}

function Legend({ layers }: { layers: PreparedLayers }) {
  const items: Array<{ key: string; label: string; swatch: string }> = [];
  if (layers.observations.length) items.push({ key: 'gps', label: `Raw GPS observations (${layers.observations.length.toLocaleString('en-US')})`, swatch: 'dot-gps' });
  if (layers.predicted.length) items.push({ key: 'pred', label: 'Predicted route (classifier + HMM)', swatch: 'line-predicted' });
  if (layers.road.length) items.push({ key: 'road', label: 'Road network', swatch: 'line-road' });
  if (layers.reference.length) items.push({ key: 'ref', label: 'Reference route (from backend)', swatch: 'line-reference' });
  if (layers.observations.length) items.push({ key: 'se', label: 'Trip start / end', swatch: 'dot-startend' });
  if (!items.length) return null;
  return (
    <div className="legend" data-testid="map-legend">
      <ul>
        {items.map((it) => (
          <li key={it.key}>
            <span className={`swatch swatch--${it.swatch}`} aria-hidden="true" />
            {it.label}
          </li>
        ))}
      </ul>
    </div>
  );
}

function formatT(t: number | null) {
  return t === null ? '' : `, t = ${t} s`;
}

export function RouteMap({ result }: { result: EvaluationResult }) {
  const layers = useMemo(() => prepareMapLayers(result), [result]);
  const { observations, predicted, road, reference, bounds, issues } = layers;

  const stride = Math.max(1, Math.ceil(observations.length / MAX_MARKERS));
  const markerObs = stride === 1 ? observations : observations.filter((_, i) => i % stride === 0);
  const track = observations.map((o) => o.position);
  const first = observations[0];
  const last = observations[observations.length - 1];

  const issueText: string[] = [];
  if (issues.invalidObservations) issueText.push(`${issues.invalidObservations} GPS observation(s) had invalid coordinates and were not plotted.`);
  if (issues.invalidRoutePoints) issueText.push(`${issues.invalidRoutePoints} route point(s) had invalid coordinates and were skipped.`);
  if (issues.droppedRouteSegments) issueText.push(`${issues.droppedRouteSegments} route segment(s) had fewer than two valid points and were not drawn.`);
  if (stride > 1) issueText.push(`Showing every ${stride}th GPS marker for performance; the GPS track line uses all points.`);

  if (!bounds) {
    return (
      <Notice tone="warning" title="No map geometry in this response">
        Laptop 1 did not return any valid coordinates for GPS observations, the predicted route or the road network, so there is nothing to plot.
      </Notice>
    );
  }

  return (
    <div className="routemap">
      {!predicted.length && (
        <Notice tone="warning" title="Predicted route is empty">
          The response contains no drawable predicted-route geometry. Other layers are shown for context.
        </Notice>
      )}
      {issueText.length > 0 && (
        <Notice tone="info" title="Geometry notes">
          <ul className="plainlist">
            {issueText.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </Notice>
      )}
      <div className="routemap__frame" data-testid="route-map">
        <MapContainer className="routemap__map" bounds={bounds} preferCanvas scrollWheelZoom zoomControl>
          <FitBounds bounds={bounds} />
          <LayersControl position="topright">
            {/* OpenStreetMap tiles need no API key; the dark variant is the same tiles with a CSS filter. */}
            <LayersControl.BaseLayer checked name="Dark">
              <TileLayer url={OSM_URL} attribution={OSM_ATTRIBUTION} maxZoom={19} className="tiles-dark" />
            </LayersControl.BaseLayer>
            <LayersControl.BaseLayer name="OpenStreetMap">
              <TileLayer url={OSM_URL} attribution={OSM_ATTRIBUTION} maxZoom={19} />
            </LayersControl.BaseLayer>

            {road.length > 0 && (
              <LayersControl.Overlay checked name="Road network">
                <Polyline positions={road.map((l) => l.positions)} pathOptions={{ color: COLORS.road, weight: 1.5, opacity: 0.55 }} />
              </LayersControl.Overlay>
            )}

            {reference.length > 0 && (
              <LayersControl.Overlay name="Reference route">
                <Polyline
                  positions={reference.map((l) => l.positions)}
                  pathOptions={{ color: COLORS.reference, weight: 4, opacity: 0.85, dashArray: '8 7' }}
                />
              </LayersControl.Overlay>
            )}

            {observations.length > 0 && (
              <LayersControl.Overlay checked name="GPS observations">
                <LayerGroup>
                  <Polyline positions={track} pathOptions={{ color: COLORS.gps, weight: 1, opacity: 0.45 }} />
                  {markerObs.map((o) => (
                    <CircleMarker
                      key={o.index}
                      center={o.position}
                      radius={2.2}
                      pathOptions={{ color: COLORS.gps, weight: 0, fillColor: COLORS.gps, fillOpacity: 0.7 }}
                    >
                      <Tooltip>
                        Observation {o.index}
                        {formatT(o.t)}
                        {o.matchedArcId !== null && <><br />Matched to arc {o.matchedArcId}</>}
                      </Tooltip>
                    </CircleMarker>
                  ))}
                </LayerGroup>
              </LayersControl.Overlay>
            )}
            {predicted.length > 0 && (
              <LayersControl.Overlay checked name="Predicted route">
                <LayerGroup>
                  {predicted.map((l, i) => (
                    <Polyline key={i} positions={l.positions} pathOptions={{ color: COLORS.predicted, weight: 4, opacity: 0.95 }}>
                      {l.arcId !== null && <Tooltip sticky>Predicted arc {l.arcId}</Tooltip>}
                    </Polyline>
                  ))}
                </LayerGroup>
              </LayersControl.Overlay>
            )}

            {observations.length > 0 && (
              <LayersControl.Overlay checked name="Start / end markers">
                <LayerGroup>
                  {first && (
                    <CircleMarker center={first.position} radius={7} pathOptions={{ color: '#0e1726', weight: 2, fillColor: COLORS.start, fillOpacity: 1 }}>
                      <Tooltip>Trip start (observation {first.index})</Tooltip>
                    </CircleMarker>
                  )}
                  {last && last !== first && (
                    <CircleMarker center={last.position} radius={7} pathOptions={{ color: '#0e1726', weight: 2, fillColor: COLORS.end, fillOpacity: 1 }}>
                      <Tooltip>Trip end (observation {last.index})</Tooltip>
                    </CircleMarker>
                  )}
                </LayerGroup>
              </LayersControl.Overlay>
            )}
          </LayersControl>
        </MapContainer>
        <Legend layers={layers} />
      </div>
    </div>
  );
}
