/**
 * Test-only stand-in for react-leaflet. jsdom has no canvas/layout engine, so
 * the mock renders each layer as a plain element exposing the data it received.
 */
import type { ReactNode } from 'react';

type LatLng = [number, number];

const Pass = ({ children }: { children?: ReactNode }) => <>{children}</>;

function countPoints(positions: LatLng[] | LatLng[][]): number {
  if (!positions.length) return 0;
  return Array.isArray(positions[0][0]) ? (positions as LatLng[][]).reduce((n, l) => n + l.length, 0) : positions.length;
}

const LayersControl = Object.assign(
  ({ children }: { children?: ReactNode }) => <div data-testid="layers-control">{children}</div>,
  {
    BaseLayer: ({ name, children }: { name: string; children?: ReactNode }) => <div data-testid={`base-${name}`}>{children}</div>,
    Overlay: ({ name, checked, children }: { name: string; checked?: boolean; children?: ReactNode }) => (
      <div data-testid={`overlay-${name}`} data-checked={String(!!checked)}>
        {children}
      </div>
    ),
  },
);

export const leafletMock = {
  MapContainer: ({ children, bounds }: { children?: ReactNode; bounds?: [LatLng, LatLng] }) => (
    <div data-testid="map-container" data-bounds={JSON.stringify(bounds)}>
      {children}
    </div>
  ),
  TileLayer: ({ url }: { url: string }) => <div data-testid="tile-layer" data-url={url} />,
  LayersControl,
  LayerGroup: Pass,
  Polyline: ({ positions, children }: { positions: LatLng[] | LatLng[][]; children?: ReactNode }) => (
    <div data-testid="polyline" data-points={countPoints(positions)}>
      {children}
    </div>
  ),
  CircleMarker: ({ center, children }: { center: LatLng; children?: ReactNode }) => (
    <div data-testid="circle-marker" data-center={center.join(',')}>
      {children}
    </div>
  ),
  Tooltip: () => null,
  useMap: () => ({ fitBounds: () => undefined }),
};
