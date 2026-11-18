"""Road network: projection, cropping, spatial index and directed shortest paths.

Each trip ships its own road network. Arcs are directed straight segments between
two nodes (a two-way street is stored as two arcs with opposite directions).

* The network is cropped to the trajectory bounding box + buffer so all downstream
  work (indexing, Dijkstra) only touches the relevant part of large networks.
  Arc ids are always reported as the ORIGINAL (global) arc ids.
* Spatial index: every arc is densified into sample points every few metres and put
  in a KD-tree; a radius query returns candidate arcs, whose exact point-to-segment
  distance is then computed. This avoids comparing each fix with every arc.
* Shortest paths: bounded Dijkstra on the directed graph (weights = arc length in m),
  memoised per source node because consecutive 1 Hz fixes share most nodes.
"""
from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from . import config
from .geo import LocalProjection, bearing_xy, point_segment_distance


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection,
                 bbox_xy: tuple[float, float, float, float] | None = None,
                 buffer_m: float = config.NETWORK_CROP_BUFFER_M,
                 index_step_m: float = config.INDEX_SAMPLE_STEP_M):
        self.proj = proj
        self.n_arcs_total = len(arcs)
        self.n_nodes_total = len(nodes_lonlat)
        nx_all, ny_all = proj.forward(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        ax_all, ay_all = nx_all[arcs[:, 0]], ny_all[arcs[:, 0]]
        bx_all, by_all = nx_all[arcs[:, 1]], ny_all[arcs[:, 1]]
        keep = arcs[:, 0] != arcs[:, 1]  # drop degenerate self-loops
        if bbox_xy is not None:
            x0, y0, x1, y1 = bbox_xy
            x0, y0, x1, y1 = x0 - buffer_m, y0 - buffer_m, x1 + buffer_m, y1 + buffer_m
            keep &= (np.maximum(ax_all, bx_all) >= x0) & (np.minimum(ax_all, bx_all) <= x1)
            keep &= (np.maximum(ay_all, by_all) >= y0) & (np.minimum(ay_all, by_all) <= y1)
        self.arc_gid = np.nonzero(keep)[0].astype(np.int64)          # local arc -> global arc id
        sub = arcs[keep]
        used_nodes, inv = np.unique(sub.ravel(), return_inverse=True)
        self.node_gid = used_nodes.astype(np.int64)                  # local node -> global node id
        self.arc_from = inv[0::2].astype(np.int64)
        self.arc_to = inv[1::2].astype(np.int64)
        self.node_x, self.node_y = nx_all[used_nodes], ny_all[used_nodes]
        self.node_lonlat = nodes_lonlat[used_nodes]
        self.ax, self.ay = self.node_x[self.arc_from], self.node_y[self.arc_from]
        self.bx, self.by = self.node_x[self.arc_to], self.node_y[self.arc_to]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_xy(self.bx - self.ax, self.by - self.ay)
        self.gid_to_local = {int(g): i for i, g in enumerate(self.arc_gid)}
        self.n_arcs = len(self.arc_gid)
        self.n_nodes = len(self.node_gid)

        # topology
        self.out_deg = np.bincount(self.arc_from, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.arc_to, minlength=self.n_nodes)
        pair_to_arc = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.arc_from, self.arc_to))}
        self.has_reverse = np.array([(int(b), int(a)) in pair_to_arc for a, b in zip(self.arc_from, self.arc_to)])
        self.reverse_arc = np.array([pair_to_arc.get((int(b), int(a)), -1) for a, b in zip(self.arc_from, self.arc_to)])
        self.adj: list[list[tuple[int, float, int]]] = [[] for _ in range(self.n_nodes)]
        for i, (a, b, w) in enumerate(zip(self.arc_from, self.arc_to, self.length)):
            self.adj[int(a)].append((int(b), float(w), i))

        self._build_index(index_step_m)
        self._dij_cache: dict[tuple[int, float], dict[int, float]] = {}

    # ------------------------------------------------------------------ factory
    @classmethod
    def for_track(cls, nodes_lonlat, arcs, proj, track_x, track_y, **kw):
        bbox = (float(np.min(track_x)), float(np.min(track_y)), float(np.max(track_x)), float(np.max(track_y)))
        return cls(nodes_lonlat, arcs, proj, bbox_xy=bbox, **kw)

    # ------------------------------------------------------------ spatial index
    def _build_index(self, step: float):
        nseg = np.maximum(1, np.ceil(self.length / step).astype(int))
        owner = np.repeat(np.arange(self.n_arcs), nseg + 1)
        # fractions 0..1 for each arc
        offs = np.concatenate([np.arange(n + 1) / n for n in nseg]) if self.n_arcs else np.zeros(0)
        sx = self.ax[owner] + offs * (self.bx - self.ax)[owner]
        sy = self.ay[owner] + offs * (self.by - self.ay)[owner]
        self._sample_owner = owner
        self._tree = cKDTree(np.c_[sx, sy]) if len(sx) else None
        self._half_step = step / 2.0

    def arcs_near(self, x: float, y: float, radius: float) -> np.ndarray:
        if self._tree is None:
            return np.zeros(0, dtype=np.int64)
        idx = self._tree.query_ball_point([x, y], radius + self._half_step)
        return np.unique(self._sample_owner[idx]) if idx else np.zeros(0, dtype=np.int64)

    def nearest_arcs(self, x: float, y: float, k: int = 8) -> np.ndarray:
        if self._tree is None:
            return np.zeros(0, dtype=np.int64)
        kk = min(len(self._sample_owner), max(4 * k, 32))
        _, idx = self._tree.query([x, y], k=kk)
        return np.unique(self._sample_owner[np.atleast_1d(idx)])

    def distance_to_arcs(self, x, y, arcs_local):
        return point_segment_distance(x, y, self.ax[arcs_local], self.ay[arcs_local], self.bx[arcs_local], self.by[arcs_local])

    # ----------------------------------------------------------- shortest paths
    def dijkstra_from(self, src: int, limit: float) -> dict[int, float]:
        """Bounded single-source Dijkstra on local node ids. Memoised by (src, limit bucket)."""
        limit = float(np.ceil(limit / 100.0) * 100.0)  # bucket the limit to improve cache hits
        key = (src, limit)
        hit = self._dij_cache.get(key)
        if hit is not None:
            return hit
        dist = {src: 0.0}
        pq = [(0.0, src)]
        adj = self.adj
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf):
                continue
            for v, w, _ in adj[u]:
                nd = d + w
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        if len(self._dij_cache) > 200_000:
            self._dij_cache.clear()
        self._dij_cache[key] = dist
        return dist

    def shortest_arc_path(self, src_node: int, dst_node: int, limit: float = 5000.0) -> list[int] | None:
        """Local arc ids of the shortest directed path src_node -> dst_node (None if unreachable)."""
        if src_node == dst_node:
            return []
        dist = {src_node: 0.0}
        prev: dict[int, tuple[int, int]] = {}
        pq = [(0.0, src_node)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst_node:
                break
            if d > dist.get(u, np.inf):
                continue
            for v, w, a in self.adj[u]:
                nd = d + w
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = (u, a)
                    heapq.heappush(pq, (nd, v))
        if dst_node not in prev:
            return None
        path, cur = [], dst_node
        while cur != src_node:
            u, a = prev[cur]
            path.append(a)
            cur = u
        return path[::-1]

    def connected_components(self):
        """Weakly / strongly connected component counts of the cropped graph."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        g = csr_matrix((np.ones(self.n_arcs), (self.arc_from, self.arc_to)), shape=(self.n_nodes, self.n_nodes))
        nw, lw = connected_components(g, directed=True, connection="weak")
        ns, ls = connected_components(g, directed=True, connection="strong")
        return nw, lw, ns, ls

    # --------------------------------------------------------------- geometry
    def arc_lonlat(self, arcs_local) -> list[list[list[float]]]:
        """[[ [lat, lon], [lat, lon] ], ...] for map drawing (Leaflet order)."""
        a = np.asarray(arcs_local, dtype=np.int64)
        f, t = self.node_lonlat[self.arc_from[a]], self.node_lonlat[self.arc_to[a]]
        return [[[float(p[1]), float(p[0])], [float(q[1]), float(q[0])]] for p, q in zip(f, t)]


def full_graph_stats(nodes: np.ndarray, arcs: np.ndarray) -> dict:
    """Connectivity statistics of an UNCROPPED trip network (EDA)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(nodes)
    g = csr_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    nw, lw = connected_components(g, directed=True, connection="weak")
    ns, ls = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(arcs[:, 0], minlength=n) + np.bincount(arcs[:, 1], minlength=n)
    pairs = set(map(tuple, arcs.tolist()))
    rev = sum((b, a) in pairs for a, b in pairs) / max(1, len(pairs))
    return {
        "n_nodes": n, "n_arcs": len(arcs),
        "isolated_nodes": int(np.sum(deg == 0)),
        "weak_components": int(nw), "largest_weak_frac": float(np.bincount(lw).max() / n),
        "strong_components": int(ns), "largest_strong_frac": float(np.bincount(ls).max() / n),
        "two_way_frac": float(rev),
    }
