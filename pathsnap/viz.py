"""Folium maps of map-matching results (generated from actual pipeline outputs)."""
from __future__ import annotations

import folium
import numpy as np
from branca.element import MacroElement, Template

from .metrics import compress
from .pipeline import matched_points


def base_map(lat: float, lon: float, zoom: int = 15) -> folium.Map:
    """Folium map WITHOUT external tiles by default.

    OpenStreetMap's tile servers reject requests without a proper Referer (HTTP 403),
    which is what happens for maps opened from disk or inside an iframe. The road
    network drawn on the map is the geographic context instead. Set the environment
    variable PATHSNAP_MAP_TILES (e.g. "OpenStreetMap" or "CartoDB positron") to add tiles.
    """
    import os
    tiles = os.environ.get("PATHSNAP_MAP_TILES") or None
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=tiles, control_scale=True, prefer_canvas=True)
    if tiles is None:
        bg = MacroElement()
        bg._template = Template("{% macro header(this, kwargs) %}<style>.leaflet-container{background:#f7f7f5}</style>{% endmacro %}")
        m.get_root().add_child(bg)
    return m


def _legend(items: list[tuple[str, str]], title: str) -> MacroElement:
    rows = "".join(f'<div><span style="background:{c};width:14px;height:6px;display:inline-block;margin-right:6px"></span>{t}</div>'
                   for t, c in items)
    html = f"""{{% macro html(this, kwargs) %}}
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white; padding: 8px 10px;
         border: 1px solid #999; border-radius: 4px; font: 12px sans-serif;"><b>{title}</b>{rows}</div>
    {{% endmacro %}}"""
    el = MacroElement()
    el._template = Template(html)
    return el


def match_map(result: dict, ref_route_gid=None, title: str = "", max_candidate_fixes: int = 400) -> folium.Map:
    net, df, feats = result["net"], result["df"], result["feats"]
    m = base_map(df.lat.mean(), df.lon.mean())

    # candidate roads (unique candidate arcs of a subsample of fixes); shown by default because
    # without map tiles they are the geographic context
    step = max(1, len(df) // max_candidate_fixes)
    cand_arcs = np.unique(feats[feats.obs % step == 0].arc.to_numpy())
    fg = folium.FeatureGroup("Candidate road segments", show=True)
    for seg in net.arc_lonlat(cand_arcs):
        folium.PolyLine(seg, color="#9e9e9e", weight=2, opacity=.7).add_to(fg)
    fg.add_to(m)

    if ref_route_gid is not None:
        rl = [net.gid_to_local[int(g)] for g in ref_route_gid if int(g) in net.gid_to_local]
        fg = folium.FeatureGroup("Reference route (server-side)")
        for seg in net.arc_lonlat(rl):
            folium.PolyLine(seg, color="#2e7d32", weight=9, opacity=.45).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup("Classifier-only route", show=True)
    for seg in net.arc_lonlat(compress(result["clf_arc_local"])):
        folium.PolyLine(seg, color="#ef6c00", weight=4, opacity=.85).add_to(fg)
    fg.add_to(m)

    if "route_local" in result:
        fg = folium.FeatureGroup("HMM + Viterbi route")
        for seg in net.arc_lonlat(result["route_local"]):
            folium.PolyLine(seg, color="#c2185b", weight=4, opacity=.9, dash_array="6,4").add_to(fg)
        fg.add_to(m)
        mp = matched_points(net, df, result["hmm_arc_local"])
        fg = folium.FeatureGroup("HMM matched positions", show=False)
        for (lo, la) in mp[::max(1, len(df) // 600)]:
            if np.isfinite(lo):
                folium.CircleMarker([la, lo], radius=2, color="#c2185b", fill=True, weight=0, fill_opacity=.9).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup("Raw GPS observations")
    for i in range(0, len(df), max(1, len(df) // 1500)):
        folium.CircleMarker([df.lat.iat[i], df.lon.iat[i]], radius=2.5, color="#1565c0", fill=True, weight=0,
                            fill_opacity=.85, tooltip=f"fix {int(df.orig_idx.iat[i])} t={df.t.iat[i]:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.Marker([df.lat.iat[0], df.lon.iat[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([df.lat.iat[-1], df.lon.iat[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    items = [("Raw GPS", "#1565c0"), ("Classifier-only", "#ef6c00"), ("HMM + Viterbi", "#c2185b"), ("Candidates", "#9e9e9e")]
    if ref_route_gid is not None:
        items.insert(1, ("Reference route", "#2e7d32"))
    m.get_root().add_child(_legend(items, title or "PathSnap"))
    m.fit_bounds([[df.lat.min(), df.lon.min()], [df.lat.max(), df.lon.max()]])
    return m
