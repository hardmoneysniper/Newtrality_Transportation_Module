#!/usr/bin/env python
"""
server_maps.py

FastAPI server to visualize:
- Clustered routes: em_delivery_clustered_routes.shp
- Individual routes: em_delivery_individual_routes.shp
- Overlap intensity (separate):
    /data/intensity_clustered   -> segments with 'count' from clustered routes only
    /data/intensity_individual  -> segments with 'count' from individual routes only

Serves:
- "/"                        -> map_index.html
- "/map_style.css"           -> map_style.css
- "/data/clustered"          -> GeoJSON (clustered routes)
- "/data/individual"         -> GeoJSON (individual routes)
- "/data/intensity_clustered"  -> GeoJSON (segments with count, clustered only)
- "/data/intensity_individual" -> GeoJSON (segments with count, individual only)

Run with:
    uvicorn server_maps:app --reload --port 8001
Then open:
    http://127.0.0.1:8001/
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

app = FastAPI()

# Paths
BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "map_index.html"
CSS_PATH = BASE_DIR / "map_style.css"
CLUSTERED_SHP = BASE_DIR / "em_delivery_clustered_routes.shp"
INDIVIDUAL_SHP = BASE_DIR / "em_delivery_individual_routes.shp"


@app.get("/")
def index():
    if not HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="map_index.html not found.")
    return FileResponse(str(HTML_PATH))


@app.get("/map_style.css")
def style():
    if not CSS_PATH.exists():
        raise HTTPException(status_code=500, detail="map_style.css not found.")
    return FileResponse(str(CSS_PATH), media_type="text/css")


@app.get("/data/clustered")
def data_clustered():
    if not CLUSTERED_SHP.exists():
        raise HTTPException(
            status_code=500,
            detail=f"{CLUSTERED_SHP.name} not found. Make sure it exists in the same directory."
        )
    gdf = gpd.read_file(CLUSTERED_SHP)
    # Ensure WGS84
    if gdf.crs and str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
        gdf = gdf.to_crs(4326)
    geojson_str = gdf.to_json()
    return Response(content=geojson_str, media_type="application/json")


@app.get("/data/individual")
def data_individual():
    if not INDIVIDUAL_SHP.exists():
        raise HTTPException(
            status_code=500,
            detail=f"{INDIVIDUAL_SHP.name} not found. Make sure it exists in the same directory."
        )
    gdf = gpd.read_file(INDIVIDUAL_SHP)
    # Ensure WGS84
    if gdf.crs and str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
        gdf = gdf.to_crs(4326)
    geojson_str = gdf.to_json()
    return Response(content=geojson_str, media_type="application/json")


# ---------- overlap / intensity helpers (used for both clustered & individual) ----------

def canonical_segment_key(x1, y1, x2, y2, ndp=6):
    """
    Create a stable key for an undirected segment (A<->B), with rounding
    so floating point noise doesn't break equality.
    """
    ax = round(float(x1), ndp)
    ay = round(float(y1), ndp)
    bx = round(float(x2), ndp)
    by = round(float(y2), ndp)
    # sort endpoints so segment AB == BA
    if (ax, ay, bx, by) <= (bx, by, ax, ay):
        return (ax, ay, bx, by)
    else:
        return (bx, by, ax, ay)


def segmentize_geoms(gdf: gpd.GeoDataFrame):
    """
    Take all LineString/MultiLineString geometries and break them into
    individual segments (between consecutive coordinates). Returns a dict:
      key -> {"count": n, "coords": ((x1,y1),(x2,y2))}
    where 'count' is the number of routes using that segment.
    """
    seg_dict = {}

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        else:
            # ignore non-line geometries just in case
            continue

        coords = list(lines[0].coords) if len(lines) == 1 else None

        # Iterate lines independently
        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                key = canonical_segment_key(x1, y1, x2, y2)
                if key not in seg_dict:
                    seg_dict[key] = {
                        "count": 0,
                        "coords": ((key[0], key[1]), (key[2], key[3]))
                    }
                seg_dict[key]["count"] += 1

    return seg_dict


def segments_to_gdf(seg_dict):
    """
    Turn segment dictionary into a GeoDataFrame with LineStrings and a 'count' field.
    """
    rows = []
    for key, data in seg_dict.items():
        (x1, y1), (x2, y2) = data["coords"]
        geom = LineString([(x1, y1), (x2, y2)])
        rows.append({"count": data["count"], "geometry": geom})

    if not rows:
        return gpd.GeoDataFrame(columns=["count", "geometry"], geometry="geometry", crs=4326)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    return gdf


def _build_intensity_from_shp(shp_path: Path) -> gpd.GeoDataFrame:
    """
    Helper: read one shapefile, segmentize, return segments with 'count'.
    """
    if not shp_path.exists():
        raise FileNotFoundError(f"{shp_path.name} not found.")
    g = gpd.read_file(shp_path)
    if g.crs and str(g.crs).lower() not in ("epsg:4326", "wgs84"):
        g = g.to_crs(4326)
    seg_dict = segmentize_geoms(g[["geometry"]])
    seg_gdf = segments_to_gdf(seg_dict)
    return seg_gdf


@app.get("/data/intensity_clustered")
def data_intensity_clustered():
    """
    Overlap intensity for clustered routes only: segments with 'count' telling
    how many clustered routes use that segment.
    """
    try:
        seg_gdf = _build_intensity_from_shp(CLUSTERED_SHP)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    geojson_str = seg_gdf.to_json()
    return Response(content=geojson_str, media_type="application/json")


@app.get("/data/intensity_individual")
def data_intensity_individual():
    """
    Overlap intensity for individual routes only: segments with 'count' telling
    how many individual vendor routes use that segment.
    """
    try:
        seg_gdf = _build_intensity_from_shp(INDIVIDUAL_SHP)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    geojson_str = seg_gdf.to_json()
    return Response(content=geojson_str, media_type="application/json")