#!/usr/bin/env python
"""
make_cluster_vs_individual_maps.py

Produce two Shapefiles from your existing pipeline:

1) em_delivery_clustered_routes.shp
   - Aggregates all cluster routes from out_clusters/cluster_*/em_farms_route.gpkg
   - One feature per cluster route
   - Has 'cluster' and 'color_hex' fields so clusters can be styled with different colors.

2) em_delivery_individual_routes.shp
   - For each vendor in stops_from_vendors_geocoded.json, requests an ORS route
     from vendor -> Eastern Market and stores the line geometry.
   - One feature per vendor trip
   - Has 'name', 'address', 'distance_km', 'duration_min', and 'color_hex' so each trip
     can be styled with different colors.

Assumptions:
- Geocoded JSON: ./stops_from_vendors_geocoded.json
- Cluster outputs: ./out_clusters/cluster_*/em_farms_route.gpkg
- Output format: Shapefile (.shp)
"""

import json
import time
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import openrouteservice as ors

# ========= CONFIG YOU CAN EDIT =========

# 1) Your OpenRouteService API key (insert your real key here)
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjNiMjA0NmE5Mzc1NDQxZmViNjUwYTUwN2VjYzYwZTA1IiwiaCI6Im11cm11cjY0In0="

# 2) Input paths (relative to this script)
JSON_PATH = Path("stops_from_vendors_geocoded.json")
CLUSTERS_DIR = Path("out_clusters")

# 3) Output prefix:
#    This base name is used for both outputs:
#      <OUT_PREFIX>_clustered_routes.shp
#      <OUT_PREFIX>_individual_routes.shp
OUT_PREFIX = "em_delivery"

# 4) ORS routing options
ORS_PROFILE = "driving-car"
ORS_PREFERENCE = "fastest"
ORS_SLEEP_BETWEEN_CALLS = 1.0  # seconds between ORS calls (rate limiting)

# 5) Color palette for clusters/trips (hex strings). Will cycle if more items than colors.
COLOR_PALETTE = [
    "#ff6b6b", "#4dabf7", "#51cf66", "#ffd43b", "#b197fc",
    "#63e6be", "#ffa94d", "#69db7c", "#74c0fc", "#ff8787"
]

# Eastern Market depot (same as your other scripts)
EM_NAME = "Eastern Market"
EM_LON = -83.0416
EM_LAT = 42.3469

# =======================================


def load_geocoded_stops(json_path: Path) -> pd.DataFrame:
    """Load geocoded vendor stops from JSON into a DataFrame."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "stops" not in data:
        raise ValueError("Geocoded JSON must be an object with key 'stops'.")
    rows = []
    for i, s in enumerate(data["stops"]):
        for k in ("name", "address", "lon", "lat"):
            if k not in s:
                raise ValueError(f"Stop index {i} missing '{k}'.")
        rows.append({
            "name": s["name"],
            "address": s["address"],
            "lon": float(s["lon"]),
            "lat": float(s["lat"])
        })
    return pd.DataFrame(rows)


def build_clustered_routes(clusters_dir: Path) -> gpd.GeoDataFrame:
    """
    Aggregate all cluster 'route' layers from cluster_*/em_farms_route.gpkg
    into a single GeoDataFrame, and assign a color_hex per cluster.
    """
    rows = []
    for gpkg in sorted(clusters_dir.glob("cluster_*/em_farms_route.gpkg")):
        cluster_name = gpkg.parent.name  # e.g. "cluster_01"
        try:
            route = gpd.read_file(gpkg, layer="route")
        except Exception as e:
            print(f"Warning: could not read 'route' layer from {gpkg}: {e}")
            continue
        route = route.copy()
        route["cluster"] = cluster_name
        rows.append(route)

    if not rows:
        raise FileNotFoundError(f"No cluster route GPKGs found under {clusters_dir}")

    gdf = pd.concat(rows, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=4326)

    # Assign a color per cluster
    unique_clusters = sorted(gdf["cluster"].unique())
    color_map = {
        cl: COLOR_PALETTE[i % len(COLOR_PALETTE)]
        for i, cl in enumerate(unique_clusters)
    }
    gdf["color_hex"] = gdf["cluster"].map(color_map)

    return gdf


def build_individual_routes(df_stops: pd.DataFrame,
                            ors_key: str,
                            profile: str = "driving-car",
                            preference: str = "fastest",
                            sleep: float = 1.0) -> gpd.GeoDataFrame:
    """
    For each vendor, request an ORS route from vendor -> Eastern Market.

    Returns a GeoDataFrame with one line feature per vendor, fields:
      - name
      - address
      - distance_m / km
      - duration_s / min
      - color_hex (per trip)
    """
    client = ors.Client(key=ors_key, timeout=120)

    rows = []
    for i, r in df_stops.iterrows():
        name = r["name"]
        addr = r["address"]
        lon = float(r["lon"])
        lat = float(r["lat"])
        print(f"Routing vendor {i+1}/{len(df_stops)}: {name} …")

        coords = [(lon, lat), (EM_LON, EM_LAT)]  # vendor -> EM
        try:
            res = client.directions(
                coordinates=coords,
                profile=profile,
                preference=preference,
                format="geojson"
            )
        except Exception as e:
            print(f"  ERROR for {name}: {e}")
            continue

        try:
            feat = res["features"][0]
            geom = shape(feat["geometry"])
            summary = feat["properties"]["summary"]
            dist_m = float(summary["distance"])
            dur_s = float(summary["duration"])
        except Exception as e:
            print(f"  ERROR parsing ORS response for {name}: {e}")
            continue

        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        rows.append({
            "name": name,
            "address": addr,
            "distance_m": dist_m,
            "distance_km": dist_m / 1000.0,
            "duration_s": dur_s,
            "duration_min": dur_s / 60.0,
            "color_hex": color,
            "geometry": geom
        })

        if sleep > 0:
            time.sleep(sleep)

    if not rows:
        raise RuntimeError("No individual routes were successfully built.")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    return gdf


def save_shapefile(gdf: gpd.GeoDataFrame, out_path: Path):
    """Save GeoDataFrame as Shapefile (.shp)."""
    if gdf.crs and str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
        gdf = gdf.to_crs(4326)
    out_file = out_path.with_suffix(".shp")
    gdf.to_file(out_file)
    print(f"Saved {out_file}")


def main():
    if not ORS_API_KEY or ORS_API_KEY == "YOUR_ORS_KEY_HERE":
        raise RuntimeError("Please set ORS_API_KEY at the top of this script to your ORS key.")

    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON not found: {JSON_PATH}")
    if not CLUSTERS_DIR.exists():
        raise FileNotFoundError(f"Clusters directory not found: {CLUSTERS_DIR}")

    print(f"Loading geocoded stops from {JSON_PATH} …")
    df_stops = load_geocoded_stops(JSON_PATH)

    print(f"Aggregating clustered routes from {CLUSTERS_DIR} …")
    clustered_routes = build_clustered_routes(CLUSTERS_DIR)

    print("Building individual vendor routes via ORS …")
    indiv_routes = build_individual_routes(
        df_stops=df_stops,
        ors_key=ORS_API_KEY,
        profile=ORS_PROFILE,
        preference=ORS_PREFERENCE,
        sleep=ORS_SLEEP_BETWEEN_CALLS
    )

    # Save both layers using OUT_PREFIX
    clustered_path = Path(f"{OUT_PREFIX}_clustered_routes")
    indiv_path = Path(f"{OUT_PREFIX}_individual_routes")

    save_shapefile(clustered_routes, clustered_path)
    save_shapefile(indiv_routes, indiv_path)

    print("Done.")


if __name__ == "__main__":
    main()