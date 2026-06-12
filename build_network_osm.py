# python build_network_osm.py --bbox="-83.953857,41.940127,-82.805786,43.190201" --out_dir net_em --simplify

import os
import json
import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import osmnx as ox


# ---------- Helpers ----------

def geocode_addresses(csv_path, addr_col="address", name_col="name"):
    df = pd.read_csv(csv_path)
    if addr_col not in df.columns or name_col not in df.columns:
        raise ValueError(f"CSV must contain '{name_col}' and '{addr_col}' columns.")
    geolocator = Nominatim(user_agent="net_builder_em")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0, swallow_exceptions=False)
    lons, lats = [], []
    for i, addr in enumerate(df[addr_col]):
        loc = geocode(addr)
        if not loc:
            raise RuntimeError(f"Could not geocode row {i}: {addr}")
        lons.append(loc.longitude)
        lats.append(loc.latitude)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(lons, lats), crs=4326)
    return gdf


def aoi_from_points(points_wgs84_gdf, buffer_m):
    # Build convex hull around points, buffer in meters (project to EPSG:3857), return polygon in WGS84
    poly_merc = points_wgs84_gdf.to_crs(3857).geometry.unary_union.convex_hull.buffer(buffer_m)
    return gpd.GeoSeries([poly_merc], crs=3857).to_crs(4326).iloc[0]


def aoi_from_gpkg(gpkg_path, layer_name):
    poly_gdf = gpd.read_file(gpkg_path, layer=layer_name)
    if poly_gdf.empty:
        raise ValueError("AOI layer is empty.")
    return poly_gdf.to_crs(4326).unary_union  # shapely (Multi)Polygon


def parse_bbox_arg(b):
    """
    Accept either:
      - one string "W,S,E,N", OR
      - four separate floats: W S E N
    Return a Shapely polygon in WGS84.
    """
    if isinstance(b, list):
        if len(b) == 1:
            w, s, e, n = map(float, b[0].split(","))
        elif len(b) == 4:
            w, s, e, n = map(float, b)
        else:
            raise ValueError('Use one string "W,S,E,N" or four numbers W S E N.')
    else:
        # Fallback if argparse gave a plain string (unlikely with nargs="+")
        w, s, e, n = map(float, str(b).split(","))
    return box(w, s, e, n)  # shapely polygon


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Build & save OSM drivable network once.")
    parser.add_argument("--out_dir", default="network_out", help="Output directory")
    # AOI sources (pick one)
    parser.add_argument("--from_csv", help="CSV with columns name,address to derive AOI (convex hull + buffer)")
    parser.add_argument("--from_gpkg", help="GeoPackage containing AOI polygon layer")
    parser.add_argument("--aoi_layer", help="Layer name inside --from_gpkg")
    parser.add_argument("--bbox", nargs="+",
                        help='AOI bbox as one string "W,S,E,N" OR four floats: W S E N (WGS84)')
    # Options
    parser.add_argument("--buffer_m", type=int, default=15000, help="Buffer (meters) for CSV-derived AOI")
    parser.add_argument("--network_type", default="drive", help='OSM network_type, e.g., "drive"')
    parser.add_argument("--simplify", action="store_true", help="Simplify network topology (recommended)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = os.path.join(args.out_dir, "_ox_cache")

    # --- Build AOI polygon ---
    if args.from_csv:
        pts = geocode_addresses(args.from_csv)
        aoi = aoi_from_points(pts, args.buffer_m)
        # Save AOI for reference
        gpd.GeoDataFrame(geometry=[aoi], crs=4326).to_file(
            os.path.join(args.out_dir, "aoi.gpkg"), layer="aoi", driver="GPKG"
        )
    elif args.from_gpkg and args.aoi_layer:
        aoi = aoi_from_gpkg(args.from_gpkg, args.aoi_layer)
    elif args.bbox:
        aoi = parse_bbox_arg(args.bbox)
    else:
        raise ValueError("Provide one AOI source: --from_csv OR --from_gpkg + --aoi_layer OR --bbox.")

    # --- Build network ---
    print("Downloading/building OSM network …")
    G = ox.graph_from_polygon(aoi, network_type=args.network_type, simplify=args.simplify)
    # Make undirected so shortest_path_length by 'length' behaves as expected
    G = ox.convert.to_undirected(G)

    # --- Save outputs ---
    graphml_path = os.path.join(args.out_dir, "network.graphml")
    ox.save_graphml(G, graphml_path)

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
    nodes_gdf.to_file(os.path.join(args.out_dir, "network_nodes.gpkg"), layer="nodes", driver="GPKG")
    edges_gdf.to_file(os.path.join(args.out_dir, "network_edges.gpkg"), layer="edges", driver="GPKG")

    meta = {
        "network_type": args.network_type,
        "simplify": bool(args.simplify),
        "n_nodes": int(len(G.nodes)),
        "n_edges": int(len(G.edges)),
        "aoi_bounds": tuple(aoi.bounds) if hasattr(aoi, "bounds") else None
    }
    with open(os.path.join(args.out_dir, "network_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"Saved: {graphml_path}")
    print("Also wrote: network_nodes.gpkg, network_edges.gpkg, network_meta.json")
    if args.from_csv:
        print("AOI saved to: aoi.gpkg (layer: aoi)")


if __name__ == "__main__":
    main()