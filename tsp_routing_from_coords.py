# tsp_routes_from_coords.py
# Network (driving) distance TSP from pre-geocoded stops, with Eastern Market as depot.
# Usage:
#   python tsp_routes_from_coords.py --graphml net_em/network.graphml --json stops_geocoded.json --out out_em
#   # add --no_return to end at last farm (no return to EM)

import os, json, argparse, itertools
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union, linemerge
from shapely.geometry import LineString
import networkx as nx
import osmnx as ox

EM_NAME = "Eastern Market"
EM_LON = -83.0416
EM_LAT = 42.3469

def load_geocoded_stops(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not (isinstance(data, dict) and "stops" in data):
        raise ValueError("Geocoded JSON must be an object with key 'stops'.")
    rows = []
    for i, s in enumerate(data["stops"]):
        for k in ("name", "address", "lon", "lat"):
            if k not in s:
                raise ValueError(f"Stop index {i} missing '{k}'.")
        rows.append({"name": s["name"], "address": s["address"],
                     "lon": float(s["lon"]), "lat": float(s["lat"])})
    if not rows:
        raise ValueError("No stops found in JSON.")
    return pd.DataFrame(rows)

def ensure_unique_names(names):
    seen = {}; out = []
    for n in names:
        if n not in seen: seen[n]=0; out.append(n)
        else: seen[n]+=1; out.append(f"{n} ({seen[n]})")
    return out

def snap_names_to_nodes(G, gdf):
    mapping = {}
    for _, r in gdf.iterrows():
        node = ox.distance.nearest_nodes(G, float(r["lon"]), float(r["lat"]))
        mapping[r["name"]] = node
    return mapping

def pairwise_distance_matrix(G, name_to_node):
    # All-pairs SHORTEST *DRIVING* DISTANCE (meters) on the directed graph
    names = list(name_to_node.keys())
    D = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        D.at[a, a] = 0.0
        for b in names[i+1:]:
            d = nx.shortest_path_length(G, name_to_node[a], name_to_node[b], weight="length")
            D.at[a, b] = d
            # NOTE: directed network ⇒ D[a,b] may != D[b,a]
            d2 = nx.shortest_path_length(G, name_to_node[b], name_to_node[a], weight="length")
            D.at[b, a] = d2
    return D

def brute_force_best(D, depot, return_to_depot=True):
    names = list(D.index)
    stops = [n for n in names if n != depot]
    best = None; best_order = None
    for perm in itertools.permutations(stops):
        seq = [depot] + list(perm)
        if return_to_depot: seq = seq + [depot]
        total = sum(D.loc[a, b] for a, b in zip(seq[:-1], seq[1:]))
        if best is None or total < best:
            best, best_order = total, seq
    return best, best_order

# ---------- robust per-route edge builder (no route_to_gdf) ----------
def edge_gdf_from_route(G, route_nodes):
    """
    Build an edges GeoDataFrame for a node route.
    - Handles Multi(Di)Graph (multiple parallel edges) and (Di)Graph.
    - Picks the shortest edge by 'length' for each (u,v).
    """
    rows = []
    is_multi = isinstance(G, (nx.MultiDiGraph, nx.MultiGraph))

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        if is_multi:
            edict = G.get_edge_data(u, v)
            if edict is None:
                raise RuntimeError(f"No directed edge between {u} -> {v}.")
            # choose shortest parallel edge
            best_k, best_attr, best_len = None, None, float("inf")
            for k, attr in edict.items():
                L = attr.get("length")
                if L is None:
                    L = ox.distance.great_circle_vec(G.nodes[u]["y"], G.nodes[u]["x"],
                                                     G.nodes[v]["y"], G.nodes[v]["x"])
                if L < best_len:
                    best_k, best_attr, best_len = k, attr, L
            geom = best_attr.get("geometry")
            if geom is None:
                geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                                   (G.nodes[v]["x"], G.nodes[v]["y"])])
            rows.append({"u": u, "v": v, "key": best_k, "edge_m": float(best_len), "geometry": geom})

        else:
            # DiGraph / Graph: single attribute dict
            attr = G.get_edge_data(u, v)
            if attr is None:
                raise RuntimeError(f"No directed edge between {u} -> {v}.")
            L = attr.get("length")
            if L is None:
                L = ox.distance.great_circle_vec(G.nodes[u]["y"], G.nodes[u]["x"],
                                                 G.nodes[v]["y"], G.nodes[v]["x"])
            geom = attr.get("geometry")
            if geom is None:
                geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                                   (G.nodes[v]["x"], G.nodes[v]["y"])])
            rows.append({"u": u, "v": v, "edge_m": float(L), "geometry": geom})

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)

def legs_edges_gdf(G, ordered_names, name_to_node):
    frames = []
    for i, (a, b) in enumerate(zip(ordered_names[:-1], ordered_names[1:]), start=1):
        route_nodes = nx.shortest_path(G, name_to_node[a], name_to_node[b], weight="length")
        segs = edge_gdf_from_route(G, route_nodes)
        segs["leg_index"] = i
        segs["from_name"] = a
        segs["to_name"] = b
        frames.append(segs)
    if not frames:
        raise RuntimeError("No legs built—check ordered_names length.")
    return pd.concat(frames, ignore_index=True)

def dissolved_route_geometry(edges_gdf):
    geom = unary_union(edges_gdf.geometry.tolist())
    try: return linemerge(geom)
    except Exception: return geom

def main():
    parser = argparse.ArgumentParser(description="Run TSP using pre-geocoded stops JSON.")
    parser.add_argument("--graphml", required=True, help="Path to saved network.graphml")
    parser.add_argument("--json", required=True, help="Path to stops_geocoded.json")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--no_return", action="store_true", help="Do not return to depot")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ox.settings.use_cache = True

    print("Loading network…")
    G = ox.load_graphml(args.graphml)

    # Keep it DIRECTED; do NOT make it undirected (to respect one-ways).
    # If your saved graph is undirected, convert to DiGraph.
    if not G.is_directed():
        G = ox.convert.to_digraph(G, weight="length")  # v2 API

    # Make sure 'length' exists on edges (some graphs already have it)
    if any(("length" not in d) for _, _, d in G.edges(data=True)):
        ox.distance.add_edge_lengths(G)  # v2 API

    print("Loading geocoded stops…")
    df = load_geocoded_stops(args.json)
    df["name"] = ensure_unique_names(df["name"].tolist())

    farms_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)

    depot_gdf = gpd.GeoDataFrame(
        {"name": [EM_NAME], "address": ["2934 Russell St, Detroit, MI 48207"],
         "lon": [EM_LON], "lat": [EM_LAT]},
        geometry=gpd.points_from_xy([EM_LON], [EM_LAT]), crs=4326
    )
    all_pts = pd.concat([depot_gdf, farms_gdf], ignore_index=True)
    all_pts.to_file(os.path.join(args.out, "stops.gpkg"), layer="stops", driver="GPKG")

    print("Snapping & computing distance matrix (DRIVING, directed)…")
    name_to_node = snap_names_to_nodes(G, all_pts)
    D = pairwise_distance_matrix(G, name_to_node)
    D.to_csv(os.path.join(args.out, "pairwise_distances_m.csv"), float_format="%.3f")

    # distance from each stop to EM (directed EM->stop)
    em_row = D.loc[EM_NAME].drop(EM_NAME)
    em_dist_df = em_row.reset_index()
    em_dist_df.columns = ["name", "dist_to_em_m"]
    em_dist_df["dist_to_em_km"] = em_dist_df["dist_to_em_m"] / 1000.0
    em_dist_df.to_csv(os.path.join(args.out, "dist_to_em.csv"), index=False, float_format="%.3f")

    print("Solving TSP on the driving-distance matrix…")
    best_m, best_order = brute_force_best(D, EM_NAME, return_to_depot=(not args.no_return))

    print("Building route geometry (directed)…")
    legs = legs_edges_gdf(G, best_order, name_to_node)
    leg_totals = legs.groupby("leg_index")["edge_m"].sum().rename("leg_m").reset_index()
    route_geom = dissolved_route_geometry(legs)

    route_gdf = gpd.GeoDataFrame(
        {"name": ["EM_Farms_Route"], "total_m": [best_m], "return_to_depot": [not args.no_return],
         "order": [">".join(best_order)]},
        geometry=[route_geom], crs=4326
    )
    legs = gpd.GeoDataFrame(legs.merge(leg_totals, on="leg_index", how="left"),
                            geometry="geometry", crs=4326)
    order_df = pd.DataFrame({"name": best_order, "visit_idx": range(len(best_order))})
    pts_ord = all_pts.merge(order_df, on="name", how="right").sort_values("visit_idx")
    pts_ord = pts_ord.merge(em_dist_df, on="name", how="left")

    out_gpkg = os.path.join(args.out, "em_farms_route.gpkg")
    route_gdf.to_file(out_gpkg, layer="route", driver="GPKG")
    legs.to_file(out_gpkg, layer="route_legs", driver="GPKG")
    gpd.GeoDataFrame(pts_ord, geometry="geometry", crs=4326)\
        .to_file(out_gpkg, layer="stops_ordered", driver="GPKG")

    pd.DataFrame({"stop_index": list(range(len(best_order))), "name": best_order})\
      .to_csv(os.path.join(args.out, "winning_itinerary.csv"), index=False)

    print("\n=== RESULT ===")
    print("Order:", " -> ".join(best_order))
    print(f"Total driving distance: {best_m/1000:.2f} km")
    print("Wrote:")
    print(f"  {out_gpkg}  (layers: route, route_legs, stops_ordered)")
    print("  pairwise_distances_m.csv, dist_to_em.csv, winning_itinerary.csv")
    print("  stops.gpkg")

if __name__ == "__main__":
    main()