# clustered_routing.py
# Cluster farms across K trucks, solve a TSP per cluster using DRIVING distances (directed),
# and write per-cluster outputs identical to the single-route workflow.

import os, json, argparse, math, itertools, sys, importlib
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

def full_distance_matrix(G, name_to_node):
    names = list(name_to_node.keys())
    D = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        D.at[a, a] = 0.0
        for b in names[i+1:]:
            d_ab = nx.shortest_path_length(G, name_to_node[a], name_to_node[b], weight="length")
            d_ba = nx.shortest_path_length(G, name_to_node[b], name_to_node[a], weight="length")
            D.at[a, b] = d_ab
            D.at[b, a] = d_ba
    return D

def tsp_bruteforce(D, depot, return_to_depot=True):
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

def tsp_2opt_heuristic(D, depot, return_to_depot=True):
    names = list(D.index)
    stops = [n for n in names if n != depot]
    unvisited = set(stops)
    route = [depot]; cur = depot
    while unvisited:
        nxt = min(unvisited, key=lambda s: D.loc[cur, s])
        route.append(nxt); unvisited.remove(nxt); cur = nxt
    if return_to_depot: route.append(depot)
    def route_len(r): return sum(D.loc[a,b] for a,b in zip(r[:-1], r[1:]))
    improved = True; best = route; best_len = route_len(route)
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                if j-i==1: continue
                new = best[:i] + best[i:j][::-1] + best[j:]
                new_len = route_len(new)
                if new_len < best_len - 1e-6:
                    best, best_len, improved = new, new_len, True
                    break
            if improved: break
    return best_len, best

# -------- FIXED: robust for MultiDiGraph and DiGraph --------
def edge_gdf_from_route(G, route_nodes):
    rows = []
    is_multi = isinstance(G, (nx.MultiDiGraph, nx.MultiGraph))
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        if is_multi:
            edict = G.get_edge_data(u, v)
            if edict is None:
                raise RuntimeError(f"No directed edge between {u} -> {v}.")
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
        segs["leg_index"] = i; segs["from_name"] = a; segs["to_name"] = b
        frames.append(segs)
    return pd.concat(frames, ignore_index=True)

def dissolved_route_geometry(edges_gdf):
    geom = unary_union(edges_gdf.geometry.tolist())
    try: return linemerge(geom)
    except Exception: return geom

def cluster_sweep(farms_df, k, depot_lon=EM_LON, depot_lat=EM_LAT):
    angles = []
    for i, r in farms_df.iterrows():
        dx = r["lon"] - depot_lon; dy = r["lat"] - depot_lat
        angles.append((i, math.atan2(dy, dx)))
    angles.sort(key=lambda t: t[1])
    n = len(angles); k = max(1, min(k, n))
    base = n // k; rem = n % k
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    clusters = []; idx = 0
    for s in sizes:
        clusters.append([angles[j][0] for j in range(idx, idx+s)]); idx += s
    return clusters

def cluster_kmeans(farms_df, k, random_state=42):
    if importlib.util.find_spec("sklearn") is None:
        raise ImportError("scikit-learn not installed")
    from sklearn.cluster import KMeans
    X = farms_df[["lon","lat"]].to_numpy()
    k = max(1, min(k, len(farms_df)))
    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
    clusters = [farms_df.index[labels==c].tolist() for c in range(k)]
    cents = []
    for idxs in clusters:
        lon = farms_df.loc[idxs, "lon"].mean(); lat = farms_df.loc[idxs, "lat"].mean()
        cents.append(math.atan2(lat-EM_LAT, lon-EM_LON))
    return [x for _, x in sorted(zip(cents, clusters), key=lambda t: t[0])]

def solve_and_write_cluster(G, D_full, name_to_node, all_pts, depot_name, stop_names, out_dir, cluster_tag, return_to_depot):
    os.makedirs(out_dir, exist_ok=True)
    names = [depot_name] + stop_names
    D = D_full.loc[names, names].copy()

    if len(names) <= 11:
        best_m, best_order = tsp_bruteforce(D, depot_name, return_to_depot)
    else:
        best_m, best_order = tsp_2opt_heuristic(D, depot_name, return_to_depot)

    legs = legs_edges_gdf(G, best_order, name_to_node)
    leg_totals = legs.groupby("leg_index")["edge_m"].sum().rename("leg_m").reset_index()
    route_geom = dissolved_route_geometry(legs)

    route_gdf = gpd.GeoDataFrame(
        {"name": [f"Route_{cluster_tag}"], "total_m": [best_m], "return_to_depot": [return_to_depot],
         "order": [">".join(best_order)]},
        geometry=[route_geom], crs=4326
    )
    legs = gpd.GeoDataFrame(legs.merge(leg_totals, on="leg_index", how="left"), geometry="geometry", crs=4326)
    order_df = pd.DataFrame({"name": best_order, "visit_idx": range(len(best_order))})
    pts_ord = all_pts.merge(order_df, on="name", how="right").sort_values("visit_idx")

    em_row = D_full.loc[depot_name].drop(depot_name).reset_index()
    em_row.columns = ["name","dist_to_em_m"]; em_row["dist_to_em_km"] = em_row["dist_to_em_m"]/1000.0
    pts_ord = pts_ord.merge(em_row, on="name", how="left")

    out_gpkg = os.path.join(out_dir, "em_farms_route.gpkg")
    route_gdf.to_file(out_gpkg, layer="route", driver="GPKG")
    legs.to_file(out_gpkg, layer="route_legs", driver="GPKG")
    gpd.GeoDataFrame(pts_ord, geometry="geometry", crs=4326)\
        .to_file(out_gpkg, layer="stops_ordered", driver="GPKG")

    D.to_csv(os.path.join(out_dir, "pairwise_distances_m.csv"), float_format="%.3f")
    pd.DataFrame({"stop_index": list(range(len(best_order))), "name": best_order})\
      .to_csv(os.path.join(out_dir, "winning_itinerary.csv"), index=False)
    pd.DataFrame({"cluster": [cluster_tag], "total_m": [best_m], "total_km": [best_m/1000.0]})\
      .to_csv(os.path.join(out_dir, "route_summary.csv"), index=False)

    return best_m, best_order

def main():
    parser = argparse.ArgumentParser(description="Cluster farms across K trucks and route each cluster (DRIVING TSP).")
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--trucks", type=int, required=True)
    parser.add_argument("--out", default="out_clusters")
    parser.add_argument("--method", choices=["auto","sweep","kmeans"], default="auto")
    parser.add_argument("--no_return", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True); ox.settings.use_cache = True

    print("Loading network…")
    G = ox.load_graphml(args.graphml)
    # Keep it DIRECTED; if undirected, convert to DiGraph (v2 API)
    if not G.is_directed():
        G = ox.convert.to_digraph(G, weight="length")
    # Ensure 'length' present on edges
    if any(("length" not in d) for _, _, d in G.edges(data=True)):
        ox.distance.add_edge_lengths(G)

    print("Loading geocoded stops…")
    df = load_geocoded_stops(args.json); df["name"] = ensure_unique_names(df["name"].tolist())
    farms_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
    depot_gdf = gpd.GeoDataFrame(
        {"name":[EM_NAME], "address":["2934 Russell St, Detroit, MI 48207"], "lon":[EM_LON], "lat":[EM_LAT]},
        geometry=gpd.points_from_xy([EM_LON],[EM_LAT]), crs=4326
    )
    all_pts = pd.concat([depot_gdf, farms_gdf], ignore_index=True)

    print("Snapping & computing full driving-distance matrix…")
    name_to_node = snap_names_to_nodes(G, all_pts)
    D_full = full_distance_matrix(G, name_to_node)

    k = max(1, min(int(args.trucks), len(farms_gdf)))
    print(f"Clustering {len(farms_gdf)} farms into {k} cluster(s)…")
    clusters = None
    if args.method in ("auto","kmeans"):
        try:
            clusters = cluster_kmeans(farms_gdf, k); print("Method: k-means")
        except Exception as e:
            if args.method=="kmeans": raise
            print(f"k-means unavailable ({e}); falling back to sweep.")
    if clusters is None:
        clusters = cluster_sweep(farms_gdf, k); print("Method: sweep")

    summary_rows = []
    for idx, idxs in enumerate(clusters, start=1):
        stops = farms_gdf.loc[idxs, "name"].tolist()
        tag = f"cluster_{idx:02d}"; out_dir = os.path.join(args.out, tag)
        print(f"Routing {tag}: {len(stops)} stop(s)…")
        best_m, best_order = solve_and_write_cluster(
            G=G, D_full=D_full, name_to_node=name_to_node, all_pts=all_pts,
            depot_name=EM_NAME, stop_names=stops, out_dir=out_dir,
            cluster_tag=tag, return_to_depot=(not args.no_return)
        )
        summary_rows.append({"cluster": tag, "n_stops": len(stops),
                             "total_m": best_m, "total_km": best_m/1000.0,
                             "order": ">".join(best_order)})

    summary = pd.DataFrame(summary_rows); summary["fleet_total_km"] = summary["total_km"].sum()
    summary.to_csv(os.path.join(args.out, "clusters_summary.csv"), index=False)
    print("\n=== CLUSTERED RESULT ===")
    print(summary.to_string(index=False))
    print(f"\nWrote per-cluster folders under: {args.out}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
        
# python tsp_routing_with_clustering.py --graphml net_em\network.graphml --json stops_from_vendors_geocoded.json  --trucks 4 --out out_clusters --method auto