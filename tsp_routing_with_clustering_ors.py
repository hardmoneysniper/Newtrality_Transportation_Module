import os, json, argparse, math, sys, importlib
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union, linemerge
from shapely.geometry import LineString

# --- ORS ---
import openrouteservice as ors

# --- OR-Tools (TSP solver) ---
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# Eastern Market depot (constant)
EM_NAME = "Eastern Market"
EM_LON = -83.0416
EM_LAT = 42.3469


# ---------- IO helpers ----------
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


# ---------- Clustering ----------
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
    # sort clusters radially around depot (stable order for outputs)
    cents = []
    for idxs in clusters:
        lon = farms_df.loc[idxs, "lon"].mean(); lat = farms_df.loc[idxs, "lat"].mean()
        cents.append(math.atan2(lat-EM_LAT, lon-EM_LON))
    return [x for _, x in sorted(zip(cents, clusters), key=lambda t: t[0])]


# ---------- ORS Matrix + Directions ----------
def build_ors_client(ors_key=None, base_url=None, timeout=120):
    # Priority: CLI arg > env var
    key = ors_key or os.environ.get("ORS_API_KEY")
    if not key:
        raise RuntimeError("Missing ORS API key. Pass --ors_key or set the ORS_API_KEY env var.")
    return ors.Client(key=key, base_url=base_url or "https://api.openrouteservice.org", timeout=timeout)


def coords_from_df(df):
    # ORS expects [lon, lat]
    return df[["lon","lat"]].to_numpy().tolist()


def full_matrix_ors(client, coords, profile="driving-car", metrics=("duration","distance")):
    res = client.distance_matrix(
        locations=coords,
        profile=profile,
        metrics=list(metrics),
    )
    out = {}
    if "durations" in res and res["durations"] is not None:
        out["duration"] = pd.DataFrame(res["durations"])
    if "distances" in res and res["distances"] is not None:
        out["distance"] = pd.DataFrame(res["distances"])
    if not out:
        raise RuntimeError("ORS Matrix returned no durations/distances.")
    return out


def ors_route_legs_geojson(client, ordered_coords, ordered_names, profile="driving-car",
                            preference="fastest", extra_options=None, units="m"):
    """
    Fetch the whole visit-ordered route in a single ORS Directions call (instead of
    one call per leg) by passing every stop as a waypoint. ORS returns one "segment"
    per leg (with its own distance/duration) plus the full route geometry; we slice
    that geometry per leg using each segment's step way_points.
    Returns (legs_gdf, total_m, total_s).
    """
    r = client.directions(
        coordinates=ordered_coords,
        profile=profile,
        preference=preference,
        format="geojson",
        units=units,
        options=extra_options or {}
    )
    feat = r["features"][0]
    full_coords = feat["geometry"]["coordinates"]
    props = feat["properties"]
    summary = props["summary"]  # {"distance": m, "duration": s} for the whole route

    rows = []
    for leg_i, seg in enumerate(props["segments"], start=1):
        steps = seg.get("steps") or []
        if steps:
            start_wp = min(s["way_points"][0] for s in steps)
            end_wp = max(s["way_points"][1] for s in steps)
        else:
            start_wp, end_wp = 0, len(full_coords) - 1
        leg_coords = full_coords[start_wp:end_wp + 1]
        rows.append({
            "leg_index": leg_i,
            "from_name": ordered_names[leg_i - 1],
            "to_name": ordered_names[leg_i],
            "edge_m": float(seg.get("distance", 0.0)),
            "edge_s": float(seg.get("duration", 0.0)),
            "geometry": LineString(leg_coords),
        })

    legs_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    return legs_gdf, float(summary.get("distance", 0.0)), float(summary.get("duration", 0.0))


# ---------- TSP (OR-Tools) ----------
def tsp_ortools(D, depot_idx, return_to_depot=True, time_limit_s=5):
    """
    Solve the (open or closed) TSP over cost matrix D with Google OR-Tools'
    routing solver (free, local, no external calls). Returns (total_cost, order_idx)
    with order_idx a list of matrix indices, depot first and, if return_to_depot,
    depot last too -- same contract as the old brute-force/2-opt helpers.
    """
    matrix = D.to_numpy(dtype=float) if hasattr(D, "to_numpy") else np.asarray(D, dtype=float)
    n = matrix.shape[0]
    if n <= 1:
        return 0.0, [depot_idx]

    scale = 1000.0  # keep sub-unit precision when rounding to the ints OR-Tools requires
    cost = np.rint(matrix * scale).astype(int)

    if return_to_depot:
        manager = pywrapcp.RoutingIndexManager(n, 1, depot_idx)
    else:
        # Open path: add a dummy end node with 0-cost arcs from every real node
        # so the route can terminate at whichever stop is cheapest, not just the depot.
        dummy = n
        padded = np.zeros((n + 1, n + 1), dtype=int)
        padded[:n, :n] = cost
        cost = padded
        manager = pywrapcp.RoutingIndexManager(n + 1, 1, [depot_idx], [dummy])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(cost[i][j])

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    # No metaheuristic: for these tiny per-cluster instances (a handful of stops),
    # plain local search finds the optimum and returns immediately once it hits a
    # local optimum, instead of a guided-local-search metaheuristic that deliberately
    # keeps searching for the entire time_limit even when there's nothing left to gain.
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools found no solution for the TSP.")

    order_idx = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order_idx.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    end_node = manager.IndexToNode(index)
    if end_node < n:  # real node (depot on a closed tour); dummy end node is dropped
        order_idx.append(end_node)

    total = sum(matrix[a, b] for a, b in zip(order_idx[:-1], order_idx[1:]))
    return total, order_idx


# ---------- Solve & Write ----------
def solve_and_write_cluster_ors(client, depot_row, stops_df, out_dir, cluster_tag,
                                profile="driving-car", metric="duration",
                                preference="fastest", return_to_depot=True,
                                avoid_options=None):
    os.makedirs(out_dir, exist_ok=True)

    # Assemble points: depot first + cluster stops
    all_pts = pd.concat([depot_row, stops_df], ignore_index=True)  # index 0 is depot
    names = all_pts["name"].tolist()
    coords = coords_from_df(all_pts)

    # Directed matrix from ORS
    mats = full_matrix_ors(client, coords, profile=profile, metrics=("duration","distance"))
    D = mats[metric]  # choose “duration” or “distance”

    # TSP
    depot_idx = 0
    best_cost, order_idx = tsp_ortools(D, depot_idx, return_to_depot)

    # Route geometry + per-leg summaries: one ORS Directions call for the whole
    # ordered route, instead of one call per leg.
    ordered_coords = [coords[i] for i in order_idx]
    ordered_names = [names[i] for i in order_idx]
    legs_gdf, total_m, total_s = ors_route_legs_geojson(
        client, ordered_coords, ordered_names, profile=profile, preference=preference,
        extra_options=avoid_options
    )
    try:
        merged = linemerge(unary_union(legs_gdf.geometry.tolist()))
    except Exception:
        merged = unary_union(legs_gdf.geometry.tolist())

    route_gdf = gpd.GeoDataFrame(
        {
            "name": [f"Route_{cluster_tag}"],
            "total_m": [total_m],
            "total_km": [total_m/1000.0],
            "total_s": [total_s],
            "total_min": [total_s/60.0],
            "metric": [metric],
            "preference": [preference],
            "return_to_depot": [return_to_depot],
            "order": [">".join([names[i] for i in order_idx])]
        },
        geometry=[merged], crs=4326
    )

    # ordered stops
    order_df = pd.DataFrame({"idx": order_idx, "visit_idx": range(len(order_idx))})
    pts_ord = all_pts.reset_index(drop=True).merge(order_df, left_index=True, right_on="idx", how="right")\
                     .sort_values("visit_idx").drop(columns=["idx"])
    pts_gdf = gpd.GeoDataFrame(pts_ord, geometry=gpd.points_from_xy(pts_ord["lon"], pts_ord["lat"]), crs=4326)

    # write GPKG + CSVs
    out_gpkg = os.path.join(out_dir, "em_farms_route.gpkg")
    route_gdf.to_file(out_gpkg, layer="route", driver="GPKG")
    legs_gdf.to_file(out_gpkg, layer="route_legs", driver="GPKG")
    pts_gdf.to_file(out_gpkg, layer="stops_ordered", driver="GPKG")

    pd.DataFrame({"stop_index": list(range(len(order_idx))), "name": [names[i] for i in order_idx]})\
      .to_csv(os.path.join(out_dir, "winning_itinerary.csv"), index=False)
    pd.DataFrame({"cluster": [cluster_tag], "total_m": [total_m], "total_km": [total_m/1000.0],
                  "total_s": [total_s], "total_min": [total_s/60.0]})\
      .to_csv(os.path.join(out_dir, "route_summary.csv"), index=False)

    return total_m, total_s, [names[i] for i in order_idx]


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Cluster farms across K trucks and route with openrouteservice.")
    ap.add_argument("--json", required=True, help="stops_geocoded.json (name,address,lon,lat)")
    ap.add_argument("--trucks", type=int, required=True, help="number of clusters/vehicles")
    ap.add_argument("--out", default="out_clusters")
    ap.add_argument("--method", choices=["auto","sweep","kmeans"], default="auto")
    ap.add_argument("--no_return", action="store_true", help="do not return to depot")
    ap.add_argument("--ors_key", help="ORS API key (or set ORS_API_KEY env var)")
    ap.add_argument("--profile", default="driving-car")
    ap.add_argument("--opt_metric", choices=["duration","distance"], default="duration",
                    help="optimize TSP by duration (fastest) or distance")
    ap.add_argument("--preference", choices=["fastest","shortest","recommended"], default="fastest")
    ap.add_argument("--avoid_tollways", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    client = build_ors_client(args.ors_key)

    # load points
    df = load_geocoded_stops(args.json); df["name"] = ensure_unique_names(df["name"].tolist())
    farms_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
    depot_gdf = gpd.GeoDataFrame({"name":[EM_NAME], "address":["2934 Russell St, Detroit, MI 48207"],
                                  "lon":[EM_LON], "lat":[EM_LAT]},
                                 geometry=gpd.points_from_xy([EM_LON],[EM_LAT]), crs=4326)

    # make clusters
    k = max(1, min(int(args.trucks), len(farms_gdf)))
    clusters = None
    if args.method in ("auto","kmeans"):
        try:
            clusters = cluster_kmeans(farms_gdf, k); print("Method: k-means")
        except Exception as e:
            if args.method == "kmeans": raise
            print(f"k-means unavailable ({e}); falling back to sweep.")
    if clusters is None:
        clusters = cluster_sweep(farms_gdf, k); print("Method: sweep")

    # routing options
    avoid_opts = {"avoid_features": ["tollways"]} if args.avoid_tollways else None

    # per cluster solve
    summary = []
    for idx, idxs in enumerate(clusters, start=1):
        stops = farms_gdf.loc[idxs].reset_index(drop=True)
        tag = f"cluster_{idx:02d}"
        out_dir = os.path.join(args.out, tag)
        print(f"Routing {tag}: {len(stops)} stop(s)…")
        total_m, total_s, order = solve_and_write_cluster_ors(
            client=client,
            depot_row=depot_gdf,
            stops_df=stops[["name","address","lon","lat"]],
            out_dir=out_dir,
            cluster_tag=tag,
            profile=args.profile,
            metric=args.opt_metric,
            preference=args.preference,
            return_to_depot=(not args.no_return),
            avoid_options=avoid_opts,
        )
        summary.append({"cluster": tag, "n_stops": len(stops),
                        "total_m": total_m, "total_km": total_m/1000.0,
                        "total_s": total_s, "total_min": total_s/60.0,
                        "order": ">".join(order)})

    sm = pd.DataFrame(summary)
    sm["fleet_total_km"] = sm["total_km"].sum()
    sm["fleet_total_min"] = sm["total_min"].sum()
    sm.to_csv(os.path.join(args.out, "clusters_summary.csv"), index=False)
    print("\n=== CLUSTERED RESULT ===")
    print(sm.to_string(index=False))
    print(f"\nWrote per-cluster folders under: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
        
        
        
# python tsp_routing_with_clustering_ors.py --json stops_from_vendors_geocoded.json --trucks 5 --out out_clusters --method auto --opt_metric duration --preference fastest
# justify the method and outputting how much weight of co2 is avoided/reduced carbon footprint, how many miles saved
# designation in higher foot traffic regions