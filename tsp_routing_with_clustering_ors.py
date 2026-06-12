import os, json, argparse, math, itertools, time, sys, importlib
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union, linemerge
from shapely.geometry import LineString

# --- ORS ---
import openrouteservice as ors

ORS_HARDCODED_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjNiMjA0NmE5Mzc1NDQxZmViNjUwYTUwN2VjYzYwZTA1IiwiaCI6Im11cm11cjY0In0="

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
    # Priority: CLI arg > env var > hardcoded
    key = ors_key or os.environ.get("ORS_API_KEY") or ORS_HARDCODED_KEY
    if not key or key == "YOUR_ORS_KEY_HERE":
        raise RuntimeError("Missing ORS API key. Pass --ors_key, set ORS_API_KEY, or edit ORS_HARDCODED_KEY.")
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


def ors_leg_geojson(client, a, b, profile="driving-car", preference="fastest", extra_options=None, units="m"):
    r = client.directions(
        coordinates=[a, b],
        profile=profile,
        preference=preference,
        format="geojson",
        units=units,
        options=extra_options or {}
    )
    feat = r["features"][0]
    geom = feat["geometry"]
    summ = feat["properties"]["summary"]  # {"distance": m, "duration": s}
    return geom, float(summ.get("distance", 0.0)), float(summ.get("duration", 0.0))


# ---------- TSP ----------
def tsp_bruteforce(D, depot_idx, return_to_depot=True):
    n = D.shape[0]
    others = [i for i in range(n) if i != depot_idx]
    best = None; order = None
    for perm in itertools.permutations(others):
        seq = [depot_idx] + list(perm)
        if return_to_depot: seq += [depot_idx]
        total = sum(D.iloc[a, b] for a, b in zip(seq[:-1], seq[1:]))
        if best is None or total < best:
            best, order = total, seq
    return best, order


def tsp_2opt(D, depot_idx, return_to_depot=True):
    n = D.shape[0]
    others = [i for i in range(n) if i != depot_idx]
    # nearest-neighbor init
    unv = set(others); route = [depot_idx]; cur = depot_idx
    while unv:
        nxt = min(unv, key=lambda j: D.iloc[cur, j]); route.append(nxt); unv.remove(nxt); cur = nxt
    if return_to_depot: route.append(depot_idx)
    def length(r): return sum(D.iloc[a,b] for a,b in zip(r[:-1], r[1:]))
    best = route; best_len = length(route); improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                if j-i == 1: continue
                cand = best[:i] + best[i:j][::-1] + best[j:]
                L = length(cand)
                if L < best_len - 1e-9:
                    best, best_len, improved = cand, L, True
                    break
            if improved: break
    return best_len, best


# ---------- Solve & Write ----------
def solve_and_write_cluster_ors(client, depot_row, stops_df, out_dir, cluster_tag,
                                profile="driving-car", metric="duration",
                                preference="fastest", return_to_depot=True,
                                avoid_options=None, per_leg_sleep=0.2):
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
    if len(names) <= 11:
        best_cost, order_idx = tsp_bruteforce(D, depot_idx, return_to_depot)
    else:
        best_cost, order_idx = tsp_2opt(D, depot_idx, return_to_depot)

    # Per-leg directions (geometry + leg summaries)
    leg_rows = []
    geoms = []
    total_m = 0.0; total_s = 0.0
    for leg_i, (a_idx, b_idx) in enumerate(zip(order_idx[:-1], order_idx[1:]), start=1):
        a = coords[a_idx]; b = coords[b_idx]
        geom, leg_m, leg_s = ors_leg_geojson(
            client, a, b, profile=profile, preference=preference,
            extra_options=avoid_options
        )
        geoms.append(geom)
        total_m += leg_m; total_s += leg_s
        leg_rows.append({
            "leg_index": leg_i,
            "from_name": names[a_idx],
            "to_name": names[b_idx],
            "edge_m": leg_m,
            "edge_s": leg_s,
            "geometry": LineString([(x, y) for x, y in geom["coordinates"]])
        })
        if per_leg_sleep > 0:
            time.sleep(per_leg_sleep)

    legs_gdf = gpd.GeoDataFrame(leg_rows, geometry="geometry", crs=4326)
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
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between per-leg directions calls")
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
            per_leg_sleep=args.sleep
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