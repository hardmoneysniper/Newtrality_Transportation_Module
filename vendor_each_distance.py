import json, os
import pandas as pd
import openrouteservice as ors

# Eastern Market (depot) – fixed coords
EM_NAME = "Eastern Market"
EM_LON = -83.0416
EM_LAT = 42.3469

# Inputs / outputs
IN_JSON = "stops_from_vendors_geocoded.json"      # expects {"stops": [{name,address,lon,lat}, ...]}
OUT_CSV = "vendor_to_em.csv"
PROFILE = "driving-car"              # or: 'driving-hgv', 'cycling-regular', etc.
PREFERENCE = "fastest"               # 'fastest' | 'shortest' | 'recommended'

def main():
    # Load stops (vendors only; no depot inside)
    with open(IN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    stops = data["stops"]
    vendors = [{"name": s["name"], "lon": float(s["lon"]), "lat": float(s["lat"])} for s in stops]

    # Build ORS client
    key = os.environ.get("ORS_API_KEY")
    if not key:
        raise RuntimeError("Missing ORS key. Set the ORS_API_KEY env var.")
    client = ors.Client(key=key, timeout=120)

    # Locations: index 0 is Eastern Market, then all vendors
    coords = [[EM_LON, EM_LAT]] + [[v["lon"], v["lat"]] for v in vendors]
    # We only need EM -> vendors, so use sources=[0] and destinations=1..n
    destinations = list(range(1, len(coords)))

    res = client.distance_matrix(
        locations=coords,
        profile=PROFILE,
        metrics=["distance", "duration"],
        sources=[0],
        destinations=destinations
    )

    # ORS returns 2D arrays: one row (source=EM) by many destinations
    dists = res.get("distances") or []
    durs  = res.get("durations") or []
    if not dists and not durs:
        raise RuntimeError("ORS Matrix returned no data.")

    out_rows = []
    for j, v in enumerate(vendors):
        dist_m = float(dists[0][j]) if dists else None
        dur_s  = float(durs[0][j])  if durs  else None
        out_rows.append({
            "name": v["name"],
            "distance_m": dist_m,
            "distance_km": None if dist_m is None else dist_m/1000.0,
            "duration_s": dur_s,
            "duration_min": None if dur_s is None else dur_s/60.0
        })

    df = pd.DataFrame(out_rows).sort_values("distance_km", na_position="last")
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV} with {len(df)} vendors.")
    print(df.to_string(index=False, justify='left', max_colwidth=40))

if __name__ == "__main__":
    main()