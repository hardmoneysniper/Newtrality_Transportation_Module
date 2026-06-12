import os, json, subprocess, sys, shutil, math
from pathlib import Path
from typing import List, Dict, Any

import geopandas as gpd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import openrouteservice as ors  # for original-vs-current distance comparison

# ---- config: defaults that mirror your CLI ----
JSON_PATH   = "stops_from_vendors_geocoded.json"
OUT_DIR     = "out_clusters"
METHOD      = "auto"
OPT_METRIC  = "duration"
PREFERENCE  = "fastest"
SCRIPT      = "tsp_routing_with_clustering_ors.py"   # your existing script

# Eastern Market location (must match clustering script)
EM_NAME = "Eastern Market"
EM_LON = -83.0416
EM_LAT = 42.3469

# Emission factors (kg CO2e per km) for a ~5t truck.
# You can override these via env vars if you have better numbers.
EF_DIESEL_KG_PER_KM   = float(os.environ.get("EF_DIESEL_KG_PER_KM", "0.55"))
EF_ELECTRIC_KG_PER_KM = float(os.environ.get("EF_ELECTRIC_KG_PER_KM", "0.15"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html on root
@app.get("/", response_class=FileResponse)
def index():
    return FileResponse("index.html")

@app.get("/style.css")
def get_style():
    return FileResponse("style.css")

# Optionally expose other static files under /static
app.mount("/static", StaticFiles(directory="."), name="static")


class RunReq(BaseModel):
    trucks: int


def run_clustering(trucks: int) -> str:
    """Invoke your clustering script with your required defaults, after clearing old outputs."""
    if trucks < 1:
        raise HTTPException(400, "trucks must be >= 1")

    # Clear out old cluster results so we don't mix runs
    out_path = Path(OUT_DIR)
    if out_path.exists() and out_path.is_dir():
        shutil.rmtree(out_path)

    cmd = [
        sys.executable, SCRIPT,
        "--json", JSON_PATH,
        "--trucks", str(trucks),
        "--out", OUT_DIR,
        "--method", METHOD,
        "--opt_metric", OPT_METRIC,
        "--preference", PREFERENCE,
    ]
    # If you want to inject ORS API key here, uncomment:
    # env = dict(os.environ)
    # env["ORS_API_KEY"] = "YOUR_ORS_KEY"
    # result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(500, f"Routing failed:\n{result.stderr or result.stdout}")
    return result.stdout


def gdf_to_featurecollection(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    # Always ensure WGS84 for the web map
    if gdf.crs and str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
        gdf = gdf.to_crs(4326)
    return json.loads(gdf.to_json())


def load_cluster_geojson(cluster_dir: Path) -> Dict[str, Any]:
    """Read route, legs, and ordered stops from the cluster's GPKG and return as GeoJSON + summary."""
    gpkg = cluster_dir / "em_farms_route.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(f"Missing {gpkg}")
    route = gpd.read_file(gpkg, layer="route")
    legs  = gpd.read_file(gpkg, layer="route_legs")
    stops = gpd.read_file(gpkg, layer="stops_ordered")

    route_row = route.iloc[0].to_dict()
    summary = {
        "name": route_row.get("name"),
        "total_km": float(route_row.get("total_km") or 0.0),
        "total_min": float(route_row.get("total_min") or 0.0),
        "order": route_row.get("order"),
        "return_to_depot": bool(route_row.get("return_to_depot")),
        "metric": route_row.get("metric"),
        "preference": route_row.get("preference"),
    }
    return {
        "cluster": cluster_dir.name,
        "summary": summary,
        "route": gdf_to_featurecollection(route),
        "route_legs": gdf_to_featurecollection(legs),
        "stops_ordered": gdf_to_featurecollection(stops),
    }


def list_cluster_dirs(base: Path):
    return sorted([p for p in base.glob("cluster_*") if p.is_dir()])


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Fallback great-circle distance if ORS key isn't available."""
    R = 6371.0088  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))


def compute_original_vendor_scenario(json_path: str, profile: str = "driving-car"):
    """
    'Original' scenario: each vendor drives directly between their location and Eastern Market.
    Returns (total_km, per_vendor_list).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stops = data.get("stops", [])
    if not stops:
        return 0.0, []

    vendors = []
    coords = [[EM_LON, EM_LAT]]  # index 0 is EM
    for s in stops:
        vendors.append({
            "name": s.get("name", "Vendor"),
            "lon": float(s["lon"]),
            "lat": float(s["lat"]),
        })
        coords.append([float(s["lon"]), float(s["lat"])])

    key = os.environ.get("ORS_API_KEY")
    per_vendor = []
    total_km = 0.0

    if key:
        client = ors.Client(key=key, timeout=60)
        dest_idx = list(range(1, len(coords)))
        res = client.distance_matrix(
            locations=coords,
            profile=profile,
            metrics=["distance"],
            sources=[0],
            destinations=dest_idx
        )
        dists = (res.get("distances") or [[]])[0]
        for v, d_m in zip(vendors, dists):
            km = float(d_m) / 1000.0
            total_km += km
            per_vendor.append({"name": v["name"], "distance_km": km})
    else:
        # Fallback: approximate with haversine if no ORS key
        for v in vendors:
            km = haversine_km(EM_LAT, EM_LON, v["lat"], v["lon"])
            total_km += km
            per_vendor.append({"name": v["name"], "distance_km": km})

    return total_km, per_vendor


def co2_from_km(km: float, ef_kg_per_km: float) -> float:
    """Convert distance (km) to kg CO2e with a given per-km factor."""
    return float(km) * float(ef_kg_per_km)


@app.post("/run")
def run(req: RunReq):
    stdout = run_clustering(req.trucks)
    base = Path(OUT_DIR)
    if not base.exists():
        raise HTTPException(500, f"Output directory {OUT_DIR} not found.")

    clusters = []
    current_total_km = 0.0

    for cdir in list_cluster_dirs(base):
        try:
            c = load_cluster_geojson(cdir)
            clusters.append(c)
            current_total_km += float(c["summary"]["total_km"] or 0.0)
        except Exception as e:
            clusters.append({"cluster": cdir.name, "error": str(e)})

    # Original scenario: N solo trips from vendors to Eastern Market
    original_km, vendor_trips = compute_original_vendor_scenario(JSON_PATH)

    delta_km = original_km - current_total_km  # >0 means distance saved

    # Diesel footprints
    original_kg_diesel = co2_from_km(original_km, EF_DIESEL_KG_PER_KM)
    current_kg_diesel  = co2_from_km(current_total_km, EF_DIESEL_KG_PER_KM)
    delta_kg_diesel    = original_kg_diesel - current_kg_diesel

    # Electric footprints
    original_kg_elec = co2_from_km(original_km, EF_ELECTRIC_KG_PER_KM)
    current_kg_elec  = co2_from_km(current_total_km, EF_ELECTRIC_KG_PER_KM)
    delta_kg_elec    = original_kg_elec - current_kg_elec

    metrics = {
        "original_km": original_km,
        "current_km": current_total_km,
        "delta_km": delta_km,
        # diesel
        "original_kgco2e_diesel": original_kg_diesel,
        "current_kgco2e_diesel": current_kg_diesel,
        "delta_kgco2e_diesel": delta_kg_diesel,
        # electric
        "original_kgco2e_electric": original_kg_elec,
        "current_kgco2e_electric": current_kg_elec,
        "delta_kgco2e_electric": delta_kg_elec,
        # counts
        "n_vendors": len(vendor_trips),
        "n_clusters": len([c for c in clusters if "error" not in c]),
    }

    return {"ok": True, "stdout": stdout, "clusters": clusters, "metrics": metrics}