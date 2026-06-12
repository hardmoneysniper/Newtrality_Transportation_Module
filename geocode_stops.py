# geocode_stops.py
# Read stops JSON (either {"stops":[...]} or a bare list). Each stop has:
#   required: name, address
#   optional: lon, lat
# Geocode ONLY those missing lon/lat (default provider: Nominatim with polite rate limits).
# Write a new JSON where (by default) all existing lon/lat are preserved and missing ones are filled.
# If some fail and you pass --strict, the script will error out; otherwise it writes an .errors.txt.
#
# Usage examples (Windows):
#   python geocode_stops.py --in_json stops_from_vendors.json --out_json stops_geocoded.json --email you@umich.edu
#   python geocode_stops.py --in_json ... --out_json ... --provider arcgis
#   python geocode_stops.py --in_json ... --out_json ... --provider google  --api_key YOUR_GOOGLE_KEY
#   python geocode_stops.py --in_json ... --out_json ... --provider geocodio --api_key YOUR_GEOCODIO_KEY
#   # Stricter: fail if any address can't be geocoded
#   python geocode_stops.py --in_json ... --out_json ... --strict
#
# Deps:
#   pip install geopy

import json, argparse, sys
from pathlib import Path
from typing import Dict, Any, Optional
from geopy.geocoders import Nominatim, ArcGIS, GoogleV3
try:
    from geopy.geocoders import Geocodio  # geopy >= 2.4
except Exception:
    Geocodio = None
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError

def validate_stop(s, i):
    if "name" not in s:
        raise ValueError(f"Stop index {i} missing 'name'.")
    if "address" not in s:
        raise ValueError(f"Stop '{s.get('name','(unnamed)')}' missing 'address'.")
    return s

def load_stops(p: Path):
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "stops" in data:
        return data["stops"], True
    elif isinstance(data, list):
        return data, False
    else:
        raise ValueError("JSON must be an object with key 'stops' or a list of stops.")

def save_stops(p: Path, stops, had_root_stops: bool):
    payload = {"stops": stops} if had_root_stops else stops
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def load_cache(p: Path) -> Dict[str, Any]:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_cache(p: Path, cache: Dict[str, Any]) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)

def norm_addr(s: str) -> str:
    return " ".join(s.strip().lower().split())

def make_geocoder(provider: str, email: Optional[str], api_key: Optional[str], timeout: int):
    provider = provider.lower()
    if provider == "nominatim":
        # Per Nominatim policy: descriptive UA; include email if possible.
        ua = f"em-routing-script/1.0 ({email})" if email else "em-routing-script/1.0"
        return Nominatim(user_agent=ua, timeout=timeout)
    if provider == "arcgis":
        return ArcGIS(timeout=timeout)
    if provider == "google":
        if not api_key:
            raise ValueError("Google provider selected but --api_key not provided.")
        return GoogleV3(api_key=api_key, timeout=timeout)
    if provider == "geocodio":
        if Geocodio is None:
            raise ValueError("Geocodio provider not available in this geopy version.")
        if not api_key:
            raise ValueError("Geocodio provider selected but --api_key not provided.")
        return Geocodio(api_key=api_key, timeout=timeout)
    raise ValueError(f"Unknown provider: {provider}")

def main():
    parser = argparse.ArgumentParser(description="Geocode stops JSON into lon/lat (for records missing coords).")
    parser.add_argument("--in_json",  required=True, help="Input JSON: {'stops':[...]} or a list of stops")
    parser.add_argument("--out_json", required=True, help="Output JSON with lon/lat added/preserved")
    parser.add_argument("--provider", default="nominatim", choices=["nominatim","arcgis","google","geocodio"],
                        help="Geocoding provider (default: nominatim)")
    parser.add_argument("--api_key",  default=None, help="API key (google/geocodio)")
    parser.add_argument("--email",    default=None, help="Your email (recommended for Nominatim UA)")
    parser.add_argument("--timeout",  type=int, default=12, help="Per-request timeout seconds (default 12)")
    parser.add_argument("--min_delay",type=float, default=1.1, help="Seconds between requests (default 1.1 for OSM)")
    parser.add_argument("--retries",  type=int, default=3, help="Max retries on error (default 3)")
    parser.add_argument("--error_wait", type=float, default=5.0, help="Seconds to wait after error before retry")
    parser.add_argument("--cache",    default="geocode_cache.json", help="Path to cache JSON (default: geocode_cache.json)")
    parser.add_argument("--strict",   action="store_true",
                        help="Fail if any stop could not be geocoded (otherwise write errors file and continue)")
    args = parser.parse_args()

    in_path  = Path(args.in_json)
    out_path = Path(args.out_json)
    cache_path = Path(args.cache)

    stops, had_root = load_stops(in_path)
    # validate required fields up front
    for i, s in enumerate(stops):
        validate_stop(s, i)

    # Only initialize geocoder if anyone needs it
    needs = [i for i, s in enumerate(stops) if not ("lon" in s and "lat" in s)]
    geocode = None
    if needs:
        geocoder = make_geocoder(args.provider, args.email, args.api_key, args.timeout)
        geocode = RateLimiter(
            geocoder.geocode,
            min_delay_seconds=args.min_delay,
            max_retries=args.retries,
            error_wait_seconds=args.error_wait,
            swallow_exceptions=False,
        )
        print(f"{len(needs)} stop(s) missing lon/lat… provider={args.provider}")

    # cache
    cache = load_cache(cache_path)

    out_stops = []
    errors = []
    for s in stops:
        name = s["name"].strip()
        addr = s["address"].strip()
        out = {"name": name, "address": addr}

        # Keep provided coords if present
        if "lon" in s and "lat" in s:
            out["lon"] = float(s["lon"])
            out["lat"] = float(s["lat"])
            print(f"✔ {name}: using provided lon/lat")
            out_stops.append(out)
            continue

        # Lookup from cache or geocode
        key = norm_addr(addr)
        cached = cache.get(key)
        if cached and "lon" in cached and "lat" in cached:
            out["lon"] = float(cached["lon"])
            out["lat"] = float(cached["lat"])
            print(f"✔ (cache) {name}: {addr} -> ({out['lat']:.6f}, {out['lon']:.6f})")
            out_stops.append(out)
            continue

        if geocode is None:
            # Should not happen since needs non-empty implies geocode is set
            errors.append(f"{name} | {addr} | No geocoder configured")
            out_stops.append(out)  # without coords for now
            continue

        try:
            loc = geocode(addr, timeout=args.timeout)  # pass explicit timeout too
            if loc is None:
                raise GeocoderServiceError("No result")
            out["lon"] = float(loc.longitude)
            out["lat"] = float(loc.latitude)
            print(f"✔ {name}: geocoded to ({out['lat']:.6f}, {out['lon']:.6f})")
            cache[key] = {"lon": out["lon"], "lat": out["lat"], "provider": args.provider}
            save_cache(cache_path, cache)  # persist as we go
        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError, Exception) as e:
            msg = f"{name} | {addr} | {type(e).__name__}: {e}"
            print(f"✖ {msg}")
            errors.append(msg)
            if args.strict:
                # fail fast in strict mode
                print("\nStrict mode: a stop failed to geocode; aborting.", file=sys.stderr)
                sys.exit(2)

        out_stops.append(out)

    # If any missing lon/lat and not strict, write an errors file but still save what we have
    if errors:
        err_path = out_path.with_suffix(".errors.txt")
        err_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"\nCompleted with {len(errors)} error(s). See: {err_path}")

    # Ensure EVERY stop has lon/lat if not strict? We keep partial results; routing scripts can decide how to handle.
    save_stops(out_path, out_stops, had_root)
    print(f"Wrote {len(out_stops)} stop(s) to {out_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)