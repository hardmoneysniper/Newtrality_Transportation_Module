# csvtransformer.py
# Convert a vendor CSV into:
# {
#   "stops": [{"name": "...", "address": "Street, City, ST ZIP"}, ...]
# }
#
# Usage (Windows examples):
#   python csvtransformer.py --in "Vendors.csv" --out "stops_from_vendors.json"
#   # If your file needs a specific encoding:
#   python csvtransformer.py --in "Vendors.csv" --out "stops.json" --encoding cp1252
#   # If it's TSV or odd-delimited:
#   python csvtransformer.py --in "Vendors.tsv" --out "stops.json" --delimiter "\t"

import argparse, json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

def read_csv_robust(path: Path, encoding_opt: Optional[str], delimiter_opt: Optional[str]) -> pd.DataFrame:
    """
    Try to read a CSV with:
      1) user-specified encoding/delimiter if provided,
      2) common encodings with C engine, then python engine (no low_memory),
      3) chardet detection (if installed) with python engine.
    """
    # If user specified both, try them first
    if encoding_opt or delimiter_opt:
        try:
            kw = dict(encoding=encoding_opt) if encoding_opt else {}
            if delimiter_opt:
                kw["sep"] = delimiter_opt
            return pd.read_csv(path, **kw)
        except Exception:
            # Try python engine for weird quoting/newlines
            kw = dict(encoding=encoding_opt) if encoding_opt else {}
            if delimiter_opt:
                kw["sep"] = delimiter_opt
            return pd.read_csv(path, engine="python", on_bad_lines="skip", **kw)

    # Otherwise, try auto encodings
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        # First: default (C) engine
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            pass
        except Exception:
            # Try python engine WITHOUT low_memory
            try:
                return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip", sep=None)
            except UnicodeDecodeError:
                pass
            except Exception:
                pass

    # Last resort: try chardet if available
    try:
        import chardet  # optional
        raw = path.read_bytes()
        enc = chardet.detect(raw).get("encoding") or "latin1"
        return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip", sep=None)
    except Exception as e:
        raise RuntimeError(f"Could not read CSV robustly. Last error: {e}")

def pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _fmt_zip(z) -> Optional[str]:
    if pd.isna(z): return None
    try:
        if isinstance(z, float) and z.is_integer():
            return f"{int(z):05d}"
        s = str(z).strip()
        # remove .0 if it looks like a floaty ZIP
        if s.endswith(".0"):
            s = s[:-2]
        return s
    except Exception:
        return str(z).strip()

def build_address(row: pd.Series) -> Optional[str]:
    # Prefer Geocodio columns
    a1   = row.get("Geocodio Address Line 1")
    city = row.get("Geocodio City")
    st   = row.get("Geocodio State")
    zipc = _fmt_zip(row.get("Geocodio Postal Code"))

    if pd.notna(a1) or pd.notna(city) or pd.notna(st) or pd.notna(zipc):
        parts = []
        if pd.notna(a1):   parts.append(str(a1).strip())
        if pd.notna(city): parts.append(str(city).strip())
        if pd.notna(st):   parts.append(str(st).strip())
        addr = ", ".join(parts) if parts else ""
        if zipc:
            addr = f"{addr} {zipc}".strip()
        return addr if addr else None

    # Generic fallbacks
    a_gen  = row.get("Address") or row.get("Street Address") or row.get("Street")
    city2  = row.get("City")
    st2    = row.get("State")
    zip2   = _fmt_zip(row.get("ZIP") or row.get("Zip") or row.get("Postal Code"))
    if pd.notna(a_gen):
        parts = [str(a_gen).strip()]
        if pd.notna(city2): parts.append(str(city2).strip())
        if pd.notna(st2):   parts.append(str(st2).strip())
        addr = ", ".join(parts)
        if zip2:
            addr = f"{addr} {zip2}"
        return addr if addr.strip() else None

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="Path to the input CSV/TSV")
    ap.add_argument("--out", dest="out_json", required=True, help="Path to write JSON")
    ap.add_argument("--encoding", dest="encoding", default=None, help="Force a specific encoding (e.g., cp1252, latin1)")
    ap.add_argument("--delimiter", dest="delimiter", default=None, help=r"Force a delimiter (e.g., ',' or '\t')")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_json)

    df = read_csv_robust(in_path, args.encoding, args.delimiter)

    # Choose a name column; fallback derives from Address Line 1 (or row index)
    name_col = pick_first(df, ["Kitchen Name", "Vendor", "Farm Name", "Business Name", "Place Name"])
    if name_col is None:
        name_col = "DerivedName"
        base = df.get("Geocodio Address Line 1", pd.Series("", index=df.index))
        df[name_col] = base.fillna("").astype(str)

    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        addr = build_address(row)
        if name and addr:
            records.append({"name": name, "address": addr})

    # Deduplicate by (name, address), case-insensitive
    seen = set(); unique = []
    for r in records:
        key = (r["name"].lower(), r["address"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(r)

    payload = {"stops": unique}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(unique)} stops to {out_path}")

if __name__ == "__main__":
    main()
