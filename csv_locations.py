# csv_locations.py
import csv
from typing import Dict, Any, Tuple, Optional

CSV_PATH = "customer_location.csv"

HEADERS = {
    "name": "Account Name",
    "city": "City",
    "address": "Address",
    "county_state": "County State",
    "zip": "Zip Code",
    "lat": "Min of Latitude",
    "lng": "Min of Longitude",
}

def _norm_key(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")

def _split_county_state(val: str) -> Tuple[Optional[str], Optional[str]]:
    """
    'Somerset PA'  -> ('Somerset', 'PA')
    gracefully handles empty/malformed values.
    """
    if not val:
        return None, None
    parts = str(val).strip().split()
    if not parts:
        return None, None
    if len(parts[-1]) in (2,):  # likely state code
        state = parts[-1].upper()
        county = " ".join(parts[:-1]) if len(parts) > 1 else None
        return county, state
    # fallback: treat entire thing as county, state unknown
    return val, None

def load_csv_locations(csv_path: str = CSV_PATH) -> Dict[str, Any]:
    """
    Returns: dict keyed by normalized account name, with:
      {
        "Account Name": str,
        "Address": str,
        "City": str,
        "County": str|None,
        "State": str|None,
        "Zip": str,
        "Latitude": float|None,
        "Longitude": float|None,
        "FormattedAddress": str
      }
    """
    index: Dict[str, Any] = {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get(HEADERS["name"]) or "").strip()
                if not name:
                    continue

                city = (row.get(HEADERS["city"]) or "").strip()
                addr = (row.get(HEADERS["address"]) or "").strip()
                county_state = (row.get(HEADERS["county_state"]) or "").strip()
                zipc = (row.get(HEADERS["zip"]) or "").strip()

                lat_raw = row.get(HEADERS["lat"])
                lng_raw = row.get(HEADERS["lng"])
                try:
                    lat = float(lat_raw) if lat_raw not in (None, "",) else None
                except:
                    lat = None
                try:
                    lng = float(lng_raw) if lng_raw not in (None, "",) else None
                except:
                    lng = None

                county, state = _split_county_state(county_state)
                formatted = ", ".join(
                    [p for p in [addr, city, state, zipc] if p]
                ) or name

                index[_norm_key(name)] = {
                    "Account Name": name,
                    "Address": addr,
                    "City": city,
                    "County": county,
                    "State": state,
                    "Zip": zipc,
                    "Latitude": lat,
                    "Longitude": lng,
                    "FormattedAddress": formatted,
                }
    except FileNotFoundError:
        # If CSV missing, return empty index; app should still run
        return {}

    return index

def to_geojson(locations_index: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the loaded locations into a GeoJSON FeatureCollection."""
    features = []
    for _, v in locations_index.items():
        lat, lng = v.get("Latitude"), v.get("Longitude")
        if lat is None or lng is None:
            continue
        props = {
            "name": v.get("Account Name"),
            "address": v.get("FormattedAddress"),
            "city": v.get("City"),
            "state": v.get("State"),
            "zip": v.get("Zip"),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}
