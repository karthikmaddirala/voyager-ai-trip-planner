"""
tools.py
────────
ALL tools use REAL FREE APIs — no mocked data, no API keys needed
(except Anthropic, which is in config.py).

APIs used:
  - Open-Meteo Geocoding & Forecast    (no key)  → coordinates, weather
  - OpenStreetMap Overpass API         (no key)  → attractions, restaurants, hotels, airports
  - REST Countries                     (no key)  → country info (visa, language, plug, etc.)
  - Nager.Date                         (no key)  → public holidays
  - Frankfurter (ECB rates)            (no key)  → currency conversion

Pure Python logic:
  - cluster_attractions   → groups by geographic proximity
  - estimate_budget       → cost calculation
  - create_itinerary      → marker tool
"""

import math
import re
import time
import requests
from datetime import datetime
from collections import defaultdict
from urllib.parse import quote

from logutil import log


def geocode_places(queries: list, near_lat=None, near_lng=None):
    """
    Nominatim (OpenStreetMap, free, no key) — POI-level geocoding so individual
    attractions (Hanging Lake, Glenwood Hot Springs Pool) can be pinned on the
    map. Open-Meteo's geocoder only does cities, hence a second geocoder here.
    Best-effort: returns {query, lat, lng} per input (lat/lng None if not found).
    Polite 1 req/s per Nominatim usage policy, so keep query lists short.
    """
    headers = {"User-Agent": "TripPlannerAgent/1.0 (trip planning demo)"}
    out = []
    for q in queries:
        hit = _geocode_one(q, near_lat, near_lng, headers)
        if hit:
            out.append({"query": q, "lat": hit[0], "lng": hit[1]})
            log("geocode", f"✓ {q[:45]}")
        else:
            out.append({"query": q, "lat": None, "lng": None})
            log("geocode", f"✗ not found: {q[:45]}")
    return out


def _geocode_one(q, near_lat, near_lng, headers):
    """Try progressively looser variants so messy attraction names still resolve.
    e.g. 'Aspen Mountain (Ajax), Colorado' -> drop '(Ajax)' -> fall back to
    'Aspen Mountain'. Each attempt is biased toward the stop's area."""
    cleaned = re.sub(r"\([^)]*\)", "", q)            # drop parentheticals
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    name_only = cleaned.split(",")[0].strip()         # name without the region suffix
    variants = []
    for v in (cleaned, name_only):
        if v and v not in variants:
            variants.append(v)
    for v in variants:
        params = {"q": v, "format": "json", "limit": 1}
        if near_lat is not None and near_lng is not None:
            d = 1.5  # soft bias toward a ~1.5° box around the stop
            params["viewbox"] = f"{near_lng-d},{near_lat+d},{near_lng+d},{near_lat-d}"
        try:
            arr = requests.get("https://nominatim.openstreetmap.org/search",
                               params=params, headers=headers, timeout=15).json()
        except Exception:
            arr = None
        time.sleep(1.0)  # Nominatim politeness
        if arr:
            return float(arr[0]["lat"]), float(arr[0]["lon"])
    return None


def place_value(query: str, near_lat=None, near_lng=None) -> dict:
    """
    Google Places API (New) Text Search — objective VALUE signal for a place:
    its rating (0-5) and review count. Needs GOOGLE_API_KEY (Places API New +
    billing). Returns {rating, reviews} or {} if no key / not found / error —
    callers must treat {} as 'no data' and fall back to LLM judgment.
    """
    from config import GOOGLE_API_KEY
    if not GOOGLE_API_KEY:
        return {}
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.rating,places.userRatingCount",
    }
    body = {"textQuery": query, "maxResultCount": 1}
    if near_lat is not None and near_lng is not None:
        body["locationBias"] = {"circle": {"center": {"latitude": near_lat, "longitude": near_lng},
                                           "radius": 50000.0}}
    try:
        r = requests.post("https://places.googleapis.com/v1/places:searchText",
                          json=body, headers=headers, timeout=15)
        p = (r.json().get("places") or [{}])[0]
        return {"rating": p.get("rating"), "reviews": p.get("userRatingCount")}
    except Exception as e:
        log("places", f"✗ {type(e).__name__}: {query[:40]}")
        return {}


def value_score(rating, reviews) -> float:
    """Objective worth-visiting score from Google data: quality × popularity.
    rating(0-5) × log10(reviews) — a 4.7/300k place far outscores a 4.6/8k one."""
    if not rating or not reviews:
        return 0.0
    return float(rating) * math.log10(float(reviews) + 10)


def _maps_url(name: str, lat: float, lng: float) -> str:
    """A Google Maps link that drops a pin on a named place at its coordinates.
    Name biases the label; the lat,lng keeps it on the right spot even when the
    name is ambiguous (e.g. several 'Union Station')."""
    return f"https://www.google.com/maps/search/?api=1&query={quote(f'{name} {lat},{lng}')}"


# ═══════════════════════════════════════════════════════════════
# HELPER — query OpenStreetMap Overpass API
# ═══════════════════════════════════════════════════════════════

_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]


def _query_overpass(query: str):
    """
    Query Overpass, trying mirror endpoints in turn. The public servers are
    frequently slow or busy, so a single host times out unpredictably — falling
    back across mirrors makes the OSM tools (attractions/restaurants/hotels/
    airports) far more reliable.
    """
    headers = {
        "User-Agent": "TripPlannerAgent/1.0 (trip planning application)",
        "Accept": "application/json",
    }
    last_err = None
    for url in _OVERPASS_ENDPOINTS:
        host = url.split("//")[1].split("/")[0]
        t0 = time.time()
        try:
            res = requests.post(url, data={"data": query}, headers=headers, timeout=40)
            res.raise_for_status()
            data = res.json()
            # Overpass signals a server-side timeout/error via a `remark` while still
            # returning HTTP 200 with no elements. raise_for_status() can't see that,
            # so we'd silently accept zero results and never try the next mirror.
            # Treat an empty-with-remark response as a failure and fall through.
            remark = (data.get("remark") or "").lower()
            if not data.get("elements") and ("timed out" in remark or "error" in remark):
                log("overpass", f"⚠ {host} remark/empty ({time.time()-t0:.1f}s) → next mirror")
                last_err = RuntimeError(f"Overpass remark from {url}: {data.get('remark').strip()}")
                continue
            log("overpass", f"✓ {host} {len(data.get('elements', []))} elements ({time.time()-t0:.1f}s)")
            return data
        except Exception as e:
            log("overpass", f"✗ {host} {type(e).__name__} ({time.time()-t0:.1f}s) → next mirror")
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Overpass query failed")


# ═══════════════════════════════════════════════════════════════
# TOOL FUNCTIONS — all real APIs
# ═══════════════════════════════════════════════════════════════

def get_coordinates(city: str, state: str | None = None, country: str | None = None):
    """
    Open-Meteo Geocoding (free, no key).

    The API matches on the city NAME only — a comma string like
    "Denver, Colorado, USA" returns 0 hits. So we fetch several candidates
    and disambiguate using the optional state/country hints, since many
    cities share a name (Denver CO vs Denver City TX vs Denver PA).
    """
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=10&format=json"
    res = requests.get(url, timeout=15).json()

    candidates = res.get("results", [])
    if not candidates:
        return {"error": f"City not found: {city}"}

    def score(r):
        """Higher = better contextual match against the state/country hints."""
        s = 0
        if state and state.lower() in r.get("admin1", "").lower():
            s += 2  # state (admin1) is the most discriminating signal
        if country:
            c = country.lower()
            if c in r.get("country", "").lower() or c == r.get("country_code", "").lower():
                s += 1
        return s

    # Pick the best-matching candidate; ties keep API relevance order (max is stable)
    r = max(candidates, key=score)
    return {
        "city": r["name"],
        "state": r.get("admin1", ""),
        "country": r.get("country", ""),
        "country_code": r.get("country_code", ""),
        "latitude": r["latitude"],
        "longitude": r["longitude"],
        "timezone": r.get("timezone", ""),
        "population": r.get("population")
    }


_COUNTRIES_CACHE = None

# Countries that drive on the LEFT (ISO-3166 alpha-2). A fixed, well-known set —
# the country dataset no longer carries this field, so we derive it statically.
_LEFT_DRIVING = {
    "GB", "IE", "JP", "AU", "NZ", "IN", "ZA", "SG", "MY", "TH", "ID", "HK", "MO",
    "PK", "BD", "LK", "NP", "BT", "BN", "TL", "KE", "TZ", "UG", "ZW", "ZM", "MW",
    "MZ", "NA", "BW", "LS", "SZ", "MU", "SC", "FJ", "PG", "CY", "MT", "JM", "TT",
    "BB", "BS", "GY", "SR", "KN", "LC", "VC", "GD", "DM", "AG",
}


def _load_countries():
    """
    Fetch the open mledoze/countries dataset once and cache it.

    REST Countries v3.1 was deprecated (now hard-fails), so we go straight to the
    open dataset it was built from — same schema, served free via GitHub's CDN,
    no key. ~1.3MB, fetched once per process.
    """
    global _COUNTRIES_CACHE
    if _COUNTRIES_CACHE is None:
        url = "https://raw.githubusercontent.com/mledoze/countries/master/countries.json"
        _COUNTRIES_CACHE = requests.get(url, timeout=25).json()
    return _COUNTRIES_CACHE


def get_country_info(country: str):
    """Country details from the open mledoze/countries dataset (free, no key)."""
    try:
        countries = _load_countries()
    except Exception as e:
        return {"error": f"Country data unavailable: {e}"}

    q = country.strip().lower()
    c = None
    # Match against ISO codes, common/official names, and alt spellings (e.g. "USA")
    for entry in countries:
        names = [entry.get("cca2", ""), entry.get("cca3", ""),
                 entry.get("name", {}).get("common", ""),
                 entry.get("name", {}).get("official", "")]
        names += entry.get("altSpellings", [])
        if q in [n.lower() for n in names if n]:
            c = entry
            break
    if c is None:  # loose fallback — substring on common name
        for entry in countries:
            if q and q in entry.get("name", {}).get("common", "").lower():
                c = entry
                break
    if c is None:
        return {"error": f"Country not found: {country}"}

    currencies = c.get("currencies", {})
    currency_code = next(iter(currencies.keys()), "USD") if currencies else "USD"
    currency_name = currencies.get(currency_code, {}).get("name", "") if currencies else ""

    languages = list(c.get("languages", {}).values())
    idd = c.get("idd", {})
    suffixes = idd.get("suffixes") or []
    root = idd.get("root", "")
    # NANP countries (US/Canada) list every area code as a suffix — just use the root
    calling_code = root + (suffixes[0] if len(suffixes) == 1 else "")

    return {
        "official_name": c.get("name", {}).get("official", country),
        "common_name": c.get("name", {}).get("common", country),
        "capital": (c.get("capital") or [None])[0],
        "region": c.get("region", ""),
        "subregion": c.get("subregion", ""),
        "languages": languages,
        "currency_code": currency_code,
        "currency_name": currency_name,
        "currency_symbol": currencies.get(currency_code, {}).get("symbol", "") if currencies else "",
        "country_code": c.get("cca2", ""),       # alias — matches get_coordinates' field name
        "country_code_iso2": c.get("cca2", ""),
        "country_code_iso3": c.get("cca3", ""),
        "calling_code": calling_code,
        "driving_side": "left" if c.get("cca2", "") in _LEFT_DRIVING else "right",
        "timezones": c.get("timezones", [])[:3],
        "tips": {
            "language_for_travelers": languages[0] if languages else "English",
            "calling_code": calling_code
        }
    }


def get_weather(latitude: float, longitude: float, month: int | None = None,
                start_date: str | None = None, end_date: str | None = None):
    """
    Open-Meteo Forecast (free, no key).

    If start_date/end_date (YYYY-MM-DD) are given AND fall inside the forecast
    window, returns the REAL per-day forecast for exactly those dates. Otherwise
    falls back to the next 7 days as a seasonal proxy (the old behaviour).
    """
    months = ["", "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    base = (f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            f"&timezone=auto")

    daily = None
    used_dates = False
    if start_date and end_date:
        try:
            data = requests.get(base + f"&start_date={start_date}&end_date={end_date}",
                                timeout=15).json()
            d = data.get("daily", {})
            if d.get("time") and any(t is not None for t in d.get("temperature_2m_max", [])):
                daily, used_dates = d, True
        except Exception:
            daily = None

    if daily is None:  # fallback — next 7 days from today
        data = requests.get(base + "&forecast_days=7", timeout=15).json()
        daily = data["daily"]

    highs = [x for x in daily["temperature_2m_max"] if x is not None]
    lows = [x for x in daily["temperature_2m_min"] if x is not None]
    rains = [x for x in daily.get("precipitation_probability_max", []) if x is not None]
    avg_max = round(sum(highs) / len(highs)) if highs else None
    avg_min = round(sum(lows) / len(lows)) if lows else None
    avg_rain = round(sum(rains) / len(rains)) if rains else 0

    # Per-day breakdown — lets the synthesizer talk about specific trip days
    times = daily.get("time", [])
    per_day = [{
        "date": times[i],
        "high_celsius": daily["temperature_2m_max"][i],
        "low_celsius": daily["temperature_2m_min"][i],
        "rain_chance_percent": daily.get("precipitation_probability_max", [None] * len(times))[i],
    } for i in range(len(times))]

    advice = []
    if avg_rain > 50:
        advice.append("Pack a waterproof jacket — rain likely")
    if avg_max is not None and avg_max > 30:
        advice.append("Hot — light clothing, sunscreen, hydration")
    elif avg_max is not None and avg_max < 10:
        advice.append("Cold — pack warm layers, jacket, gloves")
    if avg_min is not None and avg_min < 0:
        advice.append("Below freezing — winter gear essential")

    return {
        "dates": f"{start_date} to {end_date}" if used_dates else "next 7 days (forecast)",
        "month": months[month] if month and 1 <= month <= 12 else "",
        "avg_high_celsius": avg_max,
        "avg_low_celsius": avg_min,
        "rain_chance_percent": avg_rain,
        "daily": per_day,
        "summary": f"{avg_max}°C highs, {avg_min}°C lows, {avg_rain}% rain chance",
        "packing_advice": advice,
        "note": ("Real per-day forecast for your trip dates" if used_dates
                 else "Trip dates outside the forecast window — showing the current 7-day forecast as a seasonal proxy"),
    }


def get_holidays(country_code: str, year: int,
                 start_date: str | None = None, end_date: str | None = None):
    """
    Nager.Date (free, no key).

    If start_date/end_date (YYYY-MM-DD) are given, also returns the subset of
    holidays that actually fall DURING the trip — which is what the itinerary
    cares about, rather than all ~17 holidays in the year.
    """
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code.upper()}"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code != 200:
            return {"error": f"Holidays unavailable for {country_code}", "holidays": []}
        data = res.json()
    except Exception as e:
        return {"error": str(e), "holidays": []}

    holidays = [
        {
            "date": h["date"],
            "name": h["localName"],
            "english_name": h["name"],
            "global": h.get("global", True)
        }
        for h in data
    ]

    result = {
        "country_code": country_code.upper(),
        "year": year,
        "count": len(holidays),
        "holidays": holidays
    }

    if start_date and end_date:
        during = [h for h in holidays if start_date <= h["date"] <= end_date]
        result["trip_window"] = f"{start_date} to {end_date}"
        result["during_trip_count"] = len(during)
        result["during_trip"] = during

    return result


def find_airport(latitude: float, longitude: float):
    """OpenStreetMap Overpass — find nearest international airport."""
    radius_m = 100000  # 100km
    query = f"""
    [out:json][timeout:25];
    (
      node["aeroway"="aerodrome"]["iata"](around:{radius_m},{latitude},{longitude});
      way["aeroway"="aerodrome"]["iata"](around:{radius_m},{latitude},{longitude});
    );
    out center 10;
    """
    try:
        data = _query_overpass(query)
    except Exception as e:
        return {"error": str(e)}

    airports = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("name:en")
        iata = tags.get("iata")
        if not name or not iata:
            continue

        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lng is None:
            continue

        # haversine distance
        dist_km = _haversine(latitude, longitude, lat, lng)
        airports.append({
            "name": name,
            "iata": iata,
            "icao": tags.get("icao", ""),
            "distance_km": round(dist_km, 1),
            "latitude": lat,
            "longitude": lng
        })

    airports.sort(key=lambda a: a["distance_km"])
    return {"count": len(airports), "airports": airports[:5]}


def search_attractions(latitude: float, longitude: float, radius_km: float = 25):
    """
    OpenStreetMap Overpass — real tourist attractions near a point.

    Replaces the old Wikipedia-search approach (unreliable — frequently returned
    0 results or crashed on non-JSON responses). Pulls named attractions, museums,
    viewpoints, landmarks, peaks, and notable parks within radius, and prioritises
    notable ones (those tagged with wikidata/wikipedia — a good fame proxy).

    Returns each with coordinates so cluster_attractions can group them by day.
    """
    # Clamp the radius — large radii (the planner sometimes asks for 100km) make
    # the multi-tag Overpass query so expensive every mirror times out.
    radius_km = min(max(radius_km, 1), 40)
    radius_m = int(radius_km * 1000)
    # The old single 10-clause query (nodes + ways + `out center 200`) was too
    # heavy for the free public Overpass mirrors — they routinely timed out and
    # returned zero attractions. Split it: a cheap NODE-only query (museums,
    # attractions, monuments — the prime sights, which are usually nodes) plus a
    # separate heavier WAY query (needs `out center` geometry resolution). Running
    # them independently means a slow/failed way query can't wipe out the node hits.
    node_query = f"""
    [out:json][timeout:25];
    (
      node["tourism"~"attraction|museum|viewpoint|gallery|zoo|theme_park|artwork"]["name"](around:{radius_m},{latitude},{longitude});
      node["historic"~"monument|castle|memorial|ruins|archaeological_site|fort"]["name"](around:{radius_m},{latitude},{longitude});
      node["natural"~"peak|waterfall"]["name"]["wikidata"](around:{radius_m},{latitude},{longitude});
      node["information"="visitor_centre"]["name"](around:{radius_m},{latitude},{longitude});
      node["highway"="trailhead"]["name"](around:{radius_m},{latitude},{longitude});
    );
    out 150;
    """
    way_query = f"""
    [out:json][timeout:25];
    (
      way["tourism"~"attraction|museum|viewpoint|gallery|zoo|theme_park"]["name"](around:{radius_m},{latitude},{longitude});
      way["historic"~"monument|castle|memorial|ruins|archaeological_site|fort"]["name"](around:{radius_m},{latitude},{longitude});
      way["natural"="water"]["name"]["wikidata"](around:{radius_m},{latitude},{longitude});
      way["leisure"~"park|nature_reserve"]["name"]["wikidata"](around:{radius_m},{latitude},{longitude});
      way["boundary"="national_park"]["name"](around:{radius_m},{latitude},{longitude});
    );
    out center 80;
    """
    elements = []
    errors = []
    for q in (node_query, way_query):
        try:
            elements += _query_overpass(q).get("elements", [])
        except Exception as e:
            errors.append(str(e))
    if not elements:
        return {"error": "; ".join(errors) or "no results found", "attractions": []}
    data = {"elements": elements}

    # Rank by VISITOR VALUE (lower tier = better), so prime sights beat bare peaks
    tier = {
        "national_park": 0, "attraction": 0, "museum": 0, "theme_park": 0, "zoo": 0,
        "viewpoint": 1, "waterfall": 1, "lake": 1, "gallery": 1, "park": 1,
        "nature_reserve": 1, "visitor_centre": 1, "castle": 1,
        "monument": 2, "memorial": 2, "archaeological_site": 2, "ruins": 2, "fort": 2,
        "trailhead": 3, "artwork": 3, "peak": 4,
    }
    # Cap how many of any one type can appear, so a single type can't monopolise
    cap = {"peak": 3, "artwork": 3, "lake": 5, "trailhead": 4, "viewpoint": 6, "park": 6}
    default_cap = 12

    def classify(tags):
        if tags.get("tourism"):
            return tags["tourism"]
        if tags.get("historic"):
            return tags["historic"]
        if tags.get("information") == "visitor_centre":
            return "visitor_centre"
        if tags.get("highway") == "trailhead":
            return "trailhead"
        if tags.get("natural") == "water":
            return "lake"
        if tags.get("natural"):
            return tags["natural"]
        if tags.get("boundary") == "national_park":
            return "national_park"
        if tags.get("leisure"):
            return tags["leisure"]
        return "attraction"

    candidates = []
    seen = set()
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("name:en")
        if not name or name in seen:
            continue
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lng is None:
            continue
        seen.add(name)
        kind = classify(tags)
        candidates.append({
            "name": name,
            "type": kind,
            "lat": lat,
            "lng": lng,
            "notable": bool(tags.get("wikidata") or tags.get("wikipedia")),
            "dist_km": round(_haversine(latitude, longitude, lat, lng), 1),
            "google_maps_url": _maps_url(name, lat, lng),
        })

    # Order: prime types first, then notable, then nearest
    candidates.sort(key=lambda a: (tier.get(a["type"], 2), not a["notable"], a["dist_km"]))

    # Apply per-type caps while filling up to 30 — keeps the list varied
    attractions = []
    type_counts = defaultdict(int)
    for a in candidates:
        if type_counts[a["type"]] >= cap.get(a["type"], default_cap):
            continue
        type_counts[a["type"]] += 1
        attractions.append(a)
        if len(attractions) >= 30:
            break

    return {
        "source": "OpenStreetMap (Overpass API)",
        "count": len(attractions),
        "attractions": attractions
    }


def search_restaurants(latitude: float, longitude: float, radius_km: float = 3):
    """OpenStreetMap Overpass — real restaurants."""
    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="restaurant"]["name"](around:{radius_m},{latitude},{longitude});
    );
    out body 30;
    """
    try:
        data = _query_overpass(query)
    except Exception as e:
        return {"error": str(e), "restaurants": []}

    restaurants = []
    seen = set()
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name or name in seen:
            continue
        seen.add(name)

        restaurants.append({
            "name": name,
            "cuisine": tags.get("cuisine", "unknown"),
            "lat": el.get("lat"),
            "lng": el.get("lon"),
            "address": ", ".join(filter(None, [tags.get("addr:street"), tags.get("addr:city")])) or None,
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "website": tags.get("website") or tags.get("contact:website"),
            "vegetarian": tags.get("diet:vegetarian"),
            "outdoor_seating": tags.get("outdoor_seating")
        })
        if len(restaurants) >= 20:
            break

    return {
        "source": "OpenStreetMap (Overpass API)",
        "count": len(restaurants),
        "restaurants": restaurants
    }


def search_hotels(latitude: float, longitude: float, budget_style: str, radius_km: float = 4):
    """OpenStreetMap Overpass — real hotels, filtered by budget tier."""
    radius_m = int(radius_km * 1000)

    # Budget → tourism tag
    if budget_style == "budget":
        tags_filter = '["tourism"~"hostel|guest_house"]'
    elif budget_style == "luxury":
        tags_filter = '["tourism"="hotel"]["stars"~"4|5"]'
    else:  # mid-range
        tags_filter = '["tourism"="hotel"]'

    query = f"""
    [out:json][timeout:25];
    (
      node{tags_filter}["name"](around:{radius_m},{latitude},{longitude});
    );
    out body 25;
    """
    try:
        data = _query_overpass(query)
    except Exception as e:
        return {"error": str(e), "hotels": []}

    hotels = []
    seen = set()
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name or name in seen:
            continue
        seen.add(name)

        hotels.append({
            "name": name,
            "type": tags.get("tourism", "hotel"),
            "stars": tags.get("stars"),
            "lat": el.get("lat"),
            "lng": el.get("lon"),
            "address": ", ".join(filter(None, [tags.get("addr:street"), tags.get("addr:city")])) or None,
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "website": tags.get("website") or tags.get("contact:website")
        })
        if len(hotels) >= 12:
            break

    return {
        "source": "OpenStreetMap (Overpass API)",
        "budget_style": budget_style,
        "count": len(hotels),
        "hotels": hotels
    }


def convert_currency(amount: float, from_currency: str, to_currency: str):
    """Frankfurter (ECB rates, free, no key)."""
    fc = from_currency.upper()
    tc = to_currency.upper()

    if fc == tc:
        return {"amount": amount, "from": fc, "to": tc, "rate": 1.0, "result": amount}

    url = f"https://api.frankfurter.dev/v1/latest?base={fc}&symbols={tc}"
    try:
        data = requests.get(url, timeout=15).json()
        rate = data["rates"][tc]
    except Exception as e:
        return {"error": f"Currency conversion failed: {e}"}

    result = round(amount * rate, 2)
    return {
        "amount": amount,
        "from_currency": fc,
        "to_currency": tc,
        "rate": rate,
        "result": result,
        "source": "European Central Bank via Frankfurter"
    }


def get_route(from_lat: float, from_lng: float, to_lat: float, to_lng: float):
    """
    OSRM public API (free, no key) — real DRIVING distance & time between two points.

    Unlike _haversine (straight-line crow-flies), this follows actual roads, so it
    answers "how long to drive A -> B" for road-trip leg decisions.
    """
    coords = f"{from_lng},{from_lat};{to_lng},{to_lat}"
    url = f"https://router.project-osrm.org/route/v1/driving/{coords}?overview=false"
    try:
        data = requests.get(url, timeout=20).json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return {"error": f"No driving route found ({data.get('code')})"}
        route = data["routes"][0]
    except Exception as e:
        return {"error": f"Routing failed: {e}"}

    meters = route["distance"]
    seconds = route["duration"]
    hours = seconds / 3600
    km = meters / 1000
    h = int(hours)
    m = round((hours - h) * 60)
    return {
        "distance_km": round(km, 1),
        "distance_miles": round(km * 0.621371, 1),
        "duration_hours": round(hours, 2),
        "duration_text": (f"{h}h {m}m" if h else f"{m}m"),
        "source": "OSRM (OpenStreetMap routing)"
    }


def get_route_multi(waypoints: list):
    """
    OSRM (free, no key) — one driving route through MANY ordered waypoints.
    waypoints: [(lat, lng), ...] in travel order.

    Returns the full road geometry (for drawing on a map) plus per-leg distance
    and time, in a SINGLE request — used by the map view to show the whole trip.
    """
    pts = [(la, ln) for la, ln in waypoints if la is not None and ln is not None]
    if len(pts) < 2:
        return {"error": "need at least two waypoints", "legs": [], "geometry": []}
    coords = ";".join(f"{ln},{la}" for la, ln in pts)
    url = (f"https://router.project-osrm.org/route/v1/driving/{coords}"
           f"?overview=full&geometries=geojson")
    try:
        data = requests.get(url, timeout=25).json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return {"error": f"No route ({data.get('code')})", "legs": [], "geometry": []}
        route = data["routes"][0]
    except Exception as e:
        return {"error": f"Routing failed: {e}", "legs": [], "geometry": []}

    def fmt(seconds):
        h = int(seconds // 3600); m = round((seconds % 3600) / 60)
        return f"{h}h {m}m" if h else f"{m}m"

    legs = [{
        "distance_miles": round(lg["distance"] / 1000 * 0.621371, 1),
        "duration_hours": round(lg["duration"] / 3600, 2),
        "duration_text": fmt(lg["duration"]),
    } for lg in route.get("legs", [])]

    # GeoJSON is [lng, lat]; Leaflet wants [lat, lng]
    geometry = [[c[1], c[0]] for c in route.get("geometry", {}).get("coordinates", [])]
    return {
        "geometry": geometry,
        "legs": legs,
        "total_miles": round(route["distance"] / 1000 * 0.621371, 1),
        "total_duration_text": fmt(route["duration"]),
        "source": "OSRM (OpenStreetMap routing)",
    }


def drive_time_matrix(points: list):
    """
    OSRM Table service (free, no key) — real driving-time matrix (hours) between
    all points, in ONE request. points: [(lat,lng), ...]. Returns
    {durations: [[h,...],...]} symmetric N×N, or {error}. Used to let the stop
    selector reason about what actually fits the trip from real distances.
    """
    pts = [(la, ln) for la, ln in points if la is not None and ln is not None]
    if len(pts) < 2:
        return {"error": "need 2+ points", "durations": []}
    coords = ";".join(f"{ln},{la}" for la, ln in pts)
    url = f"https://router.project-osrm.org/table/v1/driving/{coords}?annotations=duration"
    try:
        d = requests.get(url, timeout=25).json()
        if d.get("code") != "Ok":
            return {"error": d.get("code"), "durations": []}
        hrs = [[round(c / 3600, 1) if c is not None else None for c in row]
               for row in d.get("durations", [])]
        log("osrm", f"✓ drive-time matrix {len(pts)}×{len(pts)}")
        return {"durations": hrs}
    except Exception as e:
        return {"error": str(e), "durations": []}


def _best_stop_order(dur, n, first_idx=None):
    """Visiting order of stops 1..n as a round trip 0 → … → 0 (0 = origin) that
    minimizes total drive on the matrix `dur`, via nearest-neighbour + 2-opt.
    If first_idx (a stop index 1..n) is given, it's LOCKED as the first stop and the
    REST is optimized around it — so "start from X" doesn't strand a stale order."""
    def d(a, b):
        v = dur[a][b]
        return v if v is not None else 1e9
    route, unv = [0], set(range(1, n + 1))
    if first_idx in unv:
        route.append(first_idx); unv.discard(first_idx)
    cur = route[-1]
    while unv:                                   # nearest-neighbour seed
        nxt = min(unv, key=lambda j: d(cur, j)); route.append(nxt); unv.discard(nxt); cur = nxt
    route.append(0)
    lock = 2 if first_idx else 1                 # keep origin (and pinned first) fixed
    improved = True
    while improved:
        improved = False
        for i in range(lock, len(route) - 2):
            for k in range(i + 1, len(route) - 1):
                a, b, c, e = route[i - 1], route[i], route[k], route[k + 1]
                if d(a, c) + d(b, e) + 1e-9 < d(a, b) + d(c, e):
                    route[i:k + 1] = reversed(route[i:k + 1]); improved = True
    return route[1:-1]                           # stop indices, in visiting order


def get_optimized_trip(origin_lat, origin_lng, stops: list, pin_first=None):
    """
    OSRM (free, no key) — solve the best VISITING ORDER for a set of stops, with the
    origin fixed as start and end (roundtrip). Returns the reordered stops, full road
    geometry, and per-leg distance/time so the map route is sensible regardless of the
    order stops were ticked.

    pin_first: name of a stop to force FIRST. The rest is still RE-OPTIMIZED around it
    (via the drive-matrix), so pinning a start never leaves the tail zig-zagging.
    """
    def fmt(seconds):
        h = int(seconds // 3600); m = round((seconds % 3600) / 60)
        return f"{h}h {m}m" if h else f"{m}m"

    pts = [(origin_lat, origin_lng)] + [(s.get("lat"), s.get("lng")) for s in stops]
    have_coords = len(pts) >= 2 and not any(la is None or ln is None for la, ln in pts)

    # Pinned start: re-optimize the rest around it (matrix order → real geometry)
    if pin_first and have_coords:
        pin_idx = next((i for i, s in enumerate(stops)
                        if (s.get("name") or "").strip().lower() == pin_first.strip().lower()), None)
        mat = drive_time_matrix(pts) if pin_idx is not None else {}
        if mat.get("durations"):
            order = _best_stop_order(mat["durations"], len(stops), first_idx=pin_idx + 1)
            ordered = [stops[i - 1] for i in order]
            r = get_route_multi([(origin_lat, origin_lng)]
                                + [(s["lat"], s["lng"]) for s in ordered]
                                + [(origin_lat, origin_lng)])
            r["ordered_stops"] = ordered
            log("osrm", f"→ pinned-first route ({pin_first}); rest re-optimized")
            return r
        # pin not found / no matrix → fall through to a normal optimize

    if have_coords:
        coords = ";".join(f"{ln},{la}" for la, ln in pts)
        url = (f"https://router.project-osrm.org/trip/v1/driving/{coords}"
               f"?source=first&roundtrip=true&geometries=geojson&overview=full")
        t0 = time.time()
        try:
            data = requests.get(url, timeout=25).json()
            if data.get("code") == "Ok" and data.get("trips"):
                log("osrm", f"✓ optimized {len(stops)}-stop trip ({time.time()-t0:.1f}s)")
                wps = data["waypoints"]
                # waypoint_index = position in the optimized trip; origin is input 0
                order = sorted(range(1, len(pts)), key=lambda i: wps[i]["waypoint_index"])
                ordered_stops = [stops[i - 1] for i in order]
                trip = data["trips"][0]
                geometry = [[c[1], c[0]] for c in trip.get("geometry", {}).get("coordinates", [])]
                legs = [{"distance_miles": round(lg["distance"] / 1000 * 0.621371, 1),
                         "duration_hours": round(lg["duration"] / 3600, 2),
                         "duration_text": fmt(lg["duration"])} for lg in trip.get("legs", [])]
                return {"ordered_stops": ordered_stops, "geometry": geometry, "legs": legs,
                        "total_miles": round(trip["distance"] / 1000 * 0.621371, 1),
                        "total_duration_text": fmt(trip["duration"]),
                        "source": "OSRM trip (optimized order)"}
        except Exception:
            pass

    # Fallback (missing coords or the trip service failed): keep the given order.
    log("osrm", f"→ fixed-order route, {len(stops)} stops (fallback)")
    r = get_route_multi([(origin_lat, origin_lng)] +
                        [(s.get("lat"), s.get("lng")) for s in stops] +
                        [(origin_lat, origin_lng)])
    r["ordered_stops"] = stops
    return r


def cluster_attractions(attractions: list, num_clusters: int = 3):
    """
    Pure Python — cluster attractions geographically using simple K-means.
    This means each day can focus on one neighborhood instead of crisscrossing.
    """
    valid = [a for a in attractions if a.get("lat") is not None and a.get("lng") is not None]

    if len(valid) < num_clusters:
        return {
            "clusters": [{"cluster_id": i, "attractions": [a]} for i, a in enumerate(valid)],
            "note": "Fewer attractions than requested clusters"
        }

    # Initialize centroids — pick evenly-spaced points
    step = len(valid) // num_clusters
    centroids = [(valid[i * step]["lat"], valid[i * step]["lng"]) for i in range(num_clusters)]

    # K-means iterations
    for _ in range(10):
        clusters = defaultdict(list)
        for a in valid:
            distances = [_haversine(a["lat"], a["lng"], cy, cx) for cy, cx in centroids]
            ci = distances.index(min(distances))
            clusters[ci].append(a)

        # Recompute centroids
        new_centroids = []
        for i in range(num_clusters):
            members = clusters.get(i, [])
            if members:
                avg_lat = sum(m["lat"] for m in members) / len(members)
                avg_lng = sum(m["lng"] for m in members) / len(members)
                new_centroids.append((avg_lat, avg_lng))
            else:
                new_centroids.append(centroids[i])

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return {
        "num_clusters": num_clusters,
        "clusters": [
            {
                "cluster_id": i,
                "centroid": {"lat": centroids[i][0], "lng": centroids[i][1]},
                "attraction_count": len(clusters[i]),
                "attractions": [{"name": a["name"], "type": a.get("type")} for a in clusters[i]]
            }
            for i in range(num_clusters) if clusters.get(i)
        ],
        "note": "Plan one neighborhood/cluster per day for efficient routing"
    }


def estimate_budget(budget_style: str, days: int, city_tier: str = "moderate"):
    """City-tier-aware budget calculator."""
    base_rates = {
        "budget":    {"hotel": 40,  "food": 20,  "transport": 10, "activities": 15},
        "mid-range": {"hotel": 120, "food": 60,  "transport": 25, "activities": 40},
        "luxury":    {"hotel": 350, "food": 150, "transport": 80, "activities": 120},
    }
    multipliers = {"cheap": 0.6, "moderate": 1.0, "expensive": 1.6}

    r = base_rates.get(budget_style, base_rates["mid-range"])
    m = multipliers.get(city_tier, 1.0)

    breakdown = {k: round(v * m) for k, v in r.items()}
    daily = sum(breakdown.values())

    return {
        "budget_style": budget_style,
        "city_tier": city_tier,
        "days": days,
        "daily_usd": daily,
        "total_usd": daily * days,
        "breakdown_per_day_usd": breakdown,
        "note": "Estimates exclude international flights"
    }


def create_itinerary(destination: str, days: int):
    """Marker tool — signals the LLM to write the final itinerary."""
    return {
        "status": "ready",
        "destination": destination,
        "days": days,
        "instruction": "All info gathered. Now write the complete itinerary as your final text response."
    }


# ═══════════════════════════════════════════════════════════════
# UTILITY — haversine distance
# ═══════════════════════════════════════════════════════════════

def _haversine(lat1, lng1, lat2, lng2):
    """Great-circle distance in km between two coordinates."""
    R = 6371  # Earth radius in km
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


