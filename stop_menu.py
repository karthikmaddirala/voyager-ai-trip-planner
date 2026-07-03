"""
stop_menu.py
────────────
INTERACTIVE SELECTION — build a "menu" of attractions for ONE stop that the
traveler picks from, rather than the agent silently choosing.

Design (the user's principle: the LLM is the brain, tools only fetch info,
and the HUMAN makes the final selection):

  1. LLM CURATES   — names the genuinely worthwhile attractions for the stop,
                     ranked by worthiness (must-see / worth-it / optional), each
                     with a one-line why and a time estimate. The LLM knows the
                     iconic spots (hot springs, Hanging Lake) that raw OSM tags
                     miss and can't rank.
  2. TOOLS GROUND  — every curated item gets a Google Maps link; we also pull
                     real OSM attractions (search_attractions) and fold in any
                     long-tail spots the LLM didn't mention, so nothing real is
                     hidden from the traveler.
  3. HUMAN PICKS   — the caller shows this menu and lets the user select; the
                     itinerary is built from the selections (see build_itinerary).
"""

import json
import re
from urllib.parse import quote

from concurrent.futures import ThreadPoolExecutor

from llm import call_llm, extract_text
from config import STOP_MENU_SYSTEM_PROMPT
from tools import get_coordinates, search_attractions, place_value, value_score
from logutil import step


def _maps_link(query: str) -> str:
    """Name-based Google Maps search link (no coords needed — Google resolves
    a real place name well, and we append the stop+region to disambiguate)."""
    return f"https://www.google.com/maps/search/?api=1&query={quote(query)}"


def _parse_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)  # extract JSON from surrounding prose
        if m:
            return json.loads(m.group(0))
        raise


def build_stop_menu(stop: str, region: str = "", state: str = "", country: str = "USA",
                    include_osm: bool = True) -> dict:
    """
    Build a pick-from menu of attractions for one stop.

    Returns:
        {
          "stop", "summary", "recommended_per_day",
          "attractions": [ {name, category, why, typical_hours, tier,
                            source: "expert"|"osm", google_maps_url, dist_km?} ]
        }
    """
    where = ", ".join(p for p in (stop, region or state, country) if p)

    # 1) LLM curates the worthwhile, worthiness-ranked list
    prompt = (f"Stop: {stop}\nRegion/State: {region or state}\nCountry: {country}\n\n"
              f"Build the attraction menu for this stop.")
    response = call_llm(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=STOP_MENU_SYSTEM_PROMPT,
        tools=None,
        label=f"menu:{stop}",
    )
    menu = _parse_json(extract_text(response))

    # 2) Ground every curated item with a Google Maps link
    curated = menu.get("attractions", [])
    seen = set()
    for a in curated:
        a["source"] = "expert"
        a["google_maps_url"] = _maps_link(f"{a['name']} {stop} {region or state}")
        seen.add(a["name"].strip().lower())

    # 3) Fold in real OSM long-tail the expert didn't mention (already carries
    #    coords + google_maps_url + a worthiness flag from tools.search_attractions)
    if include_osm:
        try:
            with step("osm", f"search_attractions near {stop}"):
                c = get_coordinates(stop, state or None, country or None)
                osm = (search_attractions(c["latitude"], c["longitude"], radius_km=30)
                       if "latitude" in c else {"attractions": []})
            if osm.get("attractions"):
                for a in osm.get("attractions", []):
                    if a["name"].strip().lower() in seen:
                        continue
                    seen.add(a["name"].strip().lower())
                    curated.append({
                        "name": a["name"],
                        "category": a["type"],
                        "why": "Listed in OpenStreetMap nearby — local/lesser-known option.",
                        "typical_hours": 1,
                        "tier": "optional",
                        "source": "osm",
                        "dist_km": a.get("dist_km"),
                        "google_maps_url": a.get("google_maps_url"),
                    })
        except Exception as e:
            menu["osm_note"] = f"OSM long-tail unavailable: {e}"

    # Objective value: Google rating + review count per attraction (no-op without key)
    _add_attraction_value(curated, stop, region or state, country)
    menu["attractions"] = curated
    menu["where"] = where
    return menu


def _add_attraction_value(attractions: list, stop: str, region: str, country: str) -> None:
    """Attach Google rating/reviews to attractions and sort within each tier by
    rating. To keep API cost low we ONLY rate the expert-curated must-see/worth-it
    picks (the few that matter) — not 'optional' or the OSM long-tail. No-op if no key.
    Runs lazily (only when a stop tab is opened), so calls are bounded to what the
    traveler actually looks at."""
    from config import GOOGLE_API_KEY
    if not GOOGLE_API_KEY or not attractions:
        return
    near = next(((a["lat"], a["lng"]) for a in attractions if a.get("lat")), None)
    to_rate = [a for a in attractions
               if a.get("source") == "expert" and a.get("tier") in ("must-see", "worth-it")]
    if not to_rate:
        return
    def fetch(a):
        v = place_value(f"{a['name']}, {stop}, {region}, {country}",
                        a.get("lat") or (near[0] if near else None),
                        a.get("lng") or (near[1] if near else None))
        a["rating"], a["reviews"] = v.get("rating"), v.get("reviews")
    with step("value", f"Google ratings for {len(to_rate)} top attractions @ {stop}"):
        with ThreadPoolExecutor(max_workers=min(6, len(to_rate))) as ex:
            list(ex.map(fetch, to_rate))
    if any(a.get("rating") for a in attractions):
        tier_order = {"must-see": 0, "worth-it": 1, "optional": 2}
        attractions.sort(key=lambda a: (tier_order.get(a.get("tier"), 3),
                                        -value_score(a.get("rating"), a.get("reviews"))))


def print_menu(menu: dict) -> None:
    """Human-readable menu for a terminal selection step."""
    tier_order = {"must-see": 0, "worth-it": 1, "optional": 2}
    print(f"\n=== {menu.get('stop')} — {menu.get('summary','')} ===")
    print(f"(realistic per day here: ~{menu.get('recommended_per_day','?')})\n")
    items = sorted(menu.get("attractions", []),
                   key=lambda a: (tier_order.get(a.get("tier"), 3),
                                  a.get("source") == "osm"))
    for i, a in enumerate(items, 1):
        tier = a.get("tier", "").upper()
        hrs = a.get("typical_hours")
        src = "🧠" if a.get("source") == "expert" else "🗺"
        print(f"{i:>2}. [{tier:<9}] {src} {a['name']}  ({a.get('category','')}, ~{hrs}h)")
        print(f"      {a.get('why','')}")
        print(f"      {a.get('google_maps_url')}")
    return items


if __name__ == "__main__":
    import sys
    stop = sys.argv[1] if len(sys.argv) > 1 else "Glenwood Springs"
    region = sys.argv[2] if len(sys.argv) > 2 else "Colorado"
    m = build_stop_menu(stop, region=region, state=region)
    print_menu(m)
