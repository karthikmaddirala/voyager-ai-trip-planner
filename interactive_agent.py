"""
interactive_agent.py
────────────────────
INTERACTIVE, HUMAN-IN-THE-LOOP trip planning for the web app.

The agent CURATES and ADVISES; the HUMAN decides. Four stages, each a plain
function returning JSON so app.py can expose them as request/response endpoints
(no long-lived paused thread — state lives in a session dict in app.py):

  1. propose_stops()      — LLM proposes a generous, iconic set of candidate
                            stops for the region (user trims). Grounded with
                            coordinates + Google Maps links.
  2. build_menus()        — per chosen stop, the curated attraction menu
                            (stop_menu.build_stop_menu): must-see/worth-it/optional
                            + Maps links + OSM long-tail. User picks attractions.
  3. assess_feasibility() — real drive legs (get_route) + the user's chosen
                            attraction hours vs their days → honest verdict.
  4. build_itinerary()    — final day-by-day itinerary from the user's picks
                            (reuses the road-trip SYNTHESIZER prompt).
"""

import json
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from llm import call_llm, extract_text, run_agent_loop
from config import (STOP_PROPOSER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT,
                    ASSISTANT_SYSTEM_PROMPT, DAY_PLAN_SYSTEM_PROMPT, STOP_SELECTOR_SYSTEM_PROMPT,
                    AGENT_PLANNER_SYSTEM_PROMPT)
from tools import (get_coordinates, get_route, get_weather, get_holidays, get_country_info,
                   place_value, value_score, drive_time_matrix, search_hotels)
from stop_menu import build_stop_menu, _maps_link
from logutil import log, step


def chat_turn(history: list, state_text: str) -> str:
    """Stateful copilot turn. `history` is the full conversation (persisted in the
    session); the live plan state is injected into the system prompt each call."""
    sys = ASSISTANT_SYSTEM_PROMPT + "\n\n## CURRENT PLAN STATE:\n" + state_text
    with step("copilot", f"turn (history={len(history)} msgs)"):
        return extract_text(call_llm(messages=history, system_prompt=sys,
                                     tools=None, label="copilot")).strip()


# Weights for blending a base's attractions into one CLUSTER value: the best
# attraction counts fully, additional strong ones add a diminishing bonus — so a
# multi-draw town (Glenwood: 2 hot springs + caverns + a lake) outscores a
# one-attraction stop, without letting many mediocre sights run away with it.
_CLUSTER_WEIGHTS = [1.0, 0.3, 0.15]


def _cluster_value(name: str, top_pois: list, lat, lng, country: str) -> dict:
    """Rate a base by the CLUSTER of its top attractions, not just one. Returns the
    blended value plus the BEST single attraction's rating/reviews (for display)."""
    pois = [p for p in (top_pois or []) if p and p.strip()]
    rated = []
    # A park/monument is rated by its own name; a town is rated by its attractions.
    if not pois or _DESTINATION_KW.search(name or ""):
        base = place_value(f"{name}, {country}", lat, lng)
        if not base.get("rating"):
            base = place_value(name, lat, lng)
        if base.get("rating"):
            rated.append(base)
    for p in pois[:3]:
        if p.strip().lower() != name.strip().lower():
            v = place_value(p, lat, lng)          # POI alone — coords bias it
            if v.get("rating"):
                rated.append(v)
    if not rated:
        return {"rating": None, "reviews": None, "value": 0.0, "n": 0}
    scored = sorted(rated, key=lambda v: value_score(v.get("rating"), v.get("reviews")), reverse=True)
    cluster = sum(w * value_score(v.get("rating"), v.get("reviews"))
                  for w, v in zip(_CLUSTER_WEIGHTS, scored))
    best = scored[0]
    return {"rating": best.get("rating"), "reviews": best.get("reviews"),
            "value": round(cluster, 1), "n": len(scored)}


def _stop_value(name: str, poi: str, lat, lng, country: str) -> dict:
    """Best Google value for a stop. Query the stop NAME (and, only if that misses,
    the name WITHOUT the ', Country' suffix — that suffix breaks some park lookups,
    e.g. 'Mesa Verde National Park, USA' returns nothing while the name alone rates),
    plus the signature POI queried ALONE (its coords bias it). Keep the highest score.
    Cities aren't rated → their value comes from the POI; parks rate by name."""
    cands = [place_value(f"{name}, {country}", lat, lng)]
    if not cands[0].get("rating"):
        cands.append(place_value(name, lat, lng))
    if poi and poi.strip().lower() != name.strip().lower():
        cands.append(place_value(poi, lat, lng))
    return max(cands, key=lambda v: value_score(v.get("rating"), v.get("reviews")))


def _parse_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # model wrapped the JSON in prose — extract the outermost {...} object
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


# ── Stage 1 (AGENTIC) — the LLM drives the tool calls itself ─────
_AGENT_TOOLS = [
    {"name": "lookup_places",
     "description": "Get coordinates + a combined VALUE for each base, reflecting its CLUSTER of "
                    "attractions. Pass {name, state, top_pois} for each. 'name' is a BASE to stay "
                    "(city, town, national park) — NOT an individual attraction. 'top_pois' is the "
                    "base's 2-3 MOST FAMOUS attractions, best first (e.g. a hot-springs town → its "
                    "main springs + caverns + nearby canyon/lake). The base's value blends these, so "
                    "a town with several strong draws scores higher than a one-attraction stop. "
                    "'state' disambiguates same-named places; for a national park, top_pois = [the park's own name].",
     "input_schema": {"type": "object",
                      "properties": {"places": {"type": "array", "items": {
                          "type": "object",
                          "properties": {"name": {"type": "string"}, "state": {"type": "string"},
                                         "top_pois": {"type": "array", "items": {"type": "string"},
                                                      "description": "the base's 2-3 most famous attractions, best first"},
                                         "top_poi": {"type": "string"}},
                          "required": ["name"]}}},
                      "required": ["places"]}},
    # (drive-time tools removed: the SYSTEM now computes distances and selects the core
    #  deterministically, so the agent only needs to brainstorm + rate the candidate pool.)
]


def propose_stops_agentic(origin: str, destination: str, days: int,
                          start_date: str = "", end_date: str = "", country: str = "USA") -> dict:
    """Stop planner: the LLM brainstorms the candidate pool and rates each base's
    attractions via lookup_places (its strength — world knowledge). The final core PICK
    is then DETERMINISTIC code (_select_core_by_value): highest value that fits the drive
    time, so the same data yields the same stops every run. Falls back to propose_stops."""
    o = get_coordinates(*_split_place(origin), country=country)
    origin_coords = {"lat": o.get("latitude"), "lng": o.get("longitude"),
                     "google_maps_url": _maps_link(origin)}
    if origin_coords["lat"] is None:
        return propose_stops(origin, destination, days, start_date, end_date, country)
    cache = {}  # name(lower) -> {lat, lng, rating, reviews, value}

    def lookup_places(places):
        out = []
        for item in places:
            name = (item.get("name") if isinstance(item, dict) else str(item)).split(",")[0].strip()
            state = item.get("state") if isinstance(item, dict) else None
            pois = (item.get("top_pois") if isinstance(item, dict) else None) or \
                   ([item["top_poi"]] if isinstance(item, dict) and item.get("top_poi") else [])
            c = _geocode_stop(name, state, country)
            lat, lng = c.get("latitude"), c.get("longitude")
            cv = _cluster_value(name, pois, lat, lng, country)
            cache[name.lower()] = {"lat": lat, "lng": lng, "rating": cv["rating"],
                                   "reviews": cv["reviews"], "value": cv["value"]}
            log("value", f"  {name}: ★{cv['rating']} ({cv['reviews']}) → cluster value "
                         f"{cv['value']} from {cv['n']} draw(s)")
            out.append({"name": name, "rating": cv["rating"], "reviews": cv["reviews"],
                        "value": cv["value"]})
        return out

    user = (f"Plan a road trip to {destination} starting from {origin}, {days} days"
            + (f", dates {start_date}..{end_date}" if start_date else "") + ".")
    with step("agent", f"planning {destination} {days}d (LLM brainstorms + rates)"):
        raw = run_agent_loop(AGENT_PLANNER_SYSTEM_PROMPT, user, _AGENT_TOOLS,
                             {"lookup_places": lookup_places},
                             label="planner", temperature=0)  # greedy → same data gives same plan
    try:
        menu = _parse_json(raw)
        if not menu.get("stops"):
            raise ValueError("no stops")
    except Exception as e:
        log("agent", f"✗ agent output unusable ({e}) — falling back to pipeline")
        return propose_stops(origin, destination, days, start_date, end_date, country)

    # Enrich each stop from the lookup cache (or geocode any the agent skipped)
    for s in menu["stops"]:
        s["name"] = s["name"].split(",")[0].strip()  # 'Denver, CO, USA' -> 'Denver'
        rec = cache.get(s["name"].lower())
        if not rec:
            c = _geocode_stop(s["name"], s.get("state"), country)
            rec = {"lat": c.get("latitude"), "lng": c.get("longitude"), "rating": None, "reviews": None}
        s["lat"], s["lng"] = rec.get("lat"), rec.get("lng")
        s["rating"], s["reviews"] = rec.get("rating"), rec.get("reviews")
        s["value_score"] = rec.get("value", value_score(s.get("rating"), s.get("reviews")))
        s["google_maps_url"] = _maps_link(f"{s['name']} {s.get('state','')}")
    menu["stops"] = _dedup_attraction_stops(menu["stops"])
    menu["stops"] = _rank_menu_by_value(menu["stops"])
    _select_core_by_value(menu["stops"], origin_coords, days)
    menu["origin_coords"] = origin_coords
    menu["days"] = days
    log("agent", f"✓ core (value+drive, deterministic): "
                 f"{', '.join(s['name'] for s in menu['stops'] if s.get('tier')=='core')}")
    return menu


# Names that mark a stop as a HEADLINE destination in its own right (you go FOR it
# and merely sleep in the nearby town), not an attraction tucked inside a city.
_DESTINATION_KW = re.compile(
    r"national park|national monument|national recreation|national forest|"
    r"national seashore|national historic|state park|wilderness", re.I)


def _dedup_attraction_stops(stops: list) -> list:
    """Collapse two candidates that are really the SAME place listed twice, where one
    is the other's signature attraction (base.top_poi names another listed stop).
    Two opposite cases, decided by what the attraction IS:
      • park + gateway town (Estes Park.top_poi = 'Rocky Mountain National Park'):
        the PARK is the headline destination, the town is just lodging → keep the
        PARK, drop the town.
      • attraction inside a city (Colorado Springs.top_poi = 'Garden of the Gods'):
        the CITY is the base, the attraction is its draw → keep the CITY (folding the
        attraction's value in), drop the attraction.
    Anything not part of such a pair is kept untouched."""
    def norm(x):
        return re.sub(r"[^a-z0-9]", "", (x or "").lower())
    by_name = {norm(s["name"]): s for s in stops}
    drop = set()
    for base in stops:
        if id(base) in drop or not base.get("top_poi"):
            continue
        attr = by_name.get(norm(base["top_poi"]))
        if attr is None or attr is base or id(attr) in drop:
            continue
        if _DESTINATION_KW.search(attr.get("name") or ""):
            # the attraction is itself a headline destination (a park): keep it,
            # drop the gateway town (base).
            drop.add(id(base))
            log("agent", f"dedup: '{base['name']}' is the gateway to {attr['name']} — "
                         f"kept the park, dropped the town")
        else:
            # ordinary in-city attraction: keep the base city, fold value in, drop attr.
            if (attr.get("value_score") or 0) > (base.get("value_score") or 0):
                base["rating"], base["reviews"] = attr.get("rating"), attr.get("reviews")
                base["value_score"] = attr.get("value_score")
            drop.add(id(attr))
            log("agent", f"dedup: '{attr['name']}' is {base['name']}'s attraction — "
                         f"merged & dropped")
    return [s for s in stops if id(s) not in drop]


_MENU_KEEP = 14  # cap the candidate menu; ratings decide which extensions make the cut


def _rank_menu_by_value(stops: list) -> list:
    """Make the candidate MENU rating-driven. The agent brainstorms a GENEROUS pool;
    here we keep every 'core' stop (the actual route) plus the HIGHEST value_score
    extensions, up to _MENU_KEEP total — so a high-rated place (e.g. a famous gorge)
    beats a low-rated one (e.g. a city whose landmark barely rates) for a '+' slot.
    Stops with no coordinates are dropped (can't be drawn/routed). Order preserved."""
    placed = [s for s in stops if s.get("lat") is not None]
    core = [s for s in placed if s.get("tier") == "core"]
    ext = sorted((s for s in placed if s.get("tier") != "core"),
                 key=lambda s: s.get("value_score") or 0, reverse=True)
    keep_ids = {id(s) for s in core + ext[:max(0, _MENU_KEEP - len(core))]}
    dropped = [f"{s['name']}({(s.get('value_score') or 0):.0f})"
               for s in placed if id(s) not in keep_ids]
    if dropped:
        log("agent", f"rating-trim: kept {len(keep_ids)}/{len(placed)}; "
                     f"dropped low-rated: {', '.join(dropped)}")
    return [s for s in stops if id(s) in keep_ids]  # keep the agent's display order


# ── Stage 1 (deterministic pipeline — fallback) ─────────────────
def propose_stops(origin: str, destination: str, days: int,
                  start_date: str = "", end_date: str = "",
                  country: str = "USA") -> dict:
    """Propose candidate stops for the region; ground each with coords + a map link."""
    prompt = (f"Origin: {origin}\nDestination region: {destination}\n"
              f"Trip length: {days} days\n"
              f"Dates: {start_date or 'unspecified'} to {end_date or 'unspecified'}\n\n"
              f"Propose the candidate stops.")
    log("propose", f"▶ {origin} → {destination}, {days}d ({start_date or '?'}..{end_date or '?'})")
    with step("propose", "LLM proposing candidate stops"):
        menu = _parse_json(extract_text(call_llm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=STOP_PROPOSER_SYSTEM_PROMPT, tools=None,
            temperature=0.2, label="proposer")))

    # Ground origin + each stop with coordinates and a Google Maps link
    with step("propose", f"geocoding origin + {len(menu.get('stops', []))} stops"):
        o = get_coordinates(*_split_place(origin), country=country)
        menu["origin_coords"] = {"lat": o.get("latitude"), "lng": o.get("longitude"),
                                 "google_maps_url": _maps_link(origin)}
        nofix = []
        for s in menu.get("stops", []):
            c = _geocode_stop(s["name"], s.get("state"), country)
            s["lat"], s["lng"] = c.get("latitude"), c.get("longitude")
            s["google_maps_url"] = _maps_link(f"{s['name']} {s.get('state','')}")
            if s["lat"] is None:
                nofix.append(s["name"])
    if nofix:
        log("propose", f"⚠ no coords for: {', '.join(nofix)}")
    log("propose", f"✓ {len(menu.get('stops', []))} stops: "
                   f"{', '.join(s['name'] for s in menu.get('stops', []))}")
    _add_signature_value(menu.get("stops", []), country)
    _select_core(menu.get("stops", []), menu.get("origin_coords", {}), days)
    menu["days"] = days
    return menu


def _add_signature_value(stops: list, country: str) -> None:
    """Attach each stop's real Google rating×reviews via its signature POI (one
    call for the stop name + one for its top_poi, keep the higher — parks rate by
    name, cities by their landmark). No-op without a key (rating stays None)."""
    if not stops:
        return
    def fetch(s):
        best = _stop_value(s["name"], s.get("top_poi"), s.get("lat"), s.get("lng"), country)
        s["rating"], s["reviews"] = best.get("rating"), best.get("reviews")
        s["value_score"] = value_score(s["rating"], s["reviews"])
    with step("value", f"signature-POI ratings for {len(stops)} stops"):
        with ThreadPoolExecutor(max_workers=min(8, len(stops))) as ex:
            list(ex.map(fetch, stops))


def _select_core(stops: list, origin_coords: dict, days: int) -> None:
    """Refine which stops are 'core' by having the LLM reason over REAL data
    together: the trip DAYS, the actual OSRM drive-time matrix, and each stop's
    Google value. Falls back to the proposer's tiers if data/LLM is unavailable."""
    valid = [s for s in stops if s.get("lat") is not None and s.get("lng") is not None]
    o = (origin_coords.get("lat"), origin_coords.get("lng"))
    if len(valid) < 2 or o[0] is None:
        return
    mat = drive_time_matrix([o] + [(s["lat"], s["lng"]) for s in valid])
    if mat.get("error") or not mat.get("durations"):
        return  # no real distances → keep proposer tiers (still day-aware)

    # Build a compact, labeled drive-time matrix for the LLM
    legend = ["0 = Origin"] + [
        f"{i} = {s['name']}  (★{s.get('rating')}/{s.get('reviews')} reviews, ~{s.get('suggested_days','?')}d)"
        for i, s in enumerate(valid, 1)]
    dur = mat["durations"]
    header = "      " + " ".join(f"{j:>5}" for j in range(len(dur)))
    rows = [f"{i:>4}  " + " ".join(f"{(v if v is not None else '-'):>5}" for v in row)
            for i, row in enumerate(dur)]
    content = (f"Trip length: {days} days.\n\n"
               f"Candidate stops (index = name, rating, suggested time):\n" + "\n".join(legend) +
               f"\n\nDRIVE-TIME MATRIX (hours; row/col index per legend; 0=Origin):\n"
               + header + "\n" + "\n".join(rows) +
               "\n\nChoose core vs extension reasoning over the days AND these real drive times AND value.")
    try:
        with step("select", f"core selection over {len(valid)} stops, {days}d + drive matrix"):
            res = _parse_json(extract_text(call_llm(
                messages=[{"role": "user", "content": content}],
                system_prompt=STOP_SELECTOR_SYSTEM_PROMPT, tools=None,
                temperature=0.2, label="selector")))
    except Exception as e:
        log("select", f"✗ selector failed ({e}) — keeping proposer tiers")
        return
    core = {n.split(",")[0].strip().lower() for n in res.get("core", [])}
    selected = [s for s in stops if s["name"].split(",")[0].strip().lower() in core]
    if not selected:  # names didn't match any stop → don't blank the core, keep tiers
        log("select", "✗ selector core names matched no stops — keeping proposer tiers")
        return
    for s in stops:
        s["tier"] = "core" if s in selected else "extension"
    log("select", f"✓ {res.get('reason','')[:120]}")
    _enforce_drive_budget(valid, dur, days)  # deterministic distance guard
    log("select", f"  core: {', '.join(s['name'] for s in stops if s['tier']=='core')}")


# A relaxed road-trip averages at most ~this many hours of actual DRIVING per day.
# This is a DRIVE-TIME limit (hours), NOT a money budget — cost is a separate phase.
_MAX_DRIVE_PER_DAY = 6.0


def _optimized_tour_hours(indices: list, dur: list) -> float:
    """Realistic round-trip drive time (origin → stops → origin) for a set of matrix
    indices, using nearest-neighbour + 2-opt so the ORDER is good. A greedy NN alone
    fakes huge detours (it would route north then double back south for a stop that's
    actually a short hop off the route); 2-opt fixes that, matching the real map loop."""
    def d(a, b):
        v = dur[a][b]
        return v if v is not None else 1e9
    if not indices:
        return 0.0
    # nearest-neighbour seed, origin (0) fixed as start and end
    route, unv, cur = [0], set(indices), 0
    while unv:
        nxt = min(unv, key=lambda j: d(cur, j)); route.append(nxt); unv.discard(nxt); cur = nxt
    route.append(0)
    # 2-opt: reverse interior segments while it shortens the loop (endpoints stay = origin)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for k in range(i + 1, len(route) - 1):
                a, b, c, e = route[i - 1], route[i], route[k], route[k + 1]
                if d(a, c) + d(b, e) + 1e-9 < d(a, b) + d(c, e):
                    route[i:k + 1] = reversed(route[i:k + 1]); improved = True
    return sum(d(route[i], route[i + 1]) for i in range(len(route) - 1))


def _select_core_by_value(stops: list, origin_coords: dict, days: int) -> None:
    """DETERMINISTIC core selection — same inputs → same core every run (an LLM re-ranking
    near-tie stops shuffles; this doesn't). Rules, in order:
      • value decides WHICH (rank by cluster value, stable name tiebreak),
      • a realistic COUNT cap (~1 base/day) keeps it from over-packing the drive budget,
      • drive time caps how many actually fit (days × ~6h round trip),
      • and the top marquee PARK is guaranteed in (iconic single-attraction destinations
        like a flagship national park rank low on cluster value but are must-sees)."""
    for s in stops:
        s["tier"] = "extension"
    valid = [s for s in stops if s.get("lat") is not None and s.get("lng") is not None]
    if not valid:
        return
    ranked = sorted(valid, key=lambda s: (-(s.get("value_score") or 0), s["name"]))
    cap = max(3, min(len(ranked), days))          # ~1 base per day, realistic count

    # real drive matrix (deterministic) so we don't over-pack the days with driving
    dur = None
    o = (origin_coords.get("lat"), origin_coords.get("lng"))
    if o[0] is not None:
        mat = drive_time_matrix([o] + [(s["lat"], s["lng"]) for s in valid])
        if not mat.get("error"):
            dur = mat.get("durations")
    idx = {id(s): i + 1 for i, s in enumerate(valid)}
    limit = days * _MAX_DRIVE_PER_DAY

    def fits(sel):
        return dur is None or _optimized_tour_hours([idx[id(x)] for x in sel], dur) <= limit

    core = []
    for s in ranked:
        if len(core) >= cap:
            break
        if not core or fits(core + [s]):
            core.append(s)

    # guarantee the single most iconic national-park-class destination
    parks = [s for s in ranked if _DESTINATION_KW.search(s.get("name") or "")]
    if parks and parks[0] not in core:
        non_parks = [s for s in core if not _DESTINATION_KW.search(s.get("name") or "")]
        if non_parks:  # swap the lowest-value non-park for the top marquee park
            drop = min(non_parks, key=lambda s: s.get("value_score") or 0)
            core = [s for s in core if s is not drop] + [parks[0]]
            log("select", f"  marquee guard: kept {parks[0]['name']} over {drop['name']}")

    if len(core) < 2 and len(ranked) >= 2:
        core = ranked[:2]
    for s in core:
        s["tier"] = "core"


def _enforce_drive_budget(valid: list, dur: list, days: int) -> None:
    """Size the core to the available DRIVE TIME — keep the highest-value stops AND fill
    the days. Two symmetric passes over the real optimized round-trip drive: (1) while
    the loop exceeds days×~6h of driving, demote the LOWEST-VALUE core stop; (2) while
    there's drive time to spare, promote the HIGHEST-VALUE extension that still fits.
    Value decides WHICH stops are core, drive time only decides HOW MANY — so a far
    iconic park is kept over a lesser town, and a 5-day trip isn't left half-empty.
    (This is time, not money — trip cost is handled in a later phase.)"""
    idx = {id(s): i + 1 for i, s in enumerate(valid)}  # matrix index (0 = origin)

    def tour_hours(core):
        return _optimized_tour_hours([idx[id(s)] for s in core], dur)

    drive_limit = days * _MAX_DRIVE_PER_DAY
    core = [s for s in valid if s.get("tier") == "core"]

    # (1) trim — drop the least worth-it core stop while the loop needs too much driving
    est = tour_hours(core)
    while est > drive_limit and len(core) > 2:
        drop = min(core, key=lambda s: s.get("value_score") or 0)
        drop["tier"] = "extension"
        core = [s for s in core if s is not drop]
        new = tour_hours(core)
        log("select", f"  ⚠ over drive-time limit ({est:.0f}h>{drive_limit:.0f}h) — dropped "
                      f"lowest-value {drop['name']} (score {drop.get('value_score',0):.0f}) → {new:.0f}h")
        est = new

    # (2) fill — promote the highest-value extensions that still fit the drive time,
    # so the trip uses the days available instead of stopping at a thin core.
    for s in sorted((x for x in valid if x.get("tier") != "core"),
                    key=lambda x: x.get("value_score") or 0, reverse=True):
        if tour_hours(core + [s]) <= drive_limit:
            s["tier"] = "core"
            core.append(s)
            log("select", f"  + drive time to spare — added {s['name']} "
                          f"(score {s.get('value_score',0):.0f}) → {tour_hours(core):.0f}h")


def _geocode_stop(name: str, state: str, country: str) -> dict:
    """Geocode a stop, tolerating compound/park names the geocoder chokes on
    (e.g. 'Rocky Mountain National Park/Estes Park') by trying simpler variants."""
    variants = [name]
    mpar = re.match(r"^(.*?)\s*\((.*?)\)\s*$", name)   # 'A (B)' -> try 'B' then 'A'
    if mpar:
        variants += [mpar.group(2).strip(), mpar.group(1).strip()]
    if "/" in name:                       # 'A/B' -> try 'B' then 'A'
        variants += [p.strip() for p in reversed(name.split("/"))]
    stripped = re.sub(r"\b(National Park|State Park|National Monument)\b", "", name).strip(" ()")
    if stripped and stripped != name:
        variants.append(stripped)
    for v in variants:
        c = get_coordinates(v, state, country)
        if c.get("latitude") is not None:
            return c
    return {}


def _split_place(place: str):
    """'Lawrence, KS' -> ('Lawrence', 'KS'); 'Denver' -> ('Denver', None)."""
    parts = [p.strip() for p in place.split(",")]
    return parts[0], (parts[1] if len(parts) > 1 else None)


# ── Stage 2 ────────────────────────────────────────────────────
def build_menus(stops: list, country: str = "USA", include_osm: bool = True) -> list:
    """stops: [{name, state}] → list of curated attraction menus, built CONCURRENTLY
    so N stops take roughly as long as one (the LLM/OSM calls are I/O-bound).
    include_osm=False skips the slow Overpass long-tail."""
    def one(s):
        return build_stop_menu(s["name"], region=s.get("state", ""),
                               state=s.get("state", ""), country=country,
                               include_osm=include_osm)
    if not stops:
        return []
    with step("menus", f"{len(stops)} stops in parallel (osm={include_osm}): "
                       f"{', '.join(s['name'] for s in stops)}"):
        with ThreadPoolExecutor(max_workers=min(8, len(stops))) as ex:
            return list(ex.map(one, stops))  # ex.map preserves input order


# ── Stage 3 ────────────────────────────────────────────────────
def assess_feasibility(origin_coords: dict, ordered_stops: list, days: int,
                       hours_per_sightseeing_day: float = 6.0) -> dict:
    """
    ordered_stops: [{name, lat, lng, chosen_hours, attractions}] in travel order.
    Computes the OBJECTIVE numbers (real drive legs + per-stop attraction hours),
    then lets the LLM judge realism holistically — no hardcoded hours-per-day or
    stops-per-day rule.
    """
    log("feasibility", f"▶ {len(ordered_stops)} stops over {days}d, computing drive legs")
    points = [{"name": "Origin", **origin_coords}] + ordered_stops + \
             [{"name": "Return to origin", **origin_coords}]
    legs, total_drive_h = [], 0.0
    for a, b in zip(points, points[1:]):
        if None in (a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng")):
            continue
        r = get_route(a["lat"], a["lng"], b["lat"], b["lng"])
        if "error" in r:
            legs.append({"from": a["name"], "to": b["name"], "error": r["error"]})
            continue
        total_drive_h += r["duration_hours"]
        legs.append({"from": a["name"], "to": b["name"],
                     "distance_miles": r["distance_miles"],
                     "duration_hours": r["duration_hours"],
                     "duration_text": r["duration_text"]})

    total_attraction_h = sum(s.get("chosen_hours", 0) for s in ordered_stops)
    per_stop = [{"name": s["name"], "hours": round(s.get("chosen_hours", 0), 1),
                 "count": len(s.get("attractions", []))} for s in ordered_stops]

    judgment = _feasibility_judge(days, total_drive_h, total_attraction_h, per_stop)
    log("feasibility", f"✓ verdict={judgment.get('verdict')} "
                       f"(drive={round(total_drive_h,1)}h attr={round(total_attraction_h,1)}h)")
    return {
        **judgment,            # verdict, headline, advice (LLM-judged)
        "days": days,
        "stops": len(ordered_stops),
        "total_drive_hours": round(total_drive_h, 1),
        "total_attraction_hours": round(total_attraction_h, 1),
        "legs": legs,
    }


def _feasibility_judge(days, drive_h, attr_h, per_stop) -> dict:
    """LLM judges realism from the objective numbers — no fixed per-day formula."""
    facts = (f"Trip length: {days} days.\n"
             f"Total driving (whole loop incl. return): {round(drive_h,1)} h.\n"
             f"Total chosen attraction time: {round(attr_h,1)} h across {len(per_stop)} stops.\n"
             "Per stop — name, chosen hours, # attractions:\n" +
             "\n".join(f"  - {p['name']}: {p['hours']}h, {p['count']} things" for p in per_stop))
    sys = """You judge whether a road trip is realistically doable in the given days.

Reason like an experienced road-tripper, HOLISTICALLY. Do NOT apply a rigid hours-per-day formula or a fixed number-of-stops-per-day rule — judge the whole picture. Keep in mind: a traveler can usually fit 2-3 attractions in a day (or one big half/full-day sight); driving days are mostly driving; several long back-to-back drives are tiring; nearby stops can be combined in a day.

Pick a verdict:
- "fits"  — comfortably doable
- "tight" — doable but busy/rushed
- "over"  — not realistic in this many days

Return JSON only, no markdown:
{"verdict":"fits|tight|over","headline":"<short phrase, e.g. 'Doable but busy'>","advice":"<2-3 candid sentences: is it realistic, and one concrete suggestion — what to cut/combine, or that there's room for more>"}"""
    try:
        return _parse_json(extract_text(call_llm(
            messages=[{"role": "user", "content": facts}],
            system_prompt=sys, tools=None, temperature=0.2, label="feasibility")))
    except Exception:
        return {"verdict": "tight", "headline": "Review the pace",
                "advice": "Couldn't auto-assess — eyeball the drive times and your days."}


# ── Stage 4 ────────────────────────────────────────────────────
def _trip_dossier(origin, ordered_stops, days, start_date, end_date, feasibility, country):
    """Shared data dossier (chosen stops + picks + drive legs + weather/holidays
    + dated day skeleton) used to build both the prose itinerary and the
    structured day plan."""
    lines = [
        "## The traveler's CHOSEN road trip (build the plan around exactly these):",
        f"Origin: {origin}",
        f"Days: {days}    Dates: {start_date or 'unspecified'} to {end_date or 'unspecified'}",
        "",
        "## Stops in travel order, with the attractions the traveler PICKED:",
    ]
    for i, s in enumerate(ordered_stops, 1):
        lines.append(f"\n### Stop {i}: {s['name']}, {s.get('state','')}")
        for a in s.get("attractions", []):
            lines.append(f"  - {a['name']} ({a.get('category','')}, ~{a.get('typical_hours','?')}h)"
                         f" — {a.get('why','')}")

    # Real lodging near each stop (OSM), so the plan can name an ACTUAL hotel to check into
    def _hotels_for(s):
        if not (s.get("lat") and s.get("lng")):
            return s["name"], []
        try:
            return s["name"], search_hotels(s["lat"], s["lng"], "mid-range", radius_km=6).get("hotels", [])[:4]
        except Exception:
            return s["name"], []
    if ordered_stops:
        with step("lodging", f"real hotels near {len(ordered_stops)} stops"):
            with ThreadPoolExecutor(max_workers=min(8, len(ordered_stops))) as ex:
                hotel_map = list(ex.map(_hotels_for, ordered_stops))
        lines.append("\n## Real lodging near each stop — name an ACTUAL one as where the traveler checks in "
                     "(pick one per overnight; if none listed, refer to lodging generically):")
        for name, hotels in hotel_map:
            if hotels:
                lines.append(f"  {name}: " + "; ".join(
                    h["name"] + (f" ({h['stars']}★)" if h.get("stars") else "") for h in hotels))
            else:
                lines.append(f"  {name}: (no listed hotels found)")

    if feasibility:
        lines.append("\n## Real drive legs (from routing):")
        for leg in feasibility.get("legs", []):
            if "error" in leg:
                continue
            lines.append(f"  - {leg['from']} → {leg['to']}: "
                         f"{leg['distance_miles']} mi, {leg['duration_text']}")
        lines.append(f"\nFeasibility: {feasibility.get('headline','')}")

    try:
        ci = get_country_info(country)
        if start_date:
            yr = int(start_date[:4])
            hol = get_holidays(ci.get("country_code", "US"), yr,
                               start_date=start_date, end_date=end_date)
            lines.append(f"\n## Holidays during trip: "
                         f"{json.dumps(hol.get('during_trip', []), default=str)}")
    except Exception:
        pass
    for s in ordered_stops:
        if s.get("lat") and start_date:
            try:
                w = get_weather(s["lat"], s["lng"], start_date=start_date, end_date=end_date)
                lines.append(f"\n## Weather at {s['name']}: {json.dumps(w, default=str)}")
            except Exception:
                pass

    date_labels = _date_list(start_date, days)
    lines.append(f"\n## DAY SKELETON — produce EXACTLY these {days} days, in order:")
    for i, dl in enumerate(date_labels, 1):
        lines.append(f"  Day {i} — {dl}")
    return "\n".join(lines)


def build_day_plan(origin, ordered_stops, days, start_date="", end_date="",
                   feasibility=None, country="USA"):
    """Structured day-by-day plan as JSON (for the day-planner UI). Returns a dict
    with title/route/total_drive/assumptions/days[]/tips[]."""
    dossier = _trip_dossier(origin, ordered_stops, days, start_date, end_date, feasibility, country)
    dossier += ("\n\nOutput the structured day plan as JSON per your system prompt. "
                f"EXACTLY {days} day objects, assign every picked attraction to one day.")
    with step("dayplan", f"building {days}-day plan from {len(ordered_stops)} stops"):
        raw = extract_text(call_llm(messages=[{"role": "user", "content": dossier}],
                                    system_prompt=DAY_PLAN_SYSTEM_PROMPT, tools=None,
                                    label="dayplan"))
    try:
        return _parse_json(raw)
    except Exception:
        # Fallback: at least return the dated skeleton so the UI shows days
        return {"title": f"{origin} road trip", "route": "", "total_drive": "",
                "assumptions": "", "tips": [],
                "days": [{"day": i + 1, "date": d, "where": "", "drive": "",
                          "blocks": [], "places": []}
                         for i, d in enumerate(_date_list(start_date, days))]}


def build_itinerary(origin: str, ordered_stops: list, days: int,
                    start_date: str = "", end_date: str = "",
                    feasibility: dict = None, country: str = "USA") -> str:
    """Prose day-by-day itinerary (markdown) from the user's selections."""
    dossier = _trip_dossier(origin, ordered_stops, days, start_date, end_date, feasibility, country)
    dossier += (
        "\n\nWrite the complete day-by-day road-trip itinerary using ONLY these chosen stops "
        "and attractions. HARD REQUIREMENTS:\n"
        f"- Output EXACTLY {days} day blocks, each headed '### Day N — <date> — <where you are that day>'.\n"
        "- Assign EVERY picked attraction to exactly ONE specific day (place every one; never repeat one).\n"
        "- Respect the route order + drive legs; a stop's attractions happen on/after the day you drive there.\n"
        "- For each day give the date, where you are, and Morning / Afternoon / Evening with the named places.")
    return extract_text(call_llm(messages=[{"role": "user", "content": dossier}],
                                 system_prompt=SYNTHESIZER_SYSTEM_PROMPT, tools=None,
                                 label="itinerary"))


def _date_list(start_date: str, days: int) -> list:
    """['Wed Jun 17, 2026', ...] for the trip, or ['Day 1', ...] if no start date."""
    try:
        d0 = datetime.strptime(start_date[:10], "%Y-%m-%d")
        return [(d0 + timedelta(days=i)).strftime("%a %b %d, %Y") for i in range(days)]
    except Exception:
        return [f"Day {i + 1}" for i in range(days)]
