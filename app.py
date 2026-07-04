"""
app.py
──────
Flask web server for the interactive trip planner. Exposes the propose → menus →
route → feasibility → finalize flow as request/response endpoints, plus the
stateful copilot chat. State lives in a per-session dict (no long-lived threads).
"""

import uuid

from flask import Flask, request, jsonify, send_from_directory
from interactive_agent import (
    propose_stops_agentic, build_menus, assess_feasibility,
    build_day_plan, chat_turn,
)
from tools import get_optimized_trip, geocode_places
from logutil import log


def _state_summary(s, state):
    """Compact live plan-state string injected into the copilot each turn."""
    lines = [f"Origin: {s['origin']}", f"Destination region: {s['destination']}",
             f"Trip length: {s['days']} days",
             f"Dates: {s.get('start_date') or '?'} to {s.get('end_date') or '?'}"]
    prop = s.get("proposal", {})
    if prop.get("stops"):
        lines.append("Candidate stops proposed: " +
                     ", ".join(f"{x['name']} ({x.get('tier')})" for x in prop["stops"]))
    sel = state.get("selected") or []
    lines.append("Currently SELECTED stops (route order): " + (", ".join(sel) if sel else "none yet"))
    if state.get("total_miles"):
        lines.append(f"Live route: {state['total_miles']} mi, {state.get('total_text', '')} driving")
    if state.get("verdict"):
        lines.append(f"Feasibility: {state['verdict']} — {state.get('feas_headline', '')}")
    return "\n".join(lines)

app = Flask(__name__, static_folder="frontend", static_url_path="")

# In-memory session store for the interactive flow (origin/dates/days persist
# across the propose → menus → feasibility → finalize requests).
SESSIONS = {}


@app.route("/")
def index():
    resp = send_from_directory("frontend", "index.html")
    # never cache the HTML, so code changes show up on a normal refresh
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


# ── INTERACTIVE FLOW (human-in-the-loop selection) ─────────────
@app.route("/i/propose", methods=["POST"])
def i_propose():
    b = request.get_json()
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {
        "origin": b.get("origin", ""),
        "destination": b.get("destination", ""),
        "days": int(b.get("days", 5)),
        "start_date": b.get("start_date", ""),
        "end_date": b.get("end_date", ""),
        "country": b.get("country", "USA"),
    }
    s = SESSIONS[sid]
    log("API", f"/i/propose [{sid[:6]}] {s['origin']} → {s['destination']} {s['days']}d")
    menu = propose_stops_agentic(s["origin"], s["destination"], s["days"],
                                 s["start_date"], s["end_date"], s["country"])
    s["origin_coords"] = menu.get("origin_coords", {})
    s["proposal"] = menu
    s["messages"] = []  # stateful copilot conversation for this session
    return jsonify({"session_id": sid, "proposal": menu})


@app.route("/i/chat", methods=["POST"])
def i_chat():
    b = request.get_json()
    s = SESSIONS.get(b.get("session_id"))
    if not s:
        return jsonify({"error": "unknown session"}), 400
    log("API", f"/i/chat [{b.get('session_id','?')[:6]}] msg={b.get('message','')[:70]!r}")
    hist = s.setdefault("messages", [])
    hist.append({"role": "user", "content": b.get("message", "")})
    reply = chat_turn(hist, _state_summary(s, b.get("state") or {}))
    hist.append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply})


@app.route("/i/menus", methods=["POST"])
def i_menus():
    b = request.get_json()
    s = SESSIONS.get(b.get("session_id"))
    if not s:
        return jsonify({"error": "unknown session"}), 400
    log("API", f"/i/menus [{b.get('session_id','?')[:6]}] "
               f"stops={len(b.get('stops', []))} quick={b.get('quick', False)}")
    # quick=True (map popup preview) skips the slow OSM long-tail
    menus = build_menus(b.get("stops", []), s["country"],
                        include_osm=not b.get("quick", False))
    return jsonify({"menus": menus})


@app.route("/i/route", methods=["POST"])
def i_route():
    """Real road geometry + per-leg distances for the map: origin → stops → origin."""
    b = request.get_json()
    s = SESSIONS.get(b.get("session_id"))
    if not s:
        return jsonify({"error": "unknown session"}), 400
    oc = s.get("origin_coords", {})
    stops = b.get("stops", [])
    pin = (b.get("pin_first") or "").strip()   # traveler pinned a start stop (rest re-optimized)
    log("API", f"/i/route [{b.get('session_id','?')[:6]}] {len(stops)} stops"
               + (f" — start pinned: {pin}" if pin else " — optimizing"))
    route = get_optimized_trip(oc.get("lat"), oc.get("lng"), stops, pin_first=pin or None)
    ordered = route.get("ordered_stops", stops)

    # Chain in the OPTIMIZED order: origin → stops → origin
    chain = ([{"name": "Origin", "lat": oc.get("lat"), "lng": oc.get("lng")}]
             + ordered
             + [{"name": "Return", "lat": oc.get("lat"), "lng": oc.get("lng")}])
    legs = []
    for i, leg in enumerate(route.get("legs", [])):
        if i + 1 >= len(chain):
            break
        a, b2 = chain[i], chain[i + 1]
        if None in (a.get("lat"), b2.get("lat")):
            continue
        legs.append({**leg, "from": a["name"], "to": b2["name"],
                     "mid": [(a["lat"] + b2["lat"]) / 2, (a["lng"] + b2["lng"]) / 2]})
    return jsonify({"geometry": route.get("geometry", []), "legs": legs,
                    "ordered": [{"name": st["name"], "lat": st.get("lat"),
                                 "lng": st.get("lng")} for st in ordered],
                    "total_miles": route.get("total_miles"),
                    "total_duration_text": route.get("total_duration_text")})


@app.route("/i/geocode", methods=["POST"])
def i_geocode():
    """POI geocoding so chosen attractions can be pinned on the map."""
    b = request.get_json()
    near = b.get("near") or {}
    log("API", f"/i/geocode {len(b.get('queries', []))} queries")
    res = geocode_places(b.get("queries", []), near.get("lat"), near.get("lng"))
    return jsonify({"results": res})


@app.route("/i/feasibility", methods=["POST"])
def i_feasibility():
    b = request.get_json()
    s = SESSIONS.get(b.get("session_id"))
    if not s:
        return jsonify({"error": "unknown session"}), 400
    log("API", f"/i/feasibility [{b.get('session_id','?')[:6]}] "
               f"stops={len(b.get('ordered_stops', []))}")
    f = assess_feasibility(s.get("origin_coords", {}), b.get("ordered_stops", []),
                           s["days"])
    return jsonify({"feasibility": f})


@app.route("/i/finalize", methods=["POST"])
def i_finalize():
    b = request.get_json()
    s = SESSIONS.get(b.get("session_id"))
    if not s:
        return jsonify({"error": "unknown session"}), 400
    log("API", f"/i/finalize [{b.get('session_id','?')[:6]}] "
               f"building day plan from {len(b.get('ordered_stops', []))} stops")
    plan = build_day_plan(s["origin"], b.get("ordered_stops", []), s["days"],
                          s["start_date"], s["end_date"],
                          feasibility=b.get("feasibility"), country=s["country"])
    return jsonify({"plan": plan})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Voyager — http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000, threaded=True)
