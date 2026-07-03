# Voyager — Agentic AI Road-Trip Planner

An interactive, map-based road-trip planner where an **LLM agent gathers real-world data through tools and edits your route live from chat**, while deterministic code guarantees the plan is drivable and reproducible. You type a region and dates; the agent proposes a route grounded in real drive times and visitor ratings; you refine it on a map or just *tell* the copilot — "add Vail, drop Denver" — and the map changes.

<!-- Add a screenshot/GIF of the map + copilot here: save it as docs/screenshot.png -->
![Voyager — map, stop tabs, and copilot](docs/screenshot.png)

> The model is the **brain** (what to gather, what to say, when to act); free real-world APIs are the **senses** (distances, ratings, weather); a thin layer of deterministic code is the **spine** (reproducible selection, real routing). Every plan is grounded in data, not vibes.

---

## What it does

```
Type "Colorado · Jun 28–Jul 3 · from Lawrence, KS"
   → the agent brainstorms iconic stops and looks up their real ratings itself
   → deterministic code picks a drivable core set (real drive-times vs your days)
   → expand any stop → curated things-to-do with ★ratings and realistic visit times
   → chat: "add Aspen, remove Denver"   → the map updates instantly
   → "Build itinerary"                  → day-by-day plan with lodging + real drive legs
```

---

## Architecture — what makes it *agentic*

"Agentic" here means one specific, testable thing: **the LLM decides on its own (a) when and what to fetch through tools, and (b) when to take an action on the world** — rather than code calling the model at fixed points and parsing the reply. Voyager has **two genuine agent loops** wrapped around a grounded, deterministic core.

### 1. The planning agent — a tool-use loop (`llm.run_agent_loop`)

Given the goal and **one tool** (`lookup_places`), the *model* drives the loop — it decides what to brainstorm, what to look up, and when it's done:

```
 system: you plan road trips; you have lookup_places({name, state, top_pois})
 user:   plan a 5-day trip to Colorado from Lawrence, KS
    │
    ▼
 ┌─────────────────────────── agent loop ───────────────────────────┐
 │  LLM turn ──► stop_reason == "tool_use" ?                         │
 │     ├─ yes → execute the tool call it CHOSE, feed result back ─┐  │
 │     │        (real Google ratings for the stops it named)      │  │
 │     └─ no  → it's finished → return its candidate pool (JSON)  │  │
 │              ▲─────────────────────────────────────────────────┘  │
 └───────────────────────────────────────────────────────────────────┘
```

The LLM itself brainstorms candidate stops from its world knowledge, decides to call `lookup_places` (and with which places + their top attractions), reads the real ratings that come back, and judges whether it has enough or should look up more — then emits the pool. **Nothing in the code says "call the tool now."** That decision is the model's.

### 2. The copilot agent — perceive → decide → act (`interactive_agent.chat_turn`)

Every turn, the copilot is handed the **live plan state** (origin, dates, selected stops, the live route distance/time, feasibility). It interprets free-form intent and decides **whether to act on the map**:

```
 "add Vail"           → ACTS:    reply + hidden directive [[ADD: Vail, CO]]   → map mutates
 "is Vail worth it?"  → ADVISES: reply only  (deliberately withholds the action)
 "swap Denver for Boulder" → two directives: [[REMOVE: Denver]] [[ADD: Boulder, CO]]
```

Choosing *when to act on the world* versus *when to only talk* is the agentic decision — it's not a fixed rule. The frontend parses the `[[ADD:…]]` / `[[REMOVE:…]]` directives, geocodes the place, and re-renders the map, so chat is a first-class editing surface, not Q&A.

### 3. The grounded, deterministic core (the "spine")

Around those two loops sits machinery that keeps the plan **real and reproducible** — and this part is deliberately *not* agentic:

- **Tools = senses.** Google Places (ratings + reviews), OSRM (drive-time matrix, TSP route order, geometry), OpenStreetMap/Overpass (attractions, hotels), Open-Meteo (weather).
- **Deterministic selection.** The LLM does *not* make the final "which stops are core" call — an algorithm does (`interactive_agent._select_core_by_value`): rank stops by **cluster value** (Google rating blended across each stop's top attractions), keep the highest-value set that fits `days × ~6h` of real driving, cap the count to the days, and guarantee the flagship national park. **Same inputs → same plan, every run.** (An LLM ranking near-tie stops shuffles between runs; code doesn't.)

**Why the split?** Agency and reproducibility trade off. The parts where creativity and judgement matter — *which places exist, what to do there, how to answer you* — are the agent's. The part that must be trustworthy and repeatable — *the drivable route* — is code's. The agent uses the deterministic selector the way a coding agent uses a compiler: a reliable tool, not a decision it re-litigates.

---

## Full request flow

```
 USER (browser: Leaflet map + chat)
        │  region · dates · origin
        ▼
┌──────────────────── Flask session API (app.py) ────────────────────┐
│                                                                     │
│  /i/propose   PLANNING AGENT (run_agent_loop, tool: lookup_places)  │  LLM drives tool use
│               → cluster value (Google Places) per stop              │  Google Places (New)
│               → DETERMINISTIC core selection (value + drive + cap)  │  code + OSRM matrix
│  /i/route     optimal visiting order + road geometry               │  OSRM /trip (TSP)
│  /i/menus     per-stop things-to-do, tiered + ★rated + hours       │  LLM + Overpass + Places
│  /i/feasibility  "fits / tight / over" verdict over real legs      │  LLM judge
│  /i/finalize  day-by-day plan: lodging, check-in pacing, drives    │  LLM + Overpass hotels
│  /i/chat      COPILOT (stateful) ──► [[ADD]] / [[REMOVE]]          │  LLM decides + acts
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
        │  JSON responses / hidden directives
        ▼
 UI re-renders the map, stop tabs, and day cards from each response
```

State lives in a per-session dict in `app.py` (no long-lived threads); every endpoint is plain request/response.

---

## Engineering highlights

- **Two real agent loops** — a tool-use planning agent (model-driven `lookup_places`) and a stateful action-taking copilot that edits the map from natural language.
- **Region-agnostic** — prompts teach the *method*, not Colorado facts; works for any region from the model's own knowledge.
- **Cluster value** — a stop is scored by its *cluster* of attractions (rating × review count, blended), so a multi-draw town isn't judged on one sight.
- **Deterministic, reproducible selection** — code picks the core from value + real OSRM drive time; the same trip comes out the same every run.
- **Grounded day plans** — real drive legs, realistic per-attraction visit times, and real OSM hotels with sane check-in pacing after long drives.
- **Free-first** — Open-Meteo, OpenStreetMap/Overpass, OSRM, Nominatim — no keys; Anthropic (required) and Google Places (optional) are the only paid pieces.

---

## Project layout

```
app.py               Flask endpoints (the session API above)
interactive_agent.py the flow: propose_stops_agentic, _select_core_by_value, menus, feasibility, day plan
llm.py               run_agent_loop (the tool-use agent) + call_llm
config.py            every system prompt (the primary tuning surface)
stop_menu.py         per-stop "things to do" curation
tools.py             real-world APIs (Google Places, OSRM, Overpass, Open-Meteo, …)
frontend/index.html  vanilla JS + Leaflet UI (map, stop tabs, copilot, day cards)
```

---

## Tech stack

**Python · Flask · Anthropic Claude · Leaflet · OSRM · OpenStreetMap/Overpass · Open-Meteo · Google Places API · vanilla JS/CSS**

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your keys into .env
#   ANTHROPIC_API_KEY=sk-ant-...   (required — the agent + LLM steps)
#   GOOGLE_API_KEY=AIza...         (optional — Google Places ratings; degrades gracefully)
python app.py                 # → http://localhost:5000  → "Plan Interactively"
```
