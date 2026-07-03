# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
pip install -r requirements.txt
# .env: ANTHROPIC_API_KEY=...   (required)
#       GOOGLE_API_KEY=...      (optional — Google Places ratings; app degrades gracefully without it)
python app.py  # Starts Flask at http://localhost:5000
```

No build step. No test suite. No linter config.

## Architecture: interactive, human-in-the-loop planning

Voyager is a map-based road-trip planner. The LLM **curates and advises**; the **human decides** on the map. The flow is a set of request/response endpoints (state lives in a per-session dict in `app.py` — no long-lived threads):

```
propose  → interactive_agent.propose_stops_agentic  — LLM brainstorms a candidate pool
route    → tools.get_optimized_trip (OSRM TSP)       — optimal stop order + geometry
menus    → stop_menu.build_stop_menu                 — per-stop "things to do" the user picks
feasibility → interactive_agent.assess_feasibility   — real drive legs + chosen hours → verdict
finalize → interactive_agent.build_day_plan          — structured day-by-day JSON
copilot  → interactive_agent.chat_turn               — stateful chat that edits the map
```

## Stop proposal: LLM brainstorms, code selects (`interactive_agent.py`)

`propose_stops_agentic` is the core. The split is deliberate:

- **The LLM** (via `llm.run_agent_loop`, one tool: `lookup_places`) brainstorms the candidate pool and names each base's 2–3 top attractions — its strength, world knowledge.
- **`lookup_places`** geocodes each base and computes a **cluster value** (`_cluster_value`): the base's attractions rated by Google Places (`tools.place_value`), blended `best + 0.3·2nd + 0.15·3rd` so a multi-draw town isn't judged on one sight.
- **Deterministic code** makes the final pick (`_select_core_by_value`): highest value → capped to ~1 base/day → must fit the real drive time (`_optimized_tour_hours`, OSRM matrix + 2-opt) → the top national-park-class stop is guaranteed in. Same inputs → same core every run (an LLM ranking near-tie stops would shuffle).

`propose_stops` (+ `_select_core`, LLM selector) is a deterministic fallback used only if the agentic path fails.

## Configuration (`config.py`)

All LLM system prompts live here — the primary place to tune behavior:
- `AGENT_PLANNER_SYSTEM_PROMPT` — the brainstorm/rate agent (calls `lookup_places`)
- `STOP_PROPOSER_SYSTEM_PROMPT` / `STOP_SELECTOR_SYSTEM_PROMPT` — the fallback pipeline
- `STOP_MENU_SYSTEM_PROMPT` — per-stop attraction menu (incl. `typical_hours` estimates)
- `DAY_PLAN_SYSTEM_PROMPT`, `SYNTHESIZER_SYSTEM_PROMPT` — final itinerary (JSON / prose)
- `ASSISTANT_SYSTEM_PROMPT` — the copilot; emits `[[ADD: …]]` / `[[REMOVE: …]]` directives

Model: `MODEL_NAME` (`claude-sonnet-4-6`), `MAX_TOKENS = 4000`. `_MAX_DRIVE_PER_DAY = 6.0` (drive-time budget, in `interactive_agent.py`) is a **time** limit, not money.

## Tools (`tools.py`) — all free, no auth (Google Places optional)

- **Open-Meteo** — `get_coordinates`, `get_weather`
- **OpenStreetMap Overpass / Nominatim** — `search_attractions`, `geocode_places`
- **OSRM** — `drive_time_matrix` (all-pairs hours), `get_optimized_trip` (TSP order + geometry), `get_route` (one leg)
- **Google Places (New)** — `place_value` (rating + review count); `value_score` = rating × log10(reviews)
- **REST Countries / Nager.Date / Frankfurter** — country info, holidays, currency

## Endpoints (`app.py`) → frontend (`frontend/index.html`, vanilla JS + Leaflet)

`/i/propose`, `/i/route`, `/i/menus`, `/i/geocode`, `/i/feasibility`, `/i/finalize`, `/i/chat`. The map + stop tabs re-render from each response; the copilot's `[[ADD]]`/`[[REMOVE]]` directives mutate the map live.
