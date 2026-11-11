# Parcel Network Prototype — Project Spec

## What this is

A Python model that simulates parcel deliveries across a national network and finds the network design that is most efficient once uncertainty in volume, delays, and driver supply is accounted for.

It's modeled on DoorDash Parcel: shippers inject packages into hubs, packages are sorted into routes, and gig drivers (Dashers) pick up routes and deliver. It's a portfolio piece for a Director of Network Strategy & Planning role, so it needs to be clear, credible, and easy for a hiring manager to click through.

Final deliverable: a live Streamlit app plus a public GitHub repo with a clear README.

## How to work on this

- Build one phase at a time. Each phase must run end to end before the next one starts.
- Start with one market (Minneapolis–St. Paul). Don't build national features until the local engine works.
- Keep every synthetic number in `assumptions.yaml`. Nothing hard-coded in the engine.
- Use only public data and stated assumptions. No data from any former employer.
- Prefer simple and working over clever. Details (routing, vehicle capacity, Dasher behavior) get refined in later passes.
- Results are always reported as ranges (typical week and a bad week, e.g. 50th and 90th percentile), never just averages.
- Every run uses a fixed random seed (set in `assumptions.yaml`) so results can be repeated exactly.
- `data/raw/` and large prepared files stay out of git. Scripts re-download them.
- API keys (e.g. the Census key) come from environment variables, never from files in the repo.

## The model

### Two layers

**National layer** (gets packages into each market)
- Shippers in real warehouse regions: Inland Empire CA, Dallas, Chicago, northern New Jersey, Atlanta.
- Each shipper sends daily volume to customers across the markets.
- Injection options per shipper and market: drop at a regional hub, truck direct to the market hub, or hand off to a partner carrier.
- Linehaul trucking between hubs: cost per mile, transit time, truck fill.
- Per market: launch or not, and build, lease, or partner for the hub.

**Local layer** (same engine in every market)
- Packages arrive at the market hub and are sorted into routes by cut-off time.
- Dashers claim routes and deliver.
- Tracks cost per package, on-time rate, Dasher earnings per hour, unclaimed routes.

### Markets

Proposed: Minneapolis–St. Paul (first), Chicago, Dallas–Fort Worth, Atlanta, Denver, Phoenix.

### Out of scope for now

Returns, air freight, international, sort-center machine layout.

## How the engine works

Goal: find the design that performs best across many simulated weeks, not the one that looks best on an average day (optimization under uncertainty).

1. **Random weeks.** Generate a few hundred versions of a week. Each draws its own values for:
   - daily volume vs. forecast
   - truck delays
   - Dasher no-shows and unclaimed routes
   - drive-time slowdowns (weather, traffic)
   - hub sort speed
   - late shipper handoffs

   Every design is tested against the same set of weeks (common random numbers) so differences come from the design, not luck.

2. **Day simulation.** For a given design and random week, play out each day step by step (SimPy): shipper pickups, linehaul arrivals, sort into routes by cut-off, Dashers claim and deliver. Captures knock-on effects like a late truck causing missed sort waves and half-full routes.

3. **Search.** Propose designs (hub count, location, size per market; injection choices; linehaul plan) and score each across all random weeks. A fast average-day estimate screens out clearly bad layouts first.
   - **First version (phases 2–3):** a short, hand-picked list of designs, e.g. 1, 2 or 3 hubs at the top-ranked candidate sites, in each size. Score them all and pick the best. No clever search yet.
   - **Later:** once that works, add a simple improvement loop (e.g. swap one hub site at a time and keep the change if it scores better). Only add anything fancier if this is clearly not good enough.

4. **Final check.** Rerun the top few designs on a fresh set of random weeks to confirm the winner.

5. **Scenarios.** Big shifts (next-day vs. same-day, a large new shipper, a hub outage, holiday peak) are named scenarios. The search runs for each.

### What "best" means (configurable)

Example objective: lowest expected cost per package, subject to
- at least 95% on-time in 90% of simulated weeks
- Dasher earnings per hour above a set floor

Targets live in `assumptions.yaml`.

### Run-time budget

- A full single-market search (all designs × all random weeks) must finish in **under 10 minutes on a laptop**.
- If it doesn't, cut work with the fast average-day screen or fewer random weeks during search. Don't make the simulation less realistic to save time.
- Real-road routing with OR-Tools runs only in the final check, never inside the search loop.

### Sanity checks

Keep a short list of public reference numbers (e.g. published last-mile cost per package ranges, typical gig driver earnings per hour, typical on-time rates) in `assumptions.yaml` with their sources. Each phase checks its outputs against them, and the README shows the comparison. If a result falls far outside a reference range, explain why or fix it.

### Routing (two levels)

- **During search:** estimate route cost from stop count, area, distance to hub, and vehicle capacity, using the standard approximation that route distance grows with the square root of (area × stops). Fast enough to score many designs.
- **During final check:** build real routes on real roads with OR-Tools.
- Details to model and refine iteratively:
  - packages per vehicle (Dashers use personal cars; capacity by volume, not just count)
  - stops per route and packages per stop (apartments vs. rural)
  - route duration (e.g. 2–4 hour blocks)
  - route attractiveness: pay vs. time and miles, which drives claim rate
- Compare the fast estimate against real routes to show when the shortcut is good enough.

## Data

| Input | Source | Real or synthetic |
| --- | --- | --- |
| Road network and drive times | OpenStreetMap via OSMnx | Real |
| Households and density by tract | US Census ACS | Real |
| Store locations (local pickups) | OpenStreetMap | Real |
| Candidate hub sites | Industrial/commercial land use in OpenStreetMap | Real locations, synthetic costs |
| Shipper locations and volume | Placed in real warehouse regions | Synthetic |
| Package demand by neighborhood | Scaled from households, with weekly and holiday patterns | Synthetic |
| Linehaul cost per mile, transit time | Assumption | Synthetic |
| Hub cost to build, lease, partner | Assumption | Synthetic |
| Dasher pay per route, claim rate, no-show rate | Assumption | Synthetic |
| Partner carrier overflow rate | Assumption | Synthetic |

## Outputs

- Recommended design per scenario, plus runner-ups and what each trades off.
- Metrics as ranges (typical and bad week):
  - cost per package, overall and by market
  - on-time rate vs. promise
  - Dasher earnings per hour, share of routes unclaimed
  - hub and truck utilization, where things backed up
  - market profitability
- Side-by-side scenario comparison.

App controls: volume and shipper mix, promise and cut-offs, hub location/size/build-lease-partner, Dasher pay and route size, disruptions. Heavy national runs are precomputed; single-market reruns can be live.

**Hosting limits.** Streamlit Community Cloud gives roughly 1 GB of memory. The deployed app:
- loads only small, precomputed result files (no raw road networks or census downloads)
- never runs OR-Tools routing on real roads
- keeps live reruns to the fast estimate for one market

**Naming.** The write-up can say the model is "modeled on DoorDash Parcel." The app's title, visuals and repo name use neutral names and no DoorDash logos or branding.

## Tech stack

- Python 3.11+
- OSMnx (road networks), Census API (households)
- pandas, NumPy
- Google OR-Tools (site selection, routing)
- SimPy (step-by-step simulation)
- Streamlit + Folium or pydeck (app and maps)
- GitHub + Streamlit Community Cloud (hosting)

## Repo layout

```
parcel-network/
├── README.md
├── PROJECT_SPEC.md
├── assumptions.yaml        # every synthetic number and target
├── requirements.txt
├── data/
│   ├── raw/                # downloaded public data
│   └── prepared/           # cleaned, per market
├── engine/
│   ├── data_prep.py        # pull and prepare map, census, sites
│   ├── demand.py           # synthetic demand generation
│   ├── randomness.py       # random week generation
│   ├── routing.py          # route cost estimate + real route building
│   ├── simulation.py       # day-by-day simulation
│   ├── search.py           # design search
│   └── scenarios.py        # scenario definitions and runner
├── scenarios/              # saved scenario settings and results
├── notebooks/              # exploration and checks
├── app/
│   └── streamlit_app.py
└── tests/
```

## Build phases

1. **Data for Minneapolis–St. Paul.** Road network, census tracts, stores, candidate hubs, synthetic demand.
   Done when: a map shows neighborhoods, daily demand, and candidate hubs.
2. **Local engine for one market.** Hub layouts, sort by cut-off, Dasher routes (fast estimate first), cost per package and Dasher earnings.
   Done when: one simulated day runs end to end with sensible numbers.
3. **Optimization under uncertainty for one market.** Random weeks, scoring each design across them, search picks the most robust design, final check.
   Done when: normal week, holiday peak, and hub outage each produce a recommended design with ranges. First shareable checkpoint.
4. **National layer.** Shippers, injection choices, linehaul, partner carrier overflow.
   Done when: packages flow from shipper regions into Minneapolis–St. Paul through each injection option.
5. **All six markets.** Local engine in the other five; launch and build/lease/partner decisions.
   Done when: the network shows which markets make money and where linehaul costs hurt.
6. **App and write-up.** Streamlit app, findings summary, README, deploy.
   Done when: someone can open a link, pick scenarios, and follow the story unassisted.

## Open decisions

- [ ] Confirm the six markets
- [ ] Hours per week available (sets the timeline)
- [ ] Same-day from local stores (DashMart-style) early, or after phase 5. *Leaning: after phase 5. It's a second engine and would slow down the core story.*
- [ ] Repo public from the start or at the end. *Leaning: private until phase 3 (first shareable checkpoint), then public.*

## Dev environment

- Work happens inside a VS Code Dev Container (`.devcontainer/`). It mounts only this project folder.
- Network access from the container is limited to an allowlist: PyPI, GitHub, the Claude API, OpenStreetMap (Overpass/Nominatim) and the Census API.
- Downloaded data is cached in a container volume at `$PARCEL_CACHE_DIR` (`/home/vscode/cache`) so it survives rebuilds and stays out of git. Engine code reads that variable instead of hard-coding a path.
- The container has no general admin (sudo) rights. It can only run the firewall script.
- Streamlit runs on port 8501, forwarded to the host.
