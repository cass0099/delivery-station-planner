---
title: Delivery Station Planner
emoji: 📦
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Simulate next-day parcel delivery and size a station network
---

# Delivery Station Planner

This project uses a basic simulation to find the lowest-cost network of delivery stations for next-day DoorDash Parcel delivery in Minneapolis–St. Paul. A delivery station receives overnight trucks, sorts packages into routes, and is where Dashers pick up their routes in the morning. The question is how many stations the metro area should run and how large each one should be.

More stations cost more in rent and staff, but they shorten the drives Dashers make from home to the station and from the station to the first delivery, and they give the network a backup when one station closes. Fewer stations cost less in a normal week but can fall behind during the holiday peak or when a station shuts down for a day. The model prices that trade-off.

## Using the app

The app walks through the assumptions in five steps: volume, station costs, Dasher pay, service targets, and disruptions. Every assumption starts at a default drawn from public data or a stated estimate. Each slider says whether the model uses its value exactly as set or treats it as an average that varies from day to day, and varying settings show the range the model actually draws.

On the Run simulation page you can choose an instant estimate, a quick run that takes about a minute, or a full run that takes several minutes on the hosted app. The results page shows the recommended network on a map and a scorecard that tests every station layout against the targets in normal weeks, holiday peak weeks and station outage weeks.

## How the simulation works

The model is a Monte Carlo simulation that runs one day at a time and follows every package. Each night, trucks bring packages to the stations, some of them late, and each station sorts at its own speed until the 7 a.m. cut-off. In the morning, packages are grouped into routes within time, package and cargo limits, and each route is offered to Dashers. The chance that a Dasher accepts a route rises with its pay per hour after car costs, and routes nobody accepts are offered again with their base pay raised. Packages that miss the cut-off or sit on an unclaimed route roll over to the next day.

Each station layout runs through the same random weeks, so differences between layouts come from the layouts and not from luck. The model screens 18 layouts on an average day, simulates the remaining ones for 200 weeks of each kind, and rechecks the closest ones on 500 to 1,000 fresh weeks. A layout must reach 98% on-time delivery (OTD) in a typical normal week and at least 96% OTD in 95% of weeks, with lower targets in holiday peak weeks (97% typical, and 94% OTD in 90% of peak weeks). Every route pays Dashers at least $15 an hour after car costs, and the cost of raising pay to that level counts against each layout. The recommendation is the cheapest layout that meets every target in all three kinds of week, with the yearly cost weighting 44 normal weeks, 6 peak weeks and 2 outage weeks.

## Results with the default assumptions

With the default assumptions, the model recommends three small delivery stations. Over a year they cost about $2.75 per package, about 13 cents less than four small stations. In a typical normal week they deliver 98.8% of packages on time, and the worst normal week in ten still reaches about 96.9%. They meet the targets in normal weeks, holiday peak weeks and station outage weeks. A single medium station is cheaper, but it fails the station outage week, because when it closes the whole network is down for the day.

These numbers depend on the assumptions, and the app lets you change any of them and rerun the simulation. The network share is the hardest assumption to pin down, so the results page also shows how the recommendation changes at a 2%, 3% and 5% share of the metro's parcels.

## Data and assumptions

The road network, drive times and candidate station sites come from OpenStreetMap. Household counts come from the US Census Bureau's American Community Survey for 2019 to 2023. Parcel volume per household comes from the Pitney Bowes Parcel Shipping Index. Warehouse rent comes from Twin Cities industrial market reports by Colliers, Transwestern and CBRE, wages from the Bureau of Labor Statistics, and car costs from the IRS mileage rate. Everything else is a stated assumption, and every number lives in `assumptions.yaml` with its source or a note that it is an assumption.

The fast route formula was checked against routes planned with OR-Tools on real streets for two stations. The real routes needed the same number of Dashers and similar hours, and the model's distance was calibrated from that check.

## Limits

This is a basic model. It covers one metro area, and the national layer of shippers, long-distance trucking and partner carriers is not built yet. Demand is scaled from household counts rather than real orders. Station sites come from a fixed shortlist rather than being chosen freely, and route lengths use a calibrated formula rather than planned routes.

## Running it locally

```
pip install -r requirements.txt
export CENSUS_API_KEY=...                      # free key from api.census.gov
python -m engine.data_prep --market msp --step all
python -m engine.demand --market msp
python -m engine.search --scenario all         # full search, a few minutes
python -m engine.live --sensitivity 0.02 0.03 0.05
streamlit run app/streamlit_app.py
```

The hosted app uses `requirements-app.txt` and the `Dockerfile`, and it loads only the small prepared files in `data/prepared` and the saved results in `scenarios`.
