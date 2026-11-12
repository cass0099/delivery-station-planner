# Delivery Station Planner

This project uses a basic simulation to find the lowest-cost network of delivery stations for next-day DoorDash Parcel delivery in Minneapolis–St. Paul. A delivery station receives overnight trucks, sorts packages into routes, and is where Dashers pick up their routes in the morning. The question is how many stations the metro area should run, how large each one should be, and, if the network leaves out some ZIP codes, how that changes the answer.

More stations cost more in rent and staff, but they shorten the drives Dashers make from home to the station and from the station to the first delivery, and they give the network a backup when one station closes. Fewer stations cost less in a normal week but can fall behind during the holiday peak or when a station shuts down for a day. The model prices that trade-off.

## Using the app

The app walks through the assumptions in six steps: volume, coverage, station costs, Dasher pay, service targets, and disruptions. Every assumption starts at a default drawn from public data or a stated estimate. Each slider says whether the model uses its value exactly as set or treats it as an average that varies from day to day, and varying settings show the range the model actually draws.

On the Run simulation page you can choose an instant estimate, a quick run that takes about a minute, or a full run that takes several minutes on the hosted app. The results page shows the recommended network on a map and a scorecard that tests every station layout against the targets in normal weeks, holiday peak weeks and station outage weeks.

## How the simulation works

The model is a Monte Carlo simulation that runs one day at a time and follows every package. Each night, trucks bring packages to the stations, some of them late, and each station sorts at its own speed until the 7 a.m. cut-off. In the morning, packages are grouped into routes within time, package and cargo limits, and each route is offered to Dashers. The chance that a Dasher accepts a route rises with its pay per hour after car costs, and routes nobody accepts are offered again with their base pay raised. Packages that miss the cut-off or sit on an unclaimed route roll over to the next day.

By default the network serves every ZIP code in the metro. The coverage step lets it serve fewer. A shipper checks each package's ZIP code against the carrier's coverage list when it buys the label, so packages for ZIP codes the network does not serve go to other carriers and never enter the simulation. Each station layout serves the ZIP codes closest to its own stations by drive time, adding whole ZIP codes until the chosen share of households is covered. A carrier that covers less of the metro also wins a smaller share of shippers' packages in the ZIP codes it does serve, because many shippers want one carrier for most of their volume. The model assumes this share falls in a straight line from the full share at 100% coverage to a set fraction at 50% coverage, the lowest coverage shippers would accept. With fewer ZIP codes, routes and Dasher commutes get shorter, but the station costs are spread over fewer packages. Cost and OTD are measured on the served ZIP codes only.

Each station layout runs through the same random weeks, so differences between layouts come from the layouts and not from luck. The model screens 18 layouts on an average day, simulates the remaining ones for 200 weeks of each kind, and rechecks the closest ones on 500 to 1,000 fresh weeks. A layout must reach 98% on-time delivery (OTD) in a typical normal week and at least 96% OTD in 95% of weeks, with lower targets in holiday peak weeks (97% typical, and 94% OTD in 90% of peak weeks). Every route pays Dashers at least $15 an hour after car costs, and the cost of raising pay to that level counts against each layout. The recommendation is the cheapest layout that meets every target in all three kinds of week, with the yearly cost weighting 44 normal weeks, 6 peak weeks and 2 outage weeks. OTD comes before cost. If no layout of up to four stations meets every target, or the cheapest one that does has four stations, the run also tries larger networks of 5 to 15 stations, because at high volume a larger network can cost less. If none of those meets every target either, the app says there is no recommendation, shows the network that comes closest on OTD, and explains what is holding OTD back, such as late trucks that no number of stations can fix.

## Results with the default assumptions

With the default assumptions, the model recommends three small delivery stations. Over a year they cost about $2.75 per package, about 13 cents less than four small stations. In a typical normal week they deliver 98.8% of packages on time, and the worst normal week in ten still reaches about 96.9%. They meet the targets in normal weeks, holiday peak weeks and station outage weeks. A single medium station is cheaper, but it fails the station outage week, because when it closes the whole network is down for the day.

These numbers depend on the assumptions, and the app lets you change any of them and rerun the simulation. The network share is the hardest assumption to pin down, so the results page also shows how the recommendation changes at a 2%, 3% and 5% share of the metro's parcels.

## Data and assumptions

The road network, drive times and candidate station sites come from OpenStreetMap. Household counts come from the US Census Bureau's American Community Survey for 2019 to 2023. ZIP codes come from the Census Bureau's 2020 relationship file between ZIP code tabulation areas and Census tracts, and each tract is assigned to the ZIP code that holds most of its land. Parcel volume per household comes from the Pitney Bowes Parcel Shipping Index. Warehouse rent comes from Twin Cities industrial market reports by Colliers, Transwestern and CBRE, wages from the Bureau of Labor Statistics, and car costs from the IRS mileage rate. Everything else is a stated assumption, and every number lives in `assumptions.yaml` with its source or a note that it is an assumption.

The fast route formula was checked against routes planned with OR-Tools on real streets for two stations. The real routes needed the same number of Dashers and similar hours, and the model's distance was calibrated from that check.

## Limits

This is a basic model. It covers one metro area, and the national layer of shippers, long-distance trucking and partner carriers is not built yet. Demand is scaled from household counts rather than real orders. ZIP codes are built from whole Census tracts, so their edges are close to, but not exactly, the postal boundaries. The coverage choice is set by the user, not optimized, and the effect of coverage on shippers' share is a stated assumption without public data behind it. Station sites come from a fixed shortlist rather than being chosen freely, and route lengths use a calibrated formula rather than planned routes.

## Testing

Besides the unit tests in `tests/`, three scripts in `tools/` test the model and the app as a visitor would use them. `tools/stress_test.py` sets every control to both ends of its range, then tries combined extremes and random combinations. For each case it runs the model, checks that every number is valid and that the pick is the cheapest layout meeting every target, and opens the app's pages with the result. `tools/widget_test.py` moves every slider in the app to both ends and checks the stored value in real units. `tools/app_flow_test.py` clicks through the app, runs a live simulation, stops a long one and resets the settings. `tools/deploy_reload_test.py` changes the engine on disk while a process is running, as a deploy does, and checks that the next run's workers use the new code.

A run that would take longer than 30 minutes on the server is turned off before it starts, and any run can be stopped, so one visitor cannot hold the server.

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

The hosted app runs on Streamlit Community Cloud. It installs only the libraries in `app/requirements.txt` and loads only the small prepared files in `data/prepared` and the saved results in `scenarios`. A `Dockerfile` is also included for hosts that run containers.
