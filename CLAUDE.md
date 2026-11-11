# Parcel Network Prototype

The full spec is in `PROJECT_SPEC.md`. Read it before starting any phase.

## Working rules

- Build one phase at a time. A phase must run end to end and meet its "Done when" test before the next one starts.
- Minneapolis–St. Paul only until phase 4. No national features before then.
- Every synthetic number, target and random seed goes in `assumptions.yaml`. Nothing hard-coded in `engine/`.
- Public data and stated assumptions only. No data from any former employer.
- Simple and working beats clever. Use a hand-picked list of designs before writing any search algorithm.
- Report results as ranges (50th and 90th percentile weeks), never just averages.
- A single-market search must finish in under 10 minutes on a laptop.
- Never commit `data/raw/`, large prepared files or secrets. API keys come from environment variables.
- The deployed Streamlit app loads only small precomputed results and must fit in about 1 GB of memory.
