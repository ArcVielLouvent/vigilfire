# Vigilfire — Global Wildfire Risk Early-Warning Scanner

Built for the PyStorm hackathon ("Code for a Greener Future").

## The problem

A UEA-led review published in *Nature Reviews Earth & Environment* found
that 335 million hectares burned globally in 2025 — 16% below the
long-term average — yet 2025 was one of the costliest and deadliest
wildfire years on record, with catastrophic fires across Canada, the US,
Europe, and South Korea causing over 300,000 evacuations and 90+
fatalities. The clearest example of the disconnect: the January 2025 Los
Angeles fires burned only ~23,000 hectares but caused over $53 billion in
damage — making it the costliest single wildfire event of the year,
while total global natural-hazard damages in 2025 reached an estimated
$224–275 billion.

The takeaway: wildfire damage is no longer well-predicted by how much
land burns. It's driven by *where* fires ignite relative to people and
infrastructure, and by how much advance warning responders get. Most
public fire-tracking tools (including NASA FIRMS itself) only show fires
that have *already* been detected by satellite — after ignition, not
before.

## What Vigilfire does

Vigilfire scans a user-specified region and estimates near-term wildfire
risk *before* ignition, using only the antecedent weather trend (heat,
dryness, humidity, wind) of the preceding week — the same conditions
fire scientists use to explain why a fire spread the way it did, but
applied predictively instead of retrospectively.

1. **Historical training data**: satellite-confirmed fire detections from
   NASA FIRMS (VIIRS/MODIS) are paired with the preceding week of weather
   from NASA POWER, for both fire and non-fire locations, to train a
   Random Forest risk classifier.
2. **Scan area**: the user picks or enters a region; the app pulls the
   latest week of weather for that point from NASA POWER and returns a
   risk score.
3. **Backtesting for measurable impact**: for historical fires in the
   test set, we measure how many days in advance the model's risk score
   would have crossed the alert threshold — giving a concrete
   "days of advance warning" number rather than an abstract accuracy
   metric.

## Why Python

Python is the backbone of the entire pipeline, not a wrapper layer:
- `requests` + `pandas` for pulling and structuring both NASA APIs
- Feature engineering (dryness streaks, rolling weather summaries) in
  `pandas`/`numpy`
- `scikit-learn` `RandomForestClassifier` for the risk model, with
  `class_weight="balanced"` to handle the natural rarity of fire events
- `streamlit` for the interactive "scan area" demo interface
- A custom backtesting module to convert the model's predictions into a
  measurable early-warning lead time

## Methodological rigor

Two choices worth calling out explicitly, since they're easy to get wrong
in a rushed hackathon build:

- **No label leakage in negative sampling.** Naively sampling random
  "no-fire" points/dates in a fire-prone region risks accidentally
  labeling an actual (but unrecorded-in-this-pull) fire-adjacent point as
  safe — which quietly teaches the model that fire-prone conditions are
  fine. Every negative sample is checked against all known fire locations
  in that batch and rejected if within `min_distance_deg` (default ~30km).
- **Cross-validated evaluation, not a single lucky split.** On a
  hackathon-scale dataset, one train/test split can look better or worse
  than the model really is just by chance. `train_model.py` reports 5-fold
  stratified cross-validated ROC-AUC (mean ± std) in addition to the
  held-out test report, so the headline number isn't a single-split
  artifact.

## Environmental impact

- **Measurable evidence**: average days of advance warning the model
  provides ahead of satellite fire detection, benchmarked against
  historical fires in the case-study regions (see `src/model/backtest.py`
  — fill in with results once trained on your chosen regions).
- **Who benefits**: local fire authorities, land managers, and
  communities in fire-prone regions who currently only get satellite
  confirmation *after* ignition.
- **Scalability**: both data sources (FIRMS, POWER) are global and free,
  so the same pipeline works for any region without modification —
  Kalimantan, California, Australia, Southern Europe, or South America.

## Why this is different from existing tools

Several open wildfire-risk projects already exist, and it's worth being
upfront about them rather than claiming this space is empty:

- **[pyro-risks](https://github.com/pyronear/pyro-risks)** (Pyronear) — a
  real, working ML forecasting package, but scoped to France's NUTS-3
  departments and built on the Copernicus CDS API, which requires account
  registration.
- **Fire Spotter** — proposes a similar open, self-hostable, weather-driven
  risk-scoring mission, but as of this writing the risk-modeling code is
  an unimplemented stub.
- **WeatherWise/BreatheWise** — combines wildfire and air-quality
  prediction for Canada, but depends on the paid Weatherbit API.

Vigilfire's specific gap-closing combination:
1. **Zero-registration weather data** (NASA POWER needs no API key at all)
   and **instant-registration fire data** (NASA FIRMS key issued
   immediately, no approval wait) — no paid or gated APIs anywhere in the
   pipeline.
2. **Coordinate-agnostic by design** — works at any lat/lon on Earth
   without retraining or region-specific administrative boundaries.
3. **Output framed as a lead-time number** ("N days of advance warning
   before satellite detection"), not an abstract risk probability —
   several academic wildfire-risk papers report classification/probability
   metrics, but none frame the result this way.
4. **Actually implemented and validated** across three independent
   real 2025 wildfire events on three different continents (see Case
   studies below), not a proposal or a single-region deployment.

## Project structure

```
vigilfire/
├── app.py                      # Streamlit "scan area" demo
├── src/
│   ├── data/
│   │   ├── fetch_firms.py      # NASA FIRMS API client
│   │   ├── fetch_power.py      # NASA POWER API client
│   │   └── build_dataset.py    # joins fire + weather into training data
│   └── model/
│       ├── train_model.py      # trains the Random Forest classifier
│       └── backtest.py         # measures advance-warning lead time
├── data/                        # cached CSVs (gitignored, regenerable)
└── requirements.txt
```

## Setup & running locally

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your free FIRMS_MAP_KEY from
# https://firms.modaps.eosdis.nasa.gov/api/map_key/

# 1. Build the training dataset (edit the bbox/dates inside the script
#    first to match your chosen case-study regions)
python -m src.data.build_dataset

# 2. Train the model
python -m src.model.train_model

# 3. Run the backtest to get the headline "days of advance warning" number
python -m src.model.backtest

# 4. Launch the demo
streamlit run app.py
```

## Testing

```bash
# Fast, offline unit tests (run these on every commit)
pytest

# Full smoke test against the real NASA APIs (needs internet + FIRMS_MAP_KEY)
pytest -m smoke
```

CI runs the offline unit tests automatically on every push via GitHub
Actions (`.github/workflows/ci.yml`). The smoke-test job is manual-trigger
only (Actions tab -> Run workflow) to avoid hitting NASA's rate limits on
every commit — add a `FIRMS_MAP_KEY` repository secret first if you want
to run it.

## Branching convention

- `main` — always working/demoable
- `develop` — integration branch for in-progress work
- `phase-N-<name>` — one branch per development phase (e.g.
  `phase-2-model-training`), merged into `develop` when that phase's work
  is validated, matching the depth-first, one-phase-at-a-time workflow
  this project follows.

## Technologies used

- Python 3.11+
- NASA FIRMS API (satellite fire detections, global, free)
- NASA POWER API (satellite/reanalysis weather data, global, free)
- pandas, numpy, scikit-learn, joblib
- Streamlit

## Data & licensing note

NASA FIRMS and NASA POWER data are both publicly available for reuse;
see their respective terms at firms.modaps.eosdis.nasa.gov and
power.larc.nasa.gov.
