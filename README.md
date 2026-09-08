# Pyrelert — Global Wildfire Risk Early-Warning Scanner

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

## What Pyrelert does

Pyrelert scans a user-specified region and estimates near-term wildfire
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

## Project structure

```
pyrelert/
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
