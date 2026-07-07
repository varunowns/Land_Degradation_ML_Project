# Land Degradation Prediction System

Professional multi-page **Streamlit** application for the Land Degradation Prediction B.Tech Major Project.

## Quick Start

```bash
cd Land_Degradation_App
pip install -r requirements.txt
streamlit run app.py
```

## Application Pages

| Page | Capability |
|------|------------|
| Home | Project summary, model status, quick navigation |
| Project Overview | Methodology, dataset, feature engineering |
| **Single Prediction** | Live inference from environmental inputs |
| **Batch Prediction** | CSV upload, validation, results download |
| Model Performance | Baseline comparison, CV scores, evaluation plots |
| **EDA Dashboard** | Interactive distributions, correlations, scatter plots |
| **SHAP Explainability** | Feature importance charts & static SHAP plots |
| **Geographic Dashboard** | Choropleth maps, KPI cards, district drill-down, animation |
| About | Tech stack, GeoJSON source, disclaimer |

## Project Structure

```
Land_Degradation_App/
├── app.py
├── pages/                    # 9 multipage modules
├── utils/
│   ├── config.py             # Paths, palette, constants
│   ├── ui.py                 # Page config, CSS, sidebar
│   ├── data_loader.py        # Cached dataset loading
│   ├── model_loader.py       # Read-only joblib loading
│   ├── preprocessing.py      # Feature validation & transformation
│   ├── prediction.py         # Single & batch inference
│   ├── plotting.py           # Plotly theme & chart helpers
│   └── geography.py          # GeoJSON, choropleth, aggregation
├── assets/geo/               # Cached district boundaries
├── tests/run_app_tests.py    # Automated test suite
└── APPLICATION_TEST_REPORT.md
```

## Important Notes

- **Models are read-only** — loaded from `../models/` (parent ML project).
- **Datasets are not regenerated** — read from `../data/ldi_dataset.csv`.
- **GeoJSON source:** [Datameet Maps](https://github.com/datameet/maps) (CC BY 4.0), Census 2011 district boundaries.

## Testing

```bash
python tests/run_app_tests.py
# Expected: 9/9 passed
```

See `APPLICATION_TEST_REPORT.md` for full test details.
