# Setup & Installation

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/Land-Degradation-ML-Project.git
cd Land-Degradation-ML-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the ML Pipeline
```bash
# Baseline models
python src/03_train_baseline_models.py

# Hyperparameter tuning
python src/04_hyperparameter_tuning.py

# Full evaluation pipeline
python src/05_model_evaluation.py
```

### 3. Launch the Streamlit App
```bash
cd Land_Degradation_App
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Project Overview

**Land Degradation Prediction System** — A B.Tech Major Project combining Google Earth Engine data with machine learning to predict land degradation across Uttar Pradesh, India.

- **Study Area**: 20 districts across Uttar Pradesh
- **Spatial Resolution**: 5 km × 5 km grid
- **Temporal Coverage**: 2020–2024
- **Target**: Degradation classification (Low / Moderate / High)

## Key Components

### ML Pipeline (`src/`)
- **01_data_loader.py** — Load and validate datasets
- **02_preprocessing.py** — Feature engineering and scaling
- **03_train_baseline_models.py** — Train 5 baseline models
- **04_hyperparameter_tuning.py** — Tune top 2 models
- **05_model_evaluation.py** — Compare and evaluate
- **06_interpretability.py** — SHAP analysis
- **07_visualization.py** — Error analysis
- **08_export_results.py** — Export artifacts

### Streamlit App (`Land_Degradation_App/`)
Multi-page dashboard with:
- System Status & Monitoring
- Project Methodology Overview
- Single & Batch Predictions
- Model Performance Comparison
- EDA Dashboard
- SHAP Explainability
- Geographic Analysis
- Settings & Configuration

## Data Files

Pre-built datasets (in `data/`):
- `master_dataset.csv` — Complete dataset from GEE extraction
- `ldi_dataset.csv` — LDI-classified observations (training data)
- `working_dataset.csv` — Processed training/test split

**Do not regenerate these files** — they are frozen for reproducibility.

## Testing

Run the app test suite:
```bash
python Land_Degradation_App/tests/run_app_tests.py
```

Expected: 9/9 tests passing

## Project Structure

```
Land_Degradation_ML_Project/
├── src/                          # ML pipeline stages
│   ├── 01_data_loader.py
│   ├── 02_preprocessing.py
│   ├── 03_train_baseline_models.py
│   ├── 04_hyperparameter_tuning.py
│   ├── 05_model_evaluation.py
│   ├── 06_interpretability.py
│   ├── 07_visualization.py
│   ├── 08_export_results.py
│   └── utils.py
├── Land_Degradation_App/        # Streamlit application
│   ├── app.py
│   ├── pages/                   # 10 feature pages
│   ├── utils/                   # App utilities
│   ├── tests/
│   └── requirements.txt
├── data/                        # Pre-built datasets (frozen)
├── models/                      # Trained model artifacts
├── plots/                       # Analysis visualizations
├── reports/                     # Academic reports
├── results/                     # Predictions & metrics
├── config/                      # Settings & config
├── README.md
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE
```

## Documentation

- **README.md** — Project overview, phases, and pipeline instructions
- **Land_Degradation_App/README.md** — App-specific documentation
- **APPLICATION_TEST_REPORT.md** — Test coverage and results
- **CONTRIBUTING.md** — Contribution guidelines

## License

MIT License — See LICENSE file for details

## Questions?

Open an issue or refer to the documentation in the project repository.
