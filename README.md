# Land Degradation Prediction using Machine Learning and Google Earth Engine

**B.Tech Major Project** | Study Area: Uttar Pradesh, India (20 districts)  
**Spatial Resolution:** 5 km × 5 km | **Temporal Coverage:** 2020–2024

**Author:** Varun Kumar Yadav  
**Description:** Add legacy phase scripts and reports

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | GEE data extraction & merging | ✓ Complete |
| 2 | EDA & data cleaning | ✓ Complete |
| 3 | LDI construction & quantile classification | ✓ Complete |
| 4 | Supervised ML classification | **This pipeline** |

Pre-built datasets (do **not** regenerate):

- `data/master_dataset.csv`
- `data/working_dataset.csv`
- `data/ldi_dataset.csv`

---

## Directory Structure

```
Land_Degradation_ML_Project/
├── data/                  # Pre-built CSV datasets
├── src/
│   ├── 01_data_loader.py
│   ├── 02_preprocessing.py
│   ├── 03_train_baseline_models.py
│   ├── 04_hyperparameter_tuning.py
│   ├── 05_model_evaluation.py
│   ├── 06_interpretability.py
│   ├── 07_visualization.py      # Error analysis (Stage 5)
│   ├── 08_export_results.py
│   └── utils.py
├── models/                # Saved joblib models & preprocessors
├── plots/                 # Evaluation, interpretability, error plots
├── reports/               # Academic report, PDF, discussion notes
└── results/               # CSV tables, predictions, export bundle
```

---

## Streamlit Monitoring App

The project includes a professional multi-page Streamlit dashboard in `Land_Degradation_App/`.

Run it with:

```bash
cd Land_Degradation_App
streamlit run app.py
```

Current app features:

- Green and earth-tone environmental AI monitoring UI with modern metric cards.
- Home System Status section showing last update, update frequency, total predictions, and latest prediction year.
- Plotly charts for district-wise degradation counts, Low / Moderate / High percentage, and latest NDVI distribution.
- Single Prediction and Batch Prediction pages for live model inference.
- EDA, SHAP, Model Performance, and Geographic Dashboard pages.
- Settings page for data update frequency and manual update requests.

Key app files:

- `config/settings.json` stores update frequency, last update timestamp, request timestamp, and pipeline status.
- `results/latest_predictions.csv` powers the Home System Status charts.
- `src/config_manager.py` reads and writes persisted settings.
- `src/update_pipeline.py` queues or triggers the future data update pipeline.

---

## Design Decisions

### Target & Leakage Prevention
- **Target:** `Degradation_Class` (Low / Moderate / High)
- **Excluded:** `LDI` (used to derive the target), `Grid_ID` (identifier)

### District Encoding
**Included via one-hot encoding (`drop_first=True`).**

| Option | Verdict |
|--------|---------|
| Exclude District | Loses spatial context across 20 agro-climatic zones |
| Label encoding | Imposes false ordinality — harmful for Logistic Regression |
| **One-hot encoding** | **Selected** — statistically correct for nominal districts; 19 dummy features is manageable for ~22 645 observations |

### Train-Test Split
Stratified 80/20 split (`random_state=42`) to preserve class balance.

---

## Running the Pipeline

Execute each stage sequentially from the project root:

```bash
# Stage 1 — Baseline models
python src/03_train_baseline_models.py

# Stage 2 — Hyperparameter tuning (top 2 models)
python src/04_hyperparameter_tuning.py

# Stage 3 — Evaluation (baseline vs tuned)
python src/05_model_evaluation.py

# Stage 4 — Interpretability (SHAP, PDP, permutation importance)
python src/06_interpretability.py

# Stage 5 — Error analysis
python src/07_visualization.py

# Stage 6 — Export all artefacts
python src/08_export_results.py
```

Or run all stages:

```bash
python run_phase4.py --stage all
python run_phase4.py --stage 1   # baseline only
```

### Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost shap
```

---

## Outputs

| Stage | Key Outputs |
|-------|-------------|
| 1 | `results/baseline_comparison.csv`, `models/baseline_*.pkl` |
| 2 | `models/tuned_*.pkl`, `results/tuned_models_summary.csv` |
| 3 | `plots/evaluation/`, `results/cv_comparison.csv` |
| 4 | `plots/interpretability/`, `results/shap_importance.csv` |
| 5 | `plots/error_analysis/`, `results/misclassified_observations.csv` |
| 6 | `results/predictions.csv`, `results/latest_predictions.csv`, `reports/comparison_report.pdf`, `reports/academic_report.txt` |

---

## Legacy Scripts

Earlier monolithic phase scripts remain at the project root for reference:

- `phase1_data_prep.py`, `phase2_eda.py`, `phase3_ldi.py`, `phase4_ml.py`

The modular `src/` pipeline supersedes `phase4_ml.py` for reproducible staged execution.
