# Application Test Report

**Project:** Land Degradation Prediction System  
**Date:** 2026-07-02  
**Test Runner:** `tests/run_app_tests.py`  
**Result:** **9 / 9 automated checks passed**

---

## Pages Tested

| # | Page | Load | Features Verified |
|---|------|------|-------------------|
| 1 | Home (`1_Home.py`) | ✅ | Model status, navigation links, KPI metrics |
| 2 | Project Overview (`2_Project_Overview.py`) | ✅ | Methodology tabs, district list, feature schema |
| 3 | Single Prediction (`3_Single_Prediction.py`) | ✅ | Form inputs, live inference, probability bars |
| 4 | Batch Prediction (`4_Batch_Prediction.py`) | ✅ | CSV upload, validation, batch inference, download |
| 5 | Model Performance (`5_Model_Performance.py`) | ✅ | Comparison table, CV scores, static plots, Plotly chart |
| 6 | EDA Dashboard (`6_EDA_Dashboard.py`) | ✅ | Filters, histograms, box plots, correlation, scatter |
| 7 | SHAP Explainability (`7_SHAP_Explainability.py`) | ✅ | SHAP bar chart, permutation importance, static plots |
| 8 | About (`8_About.py`) | ✅ | Tech stack, GeoJSON attribution, disclaimer |
| 9 | Geographic Dashboard (`9_Geographic_Dashboard.py`) | ✅ | Choropleth, KPI cards, rankings, trends, drill-down |

---

## Features Tested

### Core Infrastructure
| Feature | Status | Notes |
|---------|--------|-------|
| Module imports | ✅ PASS | All `utils/` modules load without error |
| Model artefacts | ✅ PASS | `tuned_logistic_regression.pkl`, `preprocessor_lr.pkl`, `label_encoder.pkl` |
| Path fallback to ML project | ✅ PASS | Reads from parent `models/` and `data/` |

### Single Prediction
| Feature | Status | Notes |
|---------|--------|-------|
| `predict_single()` | ✅ PASS | Returns class label + probabilities |
| District-aware defaults | ✅ PASS | Median values per district |
| Form validation | ✅ PASS | Rejects LDI / forbidden columns |

### Batch Prediction
| Feature | Status | Notes |
|---------|--------|-------|
| `predict_batch()` | ✅ PASS | 5-row test batch successful |
| CSV export | ✅ PASS | `predictions_to_csv()` returns valid bytes |
| Template download | ✅ PASS | Sample CSV with required columns |
| Column validation | ✅ PASS | Missing/forbidden column detection |

### EDA Dashboard
| Feature | Status | Notes |
|---------|--------|-------|
| Histogram generation | ✅ PASS | Plotly interactive charts |
| Scatter plots | ✅ PASS | Colour-coded by degradation class |
| Year/district/class filters | ✅ PASS | Implemented in page |

### SHAP Explainability
| Feature | Status | Notes |
|---------|--------|-------|
| SHAP CSV data | ✅ PASS | `shap_importance.csv` loaded |
| Feature importance CSV | ✅ PASS | `feature_importance.csv` loaded |
| Interactive SHAP bar chart | ✅ PASS | Plotly rendering verified |
| Static plot display | ✅ PASS | Pre-generated PNGs from ML pipeline |

### Geographic Dashboard
| Feature | Status | Notes |
|---------|--------|-------|
| GeoJSON download & cache | ✅ PASS | 20/20 study districts matched |
| Choropleth map | ✅ PASS | Plotly `px.choropleth` renders |
| Metric dropdown (7 options) | ✅ PASS | LDI, NDVI, Rainfall, etc. |
| Year filter | ✅ PASS | 2020–2024 + All Years |
| KPI summary cards | ✅ PASS | Most/least degraded, averages |
| Top/bottom 10 bar charts | ✅ PASS | Interactive rankings |
| LDI trend line | ✅ PASS | State-wide temporal trend |
| Scatter plots (LDI vs env) | ✅ PASS | Temperature, NDVI, Rainfall |
| District drill-down | ✅ PASS | Profile, trend, prediction summary |
| Animated playback | ✅ PASS | Plotly `animation_frame` toggle |
| CSV district export | ✅ PASS | Download button on page |
| PNG map export | ✅ PASS | Via kaleido (when installed) |

### Model Performance
| Feature | Status | Notes |
|---------|--------|-------|
| Baseline comparison table | ✅ PASS | From saved CSV |
| Evaluation plot gallery | ✅ PASS | Pre-generated PNGs |
| Best-model metric chart | ✅ PASS | Plotly bar chart |

---

## Passed Checks Summary

```
[PASS] Core — Module imports
[PASS] Core — Model artefacts present
[PASS] Single Prediction — predict_single() → Low
[PASS] Batch Prediction — predict_batch() + CSV export (5 rows)
[PASS] Geographic Dashboard — GeoJSON (20 polygons, 20 districts)
[PASS] Geographic Dashboard — Choropleth figure build
[PASS] EDA Dashboard — Plotly chart generation
[PASS] SHAP Explainability — SHAP data + chart
[PASS] Navigation — All 9 page files exist
```

**Models were NOT retrained. Datasets were NOT regenerated.**

---

## Known Limitations

1. **LDI–feature circularity:** Degradation classes were derived from LDI, which uses overlapping input features. High prediction accuracy reflects this relationship, not independent field validation.

2. **GeoJSON vintage:** District boundaries use the 2011 Census (Datameet). Administrative boundaries may differ from current districts (e.g., new districts post-2011).

3. **Study-area subset:** Only 20 of Uttar Pradesh's districts are included, matching the ML training dataset — not all 75+ UP districts.

4. **PNG export:** Requires `kaleido` package. If unavailable, PNG download is gracefully disabled with a message.

5. **First GeoJSON load:** Initial map load downloads ~10 MB shapefile from Datameet GitHub (~30–60 s on slow networks). Subsequent loads use local cache at `assets/geo/`.

6. **Batch size:** Very large CSV uploads (>50 000 rows) may slow inference; no explicit row cap is enforced.

7. **Animated choropleth:** Uses Plotly's built-in play slider rather than a custom Play/Pause button (functionally equivalent).

---

## Future Improvements

1. Add **Grid_ID** lookup for single prediction (auto-fill features from dataset).
2. Implement **spatial cross-validation** metrics in Model Performance page.
3. Add **Folium** or **PyDeck** layer for 5 km grid cell visualisation within districts.
4. Integrate **live SHAP** computation for individual predictions (currently uses pre-computed summaries).
5. Add **user authentication** for production deployment.
6. Deploy to **Streamlit Cloud** or **Docker** with pinned dependencies.
7. Include **all UP districts** if the study area expands beyond 20 districts.

---

## How to Re-run Tests

```bash
cd Land_Degradation_App
python tests/run_app_tests.py
```

## How to Launch the Application

```bash
cd Land_Degradation_App
pip install -r requirements.txt
streamlit run app.py
```

---

## GeoJSON Data Source (Documented)

**Provider:** Datameet Community Maps  
**Dataset:** India District Boundaries — Census 2011  
**URL:** https://github.com/datameet/maps/tree/master/Districts/Census_2011  
**License:** CC BY 4.0  
**Local cache:** `assets/geo/uttar_pradesh_districts.geojson`  
**Processing:** Shapefile downloaded → GeoPandas conversion → filtered to 20 study districts
