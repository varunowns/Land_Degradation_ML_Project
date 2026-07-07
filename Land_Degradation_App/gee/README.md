# Automated Google Earth Engine Dataset Refresh

This package reproduces the original training-data extraction workflow using **live**
Google Earth Engine data:

- Grid-based annual exports for `2020`–`2024`
- `Sentinel-2 SR` annual NDVI median composites with QA60 cloud masking
- `CHIRPS` annual rainfall totals
- `ERA5-Land` annual mean 2m temperature in °C
- `SMAP` annual mean surface soil moisture
- `ESA WorldCover` land-cover percentages

## 1. Installation

```bash
cd Land_Degradation_App
pip install -r requirements.txt
```

Required packages include `earthengine-api`, `python-dotenv`, `pandas`, `scikit-learn`, and `joblib`.

Verify Earth Engine API:

```bash
python -c "import ee; print(ee.__version__)"
```

## 2. Earth Engine Authentication

```bash
# Official Google authentication flow (opens browser)
earthengine authenticate

# Or use the project helper
python Land_Degradation_App/gee/authenticate.py
```

Credentials are stored at `~/.config/earthengine/credentials`.

## 3. Environment Variables

Copy `.env.example` to `.env` in the repository root:

```bash
copy .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GEE_PROJECT_ID` | **Yes** | GCP project ID registered with Earth Engine |
| `GEE_GRID_ASSET_ID` | **Yes** | EE asset path to 5 km grid FeatureCollection |
| `GEE_AOI_ASSET_ID` | No | Optional AOI boundary; defaults to grid geometry |

### Where to obtain values

- **GEE_PROJECT_ID:** [Google Cloud Console](https://console.cloud.google.com/) → project selector, or Earth Engine Code Editor.
- **GEE_GRID_ASSET_ID:** Upload your grid to Earth Engine Assets. Path format: `projects/YOUR_PROJECT/assets/asset_name`. Each feature needs `Grid_ID`, `District`, `Area_km2`.
- **GEE_AOI_ASSET_ID:** Optional boundary FeatureCollection for Uttar Pradesh study area.

PowerShell example:

```powershell
$env:GEE_PROJECT_ID = "your-gcp-project-id"
$env:GEE_GRID_ASSET_ID = "projects/your-gcp-project/assets/up_land_degradation_grid"
```

## 4. Running the Update

```bash
python Land_Degradation_App/gee/run_update.py
```

This executes `update_dataset()`:

```
Authenticate → Load Grid/AOI → Sentinel-2 + ancillary layers → Export
```

Outputs:

- `data/latest_dataset.csv`
- `data/metadata.json`

**Note:** Live execution requires configured assets. The legacy CSV fallback in `fetch_data.py` is only used when `GEE_GRID_ASSET_ID` is unset; operational validation requires live assets.

## 5. Running Predictions

After a successful dataset update:

```bash
python -c "from gee.run_predictions import run_predictions; run_predictions()"
```

Or as part of full validation (see below).

Outputs:

- `data/predictions.csv`
- `data/prediction_history.json`

## 6. Full Operational Validation

```bash
python Land_Degradation_App/gee/run_operational_validation.py
```

Generates:

- `results/GEE_OPERATIONAL_REPORT.md`
- `results/gee_operational_report.json`

## 7. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'ee'` | earthengine-api not installed | `pip install earthengine-api` |
| `credentials not found` | Not authenticated | `earthengine authenticate` |
| `Missing required environment variables` | `.env` not configured | Copy `.env.example` → `.env` |
| `Grid asset validation failed` | Wrong asset path or permissions | Verify asset in EE Code Editor |
| `No Sentinel-2 imagery matched` | Cloud filter too strict or date range | Check AOI and `cloud_threshold` in config |
| `Caller does not have required permission` | EE not enabled for project | Enable Earth Engine API in GCP |

## 8. Common Errors

**`ee.ee_exception.EEException: Earth Engine client library not initialized`**
→ Run authentication first, then set `GEE_PROJECT_ID`.

**Asset not found**
→ Confirm the full asset path including `projects/.../assets/...`.

**Operational validation uses legacy data**
→ Ensure `GEE_GRID_ASSET_ID` is set; do not rely on CSV fallback for production runs.

## Output Schema

`latest_dataset.csv` columns (model-compatible):

```
Grid_ID, District, Year, Area_km2, BareLand, Builtup, Cropland,
Grassland, TreeCover, Water, Wetland, Shrubland, NDVI_mean,
Rainfall_mean, Temperature_mean, SoilMoisture_mean
```
