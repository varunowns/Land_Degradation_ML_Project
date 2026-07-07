"""
Automated application test suite (non-Streamlit).

Run: python tests/run_app_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

RESULTS: list[tuple[str, str, bool, str]] = []


def record(page: str, feature: str, passed: bool, note: str = "") -> None:
    RESULTS.append((page, feature, passed, note))


def test_imports() -> None:
    try:
        import utils.config  # noqa: F401
        import utils.data_loader  # noqa: F401
        import utils.geography  # noqa: F401
        import utils.prediction  # noqa: F401
        import utils.preprocessing  # noqa: F401
        import utils.plotting  # noqa: F401
        record("Core", "Module imports", True)
    except Exception as exc:
        record("Core", "Module imports", False, str(exc))


def test_artifacts() -> None:
    from utils.model_loader import verify_artifacts_exist

    status = verify_artifacts_exist()
    ok = all(status.values())
    record("Core", "Model artefacts present", ok, str(status))


def test_single_prediction() -> None:
    from utils.data_loader import load_ldi_dataset
    from utils.prediction import predict_single
    from utils.preprocessing import REQUIRED_INPUT_COLUMNS

    row = load_ldi_dataset()[REQUIRED_INPUT_COLUMNS].iloc[0].to_dict()
    result = predict_single(row)
    ok = "degradation_class" in result and "probabilities" in result
    record("Single Prediction", "predict_single()", ok, result.get("degradation_class", ""))


def test_batch_prediction() -> None:
    from utils.data_loader import load_ldi_dataset
    from utils.prediction import predict_batch, predictions_to_csv
    from utils.preprocessing import REQUIRED_INPUT_COLUMNS

    batch = load_ldi_dataset()[REQUIRED_INPUT_COLUMNS].head(5)
    out = predict_batch(batch)
    csv_bytes = predictions_to_csv(out)
    ok = "predicted_class" in out.columns and len(csv_bytes) > 0
    record("Batch Prediction", "predict_batch() + CSV export", ok, f"{len(out)} rows")


def test_geography() -> None:
    from utils.geography import _build_study_district_geojson, aggregate_district_stats

    geo = _build_study_district_geojson()
    df = __import__("utils.data_loader", fromlist=["load_ldi_dataset"]).load_ldi_dataset()
    stats = (
        df.groupby("District")
        .agg(LDI=("LDI", "mean"), NDVI_mean=("NDVI_mean", "mean"))
        .reset_index()
    )
    ok = len(geo["features"]) >= 10 and len(stats) >= 10
    record(
        "Geographic Dashboard",
        "GeoJSON load + district aggregation",
        ok,
        f"{len(geo['features'])} polygons, {len(stats)} districts",
    )


def test_choropleth() -> None:
    from utils.geography import _build_study_district_geojson, aggregate_district_stats

    geo = _build_study_district_geojson()
    stats = aggregate_district_stats.__wrapped__(2020) if hasattr(aggregate_district_stats, "__wrapped__") else None
    if stats is None:
        df = __import__("utils.data_loader", fromlist=["load_ldi_dataset"]).load_ldi_dataset()
        df = df[df["Year"] == 2020]
        stats = df.groupby("District").agg(LDI=("LDI", "mean")).reset_index()
    stats["geo_district"] = stats["District"].map(
        __import__("utils.geography", fromlist=["dataset_to_geojson_name"]).dataset_to_geojson_name
    )
    import plotly.express as px
    fig = px.choropleth(stats, geojson=geo, locations="geo_district", featureidkey="properties.DISTRICT", color="LDI")
    ok = len(fig.data) > 0
    ok = fig.data is not None and len(fig.data) > 0
    record("Geographic Dashboard", "Choropleth figure build", ok)


def test_eda_charts() -> None:
    from utils.data_loader import load_ldi_dataset
    from utils.plotting import create_histogram, create_scatter

    df = load_ldi_dataset()
    h = create_histogram(df, "LDI")
    s = create_scatter(df.sample(500), "NDVI_mean", "LDI", color="Degradation_Class")
    ok = len(h.data) > 0 and len(s.data) > 0
    record("EDA Dashboard", "Plotly chart generation", ok)


def test_shap_data() -> None:
    from utils.config import FEATURE_IMPORTANCE_PATH, SHAP_IMPORTANCE_PATH
    from utils.plotting import create_shap_bar
    import pandas as pd

    ok = FEATURE_IMPORTANCE_PATH.exists() and SHAP_IMPORTANCE_PATH.exists()
    if ok:
        shap_df = pd.read_csv(SHAP_IMPORTANCE_PATH)
        fig = create_shap_bar(shap_df)
        ok = len(fig.data) > 0
    record("SHAP Explainability", "SHAP data + chart", ok)


def test_pages_exist() -> None:
    pages = list((APP_ROOT / "pages").glob("*.py"))
    ok = len(pages) >= 9
    record("Navigation", f"All {len(pages)} page files exist", ok)


def main() -> int:
    test_imports()
    test_artifacts()
    test_single_prediction()
    test_batch_prediction()
    test_geography()
    test_choropleth()
    test_eda_charts()
    test_shap_data()
    test_pages_exist()

    passed = sum(1 for *_, p, _ in RESULTS if p)
    total = len(RESULTS)
    print(f"\n{'='*60}\nAPPLICATION TESTS: {passed}/{total} passed\n{'='*60}")
    for page, feature, ok, note in RESULTS:
        icon = "PASS" if ok else "FAIL"
        print(f"[{icon}] {page} — {feature}" + (f" ({note})" if note else ""))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
