#!/usr/bin/env python
"""
End-to-end operational validation for the live GEE pipeline.

Usage (from repository root)::

    python Land_Degradation_App/gee/run_operational_validation.py

Generates:
    results/GEE_OPERATIONAL_REPORT.md
    results/gee_operational_report.json
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from gee.config import CONFIG, EXPORT_CSV_PATH, METADATA_PATH  # noqa: E402
from gee.env import load_project_env, require_env_vars  # noqa: E402
from gee.validate_assets import (  # noqa: E402
    check_credentials_exist,
    collect_environment_status,
    validate_assets,
    validate_authentication,
)
from gee.validate_dataset import validate_exported_dataset  # noqa: E402
from gee.run_predictions import run_predictions, PREDICTIONS_PATH, PREDICTION_HISTORY_PATH  # noqa: E402

REPORT_JSON = PROJECT_ROOT / "results" / "gee_operational_report.json"
REPORT_MD = PROJECT_ROOT / "results" / "GEE_OPERATIONAL_REPORT.md"


def _dependency_status() -> dict[str, Any]:
    status: dict[str, Any] = {"packages": {}, "errors": []}
    for package, label in [
        ("ee", "earthengine-api"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("joblib", "joblib"),
        ("dotenv", "python-dotenv"),
    ]:
        spec = importlib.util.find_spec(package)
        installed = spec is not None
        entry: dict[str, Any] = {"installed": installed}
        if installed and package == "ee":
            import ee  # type: ignore

            entry["version"] = ee.__version__
        status["packages"][label] = entry
        if not installed and label == "earthengine-api":
            status["errors"].append("Install earthengine-api: pip install earthengine-api")
    return status


def _classify_verdict(report: dict[str, Any]) -> str:
    live_ok = report.get("live_execution", {}).get("success", False)
    auth_ok = report.get("authentication", {}).get("initialized", False)
    pred_ok = report.get("prediction", {}).get("success", False)
    env_ok = report.get("environment", {}).get("GEE_PROJECT_ID", {}).get("set") and report.get(
        "environment", {}
    ).get("GEE_GRID_ASSET_ID", {}).get("set")
    minor = report.get("warnings", [])

    if live_ok and auth_ok and pred_ok and env_ok and not report.get("errors"):
        return "Fully Operational"
    if auth_ok and env_ok and (pred_ok or live_ok):
        return "Operational with Minor Issues"
    return "Not Operational"


def _write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# GEE Operational Validation Report",
        "",
        f"**Generated:** {report['generated_at']}",
        f"**Verdict:** {report['verdict']}",
        "",
        "## Dependency Status",
        "",
    ]
    for pkg, info in report["dependencies"]["packages"].items():
        ver = f" ({info['version']})" if info.get("version") else ""
        lines.append(f"- **{pkg}**: {'installed' if info['installed'] else 'MISSING'}{ver}")

    lines.extend(["", "## Authentication Status", ""])
    auth = report["authentication"]
    lines.append(f"- Credentials file present: {auth.get('credentials_file_present')}")
    lines.append(f"- Initialized: {auth.get('initialized')}")
    if auth.get("error"):
        lines.append(f"- Error: {auth['error']}")

    lines.extend(["", "## Environment Variable Status", ""])
    for key, info in report["environment"].items():
        if key == "credentials_file_present":
            continue
        lines.append(f"- **{key}**: {'set' if info.get('set') else 'NOT SET'}")
        if info.get("value"):
            lines.append(f"  - Value: `{info['value']}`")

    lines.extend(["", "## Asset Validation", ""])
    assets = report.get("assets", {})
    if assets:
        lines.append(f"- Grid readable: {assets.get('grid_readable')}")
        lines.append(f"- Grid features: {assets.get('grid_feature_count')}")
        lines.append(f"- AOI status: {assets.get('aoi_readable', 'not run')}")
    else:
        lines.append("- Not executed (prerequisites missing)")

    lines.extend(["", "## Live Execution", ""])
    live = report.get("live_execution", {})
    lines.append(f"- Success: {live.get('success', False)}")
    if live.get("processed_grid_count"):
        lines.append(f"- Processed grid count: {live['processed_grid_count']}")
    if live.get("acquisition_date"):
        lines.append(f"- Acquisition date: {live['acquisition_date']}")
    if live.get("satellite"):
        lines.append(f"- Satellite: {live['satellite']}")
    if live.get("error"):
        lines.append(f"- Error: {live['error']}")

    lines.extend(["", "## Prediction Status", ""])
    pred = report.get("prediction", {})
    lines.append(f"- Success: {pred.get('success', False)}")
    if pred.get("total_predictions"):
        lines.append(f"- Total predictions: {pred['total_predictions']}")
        lines.append(f"- Average confidence: {pred.get('average_confidence', 0):.4f}")

    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for w in report["warnings"]:
            lines.append(f"- {w}")

    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        for e in report["errors"]:
            lines.append(f"- {e}")

    lines.extend(["", "## Files Generated", ""])
    for f in report.get("files_generated", []):
        lines.append(f"- `{f}`")

    lines.extend(["", "## How to Fix (if Not Operational)", ""])
    lines.extend([
        "1. `pip install -r Land_Degradation_App/requirements.txt`",
        "2. Copy `.env.example` to `.env` and set `GEE_PROJECT_ID`, `GEE_GRID_ASSET_ID`",
        "3. Run `earthengine authenticate`",
        "4. Run `python Land_Degradation_App/gee/run_update.py`",
        "5. Re-run this validation script",
    ])

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("gee.operational_validation")

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dependencies": {},
        "authentication": {},
        "environment": {},
        "assets": {},
        "dataset_validation": {},
        "live_execution": {"success": False},
        "prediction": {"success": False},
        "metadata": {},
        "warnings": [],
        "errors": [],
        "files_generated": [],
        "files_modified": [],
    }

    # Part 1 — Dependencies
    report["dependencies"] = _dependency_status()
    report["errors"].extend(report["dependencies"].get("errors", []))

    # Part 2 & 3 — Auth + Environment
    load_project_env()
    report["environment"] = collect_environment_status()
    report["authentication"]["credentials_file_present"] = check_credentials_exist()

    try:
        require_env_vars(("GEE_PROJECT_ID", "GEE_GRID_ASSET_ID"))
    except RuntimeError as exc:
        report["errors"].append(str(exc))
        report["verdict"] = _classify_verdict(report)
        _write_markdown(report)
        REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.error("Environment not configured: %s", exc)
        return 1

    if not check_credentials_exist():
        report["errors"].append(
            "Earth Engine credentials not found. Run: earthengine authenticate"
        )
        report["verdict"] = _classify_verdict(report)
        _write_markdown(report)
        REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return 1

    # Part 4 — Asset validation + Part 5 — Live execution
    try:
        from gee.authenticate import authenticate_gee
        from gee.fetch_data import update_dataset

        ee = authenticate_gee()
        report["authentication"].update(validate_authentication(ee))
        report["assets"] = validate_assets(ee)

        logger.info("Running live update_dataset() — this may take 30+ minutes…")
        df = update_dataset()
        report["live_execution"] = {
            "success": True,
            "processed_grid_count": int(df["Grid_ID"].nunique()),
            "total_rows": int(len(df)),
            "years": sorted(df["Year"].unique().tolist()),
        }
        report["files_generated"].extend([str(EXPORT_CSV_PATH), str(METADATA_PATH)])

        if METADATA_PATH.exists():
            with METADATA_PATH.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            report["metadata"] = meta
            report["live_execution"]["acquisition_date"] = meta.get("acquisition_date")
            report["live_execution"]["satellite"] = meta.get("satellite")
            report["live_execution"]["cloud_cover"] = meta.get("cloud_cover")

    except Exception as exc:
        report["live_execution"]["error"] = str(exc)
        report["errors"].append(f"Live GEE execution failed: {exc}")
        logger.exception("Live execution failed")

    # Part 6 — Dataset validation
    if EXPORT_CSV_PATH.exists():
        report["dataset_validation"] = validate_exported_dataset(EXPORT_CSV_PATH, METADATA_PATH)
        if not report["dataset_validation"].get("passed"):
            report["errors"].extend(report["dataset_validation"].get("errors", []))
            report["warnings"].extend(report["dataset_validation"].get("warnings", []))

    # Part 7 — Model predictions (only on latest_dataset.csv)
    if EXPORT_CSV_PATH.exists() and report["dataset_validation"].get("passed"):
        try:
            _, history = run_predictions(EXPORT_CSV_PATH)
            report["prediction"] = {"success": True, **history}
            report["files_generated"].extend([str(PREDICTIONS_PATH), str(PREDICTION_HISTORY_PATH)])
        except Exception as exc:
            report["prediction"]["error"] = str(exc)
            report["errors"].append(f"Prediction failed: {exc}")
            logger.exception("Prediction failed")

    report["verdict"] = _classify_verdict(report)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_markdown(report)

    logger.info("Operational report written to %s", REPORT_MD)
    logger.info("Verdict: %s", report["verdict"])
    return 0 if report["verdict"] == "Fully Operational" else 1


if __name__ == "__main__":
    raise SystemExit(main())
