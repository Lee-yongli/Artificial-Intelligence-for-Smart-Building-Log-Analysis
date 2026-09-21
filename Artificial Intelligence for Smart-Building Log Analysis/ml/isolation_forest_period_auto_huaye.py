from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


PROJECT_DIR = Path(__file__).resolve().parents[1]
ML_READY_DIR = PROJECT_DIR / "outputs" / "floor6_zone1245_ml_ready"
ML_READY_PATH = ML_READY_DIR / "floor6_zone1245_ml_ready_2018_jul01_sep01.csv.gz"
FEATURE_MANIFEST_PATH = ML_READY_DIR / "feature_list.json"
PHYSICAL_REFERENCE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "floor6_zone1245_2018_jul01_sep01_total_ac_eda"
    / "floor6_zone1245_baseline_ready_2018_jul01_sep01.csv.gz"
)
LOF_RESULTS_PATH = (
    PROJECT_DIR
    / "outputs"
    / "floor6_zone1245_lof_modelling_gavin"
    / "lof_anomaly_results_full.csv.gz"
)
OUTPUT_DIR = PROJECT_DIR / "outputs" / "isolation_forest_huaye"

KEY_COLUMNS = ["timestamp", "floor", "zone"]
PERIOD_ORDER = ["night_off", "ramp_up", "core_work", "ramp_down", "evening_off", "weekend"]
RAW_SENSOR_COLUMNS = [
    "total_AC_power_kw",
    "light_power_kw",
    "plug_power_kw",
    "temperature_c",
    "humidity_pct",
    "lux",
]
ROLLING_COLUMNS = [f"{column}_rolling_mean_15min" for column in RAW_SENSOR_COLUMNS]
MODEL_FEATURES = [f"numerical__{column}" for column in ROLLING_COLUMNS]

MODEL_PARAMETERS: dict[str, Any] = {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}

OUTPUT_FILES = {
    "full_results": "if_period_auto_results_full.csv.gz",
    "anomalies": "if_period_auto_anomalies.csv.gz",
    "period_summary": "if_period_auto_summary.csv",
    "zone_summary": "if_period_auto_summary_by_zone.csv",
    "hour_summary": "if_period_auto_summary_by_hour.csv",
    "comparison": "if_period_auto_comparison.csv",
    "modelling_summary": "if_period_auto_modelling_summary.json",
}


def _assert_unique_keys(frame: pd.DataFrame, name: str) -> None:
    duplicate_count = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        raise AssertionError(f"{name} has {duplicate_count:,} duplicate identifier rows")


def _jaccard(left: pd.Series, right: pd.Series) -> tuple[int, int, float]:
    left_mask = left.astype(bool)
    right_mask = right.astype(bool)
    overlap = int((left_mask & right_mask).sum())
    union = int((left_mask | right_mask).sum())
    return overlap, union, float(overlap / union) if union else 0.0


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    required_paths = [ML_READY_PATH, FEATURE_MANIFEST_PATH, PHYSICAL_REFERENCE_PATH, LOF_RESULTS_PATH]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {[str(path) for path in missing]}")

    with FEATURE_MANIFEST_PATH.open(encoding="utf-8") as handle:
        feature_manifest = json.load(handle)

    manifest_features = feature_manifest["model_feature_columns"]
    missing_features = [feature for feature in MODEL_FEATURES if feature not in manifest_features]
    if missing_features:
        raise AssertionError(f"Physical features missing from shared feature manifest: {missing_features}")

    ml_usecols = (
        KEY_COLUMNS
        + ["hour", "operating_period", "baseline_anomaly_label", "baseline_anomaly_count"]
        + RAW_SENSOR_COLUMNS
        + MODEL_FEATURES
    )
    data = pd.read_csv(ML_READY_PATH, usecols=ml_usecols, parse_dates=["timestamp"])
    _assert_unique_keys(data, "ML-ready dataset")

    reference = pd.read_csv(
        PHYSICAL_REFERENCE_PATH,
        usecols=KEY_COLUMNS + ROLLING_COLUMNS,
        parse_dates=["timestamp"],
    )
    _assert_unique_keys(reference, "physical reference dataset")
    data = data.merge(reference, on=KEY_COLUMNS, how="left", validate="one_to_one", sort=False)
    if data[ROLLING_COLUMNS].isna().any().any():
        raise AssertionError("Unscaled rolling values failed to align with the ML-ready dataset")

    lof = pd.read_csv(
        LOF_RESULTS_PATH,
        usecols=KEY_COLUMNS + ["lof_anomaly_label", "lof_anomaly_score"],
        parse_dates=["timestamp"],
    )
    _assert_unique_keys(lof, "LOF results")
    data = data.merge(lof, on=KEY_COLUMNS, how="left", validate="one_to_one", sort=False)
    if data[["lof_anomaly_label", "lof_anomaly_score"]].isna().any().any():
        raise AssertionError("LOF results failed to align with the ML-ready dataset")

    if data[MODEL_FEATURES].isna().any().any():
        raise AssertionError("Model features contain missing values")
    if not np.isfinite(data[MODEL_FEATURES].to_numpy()).all():
        raise AssertionError("Model features contain non-finite values")
    if set(data["operating_period"].unique()) != set(PERIOD_ORDER):
        raise AssertionError(f"Unexpected operating periods: {sorted(data['operating_period'].unique())}")

    return data, feature_manifest


def fit_period_models(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, IsolationForest]]:
    results = data.copy()
    results["period_if_anomaly_score"] = np.nan
    results["period_if_decision_function"] = np.nan
    results["period_if_anomaly_label"] = np.int8(0)
    models: dict[str, IsolationForest] = {}

    for period in PERIOD_ORDER:
        period_mask = results["operating_period"].eq(period)
        period_index = results.index[period_mask]
        period_features = results.loc[period_index, MODEL_FEATURES]
        if period_features.empty:
            raise AssertionError(f"No rows found for operating period: {period}")

        model = IsolationForest(**MODEL_PARAMETERS)
        model.fit(period_features)
        anomaly_score = -model.score_samples(period_features)
        decision_function = model.decision_function(period_features)
        anomaly_label = (model.predict(period_features) == -1).astype(np.int8)

        if not np.isclose(model.offset_, -0.5):
            raise AssertionError(f"Unexpected auto-contamination offset for {period}: {model.offset_}")
        if not np.array_equal(anomaly_label, decision_function < 0):
            raise AssertionError(f"Prediction and decision threshold disagree for {period}")
        if not np.allclose(decision_function, 0.5 - anomaly_score):
            raise AssertionError(f"Score direction validation failed for {period}")

        results.loc[period_index, "period_if_anomaly_score"] = anomaly_score
        results.loc[period_index, "period_if_decision_function"] = decision_function
        results.loc[period_index, "period_if_anomaly_label"] = anomaly_label
        models[period] = model

    results["period_if_anomaly_label"] = results["period_if_anomaly_label"].astype(np.int8)
    results["period_if_score_percentile"] = (
        results.groupby("operating_period", observed=True)["period_if_anomaly_score"]
        .rank(method="average", pct=True)
        .astype(float)
    )
    results["period_if_rank_within_period"] = (
        results.groupby("operating_period", observed=True)["period_if_anomaly_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    results["if_lof_overlap"] = (
        results["period_if_anomaly_label"].eq(1) & results["lof_anomaly_label"].eq(1)
    ).astype(np.int8)
    results["if_baseline_overlap"] = (
        results["period_if_anomaly_label"].eq(1) & results["baseline_anomaly_label"].eq(1)
    ).astype(np.int8)

    if results[["period_if_anomaly_score", "period_if_decision_function"]].isna().any().any():
        raise AssertionError("Period-specific model left missing scores")
    return results, models


def build_summaries(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    period_summary = (
        results.groupby("operating_period", observed=True)
        .agg(
            rows=("period_if_anomaly_label", "size"),
            anomaly_count=("period_if_anomaly_label", "sum"),
            mean_anomaly_score=("period_if_anomaly_score", "mean"),
            min_decision_function=("period_if_decision_function", "min"),
            max_anomaly_score=("period_if_anomaly_score", "max"),
            lof_anomaly_count=("lof_anomaly_label", "sum"),
            baseline_anomaly_count=("baseline_anomaly_label", "sum"),
            lof_overlap=("if_lof_overlap", "sum"),
            baseline_overlap=("if_baseline_overlap", "sum"),
        )
        .reindex(PERIOD_ORDER)
        .reset_index()
    )
    period_summary["anomaly_rate_pct"] = 100 * period_summary["anomaly_count"] / period_summary["rows"]

    zone_summary = (
        results.groupby("zone", observed=True)
        .agg(rows=("period_if_anomaly_label", "size"), anomaly_count=("period_if_anomaly_label", "sum"))
        .reset_index()
    )
    zone_summary["anomaly_rate_pct"] = 100 * zone_summary["anomaly_count"] / zone_summary["rows"]

    hour_summary = (
        results.groupby("hour", observed=True)
        .agg(rows=("period_if_anomaly_label", "size"), anomaly_count=("period_if_anomaly_label", "sum"))
        .reset_index()
    )
    hour_summary["anomaly_rate_pct"] = 100 * hour_summary["anomaly_count"] / hour_summary["rows"]

    comparisons: list[dict[str, Any]] = []
    for name, column, note in [
        ("lof_k100_c050", "lof_anomaly_label", "LOF uses its existing fixed contamination 0.05 label"),
        ("statistical_baseline_v3", "baseline_anomaly_label", "Statistical baseline is a reference, not ground truth"),
    ]:
        overlap, union, jaccard = _jaccard(results["period_if_anomaly_label"], results[column])
        comparisons.append(
            {
                "comparison": name,
                "if_anomaly_count": int(results["period_if_anomaly_label"].sum()),
                "comparison_anomaly_count": int(results[column].sum()),
                "overlap_count": overlap,
                "union_count": union,
                "jaccard_overlap": jaccard,
                "note": note,
            }
        )

    return {
        "period_summary": period_summary,
        "zone_summary": zone_summary,
        "hour_summary": hour_summary,
        "comparison": pd.DataFrame(comparisons),
    }


def validate_results(
    source: pd.DataFrame,
    results: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    models: dict[str, IsolationForest],
) -> list[dict[str, Any]]:
    checks: list[tuple[str, bool, str]] = [
        ("row_count_preserved", len(results) == len(source), f"rows={len(results):,}"),
        ("expected_row_count", len(results) == 362_864, f"rows={len(results):,}"),
        ("unique_row_key", not results.duplicated(KEY_COLUMNS).any(), "timestamp + floor + zone"),
        ("six_period_models", set(models) == set(PERIOD_ORDER), f"models={sorted(models)}"),
        ("six_physical_features", len(MODEL_FEATURES) == 6, f"features={MODEL_FEATURES}"),
        ("scores_complete", not results["period_if_anomaly_score"].isna().any(), "missing=0"),
        (
            "decision_threshold_consistent",
            np.array_equal(
                results["period_if_anomaly_label"].to_numpy(dtype=bool),
                results["period_if_decision_function"].lt(0).to_numpy(),
            ),
            "label=1 exactly when decision_function < 0",
        ),
        (
            "not_fixed_five_percent",
            summaries["period_summary"]["anomaly_rate_pct"].round(6).nunique() > 1,
            str(summaries["period_summary"].set_index("operating_period")["anomaly_rate_pct"].round(6).to_dict()),
        ),
        (
            "reference_labels_excluded",
            "baseline_anomaly_label" not in MODEL_FEATURES and "lof_anomaly_label" not in MODEL_FEATURES,
            "baseline and LOF labels are comparison-only",
        ),
        (
            "all_anomaly_rows_exportable",
            int(results["period_if_anomaly_label"].sum()) > 0,
            f"anomalies={int(results['period_if_anomaly_label'].sum()):,}",
        ),
    ]
    failed = [name for name, passed, _ in checks if not passed]
    if failed:
        raise AssertionError(f"Validation failures: {failed}")
    return [{"check": name, "passed": passed, "evidence": evidence} for name, passed, evidence in checks]


def save_outputs(
    results: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    validations: list[dict[str, Any]],
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_path = OUTPUT_DIR / OUTPUT_FILES["full_results"]
    anomaly_path = OUTPUT_DIR / OUTPUT_FILES["anomalies"]

    results.to_csv(full_path, index=False, compression="gzip")
    anomalies = (
        results.loc[results["period_if_anomaly_label"].eq(1)]
        .sort_values(["operating_period", "period_if_anomaly_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    anomalies.to_csv(anomaly_path, index=False, compression="gzip")
    summaries["period_summary"].to_csv(OUTPUT_DIR / OUTPUT_FILES["period_summary"], index=False)
    summaries["zone_summary"].to_csv(OUTPUT_DIR / OUTPUT_FILES["zone_summary"], index=False)
    summaries["hour_summary"].to_csv(OUTPUT_DIR / OUTPUT_FILES["hour_summary"], index=False)
    summaries["comparison"].to_csv(OUTPUT_DIR / OUTPUT_FILES["comparison"], index=False)

    summary_payload: dict[str, Any] = {
        "author": "huaye",
        "analysis": "period-specific Isolation Forest with automatic threshold",
        "source_dataset": str(ML_READY_PATH.relative_to(PROJECT_DIR)),
        "physical_reference_dataset": str(PHYSICAL_REFERENCE_PATH.relative_to(PROJECT_DIR)),
        "feature_manifest": str(FEATURE_MANIFEST_PATH.relative_to(PROJECT_DIR)),
        "row_count": int(len(results)),
        "operating_periods": PERIOD_ORDER,
        "model_count": len(PERIOD_ORDER),
        "model_features": MODEL_FEATURES,
        "feature_count": len(MODEL_FEATURES),
        "model": "sklearn.ensemble.IsolationForest",
        "model_parameters": MODEL_PARAMETERS,
        "label_definition": "period_if_anomaly_label = 1 when model.predict(X_period) == -1, equivalently decision_function < 0",
        "score_definition": "period_if_anomaly_score = -model.score_samples(X_period); higher means more anomalous within the same period",
        "threshold_definition": "contamination='auto' uses sklearn offset -0.5; no fixed anomaly percentage is imposed",
        "anomaly_count": int(results["period_if_anomaly_label"].sum()),
        "anomaly_rate_pct": float(100 * results["period_if_anomaly_label"].mean()),
        "period_results": summaries["period_summary"].to_dict(orient="records"),
        "comparison_limitations": [
            "LOF retains its existing fixed contamination 0.05 label.",
            "The statistical baseline is a comparison reference, not ground truth.",
            "Detected rows are candidate anomalies, not confirmed equipment faults.",
            "Raw scores from separately fitted period models should not be compared directly across periods.",
        ],
        "output_files": {key: str((OUTPUT_DIR / value).relative_to(PROJECT_DIR)) for key, value in OUTPUT_FILES.items()},
        "validation": validations,
    }
    with (OUTPUT_DIR / OUTPUT_FILES["modelling_summary"]).open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    reloaded = pd.read_csv(full_path, usecols=KEY_COLUMNS + ["period_if_anomaly_label"])
    if len(reloaded) != len(results) or reloaded.duplicated(KEY_COLUMNS).any():
        raise AssertionError("Saved full results failed reload validation")
    reloaded_anomalies = pd.read_csv(anomaly_path, usecols=KEY_COLUMNS + ["period_if_anomaly_label"])
    if len(reloaded_anomalies) != int(results["period_if_anomaly_label"].sum()):
        raise AssertionError("Saved anomaly-only output has an unexpected row count")

    return summary_payload


def run_pipeline() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    source, _ = load_inputs()
    results, models = fit_period_models(source)
    summaries = build_summaries(results)
    validations = validate_results(source, results, summaries, models)
    summary_payload = save_outputs(results, summaries, validations)
    return results, summaries, summary_payload


if __name__ == "__main__":
    _, generated_summaries, generated_summary = run_pipeline()
    print("Period-specific automatic-threshold Isolation Forest complete")
    print(generated_summaries["period_summary"].to_string(index=False))
    print(
        f"Total candidate anomalies: {generated_summary['anomaly_count']:,} "
        f"({generated_summary['anomaly_rate_pct']:.4f}%)"
    )
    print("Generated outputs:")
    for output_name in OUTPUT_FILES.values():
        print(f"- {OUTPUT_DIR / output_name}")
