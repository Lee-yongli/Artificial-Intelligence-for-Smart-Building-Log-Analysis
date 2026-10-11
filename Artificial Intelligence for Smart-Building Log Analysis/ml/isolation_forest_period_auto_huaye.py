from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def ensure_runtime_dependencies() -> None:
    """Relaunch with the project virtual environment if the current Python lacks ML packages."""
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
        return
    except ModuleNotFoundError as error:
        missing_module = error.name

    if "__file__" not in globals():
        raise ModuleNotFoundError(
            f"Missing required module '{missing_module}' in {sys.executable}. "
            "Use the project .venv or Anaconda Python interpreter."
        ) from None

    if os.environ.get("SMART_BUILDING_IF_RELAUNCHED") == "1":
        raise ModuleNotFoundError(
            f"Missing required module '{missing_module}' in {sys.executable}. "
            "Install numpy, pandas, and scikit-learn in this interpreter."
        ) from None

    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[1]
    current_executable = Path(sys.executable).resolve()
    candidate_executables = [
        project_dir / ".venv" / "Scripts" / "python.exe",
        Path.home() / "anaconda3" / "python.exe",
        Path.home() / "miniconda3" / "python.exe",
    ]

    for executable in candidate_executables:
        if not executable.is_file() or executable.resolve() == current_executable:
            continue
        check = subprocess.run(
            [str(executable), "-c", "import numpy, pandas, sklearn"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if check.returncode == 0:
            env = os.environ.copy()
            env["SMART_BUILDING_IF_RELAUNCHED"] = "1"
            print(
                f"Current Python is missing '{missing_module}': {sys.executable}\n"
                f"Relaunching with: {executable}",
                file=sys.stderr,
            )
            raise SystemExit(subprocess.call([str(executable), str(script_path), *sys.argv[1:]], env=env))

    raise ModuleNotFoundError(
        f"Missing required module '{missing_module}' in {sys.executable}. "
        "No project .venv or conda Python with numpy, pandas, and scikit-learn was found."
    ) from None


ensure_runtime_dependencies()

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def resolve_project_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]

    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd / "Artificial Intelligence for Smart-Building Log Analysis", *cwd.parents]
    for candidate in candidates:
        if (candidate / "outputs").exists() and (candidate / "ml").exists():
            return candidate
    raise FileNotFoundError("Cannot locate project root. Run this script from the project folder.")


PROJECT_DIR = resolve_project_dir()
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
FALLBACK_BASELINE_REFERENCE_COLUMNS = ["baseline_anomaly_label", "baseline_anomaly_count"]

MODEL_PARAMETERS: dict[str, Any] = {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}
IF_CALIBRATED_ANOMALY_RATE = 0.05
IF_EVENT_MIN_DURATION_MINUTES = 30
IF_EVENT_MAX_GAP_MINUTES = 2

OUTPUT_FILES = {
    "full_results": "if_period_auto_results_full.csv.gz",
    "anomalies": "if_period_auto_anomalies.csv.gz",
    "events": "if_period_auto_events.csv",
    "period_summary": "if_period_auto_summary.csv",
    "event_summary": "if_period_auto_event_summary.csv",
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


def _baseline_reference_columns(feature_manifest: dict[str, Any]) -> list[str]:
    columns = feature_manifest.get("baseline_reference_columns", FALLBACK_BASELINE_REFERENCE_COLUMNS)
    return [column for column in columns if isinstance(column, str)]


def _baseline_method_label_columns(feature_manifest: dict[str, Any]) -> dict[str, str]:
    columns = feature_manifest.get("baseline_method_label_columns", {})
    if not isinstance(columns, dict):
        return {}
    return {method: column for method, column in columns.items() if isinstance(column, str)}


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

    baseline_reference_columns = _baseline_reference_columns(feature_manifest)
    baseline_method_label_columns = _baseline_method_label_columns(feature_manifest)
    ml_header = pd.read_csv(ML_READY_PATH, nrows=0).columns.tolist()
    missing_reference_columns = sorted(set(baseline_reference_columns).difference(ml_header))
    if missing_reference_columns:
        raise AssertionError(f"Baseline reference columns missing from ML-ready dataset: {missing_reference_columns}")

    ml_usecols = (
        KEY_COLUMNS
        + ["hour", "operating_period"]
        + baseline_reference_columns
        + RAW_SENSOR_COLUMNS
        + MODEL_FEATURES
    )
    data = pd.read_csv(ML_READY_PATH, usecols=ml_usecols, parse_dates=["timestamp"])
    _assert_unique_keys(data, "ML-ready dataset")
    for method, column in baseline_method_label_columns.items():
        if column not in data.columns:
            raise AssertionError(f"Baseline method label column for {method} is missing: {column}")

    reference = pd.read_csv(
        PHYSICAL_REFERENCE_PATH,
        usecols=KEY_COLUMNS + ROLLING_COLUMNS,
        parse_dates=["timestamp"],
    )
    _assert_unique_keys(reference, "physical reference dataset")
    data = data.merge(reference, on=KEY_COLUMNS, how="left", validate="one_to_one", sort=False)
    data["rolling_reference_missing_count"] = data[ROLLING_COLUMNS].isna().sum(axis=1).astype(np.int8)

    lof = pd.read_csv(
        LOF_RESULTS_PATH,
        usecols=KEY_COLUMNS + ["lof_anomaly_label", "lof_anomaly_score"],
        parse_dates=["timestamp"],
    )
    _assert_unique_keys(lof, "LOF results")
    data = data.merge(lof, on=KEY_COLUMNS, how="left", validate="one_to_one", sort=False)
    data["lof_reference_missing"] = data["lof_anomaly_label"].isna().astype(np.int8)
    data["lof_anomaly_label"] = data["lof_anomaly_label"].fillna(0).astype(np.int8)

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
    results["period_if_auto_anomaly_label"] = np.int8(0)
    results["period_if_anomaly_label"] = np.int8(0)
    results["period_if_threshold_score"] = np.nan
    results["period_if_calibrated_rate"] = IF_CALIBRATED_ANOMALY_RATE
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
        auto_anomaly_label = (model.predict(period_features) == -1).astype(np.int8)

        calibrated_count = max(1, int(np.ceil(len(anomaly_score) * IF_CALIBRATED_ANOMALY_RATE)))
        ranked_positions = np.argsort(anomaly_score)[::-1]
        calibrated_positions = ranked_positions[:calibrated_count]
        calibrated_label = np.zeros(len(anomaly_score), dtype=np.int8)
        calibrated_label[calibrated_positions] = 1
        threshold_score = float(anomaly_score[calibrated_positions].min())

        if not np.isclose(model.offset_, -0.5):
            raise AssertionError(f"Unexpected auto-contamination offset for {period}: {model.offset_}")
        if not np.array_equal(auto_anomaly_label, decision_function < 0):
            raise AssertionError(f"Prediction and decision threshold disagree for {period}")
        if not np.allclose(decision_function, 0.5 - anomaly_score):
            raise AssertionError(f"Score direction validation failed for {period}")

        results.loc[period_index, "period_if_anomaly_score"] = anomaly_score
        results.loc[period_index, "period_if_decision_function"] = decision_function
        results.loc[period_index, "period_if_auto_anomaly_label"] = auto_anomaly_label
        results.loc[period_index, "period_if_anomaly_label"] = calibrated_label
        results.loc[period_index, "period_if_threshold_score"] = threshold_score
        models[period] = model

    results["period_if_auto_anomaly_label"] = results["period_if_auto_anomaly_label"].astype(np.int8)
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
    baseline_method_label_columns = [
        column
        for column in results.columns
        if column.startswith("baseline_")
        and column.endswith("_label")
        and column != "baseline_anomaly_label"
    ]
    for column in baseline_method_label_columns:
        overlap_column = f"if_{column.removesuffix('_label')}_overlap"
        results[overlap_column] = (
            results["period_if_anomaly_label"].eq(1) & results[column].eq(1)
        ).astype(np.int8)

    if results[["period_if_anomaly_score", "period_if_decision_function"]].isna().any().any():
        raise AssertionError("Period-specific model left missing scores")
    return results, models


def build_if_events(results: pd.DataFrame) -> pd.DataFrame:
    anomaly_rows = (
        results.loc[results["period_if_anomaly_label"].eq(1)]
        .sort_values(["zone", "operating_period", "timestamp"])
        .copy()
    )
    event_columns = [
        "if_event_id",
        "zone",
        "operating_period",
        "start_time",
        "end_time",
        "duration_minutes",
        "anomaly_points",
        "required_duration_minutes",
        "max_gap_minutes",
        "mean_anomaly_score",
        "max_anomaly_score",
        "min_decision_function",
        "lof_overlap_points",
        "baseline_overlap_points",
        "baseline_overlap_pct",
    ]
    if anomaly_rows.empty:
        return pd.DataFrame(columns=event_columns)

    group_columns = ["zone", "operating_period"]
    gap_minutes = (
        anomaly_rows.groupby(group_columns, observed=True)["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )
    new_event = gap_minutes.isna() | gap_minutes.gt(IF_EVENT_MAX_GAP_MINUTES)
    anomaly_rows["_if_event_group"] = new_event.groupby(
        [anomaly_rows["zone"], anomaly_rows["operating_period"]], observed=True
    ).cumsum()

    events = (
        anomaly_rows.groupby(group_columns + ["_if_event_group"], observed=True)
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            anomaly_points=("period_if_anomaly_label", "sum"),
            mean_anomaly_score=("period_if_anomaly_score", "mean"),
            max_anomaly_score=("period_if_anomaly_score", "max"),
            min_decision_function=("period_if_decision_function", "min"),
            lof_overlap_points=("if_lof_overlap", "sum"),
            baseline_overlap_points=("if_baseline_overlap", "sum"),
        )
        .reset_index()
        .drop(columns="_if_event_group")
    )
    events["duration_minutes"] = (
        (events["end_time"] - events["start_time"]).dt.total_seconds().div(60).astype(int) + 1
    )
    events = events.loc[events["duration_minutes"].ge(IF_EVENT_MIN_DURATION_MINUTES)].copy()
    events["required_duration_minutes"] = IF_EVENT_MIN_DURATION_MINUTES
    events["max_gap_minutes"] = IF_EVENT_MAX_GAP_MINUTES
    events["baseline_overlap_pct"] = (
        100 * events["baseline_overlap_points"] / events["anomaly_points"].replace(0, np.nan)
    ).fillna(0)
    events = events.sort_values(
        ["max_anomaly_score", "duration_minutes", "anomaly_points"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    events.insert(0, "if_event_id", np.arange(1, len(events) + 1, dtype=int))
    return events[event_columns]


def build_summaries(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline_method_label_columns = [
        column
        for column in results.columns
        if column.startswith("baseline_")
        and column.endswith("_label")
        and column != "baseline_anomaly_label"
    ]
    method_period_aggs: dict[str, tuple[str, str]] = {}
    for column in baseline_method_label_columns:
        overlap_column = f"if_{column.removesuffix('_label')}_overlap"
        method_period_aggs[f"{column}_count"] = (column, "sum")
        if overlap_column in results.columns:
            method_period_aggs[f"{overlap_column}_count"] = (overlap_column, "sum")

    events = build_if_events(results)

    period_summary = (
        results.groupby("operating_period", observed=True)
        .agg(
            rows=("period_if_anomaly_label", "size"),
            anomaly_count=("period_if_anomaly_label", "sum"),
            auto_anomaly_count=("period_if_auto_anomaly_label", "sum"),
            calibrated_threshold=("period_if_threshold_score", "first"),
            mean_anomaly_score=("period_if_anomaly_score", "mean"),
            min_decision_function=("period_if_decision_function", "min"),
            max_anomaly_score=("period_if_anomaly_score", "max"),
            lof_anomaly_count=("lof_anomaly_label", "sum"),
            baseline_anomaly_count=("baseline_anomaly_label", "sum"),
            lof_overlap=("if_lof_overlap", "sum"),
            baseline_overlap=("if_baseline_overlap", "sum"),
            **method_period_aggs,
        )
        .reindex(PERIOD_ORDER)
        .reset_index()
    )
    period_summary["anomaly_rate_pct"] = 100 * period_summary["anomaly_count"] / period_summary["rows"]
    period_summary["auto_anomaly_rate_pct"] = 100 * period_summary["auto_anomaly_count"] / period_summary["rows"]
    if events.empty:
        period_summary["if_event_count"] = 0
    else:
        event_counts = events.groupby("operating_period", observed=True).size()
        period_summary["if_event_count"] = (
            period_summary["operating_period"].map(event_counts).fillna(0).astype(int)
        )

    point_event_context = period_summary[
        ["operating_period", "anomaly_count", "anomaly_rate_pct"]
    ].rename(
        columns={
            "anomaly_count": "point_anomaly_count",
            "anomaly_rate_pct": "point_anomaly_rate_pct",
        }
    )
    if events.empty:
        event_summary = pd.DataFrame({"operating_period": PERIOD_ORDER})
    else:
        event_summary = (
            events.groupby("operating_period", observed=True)
            .agg(
                event_count=("if_event_id", "size"),
                persistent_event_points=("anomaly_points", "sum"),
                mean_duration_minutes=("duration_minutes", "mean"),
                max_duration_minutes=("duration_minutes", "max"),
                mean_anomaly_points=("anomaly_points", "mean"),
                max_anomaly_score=("max_anomaly_score", "max"),
                baseline_overlap_points=("baseline_overlap_points", "sum"),
                lof_overlap_points=("lof_overlap_points", "sum"),
            )
            .reindex(PERIOD_ORDER)
            .reset_index()
        )
    event_summary = point_event_context.merge(event_summary, on="operating_period", how="left")
    zero_when_no_event_columns = [
        "event_count",
        "persistent_event_points",
        "mean_duration_minutes",
        "max_duration_minutes",
        "mean_anomaly_points",
        "max_anomaly_score",
        "baseline_overlap_points",
        "lof_overlap_points",
    ]
    for column in zero_when_no_event_columns:
        if column not in event_summary.columns:
            event_summary[column] = 0
        event_summary[column] = event_summary[column].fillna(0)
    int_event_columns = [
        "point_anomaly_count",
        "event_count",
        "persistent_event_points",
        "baseline_overlap_points",
        "lof_overlap_points",
    ]
    event_summary[int_event_columns] = event_summary[int_event_columns].astype(int)
    event_summary["event_point_capture_pct"] = (
        100
        * event_summary["persistent_event_points"]
        / event_summary["point_anomaly_count"].replace(0, np.nan)
    ).fillna(0)
    event_summary = event_summary[
        [
            "operating_period",
            "point_anomaly_count",
            "point_anomaly_rate_pct",
            "event_count",
            "persistent_event_points",
            "event_point_capture_pct",
            "mean_duration_minutes",
            "max_duration_minutes",
            "mean_anomaly_points",
            "max_anomaly_score",
            "baseline_overlap_points",
            "lof_overlap_points",
        ]
    ]

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
    for column in baseline_method_label_columns:
        method_name = column.removeprefix("baseline_").removesuffix("_label")
        overlap, union, jaccard = _jaccard(results["period_if_anomaly_label"], results[column])
        comparisons.append(
            {
                "comparison": f"statistical_baseline_v3_{method_name}",
                "if_anomaly_count": int(results["period_if_anomaly_label"].sum()),
                "comparison_anomaly_count": int(results[column].sum()),
                "overlap_count": overlap,
                "union_count": union,
                "jaccard_overlap": jaccard,
                "note": "Method-specific statistical baseline reference, not ground truth",
            }
        )

    return {
        "period_summary": period_summary,
        "events": events,
        "event_summary": event_summary,
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
        ("expected_row_count", len(results) == 362_880, f"rows={len(results):,}"),
        ("unique_row_key", not results.duplicated(KEY_COLUMNS).any(), "timestamp + floor + zone"),
        ("six_period_models", set(models) == set(PERIOD_ORDER), f"models={sorted(models)}"),
        ("six_physical_features", len(MODEL_FEATURES) == 6, f"features={MODEL_FEATURES}"),
        ("scores_complete", not results["period_if_anomaly_score"].isna().any(), "missing=0"),
        (
            "auto_decision_threshold_consistent",
            np.array_equal(
                results["period_if_auto_anomaly_label"].to_numpy(dtype=bool),
                results["period_if_decision_function"].lt(0).to_numpy(),
            ),
            "auto label=1 exactly when decision_function < 0",
        ),
        (
            "calibrated_period_rate",
            np.allclose(
                summaries["period_summary"]["anomaly_rate_pct"].to_numpy(dtype=float),
                100 * IF_CALIBRATED_ANOMALY_RATE,
                atol=0.01,
            ),
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
        (
            "event_table_available",
            "events" in summaries and "event_summary" in summaries,
            f"events={len(summaries.get('events', [])):,}",
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
    event_path = OUTPUT_DIR / OUTPUT_FILES["events"]

    results.to_csv(full_path, index=False, compression="gzip")
    anomalies = (
        results.loc[results["period_if_anomaly_label"].eq(1)]
        .sort_values(["operating_period", "period_if_anomaly_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    anomalies.to_csv(anomaly_path, index=False, compression="gzip")
    summaries["events"].to_csv(event_path, index=False)
    summaries["period_summary"].to_csv(OUTPUT_DIR / OUTPUT_FILES["period_summary"], index=False)
    summaries["event_summary"].to_csv(OUTPUT_DIR / OUTPUT_FILES["event_summary"], index=False)
    summaries["zone_summary"].to_csv(OUTPUT_DIR / OUTPUT_FILES["zone_summary"], index=False)
    summaries["hour_summary"].to_csv(OUTPUT_DIR / OUTPUT_FILES["hour_summary"], index=False)
    summaries["comparison"].to_csv(OUTPUT_DIR / OUTPUT_FILES["comparison"], index=False)

    summary_payload: dict[str, Any] = {
        "author": "huaye",
        "analysis": "period-specific Isolation Forest with calibrated period thresholds and event aggregation",
        "source_dataset": str(ML_READY_PATH.relative_to(PROJECT_DIR)),
        "physical_reference_dataset": str(PHYSICAL_REFERENCE_PATH.relative_to(PROJECT_DIR)),
        "lof_reference_dataset": str(LOF_RESULTS_PATH.relative_to(PROJECT_DIR)),
        "lof_reference_missing_rows": int(results["lof_reference_missing"].sum()),
        "feature_manifest": str(FEATURE_MANIFEST_PATH.relative_to(PROJECT_DIR)),
        "row_count": int(len(results)),
        "operating_periods": PERIOD_ORDER,
        "model_count": len(PERIOD_ORDER),
        "model_features": MODEL_FEATURES,
        "rolling_reference_missing_values": int(results[ROLLING_COLUMNS].isna().sum().sum()),
        "baseline_reference_columns": [
            column
            for column in results.columns
            if column.startswith("baseline_")
            or column in {"baseline_anomaly_methods", "baseline_anomaly_variables", "baseline_anomaly_directions"}
        ],
        "baseline_method_label_columns": [
            column
            for column in results.columns
            if column.startswith("baseline_")
            and column.endswith("_label")
            and column != "baseline_anomaly_label"
        ],
        "feature_count": len(MODEL_FEATURES),
        "model": "sklearn.ensemble.IsolationForest",
        "model_parameters": MODEL_PARAMETERS,
        "calibrated_anomaly_rate": IF_CALIBRATED_ANOMALY_RATE,
        "event_min_duration_minutes": IF_EVENT_MIN_DURATION_MINUTES,
        "event_max_gap_minutes": IF_EVENT_MAX_GAP_MINUTES,
        "label_definition": (
            "period_if_anomaly_label = 1 for the top calibrated anomaly scores within each operating period; "
            "period_if_auto_anomaly_label retains sklearn's decision_function < 0 threshold for comparison"
        ),
        "score_definition": "period_if_anomaly_score = -model.score_samples(X_period); higher means more anomalous within the same period",
        "threshold_definition": (
            f"Final labels use the top {IF_CALIBRATED_ANOMALY_RATE:.1%} scores inside each operating period; "
            "sklearn contamination='auto' is retained only for the auto-label reference."
        ),
        "anomaly_count": int(results["period_if_anomaly_label"].sum()),
        "anomaly_rate_pct": float(100 * results["period_if_anomaly_label"].mean()),
        "auto_anomaly_count": int(results["period_if_auto_anomaly_label"].sum()),
        "auto_anomaly_rate_pct": float(100 * results["period_if_auto_anomaly_label"].mean()),
        "event_count": int(len(summaries["events"])),
        "period_results": summaries["period_summary"].to_dict(orient="records"),
        "event_results": summaries["event_summary"].to_dict(orient="records"),
        "comparison_limitations": [
            "LOF retains its existing fixed contamination 0.05 label.",
            "The statistical baseline is a comparison reference, not ground truth.",
            "Detected rows are candidate anomalies, not confirmed equipment faults.",
            "Raw scores from separately fitted period models should not be compared directly across periods.",
            "Event rows are persistence-filtered clusters of calibrated point anomalies.",
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
    reloaded_events = pd.read_csv(event_path)
    if len(reloaded_events) != len(summaries["events"]):
        raise AssertionError("Saved event output has an unexpected row count")

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
    print("Period-specific calibrated-threshold Isolation Forest complete")
    print(generated_summaries["period_summary"].to_string(index=False))
    print(
        f"Total candidate anomalies: {generated_summary['anomaly_count']:,} "
        f"({generated_summary['anomaly_rate_pct']:.4f}%)"
    )
    print(f"Persistent IF events: {generated_summary['event_count']:,}")
    print("Generated outputs:")
    for output_name in OUTPUT_FILES.values():
        print(f"- {OUTPUT_DIR / output_name}")
